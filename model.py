#!/usr/bin/env python3
"""
model_enhanced.py — Mechanistic performance model with SPAD & classic DRAM roofline

This model predicts conv2d timings combining:
  • Compute-bound roof (T_comp)
  • DRAM-bound roofline (T_dram) using fixed intensity (0.25 FLOP/B)
  • Shared-mem (SPAD) path with:
      – per-SM resource limits (threads + SPAD capacity)
      – wave-based DRAM loads
      – measured SPAD inefficiency factor on throughput
      – per-block shared-mem latency overhead
  • Launch overhead (T_launch)
  • Memory-latency penalty (T_lat)

Decision:
  Use SPAD path when tile fits in SPAD; otherwise DRAM-only path.

Outputs min/median/max error cases and summary stats.
"""
import csv, math

# ───── Hardware specs ─────
PEAK_FP_nom             = 13.8e12    # FP/s Titan V
BW_DRAM_nom             = 652e9      # B/s DRAM peak
BW_SPAD_nom             = 1.5e12     # B/s shared mem peak
BYTE_PER_ELT            = 4          # bytes per float
LAUNCH_OVERHEAD         = 5e-6       # s kernel launch overhead (classic model)
BLOCK_LATENCY           = 0.2e-6     # s per-block dispatch
SPAD_LATENCY_PER_BLOCK  = 0.5e-6     # s per-block shared-mem overhead
LATENCY_CYCLES          = 200        # DRAM latency cycles
CORE_CLOCK              = 1455e6     # Hz
MAX_THREADS_PER_SM      = 2048
MAX_BLOCKS_PER_SM       = 32
FMA_LATENCY_CYCLES      = 32
SPAD_CAP                = 48 * 1024  # bytes per block (48 KiB)
NUM_SMS                 = 80         # number of SMs on Titan V

# ───── SPAD inefficiency from nvprof ─────
# measured shared_store_throughput ≈ 120 GB/s at warp_hiding≈0.3027
SPAD_STORE_EFF          = 120.41e9 / (BW_SPAD_nom * 0.3027)

MEAS_DRAM_READ_BW  = 5.26e9   # from nvprof
MEAS_DRAM_WRITE_BW = 4.46e9   # from nvprof


def compute_times(row):
    # Parse inputs
    Nx, Ny = int(row['Nx']), int(row['Ny'])
    Ni, Nn = int(row['Ni']), int(row['Nn'])
    Kx, Ky = int(row['Kx']), int(row['Ky'])
    Tx, Ty = int(row['Tx']), int(row['Ty'])
    Tn, Ti = int(row['Tn']), int(row['Ti'])

    # Total operations and outputs
    flops = 2 * Nx * Ny * Ni * Nn * Kx * Ky
    Nout  = Nx * Ny * Nn

    # Grid tiling
    NBx = math.ceil(Nx / Tx)
    NBy = math.ceil(Ny / Ty)
    NBn = math.ceil(Nn / Tn)
    NB_blocks = NBx * NBy * NBn

    # Threads & warps per block
    threads_per_block = Tx * Ty
    warps_per_block   = threads_per_block / 32

    # SPAD footprint per block
    padX = Tx + Kx - 1
    padY = Ty + Ky - 1
    sm_act_bytes = padX * padY * Ti * BYTE_PER_ELT
    sm_wt_bytes  = Kx * Ky * Ti * BYTE_PER_ELT * Tn
    sm_bytes     = sm_act_bytes + sm_wt_bytes

    # Blocks/SM resource limits
    blocks_thr  = min(MAX_BLOCKS_PER_SM, MAX_THREADS_PER_SM // threads_per_block)
    blocks_spad = min(MAX_BLOCKS_PER_SM, SPAD_CAP // sm_bytes)

    # Choose path
    use_spad = (sm_bytes <= SPAD_CAP)
    blocks_per_sm = min(blocks_thr, blocks_spad) if use_spad else blocks_thr

    # Occupancy & warp-hiding
    threads_active_sm = threads_per_block * blocks_per_sm
    occupancy        = min(1.0, threads_active_sm / MAX_THREADS_PER_SM)
    warps_active_sm  = warps_per_block * blocks_per_sm
    warp_hiding      = min(1.0, warps_active_sm / FMA_LATENCY_CYCLES)

    # Compute time
    eff_fp = PEAK_FP_nom * occupancy * warp_hiding
    T_comp = flops / eff_fp

    # Fixed-intensity DRAM roofline (0.25 FLOP/B → 4 B/F)
    T_dram = flops * BYTE_PER_ELT / BW_DRAM_nom  # bytes=flops*4

    # SPAD path transfers
    if use_spad:
        # SPAD roofline
        eff_spad_bw = BW_SPAD_nom * warp_hiding * SPAD_STORE_EFF
        T_spad = NB_blocks * sm_bytes / eff_spad_bw
        T_spad += NB_blocks * SPAD_LATENCY_PER_BLOCK

        # Wave-based DRAM load using spec BW
        blocks_per_wave = blocks_per_sm * NUM_SMS
        waves = math.ceil(NB_blocks / blocks_per_wave)
        bytes_per_wave = min(blocks_per_wave, NB_blocks) * sm_bytes
      
        T_read_spad  = waves * (bytes_per_wave  / (MEAS_DRAM_READ_BW  * warp_hiding))
        T_write_spad = (Nout * BYTE_PER_ELT) / (MEAS_DRAM_WRITE_BW * warp_hiding)
        mem_bound    = max(T_spad, T_read_spad) + T_write_spad
    else:
        T_spad = float('inf')
        mem_bound = T_dram

    # Final prediction: classic roofline + launch + latency
    T_pred = LAUNCH_OVERHEAD + max(T_comp, mem_bound)

    # return ms
    return T_comp*1e3, T_dram*1e3, T_spad*1e3, T_pred*1e3, use_spad


def main():
    data = list(csv.DictReader(open('sweep_results.csv')))
    results = []
    for row in data:
        t_comp, t_dram, t_spad, t_pred, use_spad = compute_times(row)
        actual = float(row['time_ms'])
        err    = (t_pred - actual) / actual * 100
        results.append({ 'row': row, 'actual': actual,
                         't_comp': t_comp, 't_dram': t_dram, 't_spad': t_spad,
                         'pred': t_pred, 'err': err, 'abs': abs(err),
                         'use_spad': use_spad })

    # Summary
    n        = len(results)
    mean_err = sum(r['err'] for r in results)/n
    mean_abs = sum(r['abs'] for r in results)/n
    sorted_r = sorted(results, key=lambda r: r['abs'])
    min_r, med_r, max_r = sorted_r[0], sorted_r[n//2], sorted_r[-1]

    print(f"Points eval’d    : {n}")
    print(f"Mean % error     : {mean_err:.2f}%")
    print(f"Mean abs % error : {mean_abs:.2f}%\n")
    def print_case(label, rec):
        r=rec['row']; path = 'SPAD' if rec['use_spad'] else 'DRAM'
        print(f"{label} abs% err = {rec['abs']:.2f}% ({path})")
        print(f"  Params: Tx={r['Tx']} Ty={r['Ty']} Tn={r['Tn']} Ti={r['Ti']} | Nx={r['Nx']} Ny={r['Ny']} Ni={r['Ni']} Nn={r['Nn']}")
        print(f"  actual   = {rec['actual']:.3f} ms")
        print(f"    T_comp = {rec['t_comp']:.3f} ms | T_dram = {rec['t_dram']:.3f} ms | T_spad = {rec['t_spad']:.3f} ms")
        print(f"  pred     = {rec['pred']:.3f} ms\n")
    print_case("Min   ", min_r)
    print_case("Median", med_r)
    print_case("Max   ", max_r)

    # CSV output
    with open('model_predictions.csv','w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['Tx','Ty','Tn','Ti','Nx','Ny','Ni','Nn','t_comp','t_dram','t_spad','t_pred','err%','use_spad'])
        for rec in results:
            r=rec['row']; w.writerow([r['Tx'],r['Ty'],r['Tn'],r['Ti'],r['Nx'],r['Ny'],r['Ni'],r['Nn'],
                                      f"{rec['t_comp']:.6f}",f"{rec['t_dram']:.6f}",f"{rec['t_spad']:.6f}",
                                      f"{rec['pred']:.6f}",f"{rec['err']:.2f}",rec['use_spad']])
    print("Wrote detailed predictions to model_predictions.csv")

if __name__=='__main__':
    main()