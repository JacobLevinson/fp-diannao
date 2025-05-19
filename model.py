#!/usr/bin/env python3
"""
model_enhanced.py — Mechanistic performance model with SPAD/global‑mem branch

This model predicts conv2d timings using:
  • Compute roof (T_comp)
  • DRAM-bound roofline (T_dram)
  • Shared‑mem roofline (T_spad) if tile fits
  • Launch overhead (T_launch)
  • Memory‑latency penalty (T_lat)

Decision:
  If shared_bytes ≤ SPAD_capacity, use T_spad; else treat as global mem (no SPAD reuse).

Report: min/median/max error cases and summary stats.
"""
import csv, math

# ───── Hardware specs ─────
PEAK_FP_nom      = 13.8e12      # FP/s Titan V
BW_DRAM_nom      = 652e9        # B/s DRAM
BW_SPAD_nom      = 1.5e12       # B/s shared mem
SPAD_CAP         = 48 * 1024    # bytes per block (e.g. 48 KiB)
BYTE_PER_ELT     = 4            # bytes per float
LAUNCH_OVERHEAD  = 6e-6         # s kernel launch overhead
BLOCK_LATENCY    = 0.2e-6       # s per-block dispatch
LATENCY_CYCLES   = 200          # DRAM latency cycles
CORE_CLOCK       = 1455e6       # Hz
# occupancy parameters
MAX_THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM  = 32
FMA_LATENCY_CYCLES = 32


def compute_times(row):
    # parse inputs
    Nx, Ny = int(row['Nx']), int(row['Ny'])
    Ni, Nn = int(row['Ni']), int(row['Nn'])
    Kx, Ky = int(row['Kx']), int(row['Ky'])
    Tx, Ty = int(row['Tx']), int(row['Ty'])
    Tn, Ti = int(row['Tn']), int(row['Ti'])

    # total flops
    flops = 2 * Nx * Ny * Ni * Nn * Kx * Ky

    # compute roof
    threads_per_block = Tx * Ty
    occ = min(1.0, (threads_per_block * MAX_BLOCKS_PER_SM) / MAX_THREADS_PER_SM)
    warps_per_block = threads_per_block / 32
    warps_per_sm    = warps_per_block * MAX_BLOCKS_PER_SM
    warp_hiding     = min(1.0, warps_per_sm / FMA_LATENCY_CYCLES)
    eff_fp = PEAK_FP_nom * occ * warp_hiding
    T_comp = flops / eff_fp

    # grid dimensions
    NBx = math.ceil(Nx / Tx)
    NBy = math.ceil(Ny / Ty)
    NBn = math.ceil(Nn / Tn)
    NB_blocks = NBx * NBy * NBn

    # DRAM traffic
    # activations: each patch (padX×padY×Ni) fetched per spatial block
    padX = Tx + Kx - 1
    padY = Ty + Ky - 1
    total_act_bytes = NBx * NBy * padX * padY * Ni * BYTE_PER_ELT
    # weights: NBi * NBn distinct tiles
    NBi = math.ceil(Ni / Ti)
    wt_tile_bytes = Kx * Ky * Tn * Ti * BYTE_PER_ELT
    weight_tiles = NBi * NBn
    # L2 reuse: if weight_tiles * wt_tile_bytes < L2 capacity, single load
    # approximate L2 cap >> footprints, assume reuse; else full streams
    total_wt_bytes = weight_tiles * wt_tile_bytes
    total_dram_bytes = total_act_bytes + total_wt_bytes
    T_dram = total_dram_bytes / BW_DRAM_nom

    # SPAD traffic and decision
    sm_bytes = (padX * padY * Ti + Kx * Ky * Ti * Tn) * BYTE_PER_ELT
    if sm_bytes <= SPAD_CAP:
        # each block loads its tile once
        T_spad = (NB_blocks * sm_bytes) / BW_SPAD_nom
        use_spad = True
    else:
        T_spad = float('inf')
        use_spad = False

    # launch and latency
    T_launch = NB_blocks * BLOCK_LATENCY
    T_lat = 0.0
    if warps_per_sm < FMA_LATENCY_CYCLES:
        T_lat = (NB_blocks * LATENCY_CYCLES) / CORE_CLOCK

    # final predicted time with roofline over chosen data path
    mem_bound = T_spad if use_spad else T_dram
    T_pred = LAUNCH_OVERHEAD + T_launch + max(T_comp, mem_bound, T_lat)

    # return all terms in ms
    return (T_comp*1e3, T_dram*1e3, T_spad*1e3,
            T_launch*1e3, T_lat*1e3, T_pred*1e3, use_spad)


def main():
    data = list(csv.DictReader(open('sweep_results.csv')))
    results = []
    for row in data:
        t_comp, t_dram, t_spad, t_launch, t_lat, t_pred, use_spad = compute_times(row)
        actual = float(row['time_ms'])
        err    = (t_pred - actual) / actual * 100
        results.append({'row':row,'actual':actual,
                        't_comp':t_comp,'t_dram':t_dram,'t_spad':t_spad,
                        't_launch':t_launch,'t_lat':t_lat,
                        'pred':t_pred,'err':err,'abs':abs(err),
                        'use_spad':use_spad})
    n = len(results)
    mean_err = sum(r['err'] for r in results)/n
    mean_abs = sum(r['abs'] for r in results)/n
    sorted_r  = sorted(results, key=lambda r: r['abs'])
    min_r, med_r, max_r = sorted_r[0], sorted_r[n//2], sorted_r[-1]

    def print_case(label, rec):
        r = rec['row']
        print(f"{label} abs% err = {rec['abs']:.2f}%  (" +
              ("SPAD" if rec['use_spad'] else "DRAM") + ")")
        print(f"  params: Tx={r['Tx']} Ty={r['Ty']} Tn={r['Tn']} Ti={r['Ti']} " +
              f"| Nx={r['Nx']} Ny={r['Ny']} Ni={r['Ni']} Nn={r['Nn']}")
        print(f"  actual   = {rec['actual']:.3f} ms")
        print(f"    t_comp   = {rec['t_comp']:.3f} ms" +
              f" | t_dram = {rec['t_dram']:.3f} ms" +
              f" | t_spad = {rec['t_spad']:.3f} ms")
        print(f"    t_launch = {rec['t_launch']:.3f} ms" +
              f" | t_lat  = {rec['t_lat']:.3f} ms")
        print(f"  pred     = {rec['pred']:.3f} ms\n")

    print(f"Points eval’d    : {n}")
    print(f"Mean % error     : {mean_err:.2f}%")
    print(f"Mean abs % error : {mean_abs:.2f}%\n")
    print_case("Min   ", min_r)
    print_case("Median", med_r)
    print_case("Max   ", max_r)

    # write detailed CSV
    with open('model_predictions.csv','w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Tx','Ty','Tn','Ti','Nx','Ny','Ni','Nn',
                    't_comp','t_dram','t_spad','t_launch','t_lat','t_pred','err%','use_spad'])
        for rec in results:
            r = rec['row']
            w.writerow([r['Tx'],r['Ty'],r['Tn'],r['Ti'],
                        r['Nx'],r['Ny'],r['Ni'],r['Nn'],
                        f"{rec['t_comp']:.6f}",
                        f"{rec['t_dram']:.6f}",
                        f"{rec['t_spad']:.6f}",
                        f"{rec['t_launch']:.6f}",
                        f"{rec['t_lat']:.6f}",
                        f"{rec['pred']:.6f}",
                        f"{rec['err']:.2f}",
                        rec['use_spad']])
    print("Wrote detailed predictions to model_predictions.csv")

if __name__=='__main__':
    main()
