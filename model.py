#!/usr/bin/env python3
"""
model.py — Memory-bound fixed-intensity roofline model with error stats

This model targets the optimized conv2d kernel which:
  • Does no spatial reuse in software (no shared-memory tiling),
    so its arithmetic intensity is fixed at 0.25 FLOP/Byte.
      - Each output element computes Ni*Kx*Ky multiplies + adds ⇒ 2·Ni·Kx·Ky FLOPs
      - And loads Ni·Kx·Ky weights + Ni·Kx·Ky activations = 2·Ni·Kx·Ky elements
        at 4 bytes each ⇒ 8·Ni·Kx·Ky bytes
      - Arithmetic intensity = (2·Ni·Kx·Ky FLOPs) / (8·Ni·Kx·Ky bytes)
                           = 0.25 FLOP/Byte
  • Is purely memory-bound: DRAM traffic (bytes) dominates compute.
  • Still honors occupancy and warp-based latency hiding for completeness,
    but those only affect the tiny compute term (<1% of total).

Model formulas:
    flops   = 2 * Nx * Ny * Ni * Nn * Kx * Ky
      # total FP operations

    T_comp  = flops / (PEAK_FP_nominal × occupancy × warp_hiding)
      # time to compute all FLOPs

    T_dram  = flops * BYTE_PER_ELT / BW_DRAM_nominal
      # time to transfer all required bytes from DRAM
      # since bytes = flops × (1 / 0.25 FLOP/B) = flops × 4 B/F

    T_pred  = LAUNCH_OVERHEAD + max(T_comp, T_dram)
      # roofline: the slowest of compute vs memory-bound, plus launch latency
"""

import csv

# ───── Hardware specs ─────
PEAK_FP_nom        = 13.8e12    # peak single-precision FLOP/s (Titan V)
BW_DRAM_nom        = 652e9      # peak DRAM bandwidth in bytes/s
BYTE_PER_ELT       = 4          # bytes per float element

# Occupancy / latency-hiding parameters (for compute term)
MAX_THREADS_PER_SM = 2048       # threads per SM
MAX_BLOCKS_PER_SM  = 32         # blocks per SM
FMA_LATENCY_CYCLES = 32         # FMA pipeline latency in cycles
CORE_CLOCK         = 1455e6     # Hz (if more precise warp_hiding needed)

# Kernel launch overhead (measured)
LAUNCH_OVERHEAD    = 5e-6       # seconds

def compute_times(row):
    # ─── parse kernel/prob parameters ───
    Nx, Ny = int(row['Nx']), int(row['Ny'])
    Ni, Nn = int(row['Ni']), int(row['Nn'])
    Kx, Ky = int(row['Kx']), int(row['Ky'])
    Tx, Ty = int(row['Tx']), int(row['Ty'])
    Tn, Ti = int(row['Tn']), int(row['Ti'])

    # ─── compute total FLOPs ───
    # each output does Ni*Kx*Ky multiplies + Ni*Kx*Ky adds = 2·Ni·Kx·Ky
    flops = 2 * Nx * Ny * Ni * Nn * Kx * Ky

    # ─── occupancy factor ───
    # how many threads per block, scaled by SM capacity
    threads_per_block = Tx * Ty * Tn
    occ = min(1.0,
              (threads_per_block * MAX_BLOCKS_PER_SM) /
              MAX_THREADS_PER_SM)

    # ─── warp‐based latency hiding ───
    # need enough warps to hide FMA latency
    warps_per_block = threads_per_block / 32
    warps_per_SM    = warps_per_block * MAX_BLOCKS_PER_SM
    warp_hiding     = min(1.0,
                         warps_per_SM / FMA_LATENCY_CYCLES)

    # ─── effective compute throughput (FLOP/s) ───
    eff_fp = PEAK_FP_nom * occ * warp_hiding

    # ─── effective memory throughput (B/s) ───
    # kernel is memory‐bound, assume full DRAM saturation
    eff_bw = BW_DRAM_nom

    # ─── time components (seconds) ───
    T_comp = flops / eff_fp
    # since arithmetic intensity is 0.25 F/B, bytes = flops × 4
    T_dram = flops * BYTE_PER_ELT / eff_bw

    # ─── final predicted time ───
    T_pred = LAUNCH_OVERHEAD + max(T_comp, T_dram)

    # return all timings in milliseconds
    return T_comp*1e3, T_dram*1e3, T_pred*1e3

def main():
    input_csv = 'sweep_results.csv'
    data = list(csv.DictReader(open(input_csv)))
    results = []

    # ─── evaluate model on each sweep point ───
    for row in data:
        t_comp, t_dram, t_pred = compute_times(row)
        actual = float(row['time_ms'])
        error  = (t_pred - actual) / actual * 100
        results.append({
            'row': row,
            'actual':      actual,
            'pred':        t_pred,
            'error':       error,
            'abs_error':   abs(error),
            't_comp':      t_comp,
            't_dram':      t_dram
        })

    # ─── compute summary statistics ───
    n        = len(results)
    mean_err = sum(r['error'] for r in results) / n
    mean_abs = sum(r['abs_error'] for r in results) / n

    # sort by absolute error for min/median/max cases
    sorted_results = sorted(results, key=lambda r: r['abs_error'])
    min_rec = sorted_results[0]
    med_rec = sorted_results[n//2]
    max_rec = sorted_results[-1]

    # ─── print high-level summary ───
    print(f"Points evaluated : {n}")
    print(f"Mean error       : {mean_err:.2f}%")
    print(f"Mean abs % error : {mean_abs:.2f}%\n")

    # helper to print detailed case breakdown
    def print_case(label, rec):
        r = rec['row']
        print(f"{label} abs error: {rec['abs_error']:.2f}%")
        print(f"  Params: Tx={r['Tx']}, Ty={r['Ty']}, Tn={r['Tn']}, Ti={r['Ti']}, "
              f"Nx={r['Nx']}, Ny={r['Ny']}, Ni={r['Ni']}, Nn={r['Nn']}")
        print(f"  actual    = {rec['actual']:.3f} ms")
        print(f"  predicted = {rec['pred']:.3f} ms")
        print(f"    breakdown: T_comp = {rec['t_comp']:.3f} ms, "
              f"T_dram = {rec['t_dram']:.3f} ms\n")

    # ─── print min, median, max error cases ───
    print_case("Min   ",    min_rec)
    print_case("Median",    med_rec)
    print_case("Max   ",    max_rec)

    # ─── save detailed CSV of predictions ───
    output_csv = 'model_predictions.csv'
    with open(output_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow([
            'Tx','Ty','Tn','Ti',
            'Nx','Ny','Ni','Nn','Kx','Ky',
            'actual_ms','t_comp_ms','t_dram_ms','pred_ms','error_pct'
        ])
        for rec in results:
            r = rec['row']
            writer.writerow([
                r['Tx'], r['Ty'], r['Tn'], r['Ti'],
                r['Nx'], r['Ny'], r['Ni'], r['Nn'],
                r['Kx'], r['Ky'],
                f"{rec['actual']:.6f}",
                f"{rec['t_comp']:.6f}",
                f"{rec['t_dram']:.6f}",
                f"{rec['pred']:.6f}",
                f"{rec['error']:.2f}"
            ])

    print(f"Detailed results saved to {output_csv}")

if __name__ == '__main__':
    main()
