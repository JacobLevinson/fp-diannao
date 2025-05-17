#!/usr/bin/env python3
"""
calibrate.py — Offline calibration of effective roofline constants
Reads sweep_results.csv and prints out median effective compute & DRAM rates.
"""

import csv
from statistics import median

# ─────────── Nominal hardware parameters ───────────
PEAK_FP_nom = 13.8e12  # TFLOP/s
BW_DRAM_nom = 652e9    # GB/s
BYTE_PER_ELT = 4

def classify_and_collect():
    comp_rates = []
    mem_rates = []
    with open('sweep_results.csv', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # parse row
            Nx,Ny,Ni,Nn,Kx,Ky = (int(row[c]) for c in ['Nx','Ny','Ni','Nn','Kx','Ky'])
            actual_ms = float(row['time_ms'])
            actual_s = actual_ms * 1e-3

            # compute FLOPs and DRAM bytes
            flops = 2 * Nx * Ny * Ni * Nn * Kx * Ky
            bytes_acts = (Nx + Kx)*(Ny + Ky)*Ni * BYTE_PER_ELT
            bytes_wt   = Kx * Ky * Ni * Nn    * BYTE_PER_ELT
            bytes_total = bytes_acts + bytes_wt

            # nominal times
            T_comp_nom = flops / PEAK_FP_nom
            T_mem_nom  = bytes_total / BW_DRAM_nom

            # classify and record effective rates
            if T_comp_nom >= T_mem_nom:
                # compute-bound
                comp_rates.append(flops / actual_s)
            else:
                # memory-bound
                mem_rates.append(bytes_total / actual_s)

    return comp_rates, mem_rates

def main():
    comp_rates, mem_rates = classify_and_collect()
    eff_peak_fp = median(comp_rates) if comp_rates else PEAK_FP_nom
    eff_bw_dram = median(mem_rates)  if mem_rates  else BW_DRAM_nom

    print("# Calibration results:")
    print(f"PEAK_FP_eff = {eff_peak_fp:.6e}  # effective GFLOP/s")
    print(f"BW_DRAM_eff = {eff_bw_dram:.6e}  # effective B/s")

if __name__ == '__main__':
    main()

# -----------------------------------------
# Usage:
#   python3 calibrate.py > calibration.txt
# Then copy the printed values into model.py constants.

