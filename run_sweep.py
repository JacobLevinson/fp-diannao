#!/usr/bin/env python3
"""
Sweep driver   —   (Tx=Ty × Tn × Ti × problem sizes)  →  CSV
─────────────────────────────────────────────────────────
* For each square tile (Tx) and channel tiles (Tn,Ti) ➜ compile conv2d
* For each problem shape                              ➜ run, log time
"""

import csv, itertools, subprocess, re, pathlib, shutil, sys

# ────────── sweep configuration ──────────
square_tiles = [8, 16, 32]          # Tx = Ty  ⇒ 3 spatial-tile sizes
tile_Tn      = [4, 8, 16]          # output-channel tile sizes
tile_Ti      = [4, 8, 16]              # input-channel  tile sizes

Nx_list      = [224, 112, 56, 28]   # ← dropped 14 to shorten sweep
Ny_list      = Nx_list              # keep square inputs
NiNn         = [(64, 64), (128, 128), (256, 256)]   # dropped 512×512
KxKy         = (3, 3)               # kernel size

CSV_PATH = pathlib.Path("sweep_results.csv")
NVCC     = shutil.which("nvcc") or "nvcc"
ARCHFLAG = "-arch=sm_70"            # change if needed
time_rx  = re.compile(r"Kernel avg time\s*:\s*([\d.]+)")

# ────────── helpers ──────────
def nvcc_compile(Tx, Tn, Ti):
    macros = f"-DTx={Tx} -DTy={Tx} -DTn={Tn} -DTi={Ti}"
    cmd    = f"{NVCC} conv2d.cu -o conv2d {ARCHFLAG} -O3 {macros}"
    print(f"\n🛠  {cmd}")
    return subprocess.call(cmd.split()) == 0

def run_one(args):
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        print("  ! conv2d failed"); sys.stderr.write(proc.stderr); return None
    m = time_rx.search(proc.stdout);  return float(m.group(1)) if m else None

# ────────── CSV setup ──────────
with CSV_PATH.open("w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(["Tx","Ty","Tn","Ti",
                     "Nx","Ny","Ni","Nn","Kx","Ky","time_ms"])

    # ────────── sweep ──────────
    for Tx, Tn, Ti in itertools.product(square_tiles, tile_Tn, tile_Ti):

        if not nvcc_compile(Tx, Tn, Ti):
            print("✖︎ compile failed — skipping")
            continue

        for Nx, Ny in itertools.product(Nx_list, Ny_list):
            for Ni, Nn in NiNn:
                cmd = ["./conv2d", str(Nx), str(Ny),
                       str(Ni), str(Nn), str(KxKy[0]), str(KxKy[1])]
                print("⇢", " ".join(cmd))
                t_ms = run_one(cmd)
                if t_ms is None: continue
                writer.writerow([Tx, Tx, Tn, Ti,
                                 Nx, Ny, Ni, Nn, *KxKy,
                                 f"{t_ms:.6f}"])
                print(f"  ↳ {t_ms:.3f} ms")

print(f"\n✅  sweep complete — results in {CSV_PATH.resolve()}")
