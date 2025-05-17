#!/usr/bin/env python3
"""
Analytical performance model for a single-precision 2-D convolution kernel.

* Everything is parameterized: problem size, tile size, hardware limits.
* For now we implement:
    – FLOP count
    – Nominal DRAM bytes (1 load / 1 store per element)
    – Compute-bound time  (FLOP / peak_flops)
    – DRAM-bound time     (bytes / dram_bw)
    – Predicted runtime   = max(compute, dram) + launch_overhead
* Later you can add L2/L1 stages, latency penalties, etc.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import json
import math


# ──────────────────────────────────────────────────────────────
# 1.  Data classes for clean parameter passing
# ──────────────────────────────────────────────────────────────

@dataclass
class Problem:
    Nx: int; Ny: int         # spatial size
    Ni: int; Nn: int         # channels   (in/out)
    Kx: int = 3; Ky: int = 3 # kernel
    Sx: int = 1; Sy: int = 1 # stride

@dataclass
class Tile:
    Tx: int; Ty: int
    Ti: int; Tn: int

@dataclass
class Hardware:
    peak_flops: float = 13.8e12
    dram_bw:    float = 652e9
    launch_us:  float = 5.0
    dtype_bytes:int   = 4
    util_compute: float = 0.75     # 75 % of peak
    traffic_mul: float = 3.0       # see text

# ──────────────────────────────────────────────────────────────
# 2.  Core model
# ──────────────────────────────────────────────────────────────

class ConvModel:
    def __init__(self, prob: Problem, tile: Tile, hw: Hardware):
        self.p = prob
        self.t = tile
        self.hw = hw

    # ---------- basic geometry ----------

    @property
    def flops(self) -> int:
        p = self.p
        # 2 = FMA   (mul + add)  per overlapping window
        return p.Ny * p.Nx * p.Nn * 2 * p.Ky * p.Kx * p.Ni

    @property
    def bytes_dram(self) -> int:
        """Naïve traffic (no reuse counted)."""
        p, B = self.p, self.hw.dtype_bytes
        weights = p.Ky * p.Kx * p.Ni * p.Nn * B
        activ_in  = (p.Ny + p.Ky) * (p.Nx + p.Kx) * p.Ni * B
        activ_out = p.Ny * p.Nx * p.Nn * B
        return weights + activ_in + activ_out

    # ---------- simple roofline-style latency ----------

    @property
    def t_compute(self) -> float:
        return self.flops / (self.hw.peak_flops * self.hw.util_compute)

    @property
    def t_dram(self) -> float:
        bytes_eff = self.bytes_dram * self.hw.traffic_mul
        return bytes_eff / self.hw.dram_bw

    # ---------- predicted total runtime ----------

    def runtime_pred_ms(self) -> float:
        base_s = max(self.t_compute, self.t_dram)
        total_s = base_s + self.hw.launch_us * 1e-6
        return total_s * 1e3

    # ---------- convenience dump ----------

    def dict(self) -> Dict[str, Any]:
        d = asdict(self.p) | asdict(self.t)
        d.update({
            "flops": self.flops,
            "bytes_dram": self.bytes_dram,
            "t_compute_ms": self.t_compute * 1e3,
            "t_dram_ms":    self.t_dram    * 1e3,
            "pred_ms":      self.runtime_pred_ms()
        })
        return d


# ──────────────────────────────────────────────────────────────
# 3.  Quick demo / CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, textwrap, csv, sys

    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        Simple test run:
            $ python model.py --meas_ms 19.0

        Pass  --json  to dump all intermediate numbers.
        """))

    ap.add_argument("--Nx", type=int, default=224)
    ap.add_argument("--Ny", type=int, default=224)
    ap.add_argument("--Ni", type=int, default=64)
    ap.add_argument("--Nn", type=int, default=64)
    ap.add_argument("--Tx", type=int, default=7)
    ap.add_argument("--Ty", type=int, default=7)
    ap.add_argument("--Ti", type=int, default=16)
    ap.add_argument("--Tn", type=int, default=16)
    ap.add_argument("--meas_ms", type=float, help="Measured kernel time to compare against")
    ap.add_argument("--json", action="store_true", help="Print full JSON dump")

    args = ap.parse_args()

    prob = Problem(args.Nx, args.Ny, args.Ni, args.Nn)
    tile = Tile(args.Tx, args.Ty, args.Ti, args.Tn)
    model = ConvModel(prob, tile, Hardware())

    if args.json:
        print(json.dumps(model.dict(), indent=2))
    else:
        print(f"Predicted: {model.runtime_pred_ms():.3f} ms")

    if args.meas_ms:
        err = (model.runtime_pred_ms() - args.meas_ms) / args.meas_ms * 100
        print(f"Measured : {args.meas_ms:.3f} ms   →  error = {err:+.1f}%")
