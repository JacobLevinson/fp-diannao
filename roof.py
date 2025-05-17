import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
#  GPU-peak numbers  (TITAN-V, single-precision)
FLOP_peak = 13.8e12          # 13.8 TFLOP/s
BW_peak   = 652e9            # 652 GB/s  (TechPowerUp spec)
AI_tilt   = FLOP_peak / BW_peak
# ────────────────────────────────────────────────

kernels  = ["Conv1", "Conv2", "FC1", "FC2"]
markers  = ["o", "s", "^", "v"]
colors   = ["crimson", "orange", "forestgreen", "royalblue"]

# performance (GFLOP/s  →  FLOP/s)
gflops   = np.array([72.6, 61.0, 20.7, 10.0]) * 1e9

# effective DRAM BW = read + write  (GB/s → B/s)
bw_bytes = np.array([94.21, 2.088, 73.02, 20.42]) * 1e9

# arithmetic intensity  (F/B)
ai = gflops / bw_bytes

# ─── Roof-line curve ────────────────────────────
ai_line  = np.logspace(-1, 4, 256)
roof     = np.minimum(BW_peak * ai_line, FLOP_peak)

fig, ax = plt.subplots(figsize=(8, 6))

# roof & diagonal
ax.loglog(ai_line, roof / 1e12, '-', lw=1.5, color='black', label="Roof-line")

# vertical AI_tilt guide
ax.axvline(AI_tilt, ls='--', lw=2.0, color='dodgerblue', zorder=1)
ax.text(AI_tilt*1.03, 0.07,
        fr"$\mathrm{{AI_{{tilt}}}}\!\approx\!{AI_tilt:.0f}$ F/B",
        rotation=90, va='bottom', ha='left',
        color='dodgerblue', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="dodgerblue", lw=0.8))

# kernel points & labels
for k, m, c, x, y in zip(kernels, markers, colors, ai, gflops/1e12):
    ax.scatter(x, y, marker=m, s=80, color=c, zorder=5)
    ax.text(x*1.15, y*1.25, k, color=c, fontsize=9, zorder=6)

# axis limits
y_min = (gflops/1e12).min() * 0.5
ax.set_xlim(1e-1, 1e4)
ax.set_ylim(y_min, FLOP_peak/1e12*1.2)

ax.set_xlabel("Arithmetic intensity  [FLOP / Byte]")
ax.set_ylabel("Performance  [TFLOP s⁻¹]")
ax.set_title("Roof-line — NVIDIA TITAN V  (batch = 1)")
ax.grid(True, which='both', ls=':', lw=0.4)

plt.tight_layout()
plt.savefig("roofline_titanV.png", dpi=250)
print("saved → roofline_titanV.png")
