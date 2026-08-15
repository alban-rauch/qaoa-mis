from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

script_dir = Path(__file__).parent
data_path = script_dir / Path("speed_data.npz")
data = np.load(data_path)

speed_mean = data["speed_mean"]
speed_stderr = data["speed_stderr"]
evals_mean = data["evals_mean"]
evals_stderr = data["evals_stderr"]
aratio_mean = data["aratio_mean"]
aratio_stderr = data["aratio_stderr"]

N_values = data["N_values"]
p_values = data["p_values"]

speed_stderr = np.nan_to_num(speed_stderr, nan=0.0)
evals_stderr = np.nan_to_num(evals_stderr, nan=0.0)
aratio_stderr = np.nan_to_num(aratio_stderr, nan=0.0)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), sharex=True)

for p_idx, p in enumerate(p_values):
    ax1.errorbar(
        N_values,
        speed_mean[:, p_idx],
        yerr=speed_stderr[:, p_idx],
        fmt="-o",
        capsize=4,
        capthick=1.5,
        linewidth=1.8,
        label=f"p = {p}",
    )

ax1.set_xlabel("System Size $N$", fontsize=11)
ax1.set_ylabel("Time", fontsize=11)
ax1.set_title("QAOA Speed vs. $N$", fontsize=12, fontweight="bold")
ax1.set_xticks(N_values)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(title="Depth", frameon=True)

for p_idx, p in enumerate(p_values):
    ax2.errorbar(
        N_values,
        aratio_mean[:, p_idx],
        yerr=aratio_stderr[:, p_idx],
        fmt="-s",
        capsize=4,
        capthick=1.5,
        linewidth=1.8,
        label=f"p = {p}",
    )

ax2.set_xlabel("System Size $N$", fontsize=11)
ax2.set_ylabel("Approximation Ratio $\\alpha$", fontsize=11)
ax2.set_title("Approximation Ratio vs. $N$", fontsize=12, fontweight="bold")
ax2.set_xticks(N_values)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(title="Depth", frameon=True)

for p_idx, p in enumerate(p_values):
    ax3.errorbar(
        N_values,
        evals_mean[:, p_idx],
        yerr=evals_stderr[:, p_idx],
        fmt="-o",
        capsize=4,
        capthick=1.5,
        linewidth=1.8,
        label=f"p = {p}",
    )

ax3.set_xlabel("System Size $N$", fontsize=11)
ax3.set_ylabel("Cost Circuit Evaluations", fontsize=11)
ax3.set_title("Number of evals vs. $N$", fontsize=12, fontweight="bold")
ax3.set_xticks(N_values)
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.legend(title="Depth", frameon=True)


plt.tight_layout()
output_path = Path("qaoa_benchmark_results.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()