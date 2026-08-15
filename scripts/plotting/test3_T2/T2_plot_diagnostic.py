import numpy as np
import matplotlib.pyplot as plt

data = np.load("diagnostic.npz")


ar_samples = data["ar_samples"]
sr_samples = data["sr_samples"]

ar_mean = np.mean(ar_samples, axis=2)
ar_sder = np.std(ar_samples, axis=2, ddof=1) / np.sqrt(ar_samples.shape[2])
sr_mean = np.mean(sr_samples, axis=2)
sr_sder = np.std(sr_samples, axis=2, ddof=1) / np.sqrt(sr_samples.shape[2])

p_values = data["p_values"]
N_values = data["N_values"]

# approximation ratio
plt.figure(figsize=(8, 5))
for p_idx, p in enumerate(p_values):
    plt.errorbar(
        N_values,
        ar_mean[p_idx],
        yerr=ar_sder[p_idx],
        marker="o",
        linestyle="-",
        capsize=4,
        label=f"p = {p}"
    )
plt.xlabel("Number of Qubits (N)")
plt.ylabel("Approximation Ratio")
plt.title("QAOA Approximation Ratio by Depth")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend(title="QAOA Depth (p)")
plt.show()

# success rate
plt.figure(figsize=(8, 5))
for p_idx, p in enumerate(p_values):
    plt.errorbar(
        N_values,
        sr_mean[p_idx],
        yerr=sr_sder[p_idx],
        marker="o",
        linestyle="-",
        capsize=4,
        label=f"p = {p}"
    )
plt.xlabel("Number of Qubits (N)")
plt.ylabel("Success Rate")
plt.title("QAOA Success Rate by Depth")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend(title="QAOA Depth (p)")
plt.show()