from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
foldername = "3reg-N12-p5-(20x20)-right"
filename = "data_mat_3-reg_avg.npz"
file_path = SCRIPT_DIR / foldername / filename

data_file = np.load(file_path)
mean_mat = data_file["mean"]
std_mat = data_file["std"]
p = len(mean_mat)

def superposed_heatmap(mean_mat, std_mat, name):
    fig, ax = plt.subplots(figsize=(7, 6))

    std_min, std_max = std_mat.min(), std_mat.max()
    norm_std = (std_mat - std_min) / (std_max - std_min + 1e-8)
    alpha_mat = 1.0 - norm_std * 0.9  # Transparency btw 0.3 and 1.0

    im = ax.imshow(
        mean_mat, 
        cmap="RdYlGn", 
        vmin=0.0, 
        vmax=1.0, 
        aspect="auto", 
        origin="lower",
        alpha=alpha_mat
    )
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Value (Faded = Higher Std Err)")

    ax.set_xlabel("gamma")
    ax.set_ylabel("beta")
    ax.set_title("Mean (Color) superposed with Std Err (Fading)")

    plt.savefig(name, bbox_inches="tight")
    plt.close()

for q in range(1, p+1):
    image_path = SCRIPT_DIR / foldername / f"{filename}-p={q}.png"
    superposed_heatmap(mean_mat[q-1], std_mat[q-1], image_path)