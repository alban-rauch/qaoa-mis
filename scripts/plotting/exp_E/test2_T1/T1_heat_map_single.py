from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
foldername = "Gilb0.25-N12-p5-(20x20)-right"
filename = "data_mat_Gilb_7.npz"
file_path = SCRIPT_DIR / foldername / filename

data_file = np.load(file_path)
data_mat = data_file["data_mat"]
p = len(data_mat)

def heatmap(data_mat, name):
    plt.figure()
    im = plt.imshow(
        data_mat, 
        cmap="RdYlGn", 
        vmin=0.0, 
        vmax=1.0, 
        aspect="auto",
        origin="lower"
    )
    plt.colorbar(im)
    plt.xlabel("gamma")
    plt.ylabel("beta")
    plt.savefig(name)
    plt.close()

for q in range(1, p+1):
    image_path = SCRIPT_DIR / foldername / f"{filename}-p={q}.png"
    heatmap(data_mat[q-1], image_path)