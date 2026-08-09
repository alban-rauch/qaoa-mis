import numpy as np
from matplotlib import pyplot as plt

foldername = "~/penn_cluster/plots/T1/3reg-N12-p5-(20x20)/"
filename = "data_mat3reg_0.npz"

data_file = np.load(foldername + filename)
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
    heatmap(data_mat[q-1], f"{filename}-p={q}")