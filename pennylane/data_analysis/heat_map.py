import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

data_file = np.load("data_mat.npz")
data_mat = data_file["data_mat"]
p = len(data_mat)

def heatmap(data_mat, name):
    plt.figure()
    sns.heatmap(
        data_mat, 
        annot=False, 
        cmap="RdYlGn", 
        vmin=0.0, 
        vmax=1.0, 
        cbar=True
    )
    plt.xlabel("gamma")
    plt.ylabel("beta")
    plt.savefig(name)
    plt.close()

for q in range(1, p+1):
    heatmap(data_mat[q-1], f"data_mat_p={q}")