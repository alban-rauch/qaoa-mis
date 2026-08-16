from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
foldername = "3reg-N12-p5-(20x20)-right"

data_mat_list = []
for idx in range(8):
    filename = f"data_mat_3-reg_{idx}.npz"
    file_path = SCRIPT_DIR / foldername / filename
    data_file = np.load(file_path)
    data_mat = data_file["data_mat"]
    data_mat_list.append(data_mat)

A = np.stack(data_mat_list)
mean = np.mean(A, axis=0)
std_error = np.std(A, axis=0, ddof=1) / np.sqrt(len(A))

filename = f"data_mat_3-reg_avg.npz"
file_path = SCRIPT_DIR / foldername / filename
np.savez(file_path, mean=mean, std=std_error)