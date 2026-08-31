"""
time_lookup.py
==============
Check the average QAOA running time for sample parameters.
"""

import pickle

from source.paths import DATA_DIR

DATASET_DIR = DATA_DIR / "dataset"

for path in sorted(DATASET_DIR.glob("*.pkl")):

    with open(path, "rb") as f:
        data = pickle.load(f)

    total = 0
    runs = 0

    for sample in data["samples"].values():
        for run in sample["runs"].values():
            total += sum(run["times"])
            runs += 1

    print(
        f"{path.stem:30s} "
        f"{total/3600:6.2f} h "
        f"({runs} runs)"
    )