"""
repair_miswritten.py
====================
Correction of a storage mistake oticed after run.
"""

import pickle
from pathlib import Path


DATASET_DIR = Path("DATA/dataset")


def repair_file(path):

    with open(path, "rb") as f:
        cache = pickle.load(f)

    N = cache["params"]["N"]

    for sample in cache["samples"].values():

        # graph = gph.get_graph_from_edges(sample["edges"], N=N)

        for (method, p), run in sample["runs"].items():

            run["problem"] = {
                **run["problem"],
                "N": N,
                "graph": None,
            }

            run["apparatus"] = {
                **run["apparatus"],
                "p": p,
            }

    tmp_path = Path(str(path) + ".tmp")

    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    tmp_path.replace(path)

    print(f"Repaired: {path}")


for path in DATASET_DIR.glob("*.pkl"):
    repair_file(path)