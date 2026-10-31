import json, pickle
import numpy as np
from pathlib import Path

def create_subfold_name(graph_type, param_transfer_type):
    subfolder_name = (
        f'{graph_type["family"]}_'
        f'{"-".join(map(str, graph_type["param"]))}_'
        f's{graph_type["s"]}_'
        f'{param_transfer_type}'
    )
    return subfolder_name


def make_run_dir(base, name=None):
    run_dir = Path(base) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

# --- meta --------------------------------------------------------- #

def save_meta(run_dir, meta):               # can update strat_cond
    path = run_dir / "meta.json"
    meta = json.loads(json.dumps(meta, default=str))

    if path.exists():
        stored = json.loads(path.read_text())
        stored["p"] = max(stored["p"], meta["p"])
    else:
        stored = {**meta, "strat_cond": []}

    new_pos, stored_idx = [], []
    for pos, cond in enumerate(meta["strat_cond"]):
        if cond not in stored["strat_cond"]:
            stored["strat_cond"].append(cond)
        i = stored["strat_cond"].index(cond)
        stored_idx.append(i)
        if not (run_dir / f"histories_c{i}.pkl").exists():
            new_pos.append(pos)

    path.write_text(json.dumps(stored, indent=2, default=str))
    return new_pos, stored_idx    
    # (conds for which we need to compute, idx of the c of the files)

def load_meta(run_dir):
    path = run_dir / "meta.json"
    if path.exists():
        return json.loads(path.read_text())
    else:
        raise FileNotFoundError(f"No meta.json in {run_dir}")

# --- history ------------------------------------------------------ #

def save_histories(run_dir, cond_idxs, all_aratio_ps, all_param_ps, all_energy_ps):
    for i in range(len(cond_idxs)):
        with open(run_dir / f"histories_c{cond_idxs[i]}.pkl", "wb") as f:
                pickle.dump(dict(
                    aratios=all_aratio_ps[i],
                    params=all_param_ps[i],
                    energies=all_energy_ps[i],
                ), f)

def load_histories(run_dir, cond_idx):
    with open(Path(run_dir) / f"histories_c{cond_idx}.pkl", "rb") as f:
        return pickle.load(f)

# --- landscape ---------------------------------------------------- #

def save_landscape(run_dir, cond_idx, p, row, layer, land):
    # land = dict(theta, r, G, E, its)
    rowname = "gamma" if row == 0 else "beta"
    np.savez_compressed(
        run_dir / f"landscape_c{cond_idx}_p{p}_{rowname}-{layer}.npz", **land)

def load_landscape(run_dir, cond_idx, p, row, layer):
    rowname = "gamma" if row == 0 else "beta"
    with np.load(
        Path(run_dir) / f"landscape_c{cond_idx}_p{p}_{rowname}-{layer}.npz"
    ) as d:
        return {k: d[k] for k in d.files}