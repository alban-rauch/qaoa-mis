"""
cache_dataset.py
================
Pipeline to generate a large dataset of QAOA runs.
"""

import itertools
import os
import pickle

from source.paths import GRAPHS_DIR
from source.utils import graph_gen as gph


def make_graph_id(family, axis_dict):

    parts = [family]

    for name in sorted(axis_dict):
        parts.append(f"{name}{axis_dict[name]}")

    return "_".join(parts)


def filter_cache(run):

    excluded = {"cost_function", "cost_hamiltonian"}

    return {key: value for key, value in run.items() if key not in excluded}


def load_or_create_cache(path, family, axis_dict):
    # Load existing cache or create new one

    if path.exists():
        with open(path, "rb") as file:
            cache = pickle.load(file)

        print(f"Resuming: {path}")
        return cache

    return {
        "family": family,
        "params": dict(axis_dict),
        "samples": {},
    }

def save_cache(cache, path):
    # Save cache to disk, atomically
    # (prevents half-written pickles)
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "wb") as file:
        pickle.dump(cache, file)
    os.replace(tmp_path, path)

def valid_combo(family, axis_dict):
    if family == 'DRegular':
        N, d = axis_dict['N'], axis_dict['d']
        return N > d and (N * d) % 2 == 0
    return True


def run_dataset(
    SWEEP_CONFIG,       # [dict] graph families, graph parameters, p values and number of samples
    exp_configs,        # [tuple] (problem_config, strategy_config, apparatus_config)
    qaoa_methods,       # [dict] QAOA configurations
    outdir,             # [path] storage directory
    save_every=50,      # [int] flush to disk every N completed mutations (new sample or run), not every single one
    overwrite=False,    # Replace or not the current files while sweeping
):

    problem_config, _, apparatus_config = exp_configs
    outdir.mkdir(parents=True, exist_ok=True)

    # === Load graphs ===

    DETERM_FAMILIES = {'complete', 'linear', 'circular'}

    random_graphs = {}
    for family, cond in SWEEP_CONFIG.items():
        axis_names = list(cond['axes'].keys())
        axis_value_lists = [cond['axes'][name] for name in axis_names]
        combos = list(itertools.product(*axis_value_lists))
        combos = [c for c in combos if valid_combo(family, dict(zip(axis_names, c)))]

        if family in DETERM_FAMILIES:
            params_needed = [combo[axis_names.index('N')] for combo in combos]
        else:
            params_needed = combos

        random_graphs[family] = gph.load_family(
            GRAPHS_DIR / f'{family}.npz', 
            family, 
            params=params_needed
        )

    # === Run dataset ===

    for family, cond in SWEEP_CONFIG.items():

        p_values = cond['p_values']
        num_samples = cond['num_samples']

        axis_names = list(cond['axes'].keys())

        axis_value_lists = [cond['axes'][name] for name in axis_names]

        # Combinations eg (N, q) = (12, 0.25)
        combos = list(itertools.product(*axis_value_lists))
        combos = [c for c in combos if valid_combo(family, dict(zip(axis_names, c)))]

        # One file: DATA/qaoa_data/Gilbert_N10_q0.25
        for combo in combos:

            axis_dict = dict(zip(axis_names, combo))
            graph_id = make_graph_id(family, axis_dict)

            path = outdir / f"{graph_id}.pkl"
            cache = load_or_create_cache(
                path,
                family,
                axis_dict,
            )

            N = axis_dict.get('N')

            key = (
                axis_dict["N"]
                if family in DETERM_FAMILIES
                else combo
            )

            pending_saves = 0 

            for sample_idx in range(num_samples):

                # --- Resume support ---

                if sample_idx not in cache["samples"]:

                    # If sample does not exist, add it:

                    graph = gph.get_graph_from_edges(
                        gph.get_sample(
                            random_graphs[family], 
                            key, 
                            s=sample_idx), 
                        N=N,
                    )

                    cache["samples"][sample_idx] = {
                        "sample_idx": sample_idx,
                        "edges": list(graph.edges),
                        "runs": {},
                    }

                    pending_saves += 1 
                    if pending_saves >= save_every:   # batched flush
                        save_cache(cache, path)
                        pending_saves = 0

                    print(
                        f"{graph_id}: "
                        f"sample {sample_idx} graph created"
                    )

                else:
                    # Simply reconstruct from edges
                    edges = cache["samples"][sample_idx]["edges"]
                    graph = gph.get_graph_from_edges(edges, N=N)

                sample = cache["samples"][sample_idx]

                # --- QAOA configuration ---

                for qaoa_name, qaoa_method in qaoa_methods.items():

                    for p in p_values:

                        run_key = (qaoa_name, p)

                        # --- Resume support ---

                        if run_key in sample["runs"]:
                            if not overwrite:
                                print(
                                    f"{graph_id}: sample {sample_idx}: "
                                    f"{qaoa_name}, p={p} already exists. "
                                    f"Sample will be skipped. "
                                )
                                continue
                            else:
                                print(
                                    f"{graph_id}: sample {sample_idx}: "
                                    f"{qaoa_name}, p={p} already exists "
                                    f"Sample will be overwritten. "
                                )

                        # --- Configure run ---

                        problem_config["N"] = N
                        problem_config["graph"] = graph

                        apparatus_config["p"] = p

                        run_problem_config = dict(problem_config)
                        run_apparatus_config = dict(apparatus_config)


                        # --- Run QAOA ---

                        print(
                            f"Running {graph_id}: "
                            f"sample {sample_idx}, "
                            f"{qaoa_name}, p={p}"
                        )

                        result = run_method(
                            method=qaoa_method,
                            problem=run_problem_config,
                            apparatus=run_apparatus_config,
                        )

                        # --- Cache ---

                        sample["runs"][run_key] = {
                            "method": qaoa_name,
                            "p": p,
                            **filter_cache(result),
                        }

                        pending_saves += 1
                        if pending_saves >= save_every:
                            save_cache(cache, path)
                            pending_saves = 0

                        print(
                            f"Done {graph_id}: "
                            f"sample {sample_idx}, "
                            f"{qaoa_name}, p={p}"
                        )

            if pending_saves > 0:           # Unconditional final flush
                save_cache(cache, path)
                pending_saves = 0

            print(f"Finished {graph_id} → {path}")


def run_method(method, problem, apparatus):
    return method["runner"](
        problem=problem,
        strategy=method["config"],
        apparatus=apparatus,
        silence=True,
    )

# ==== Result design ====

# Gilbert_N10_q0.25.pkl
# │
# ├── family
# ├── params
# │
# └── samples
#     │
#     ├── 0
#     │   ├── sample_idx
#     │   ├── edges
#     │   └── runs
#     │       ├── ("standard_cold", 1)
#     │       ├── ("standard_cold", 2)
#     │       ├── ("standard_warm", 1)
#     │       ├── ("standard_warm", 2)
#     │       └── ...
#     │
#     ├── 1
#     │   └── ...
#     │
#     └── 9
#         └── ...


# ==== Usage ====

# data = load_cache("Gilbert_N10_q0.25.pkl")
# sample = data["samples"][3]
# run = sample["runs"][("warm", 5)]


# ==== Note on qaoa_methods ====

# The function above accepts other methods:
#
#     methods = {
#         "standard_cold": {
#             "runner": run_qaoa,
#             "config": cold_strategy_config,
#         },
#
#         "standard_warm": {
#             "runner": run_qaoa,
#             "config": warm_strategy_config,
#         },
#    
#         "multi_angle": {
#             "runner": ...,
#             "config": ...,
#         },   
#     }