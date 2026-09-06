"""
"""

import pickle

def make_graph_id(family, axis_dict):
    parts = [family]
    for name in sorted(axis_dict):
        parts.append(f"{name}{axis_dict[name]}")
    return "_".join(parts)

def load_cache(cache_dir, family, axis_dict):
    graph_id = make_graph_id(family, axis_dict)
    path = cache_dir / f"{graph_id}.pkl"
    with open(path, "rb") as file:
        cache = pickle.load(file)
    return cache

def load_sample(cache, sample_idx):
    return cache["samples"].get(sample_idx)

def collect_run(sample, qaoa_name, p):
    run = sample["runs"].get((qaoa_name, p))
    return run