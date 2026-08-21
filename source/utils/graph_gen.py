"""
graphs_gen.py
=============
Graph generation.
"""

import numpy as np
import networkx as nx
import multiprocessing as mp

from source.paths import GRAPHS_DIR


####################################################################################
###                               Useful functions                               ###
####################################################################################

def is_legal(graph, x):
    for u, v in graph.edges:
        if int(x[u]) == 1 and int(x[v]) == 1:
            return False
    return True

def edges_to_adjmat(edges, N, dtype=np.int8):
    adjmat = np.zeros((N, N), dtype=dtype)
    adjmat[edges[:, 0], edges[:, 1]] = 1
    adjmat[edges[:, 1], edges[:, 0]] = 1
    return adjmat

def get_graph_from_edges(edges, N=None):
    # Assumes edges is np.array of dim (E, 2)
    graph = nx.Graph()
    if N is not None:
        graph.add_nodes_from(range(N))
    graph.add_edges_from((int(u), int(v)) for u, v in edges)
    return graph


####################################################################################
###                            Generate simple graphs                            ### 
####################################################################################

### Deterministic ###

def complete_graph(N):
    return nx.complete_graph(N)

def complete_edges(N):
    iu = np.triu_indices(N, k=1)
    return np.stack(iu, axis=1)

def linear_graph(N):
    return nx.path_graph(N)

def linear_edges(N):
    return np.array(
        [(i, i+1) for i in range(N-1)]
    )

def circular_graph(N):
    return nx.circulant_graph(N, [1])

def circular_edges(N):
    return np.array(
        [(i, i+1) for i in range(N-1)] + [(N-1, 0)]
    )

def complete_bipartite_graph(a, b):
    return nx.complete_bipartite_graph(a, b)

def complete_bipartite_edges(a, b):
    ii, jj = np.meshgrid(np.arange(a), np.arange(a, a + b), indexing='ij')
    return np.stack([ii.ravel(), jj.ravel()], axis=1)

def simple():
    edges = [(0, 1), (1,2)]
    return nx.Graph(edges)

def paragon():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), 
            (4, 5), (5, 6), (6, 7), (7, 4), 
            (0, 4), (1, 5), (2, 6), (3, 7),
            ]
    graph = nx.Graph(edges)

    # positions = nx.spring_layout(graph, seed=1)
    # nx.draw(graph, with_labels=True, pos=positions)
    # plt.show()

    return graph


### Random ###

def randomDRegular(N, d):                   # N nodes with same degree d (N * d must be even)
    return nx.random_regular_graph(d, N)

def randomGilbert(N, q):
    return nx.fast_gnp_random_graph(N, q)


####################################################################################
###               Batch generation for benchmark and repeatability               ###
####################################################################################


### Complete, linear, circular ###

determ_graphs = {
    'complete': complete_edges,
    'linear': linear_edges,
    'circular': circular_edges,
}

def gen_determ(fn, N_range):
    return {
        N: np.array(fn(N), dtype=np.uint8) 
                for N in N_range
    }

def save_determ(data_dict, path):
    flat = {
        str(k): v
        for k, v in data_dict.items()
    }
    np.savez_compressed(path, **flat)

def load_determ(path, N_vals=None):
    data = np.load(path)
    if N_vals is None:
        N_vals = [int(name) for name in data.files]
    out = {}
    for N in N_vals:
        out[N] = data[str(N)]
    return out


### Complete bipartite ###

def gen_bip(a_range, b_range):
    return {
        (a, b): np.array(complete_bipartite_edges(a, b), dtype=np.uint8) 
                for a in a_range for b in b_range if a <= b
    }

### Random D-Regular (multi-processing) ###

def _regular_sample(args):
    N, d, seed = args
    return np.array(nx.random_regular_graph(d, N, seed=seed).edges(), dtype=np.uint8)


def gen_DRegular_singleprocess(N_range, d_values, num_sample=100):
    # Storage for gen_DReg uses the fact that for a graph DReg(N, d), the number 
    #   of edges is E = N * d / 2, so we can use an array of dimension (num_sample, E, 2)
    out = {}
    for d in d_values:
        for N in N_range:
            if N <= d or (N * d) % 2 != 0:
                continue
            E = N * d // 2
            arr = np.empty((num_sample, E, 2), dtype=np.int32)
            for s in range(num_sample):
                arr[s] = np.array(nx.random_regular_graph(d, N, seed=s).edges(), dtype=np.int32)
            out[(N, d)] = arr
    return out


def gen_DRegular_multiprocess(N_range, d_values, num_sample=100, n_workers=None):
    # More efficient than other version by dividing the work. The total number of jobs, 
    #   i.e. tuples (N, d, s) is split between workers (in my case 6 workers) and I further
    #   divided the patches by 4 since the jobs are highly uneven (eg (N=3, d=2) vs (N=100, d=50)).

    valid_params = [
        (N, d) for d in d_values for N in N_range
        if N > d and (N * d) % 2 == 0
    ]

    jobs = [(N, d, s) for (N, d) in valid_params for s in range(num_sample)]
    if not jobs:
        return {}
 
    n_workers = n_workers or mp.cpu_count()
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(
                _regular_sample, 
                jobs, 
                chunksize=max(1, len(jobs) // (n_workers * 4))
            )
    else:
        results = [_regular_sample(j) for j in jobs]
 
    out = {}
    idx = 0
    for (N, d) in valid_params:
        E = N * d // 2
        arr = np.empty((num_sample, E, 2), dtype=np.uint8)
        for s in range(num_sample):
            arr[s] = results[idx]
            idx += 1
        out[(N, d)] = arr

    return out


def save_double(data_dict, path):
    flat = {
        f"{k[0]}_{k[1]}": v
        for k, v in data_dict.items()
    }
    np.savez_compressed(path, **flat)

def load_double(path, params=None):
    data = np.load(path)
    out = {}
    if params is None:
        for name in data.files:
            a, b = name.split('_')
            key = (int(a), int(b))
            if params is None or key == params:
                out[key] = data[name]
    else:
        for (a, b) in params:
            out[(a, b)] = data[f"{a}_{b}"]
    return out


### Random Gilbert (vectorized) ###

def gen_Gilbert_unvect(N_range, q_values, num_sample=100):
    # Storage for gen_Gilb is done in a long array "N_q_edges" of dim (TotE, 2)
    # Distinction between graphs is done thanks to "N_q_lengths" array
    out = {}
    for q in q_values:
        for N in N_range:
            all_edges = []
            lengths = np.empty(num_sample, dtype=np.int32)
            for s in range(num_sample):
                e = np.array(list(nx.fast_gnp_random_graph(N, q, seed=s).edges()), dtype=np.int32).reshape(-1, 2)
                lengths[s] = len(e)
                all_edges.append(e)
            out[(N, q)] = {
                'edges': np.concatenate(all_edges, axis=0) if all_edges else np.empty((0, 2), dtype=np.int32),
                'lengths': lengths,
            }
    return out

def gen_Gilbert_vect(N_range, q_values, num_sample=100, seed=0):
    # The same seed is reused throughout. Given a size N, the possible 
    #   edges and num_sample do not vary, hence ii, jj remain fixed.
    # For every q, a mask of size (num_sample, num_possible_edges) determines, 
    #   at pos [s, e] if edge e in sample s exists.
    rng = np.random.default_rng(seed)
    out = {}
    for N in N_range:
        iu = np.triu_indices(N, k=1)  # [(i, j) | i < j] in the form [[0, 0, ..., N-1], [1, 2, ..., N]]
        num_possible_edges = len(iu[0])     # N*(N-1)/2 possible edges
        ii = np.broadcast_to(iu[0], (num_sample, num_possible_edges))
        jj = np.broadcast_to(iu[1], (num_sample, num_possible_edges))
        for q in q_values:
            mask = rng.random((num_sample, num_possible_edges)) < q    
            lengths = mask.sum(axis=1).astype(np.uint16)
            all_i = ii[mask]
            all_j = jj[mask]
            edges = np.stack([all_i, all_j], axis=1).astype(np.uint8)
            out[(N, q)] = {'edges': edges, 'lengths': lengths}
    return out


def get_Gilbert(entry, s):
    starts = np.concatenate([[0], np.cumsum(entry['lengths'], dtype=np.int64)])
    return entry['edges'][starts[s]:starts[s+1]]

def save_Gilbert(data_dict, path):
    flat = {}
    for (N, p), entry in data_dict.items():
        flat[f"{N}_{p}_edges"] = entry['edges']
        flat[f"{N}_{p}_lengths"] = entry['lengths']
    np.savez_compressed(path, **flat)

def load_Gilbert(path, params=None):
    data = np.load(path)
    out = {}
    if params is None:
        for name in data.files:
            if name.endswith('_edges'):
                N, p, _ = name.split('_')
                key = (int(N), float(p))
                out.setdefault(key, {})['edges'] = data[name]
            elif name.endswith('_lengths'):
                N, p, _ = name.split('_')
                key = (int(N), float(p))
                out.setdefault(key, {})['lengths'] = data[name]
        return out
    else:
        for (N, q) in params:
            out[(N, q)] = {
                'edges': data[f"{N}_{q}_edges"], 
                'lengths': data[f"{N}_{q}_lengths"]
            }
        return out


### General ###

def gen_save_samples(instructions, folder):
    for family, info in instructions.items():
        path = folder / f'{family}.npz'
        if family in ['complete', 'linear', 'circular']:
            data_dict = gen_determ(determ_graphs[family], info)
            save_determ(data_dict, path)
        elif family == 'complete_bipartite':
            data_dict = gen_bip(*info)
            save_double(data_dict, path)
            print("determ done")
        elif family == 'DRegular':
            data_dict = gen_DRegular_multiprocess(*info)
            save_double(data_dict, path)
            print("dreg done")
        elif family == 'Gilbert':
            data_dict = gen_Gilbert_vect(*info)
            save_Gilbert(data_dict, path)
            print("gilbert done")


def load_family(path, family, params=None):
    if family == 'DRegular' or family == 'complete_bipartite':
        data_dict = load_double(path, params)
    elif family == 'Gilbert':
        data_dict = load_Gilbert(path, params)
    else:
        data_dict = load_determ(path, params)
    data_dict['type'] = family
    return data_dict

def get_sample(data_dict, param, s=None):
    entry = data_dict[param]
    gtype = data_dict['type']
    if gtype == 'Gilbert':
        return get_Gilbert(entry, s)
    elif gtype == 'DRegular':
        return entry[s]
    else:
        return entry


# Usage:
#  data_dict = load_family(path, family, params=None)
#  get_sample(data_dict, param, s=None)

####################################################################################
###                                  Generation                                  ###
####################################################################################

if __name__ == "__main__":
    instructions = {
        'complete': range(3, 201),                              # N
        'linear': range(3, 201),                                # N
        'circular': range(3, 201),                              # N
        'complete_bipartite': (range(3, 101), range(3, 101)),   # (a, b)
        'DRegular': (range(3, 101), [2, 3, 5, 10, 50], 100),    # (num_sample, N, d)
        'Gilbert': (range(3, 101), np.linspace(0, 1, 21), 100), # (num_sample, N, q)
    }
    gen_save_samples(instructions, GRAPHS_DIR)