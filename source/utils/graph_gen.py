"""
graphs.py
=========
Graph generation.
"""

import numpy as np
import networkx as nx

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

def get_graph_from_edges(edges):
    # Assumes edges is np.array of dim (E, 2)
    return nx.Graph(edges)


####################################################################################
###                            Generate simple graphs                            ### 
####################################################################################

### Deterministic ###

def complete_graph(N):
    return nx.complete_graph(N)

def linear_graph(N):
    return nx.path_graph(N)

def circular_graph(N):
    return nx.circulant_graph(N, [1])

def complete_bipartite_graph(a, b):
    return nx.complete_bipartite_graph(a, b)

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
    'complete': complete_graph,
    'linear': linear_graph,
    'circular': circular_graph,
}

def gen_determ(fn, N_range):
    return {
        N: np.array(fn(N).edges(), dtype=np.int32) 
                for N in N_range
    }

def save_determ(data_dict, path):
    flat = {
        str(k): v
        for k, v in data_dict.items()
    }
    np.savez_compressed(path, **flat)

def load_determ(path):
    data = np.load(path)
    out = {}
    for name in data.files:
        key = int(name)
        out[key] = data[name]
    return out


### Complete bipartite and Random D-Regular ###

def gen_bip(a_range, b_range):
    return {
        (a, b): np.array(nx.complete_bipartite_graph(a, b).edges(), dtype=np.int32) 
                for a in a_range for b in b_range if a <= b
    }

def gen_DRegular(N_range, d_values, num_sample=100):
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

    # Storage for gen_DReg uses the fact that for a graph DReg(N, d), the number 
    # of edges is E = N * d / 2, so we can use an array of dimension (num_sample, E, 2)

def save_double(data_dict, path):
    flat = {
        f"{k[0]}_{k[1]}": v
        for k, v in data_dict.items()
    }
    np.savez_compressed(path, **flat)

def load_double(path):
    data = np.load(path)
    out = {}
    for name in data.files:
        a, b = name.split('_')
        key = (int(a), int(b))
        out[key] = data[name]
    return out


### Random Gilbert ###

def gen_Gilbert(N_range, q_values, num_sample=100):
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

    # Storage for gen_Gilb is done in a long array "N_q_edges" of dim (TotE, 2)
    # Distinction between graphs is done thanks to "N_q_lengths" array


def get_Gilbert(entry, s):
    starts = np.concatenate([[0], np.cumsum(entry['lengths'])])
    return entry['edges'][starts[s]:starts[s+1]]

def save_Gilb(data_dict, path):
    flat = {}
    for (N, p), entry in data_dict.items():
        flat[f"{N}_{p}_edges"] = entry['edges']
        flat[f"{N}_{p}_lengths"] = entry['lengths']
    np.savez_compressed(path, **flat)

def load_Gilbert(path):
    data = np.load(path)
    out = {}
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
        elif family == 'DRegular':
            data_dict = gen_DRegular(*info)
            save_double(data_dict, path)
        elif family == 'Gilbert':
            data_dict = gen_Gilbert(*info)
            save_Gilb(data_dict, path)


def load_family(path, family):
    if family == 'DRegular' or family == 'complete_bipartite':
        data_dict = load_double(path)
    elif family == 'Gilbert':
        data_dict = load_Gilbert(path)
    else:
        data_dict = load_determ(path)
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