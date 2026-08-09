"""
circuit/warm_start.py
=============
Warm starting via relaxation for improved reference state initialization.
"""

from pennylane import numpy as np

from scipy.optimize import linprog

# ================================================================ #
#                        INITIAL PARAMETERS                        #
# ================================================================ #

def mixed_init_param(p, init_param):
    nb_param = len(init_param)
    raw_params = [[init_param[i] * np.pi] * p for i in range(nb_param)]
    raw_params_ar = np.array(raw_params, requires_grad=True)
    return raw_params_ar

def random_init_param(p, nb_params, nb_twopi=1):
    rng = np.random.default_rng()
    raw_params = []
    for _ in range(nb_twopi):
        raw_params.append(rng.uniform(0, 2 * np.pi, p))
    for _ in range(nb_params - nb_twopi):
        raw_params.append(rng.uniform(0, np.pi, p))
    raw_params_ar = np.array(raw_params, requires_grad=True)
    return raw_params_ar


# ================================================================ #
#                           INITIAL STATE                          #
# ================================================================ #


def relaxation(graph, wires):
    nodes = wires
    edges = list(graph.edges)
    n = len(nodes)
    num_edge = len(edges)
    
    c = - np.ones(n)
    b_ub = np.ones(num_edge)
    A_ub = np.zeros((num_edge, n))
    for k, (i, j) in enumerate(edges):
        A_ub[k, i] = 1
        A_ub[k, j] = 1
    
    bounds = [(0, 1)] * n
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    x_star = res.x
    
    return dict(zip(nodes, x_star)), -res.fun



def relaxation_angles(graph, wires, eps=0.25):
    d, _ = relaxation(graph, wires)
    angles = {}
    for node in d:
        x = d[node]
        if x < eps:
            x = eps
        elif x > 1-eps:
            x = 1-eps
        th = 2 * np.arcsin(np.sqrt(x))
        angles[node] = th
    return angles