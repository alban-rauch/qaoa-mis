"""
classical.py
============
Classical resolution.
"""

import auxiliary.graphs as gph

def list_to_string(lst):
    binary = ''
    for i in lst:
        binary += f'{i}'
    return binary

def generate_bitstrings(n):
    if n == 1:
        return ['0', '1']
    small = generate_bitstrings(n-1)
    one = ['0' + b for b in small]
    two = ['1' + b for b in small]
    return one + two

def best_config_brute(graph):
    n = len(graph)
    pool = generate_bitstrings(n)
    best_score = None
    best_bit = []
    for b in pool:
        if gph.is_legal(graph, b):
            score = sum(int(b[i]) for i in range(n))
            if best_score is None or best_score < score:
                best_score = score
                best_bit = [b]
            elif best_score == score:
                best_bit.append(b)
    return (best_score, best_bit)


def best_config_branch_bound(graph):
    nodes = sorted(list(graph.nodes))
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    neighb = [0] * n
    for u, v in graph.edges:
        neighb[idx[u]] |= (1 << idx[v])
        neighb[idx[v]] |= (1 << idx[u])
    
    best_score = -1
    best_sets = []

    def branch(candidates, chosen, size):
        nonlocal best_score, best_sets
        if candidates == 0:
            if size > best_score:
                best_score, best_sets = size, [chosen]
            elif size == best_score:
                best_sets.append(chosen)
            return
        if size + bin(candidates).count("1") < best_score:       # Cannot lead to optimum, so prune the branch
            return
        v = (candidates & - candidates).bit_length() - 1         # Gives first available candidate
        rest = candidates & ~(1 << v)
        branch(rest, chosen, size)                               # Exclude node v from the clique
        branch(rest & ~neighb[v], chosen | (1 << v), size + 1)   # Include node v  in  the clique

    branch((1 << n) - 1, 0, 0)

    best_bit = [''.join(str((m >> i) & 1) for i in range(n)) for m in best_sets]
    return best_score, best_bit

