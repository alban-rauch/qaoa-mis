"""
graphs.py
=========
Graph generation.
"""

import networkx as nx


##############################          Tests          ##############################

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


############################## Random d-regular (N, d) ##############################

def randomDRegular(N, d):                   # N nodes with same degree d (N * d must be even)
    return nx.random_regular_graph(d, N)


##############################  Random Gilbert (N, q)  ##############################

def randomGilbert(N, q):
    return nx.fast_gnp_random_graph(N, q)


##############################      Deterministic      ##############################

def complete_graph(N):
    return nx.complete_graph(N)

def linear_graph(N):
    return nx.path_graph(N)

def circular_graph(N):
    return nx.circulant_graph(N, [1])

def complete_bipartite_graph(a, b):
    return nx.complete_bipartite_graph(a, b)

def is_legal(graph, x):
    for u, v in graph.edges:
        if int(x[u]) == 1 and int(x[v]) == 1:
            return False
    return True