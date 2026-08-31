"""
relax_pipeline.py
============
"""

import numpy as np

import source.qaoa_run as qr
import source.utils.classical as clas

def random_gen_from_relax(d, N, seed=0):
    rng = np.random.default_rng(seed)
    draws = rng.random(N)
    bitstring = np.zeros(N)
    for i in range(N):
        bitstring[i] = -1 if draws[i] < d[i] else 1
    return bitstring

def energy_from_bitstring(bitstring, N, graph, penalizer):
    relax_energy = sum((1 - graph.degree(i) * penalizer / 2) * bitstring[i] for i in range(N))
    for edge in graph.edges:
        relax_energy += penalizer / 2 * bitstring[edge[0]] * bitstring[edge[1]]
    return relax_energy

def extract_ratios(graph, considered_energies, penalizer):
    theo_best_cost, _ = clas.best_config_branch_bound(graph)
    considered_ratios = [
        qr.approx_ratio(graph, energy, penalizer, theo_best_cost) 
                if energy is not None else None
                for energy in considered_energies
    ]
    return considered_ratios

def compare_pipeline(warm_qaoa_run, cold_qaoa_run=None, seed=0):
    problem_config = warm_qaoa_run["problem"]
    N = problem_config["N"]
    graph = problem_config["graph"]
    penalizer = 1.5

    x_d = warm_qaoa_run["relax_values"]
    bitstring_rounded = random_gen_from_relax(x_d, N, seed=seed)
    relax_core_energy = energy_from_bitstring(x_d, N, graph, penalizer)
    relax_rounded_energy = energy_from_bitstring(bitstring_rounded, N, graph, penalizer)

    opt_warm_energy = warm_qaoa_run["best_energy"]
    opt_cold_energy = cold_qaoa_run["best_energy"] if cold_qaoa_run is not None else None

    relax_core_ratio, relax_rounded_ratio, opt_warm_ratio, opt_cold_ratio = extract_ratios(
        graph, 
        [relax_core_energy, relax_rounded_energy, opt_warm_energy, opt_cold_energy], 
        penalizer
    )

    return relax_core_ratio, relax_rounded_ratio, opt_warm_ratio, opt_cold_ratio