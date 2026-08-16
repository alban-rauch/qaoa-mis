"""
optimization/pt_random.py
===============
"RANDOM" parameter restart framework.
"""

# ======================================================================================== #
#                                         'random'                                         #
# ======================================================================================== #

import source.circuit.warm_start as ws
from .optimization_process import run_optimization

def param_restarts(cost_function, n_restarts, p, optimizer, opt_steps, nb_params, nb_sin, silence):
    best_params = None
    best_energies = None
    best_energy_ps = []
    for i in range(n_restarts):
        if not silence: print(f"Restart {i}")
        init_params_i = ws.random_init_param(p, nb_params, nb_sin)
        optimal_params, energies = run_optimization(
            cost_function=cost_function,
            init_params=init_params_i, 
            optimizer=optimizer, 
            steps=opt_steps,
            silence=silence
            )
        if best_energies is None or energies[-1] < best_energies[-1]:
            best_params = optimal_params
            best_energies = energies
        best_energy_ps.append(energies)
        if not silence: print("----------------------------")
    # if not silence: plot.plot_energies(best_energies)
    return best_params, best_energies, best_energy_ps


def random_pt(cost_function_p, strategy, apparatus, silence=True):
    p = apparatus["p"]
    cost_function = cost_function_p(p)
    return param_restarts(
        cost_function=cost_function, 
        n_restarts=20,
        p=p,
        optimizer=apparatus["optimizer"], 
        opt_steps=apparatus["opt_steps"],
        nb_params=1+len(strategy["mixers"]),
        nb_sin=1, 
        silence=silence
        )