"""
optimization/pt_given.py
===============
Optimization with given parameter
"""

# ======================================================================================== #
#                                         'given'                                          #
# ======================================================================================== #

import circuit.warm_start as ws
from .optimization_process import run_optimization
from plotting import plot_energies

def given_opt(cost_function_p, strategy, apparatus, silence=True):
    p = apparatus["p"]
    init_params = ws.mixed_init_param(p, strategy["init_param"])
    cost_function = cost_function_p(p)
    optimal_params, energies = run_optimization(
        cost_function=cost_function,
        init_params=init_params, 
        optimizer=apparatus["optimizer"], 
        steps=apparatus["opt_steps"],
        silence=silence
        )
    if not silence: plot_energies(energies)
    return optimal_params, energies, [energies]