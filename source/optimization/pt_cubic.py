"""
pt_cubic.py
===============
"CUBIC" parameter transfer framework.
"""

# ======================================================================================== #
#                                         'cubic'                                          #
# ======================================================================================== #

from pennylane import numpy as np

import source.circuit.warm_start as ws
from .optimization_process import run_optimization
from scipy.interpolate import make_interp_spline


def cubic_interpolation(params_list, p):
    # Convert from p-dimensional gamma and beta to (p+1)-dimensional
    params_arr = np.array(params_list)
    x_old = np.linspace(0, 1, p)
    x_new = np.linspace(0, 1, p+1)

    k = min(3, p - 1) if p > 1 else 0    # Cubic splines need >= 4 pts

    new_params = np.zeros((params_arr.shape[0], p + 1))
    for row_idx, row in enumerate(params_arr):
        if p == 1:
            new_params[row_idx] = row[0]
        else:
            spline = make_interp_spline(x_old, row, k=k)
            new_params[row_idx] = spline(x_new)
    return new_params.tolist()

def interp_params(
        cost_function_p, init_param, optimizer, opt_steps, q, silence
        ):
    if q == 1:
        init_params = ws.mixed_init_param(1, init_param)
        if not silence: print("Layer 1 optimization...")
        opt_params, energies, params_history = run_optimization(
            cost_function=cost_function_p(1),
            init_params=init_params, 
            optimizer=optimizer,
            steps=opt_steps,
            silence=silence
            )
        # if not silence: plot.plot_energies(energies)
        best_energy_ps = [energies]
        best_params_ps = [params_history]
        return opt_params, best_energy_ps, best_params_ps
    
    prev_layer_params, best_energy_ps, best_params_ps = interp_params(
        cost_function_p, 
        init_param=init_param,
        optimizer=optimizer, 
        opt_steps=opt_steps, 
        q=q-1, 
        silence=silence
        )
    if not silence: print(f"Layer {q} optimization...")
    new_params = cubic_interpolation(prev_layer_params, q-1)
    init_params = np.array(new_params, requires_grad=True)
    opt_params, energies, params_history = run_optimization(
            cost_function=cost_function_p(q),
            init_params=init_params,
            optimizer=optimizer,
            steps=opt_steps,
            silence=silence
            )
    best_energy_ps.append(energies)
    best_params_ps.append(params_history)
    # if not silence: plot.plot_energies(energies)
    return opt_params, best_energy_ps, best_params_ps

def interp_pt(cost_function_p, strategy, apparatus, silence=True):
    best_params, best_energy_ps, best_params_ps = interp_params(
        cost_function_p=cost_function_p, 
        init_param=strategy["init_param"], 
        optimizer=apparatus["optimizer"], 
        opt_steps=apparatus["opt_steps"], 
        q=apparatus["p"], 
        silence=silence,
        )
    best_energies = best_energy_ps[-1]
    return best_params, best_params_ps, best_energies, best_energy_ps
