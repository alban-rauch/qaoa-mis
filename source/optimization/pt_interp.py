"""
pt_interp.py
===============
"INTERP" parameter transfer framework.
"""

# ======================================================================================== #
#                                         'interp'                                         #
# ======================================================================================== #

from pennylane import numpy as np

import source.circuit.warm_start as ws
from .optimization_process import run_optimization


def linear_interpolation(params_list, p):
    # Convert from p-dimensional gamma and beta to (p+1)-dimensional
    params_arr = np.array(params_list)
    new_params = np.zeros((params_arr.shape[0], p+1))
    new_params[:, 0] = params_arr[:, 0]
    new_params[:, -1] = params_arr[:, -1]
    for i in range(1, p):
        new_params[:, i] = (i / p) * params_arr[:, i-1] + ((p-i) / p) * params_arr[:, i]
    return new_params.tolist()

def general_linear_interpolation(params_list, p, q):
    # Convert from p-dimensional gamma and beta to (p+1)-dimensional
    params_arr = np.array(params_list)
    if q == 1:
        return np.mean(params_arr, axis=1, keepdims=True).tolist()
    positions = np.linspace(0, p-1, q)
    floors = np.floor(positions).astype(int)
    ceilings = np.minimum(floors + 1, p-1)
    distances = positions - floors
    new_params = (1 - distances) * params_arr[:, floors] + distances * params_arr[:, ceilings]
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
    new_params = linear_interpolation(prev_layer_params, q-1)
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
