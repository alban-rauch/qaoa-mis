"""
pt_interp.py
===============
"INTERP" general parameter transfer framework.
"""

# ======================================================================================== #
#                                         'interp'                                         #
# ======================================================================================== #

from pennylane import numpy as np

import source.circuit.warm_start as ws
from .optimization_process import run_optimization


### Application ###

def perturb_params(params, strength, rng):
    # Zero mean Gaussian noise perturbation (to avoid being stuck on a plateau)
    params_arr = np.array(params)
    noise = rng.normal(loc=0.0, scale=strength, size=params_arr.shape)
    return (params_arr + noise).tolist()


def interp_params(
        cost_function_p, 
        init_param, 
        optimizer, 
        opt_steps, 
        q, 
        interpolator,
        R_perturb=0,
        noise_strength=0.3,
        rng=None,
        silence=True,
    ):

    if rng is None:
        rng = np.random.default_rng()

    if q == 1:
        init_params = ws.mixed_init_param(1, init_param)
        if not silence: print("Layer 1 optimization...")
        opt_params, energies, params_history = run_optimization(
            cost_function=cost_function_p(1),
            init_params=init_params, 
            optimizer=optimizer,
            steps=opt_steps,
            silence=silence,
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
        interpolator=interpolator,
        R_perturb=R_perturb,
        noise_strength=noise_strength,
        rng=rng,
        silence=silence,
        )
    if not silence: print(f"Layer {q} optimization...")

    base_new_params = interpolator(prev_layer_params, q-1)

    candidates = [base_new_params]
    for _ in range(R_perturb):
        candidates.append(perturb_params(base_new_params, noise_strength, rng))

    best_final_energy = None
    for cand in candidates:
        init_params = np.array(cand, requires_grad=True)
        cand_opt_params, cand_energies, cand_params_history = run_optimization(
            cost_function=cost_function_p(q),
            init_params=init_params,
            optimizer=optimizer,
            steps=opt_steps,
            silence=silence,
        )
        final_energy = cand_energies[-1]
        if best_final_energy is None or final_energy < best_final_energy:
            best_final_energy = final_energy
            opt_params, energies, params_history = cand_opt_params, cand_energies, cand_params_history

    best_energy_ps.append(energies)
    best_params_ps.append(params_history)
    # if not silence: plot.plot_energies(energies)
    return opt_params, best_energy_ps, best_params_ps

def interp_pt(cost_function_p, strategy, apparatus, interpolator, silence=True):
    rng = np.random.default_rng(seed=0)
    best_params, best_energy_ps, best_params_ps = interp_params(
        cost_function_p=cost_function_p, 
        init_param=strategy["init_param"], 
        optimizer=apparatus["optimizer"], 
        opt_steps=apparatus["opt_steps"], 
        q=apparatus["p"],                           # Current layer
        R_perturb=strategy["fourier_qR"][1],        # Random R
        noise_strength=strategy.get("noise_strength", 0.3),
        interpolator=interpolator,
        rng=rng,
        silence=silence,
        )
    best_energies = best_energy_ps[-1]
    return best_params, best_params_ps, best_energies, best_energy_ps
