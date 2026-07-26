"""
optimization.py
===============
Classical parameter optimizer.
"""

import pennylane as qp
from pennylane import numpy as np
from scipy.optimize import minimize
from scipy.fft import dst, idst
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt



import copy
import qaoa_pipeline.warm_start as ws

def plot_energies(energies):
    plt.plot(energies)
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy vs. Optimization Step")
    plt.grid(alpha=0.3)
    plt.show()

def run_optimization_steps(cost_function, init_params, optimizer, steps=70):
    params = np.array(init_params, requires_grad=True)
    energies = []
    for i in range(steps):
        params, energy = optimizer.step_and_cost(cost_function, params)
        energies.append(energy)
        if i % 10 == 0: print(f"Step {i:3d} | Energy: {energy:.6f}")
    return params, energies

def run_optimization(cost_function, init_params, optimizer, steps=300, precision=1e-8, silence=True):
    if optimizer == "L-BFGS-B":
        return run_lbfgsb(
            cost_function=cost_function, 
            init_params=init_params, 
            maxiter=steps, 
            gtol=precision,
            silence=silence
            )
    else:
        if optimizer == "Adam": optimizer = qp.AdamOptimizer(stepsize=0.03)
        return run_native(
            cost_function=cost_function, 
            init_params=init_params, 
            optimizer=optimizer, 
            steps=steps, 
            precision=1e-6,
            silence=silence
            )
        

# steps=opt_steps

def run_native(cost_function, init_params, optimizer, steps=400, precision=1e-6, silence=True):
    params = np.array(init_params, requires_grad=True)
    energies = []
    flag = True
    near_end = 0
    i = 0
    while flag:
        params, energy = optimizer.step_and_cost(cost_function, params)
        energies.append(energy)
        if not silence and i % 10 == 0: print(f"Step {i:3d} | Energy: {energy:.6f}")
        if i > 2 and abs(energies[-1] - energies[-2]) < precision: near_end += 1
        else: near_end = 0
        if near_end == 20 or i >= steps: flag = False
        i += 1
    return params, energies


def run_lbfgsb(cost_function, init_params, maxiter=300, gtol=1e-8, silence=True):

    init_params = np.array(init_params, requires_grad=True)
    shape = init_params.shape          # (2, p): [gammas, betas]

    grad_fn = qp.grad(cost_function)
    energies = []

    def flat_cost(flat_params):
        params = np.array(flat_params.reshape(shape), requires_grad=True)
        energy = cost_function(params)
        energies.append(float(energy))
        if not silence and len(energies) % 10 == 0:
            print(f"Eval {len(energies):3d} | Energy: {float(energy):.6f}")
        return float(energy)

    def flat_grad(flat_params):
        params = np.array(flat_params.reshape(shape), requires_grad=True)
        grad = grad_fn(params)
        return np.array(grad).reshape(-1)

    x0 = np.array(init_params).reshape(-1)

    result = minimize(
        flat_cost,
        x0,
        jac=flat_grad,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "gtol": gtol, "disp": False},
    )

    best_params = np.array(result.x.reshape(shape), requires_grad=True)
    return best_params, energies


# ======================================================================================== #
#                                         'random'                                         #
# ======================================================================================== #

def param_restarts(cost_function, n_restarts, p, optimizer, opt_steps, silence):
    best_params = None
    best_energies = None
    best_energy_ps = []
    for i in range(n_restarts):
        if not silence: print(f"Restart {i}")
        init_params_i = ws.random_init_param(p)
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
    if not silence: plot_energies(best_energies)
    return best_params, best_energies, best_energy_ps

# ======================================================================================== #
#                                         'interp'                                         #
# ======================================================================================== #

def linear_interpolation(params_list, p):     # Convert from p-dimensional gamma and beta to (p+1)-dimensional
    params_arr = np.array(params_list)
    new_params = np.zeros((params_arr.shape[0], p+1))
    new_params[:, 0] = params_arr[:, 0]
    new_params[:, -1] = params_arr[:, -1]
    for i in range(1, p):
        new_params[:, i] = (i / p) * params_arr[:, i-1] + ((p-i) / p) * params_arr[:, i]
    return new_params.tolist()

def general_linear_interpolation(params_list, p, q):     # Convert from p-dimensional gamma and beta to (p+1)-dimensional
    params_arr = np.array(params_list)
    if q == 1:
        return np.mean(params_arr, axis=1, keepdims=True).tolist()
    positions = np.linspace(0, p-1, q)
    floors = np.floor(positions).astype(int)
    ceilings = np.minimum(floors + 1, p-1)
    distances = positions - floors
    new_params = (1 - distances) * params_arr[:, floors] + distances * params_arr[:, ceilings]
    return new_params.tolist()

def interp_params(cost_function_p, init_param, optimizer, opt_steps, q, silence):
    if q == 1:
        init_params = ws.mixed_init_param(1, init_param)
        print("Layer 1 optimization...")
        opt_params, energies = run_optimization(
            cost_function=cost_function_p(1),
            init_params=init_params, 
            optimizer=optimizer,
            steps=opt_steps,
            silence=silence
            )
        if not silence: plot_energies(energies)
        best_energy_ps = [energies]
        return opt_params, best_energy_ps
    
    prev_layer_params, best_energy_ps = interp_params(
        cost_function_p, 
        init_param=init_param,
        optimizer=optimizer, 
        opt_steps=opt_steps, 
        q=q-1, 
        silence=silence)
    print(f"Layer {q} optimization...")
    new_params = linear_interpolation(prev_layer_params, q-1)
    init_params = np.array(new_params, requires_grad=True)
    opt_params, energies = run_optimization(
            cost_function=cost_function_p(q),
            init_params=init_params,
            optimizer=optimizer,
            steps=opt_steps,
            silence=silence
            )
    best_energy_ps.append(energies)
    if not silence: plot_energies(energies)
    return opt_params, best_energy_ps


# ======================================================================================== #
#                                     'fourier(q, r)'                                      #
# ======================================================================================== #

def uv_to_gammabeta(u, v, p):
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    q = len(u)

    k = np.arange(1, q+1)
    i = np.arange(1, p+1)

    angle = np.outer(k - 0.5, i - 0.5) * np.pi / p     # (q, p)

    gamma = np.dot(u, np.sin(angle))
    beta = np.dot(v, np.cos(angle))

    return gamma, beta


def cost_function_uv(base_cost_function_p, p):
    base_cost = base_cost_function_p(p)

    def cost(params_uv):
        u, v = params_uv[0], params_uv[1]
        gamma, beta = uv_to_gammabeta(u, v, p)
        return base_cost(np.array([gamma, beta]))
    
    return cost


def pad_uv(u, v, q_new):
    u = np.array(u)
    v = np.array(v)
    pad_len = q_new - len(u)
    if pad_len <= 0:
        return u[:q_new], v[:q_new]
    new_u = np.concatenate([u, np.zeros(pad_len)])
    new_v = np.concatenate([v, np.zeros(pad_len)])
    return np.array(new_u, requires_grad=True), np.array(new_v, requires_grad=True)


def perturb_uv(u, v, alp=0.6):
    u = np.array(u)
    v = np.array(v)
    noise_u = np.random.normal(loc=0.0, scale=np.abs(u))
    noise_v = np.random.normal(loc=0.0, scale=np.abs(v))
    new_u = u + alp * noise_u
    new_v = v + alp * noise_v
    return np.array(new_u, requires_grad=True), np.array(new_v, requires_grad=True)


def fourier_params(cost_function_p, init_param, optimizer, opt_steps, p_max, q_max=None, R=0, alp=0.6, silence=True):
    uL = vL = uB = vB = None
    energy_history = []
    for p in range(1, p_max + 1):
        q_p = p if q_max is None else min(p, q_max)
        cost_fn = cost_function_uv(cost_function_p, p)

        # --- L-chain --- #
        
        if p == 1:
            init_params_1 = np.array(ws.random_init_param(1, init_param), requires_grad=True)
            init_u, init_v = init_params_1[0], init_params_1[1]
        else:
            init_u, init_v = pad_uv(uL, vL, q_p)

        init_params_L = np.array([init_u, init_v], requires_grad=True)
        opt_params_L, energies_L = run_optimization(
            cost_function=cost_fn,
            init_params=init_params_L,
            optimizer=optimizer,
            steps=opt_steps,
            silence=True,
        )
        uL, vL = opt_params_L[0], opt_params_L[1]
        level_energies = [energies_L]

        if R == 0:
            uB, vB = uL, vL
            if not silence:
                print(f"p={p:3d} | q={q_p:3d} | Energy: {energies_L[-1]:.6f}")
            energy_history.append(energies_L)
            continue

        # --- B-chain --- #
        candidates = [(opt_params_L, energies_L[-1], energies_L)]

        if p == 1:
            b_init_u, b_init_v = init_u, init_v
        else:
            b_init_u, b_init_v = pad_uv(uB, vB, q_p)

        init_params_B0 = np.array([b_init_u, b_init_v], requires_grad=True)
        opt_params_B0, energies_B0 = run_optimization(
            cost_function=cost_fn,
            init_params=init_params_B0,
            optimizer=optimizer,
            steps=opt_steps,
            silence=True,
        )
        candidates.append((opt_params_B0, energies_B0[-1], energies_B0))

        for r in range(1, R+1):
            pert_u, pert_v = perturb_uv(b_init_u, b_init_v, alp=alp)
            init_params_Br = np.array([pert_u, pert_v], requires_grad=True)
            opt_params_Br, energies_Br = run_optimization(
                cost_function=cost_fn,
                init_params=init_params_Br,
                optimizer=optimizer,
                steps=opt_steps,
                silence=True,
            )
            candidates.append((opt_params_Br, energies_Br[-1], energies_Br))
            level_energies.append(energies_Br)

        best_params, best_energy, best_energies = min(candidates, key=lambda c: c[1])
        uB, vB = best_params[0], best_params[1]

        if not silence:
            print(f"p={p:3d} | q={q_p:3d} | best of {len(candidates)}: {best_energy:.6f}")
            plot_energies(best_energies)
 
        energy_history.append(best_energies)
 
    return (uL, vL), (uB, vB), energy_history

