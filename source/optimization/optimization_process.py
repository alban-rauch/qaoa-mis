"""
optimization/optimization_process.py
===============
Classical parameter optimizer.
"""

import pennylane as qml
from pennylane import numpy as np

from scipy.optimize import minimize

def run_optimization(
        cost_function, 
        init_params, 
        optimizer, 
        steps=300, 
        precision=1e-8, 
        silence=True,
):
    if optimizer == "L-BFGS-B":
        return run_lbfgsb(
            cost_function=cost_function, 
            init_params=init_params, 
            maxiter=steps, 
            gtol=precision,
            silence=silence,
        )
    
    else:
        if optimizer == "Adam":
            optimizer = qml.AdamOptimizer(stepsize=0.03)

        return run_native(
            cost_function=cost_function, 
            init_params=init_params, 
            optimizer=optimizer, 
            steps=steps, 
            precision=1e-6,
            silence=silence,
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

        if not silence and i % 10 == 0:
            print(f"Step {i:3d} | Energy: {energy:.6f}")

        if i > 2 and abs(energies[-1] - energies[-2]) < precision:
            near_end += 1
        else:
            near_end = 0

        if near_end == 20 or i >= steps:
            flag = False

        i += 1

    return params, energies


def run_lbfgsb(cost_function, init_params, maxiter=300, gtol=1e-8, silence=True):

    init_params = np.array(init_params, requires_grad=True)
    shape = init_params.shape          # (2, p): [gammas, betas]

    grad_fn = qml.grad(cost_function)
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


