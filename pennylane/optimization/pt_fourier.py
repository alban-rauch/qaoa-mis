# ======================================================================================== #
#                                     'fourier(q, r)'                                      #
# ======================================================================================== #

from pennylane import numpy as np

from circuit.warm_start import mixed_init_param
from .optimization_process import run_optimization, plot_energies

def fourier_to_params(uvetc, p, sin_nb=1):
    uvetc = [np.atleast_1d(u) for u in uvetc]
    q = len(uvetc[0])

    k = np.arange(1, q+1)
    i = np.arange(1, p+1)
    angle = np.outer(k - 0.5, i - 0.5) * np.pi / p     # (q, p)

    params = []
    for j, u in enumerate(uvetc):
        basis = np.sin if j < sin_nb else np.cos
        params.append(np.dot(u, basis(angle)))

    return np.array(params)


def cost_function_fourier(base_cost_function_p, p):
    base_cost = base_cost_function_p(p)

    def cost(params_uv):
        params = fourier_to_params(params_uv, p)
        return base_cost(params)
    
    return cost


def pad_uv(uvetc, q_new):
    uvetc = [np.array(u) for u in uvetc]
    pad_len = q_new - len(uvetc[0])
    if pad_len <= 0:
        return [u[:q_new] for u in uvetc]
    new_uvetc = []
    for u in uvetc:
        new_uvetc.append(np.concatenate([u, np.zeros(pad_len)]))
    return [np.array(new_u, requires_grad=True) for new_u in new_uvetc]


def perturb_uv(uvetc, alp=0.6):
    uvetc = [np.array(u) for u in uvetc]
    noise_uvetc = [np.random.normal(0.0, np.abs(u)) for u in uvetc]
    new_uvetc = [
        u + alp * noise_u 
        for u, noise_u in zip(uvetc, noise_uvetc)
    ]
    return [np.array(new_u, requires_grad=True) for new_u in new_uvetc]


def fourier_params(
        cost_function_p, 
        init_param, 
        optimizer, opt_steps, 
        p_max, 
        q_max=None, R=0, 
        alp=0.6, 
        sin_nb=1, 
        silence=True
        ):
    coeffsL = coeffsB = None
    energy_history = []
    for p in range(1, p_max + 1):
        q_p = p if q_max is None else min(p, q_max)
        cost_fn = cost_function_fourier(cost_function_p, p, sin_nb)

        # --- L-chain --- #
        
        if p == 1:
            init_coeffs = np.array(mixed_init_param(1, init_param), requires_grad=True)
        else:
            init_coeffs = pad_uv(coeffsL, q_p)

        init_coeffs_L = np.array(init_coeffs, requires_grad=True)
        opt_coeffs_L, energies_L = run_optimization(
            cost_function=cost_fn,
            init_params=init_coeffs_L,
            optimizer=optimizer,
            steps=opt_steps,
            silence=True,
        )
        coeffsL = opt_coeffs_L
        level_energies = [energies_L]

        if R == 0:
            coeffsB = coeffsL
            if not silence:
                print(f"p={p:3d} | q={q_p:3d} | Energy: {energies_L[-1]:.6f}")
            energy_history.append(energies_L)
            continue

        # --- B-chain --- #
        candidates = [(opt_coeffs_L, energies_L[-1], energies_L)]

        if p == 1:
            b_init_coeffs = init_coeffs
        else:
            b_init_coeffs = pad_uv(coeffsB, q_p)

        init_coeffs_B0 = np.array(b_init_coeffs, requires_grad=True)
        opt_coeffs_B0, energies_B0 = run_optimization(
            cost_function=cost_fn,
            init_params=init_coeffs_B0,
            optimizer=optimizer,
            steps=opt_steps,
            silence=True,
        )
        candidates.append((opt_coeffs_B0, energies_B0[-1], energies_B0))

        for r in range(1, R+1):
            pert_coeffs = perturb_uv(b_init_coeffs, alp=alp)
            init_coeffs_Br = np.array(pert_coeffs, requires_grad=True)
            opt_coeffs_Br, energies_Br = run_optimization(
                cost_function=cost_fn,
                init_params=init_coeffs_Br,
                optimizer=optimizer,
                steps=opt_steps,
                silence=True,
            )
            candidates.append((opt_coeffs_Br, energies_Br[-1], energies_Br))
            level_energies.append(energies_Br)

        best_coeffs_B, best_energy, best_energies = min(candidates, key=lambda c: c[1])
        coeffsB = best_coeffs_B
        
        if not silence:
            print(f"p={p:3d} | q={q_p:3d} | best of {len(candidates)}: {best_energy:.6f}")
            plot_energies(best_energies)
 
        energy_history.append(best_energies)
 
    return coeffsL, coeffsB, energy_history

def fourier_pt(cost_function_p, strategy, apparatus, silence=True):
    p = apparatus["p"]
    bestL, bestB, best_energy_ps = fourier_params(
        cost_function_p, 
        init_param=strategy["init_param"], 
        optimizer=apparatus["optimizer"], 
        opt_steps=apparatus["opt_steps"], 
        p_max=p, 
        q_max=strategy["fourier_qR"][0], 
        R=strategy["fourier_qR"][1], 
        alp=0.6, 
        sin_nb=1, 
        silence=silence
        )
    best_params = fourier_to_params(bestB, p)
    best_energies = best_energy_ps[-1]
    return best_params, best_energies, best_energy_ps