
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt

import src.auxiliary.graphs as gph
import src.circuit.warm_start as ws
import src.circuit.ansatz as qa
import src.qaoa_run as qr


# ======================================================================================== #
#                                       Performance                                        #
# ======================================================================================== #

device = "lightning.qubit" 

estimator_shots = 10000
sampler_shots = 10000


optimizer = "L-BFGS-B"
opt_steps = 400

constrained = False

relaxation_type = None
param_transfer_type = 'interp'
(fourier_q, fourier_R) = (None, 5)

conditions = [
    device,
    estimator_shots, sampler_shots,
    optimizer, opt_steps,
    constrained,
    relaxation_type,
    param_transfer_type,
    fourier_q,
    fourier_R,
    ]


def make_qaoa_pipeline(
        device,
        estimator_shots, sampler_shots,
        optimizer, opt_steps,
        constrained,
        relaxation_type,
        param_transfer_type,
        fourier_q,
        fourier_R,
        ):

    problem_config = {
        "N": None,
        "graph": None,
    }

    strategy_config = {
        "constrained": constrained,
        "relaxation_type": relaxation_type,
        "param_transfer_type": param_transfer_type,
        "fourier_qR": (fourier_q, fourier_R)
    }

    apparatus_config = {
        "p": None,
        "device": device,
        "estimator_shots": estimator_shots,
        "sampler_shots": sampler_shots,
        "optimizer": optimizer,
        "opt_steps": opt_steps,
    }

    def qaoa_pipeline(p, N):
        problem_config["N"] = N
        apparatus_config["p"] = p
        graph = gph.complete_graph(N)
        problem_config["graph"] = graph
        return qr.standard_qaoa(problem_config, strategy_config, apparatus_config, silence=True)
    
    return qaoa_pipeline


def repeated_qaoa(p_values, N_values, n_graphs, *conditions):
    print(*conditions)
    pipeline = make_qaoa_pipeline(*conditions)
    
    n_p = len(p_values)
    n_N = len(N_values)

    ar_samples = np.zeros((n_p, n_N, n_graphs))
    sr_samples = np.zeros((n_p, n_N, n_graphs))

    for p_idx in range(n_p):
        p_val = int(p_values[p_idx])
        for N_idx in range(n_N):
            N_val = int(N_values[N_idx])
            for g_idx in range(n_graphs):
                result = pipeline(p_val, N_val)
                ar_samples[p_idx, N_idx, g_idx] = result["approximation_ratio"]
                sr_samples[p_idx, N_idx, g_idx] = result["success"]
                print(f"{(p_val, N_val)} completed")

    return {
        "ar_samples": ar_samples,
        "sr_samples": sr_samples
    }

p_values = np.arange(1, 7)
N_values = np.arange(4, 15)
results = repeated_qaoa(p_values, N_values, 1, *conditions)
ar_samples = results["ar_samples"]
sr_samples = results["sr_samples"]
np.savez(
    "diagnostic.npz", 
    p_values=p_values, 
    N_values=N_values, 
    ar_samples=ar_samples, 
    sr_samples=sr_samples
    )


# ======================================================================================== #
#                                     Barren landscape                                     #
# ======================================================================================== #

def make_cost_function_builder(constrained, relaxation_type, device):
    
    def cost_function_builder(p, N):
        graph = gph.circular_graph(N)
        penalizer = 1.5 if not constrained else 0.0
        wires = range(N)

        cost_h, mixer_layer, angles = qr.build_hamiltonians(
            graph,
            penalizer, 
            constrained, 
            relaxation_type, 
            )
        
        dev = qml.device(device, wires=wires)
        _, cost_function = qr.estimation_framework(
            wires,
            p, 
            dev, 
            cost_h, 
            mixer_layer, 
            angles
            )
        return cost_function
    
    return cost_function_builder
        

def gradient_variance_scan(cost_function_builder, N_values, p_values, n_graphs=5, n_samples=10):
    variances = []
    for p in p_values:
        variances_p = []
        for N in N_values:
            grad0_samples = []
            for _ in range(n_graphs):
                cost_fn = cost_function_builder(p, N)
                grad_fn = qml.grad(cost_fn)
                for _ in range(n_samples):
                    params = ws.random_init_param(p)
                    g = grad_fn(params)
                    grad0_samples.append(g[0][0])
            variances_p.append(np.var(grad0_samples))
            print(f"{(p, N)} done")
        variances.append(variances_p)
    return variances



# builder = make_cost_function_builder(constrained=False, relaxation_type=None, device="lightning.qubit")
# N_vals = range(3, 16)
# p_vals = range(1, 8)
# variances = gradient_variance_scan(builder, N_vals, p_vals, n_graphs=20, n_samples=20)


# plt.figure(figsize=(8, 5))

# for p, vars_for_p in zip(p_vals, variances):
#     plt.semilogy(
#         N_vals, 
#         vars_for_p, 
#         marker="o", 
#         linestyle="-", 
#         label=f"p = {p}"
#     )

# plt.xlabel("Number of Qubits (N)")
# plt.ylabel(r"Gradient Variance $\text{Var}(\partial E)$")
# plt.title("QAOA Gradient Variance Scaling by Depth")
# plt.grid(True, which="both", linestyle="--", alpha=0.3)
# plt.legend(title="QAOA Depth (p)")
# plt.show()