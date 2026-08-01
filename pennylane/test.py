import pennylane as qml
from pennylane import numpy as np
from functools import partial
from matplotlib import pyplot as plt

import qaoa_run as qr
import circuit.ansatz as ans

from plotting import energy_evolution
import circuit.ansatz as ans
from optimization.parameter_transfer import PARAM_TRANSFER_REGISTRY

import seaborn as sns

def run_qaoa(problem, strategy, apparatus, silence=True):

    # ----------------------  Expand variables  ---------------------- #

    graph = problem["graph"]
    N = problem["N"]
    wires = range(N)

    constrained = strategy["constrained"]
    penalizer = 1.5 if not constrained else 0.0
    relaxation_type = strategy["relaxation_type"]
    mixer_fns = strategy["mixers"]

    p = apparatus["p"]
    device = apparatus["device"]
    estimator_shots = apparatus["estimator_shots"]
    sampler_shots = apparatus["sampler_shots"]


    # -----------------  STEP 1:  Build QAOA ansatz  ----------------- #

    cost_h, mixer_fns, angles = qr.build_hamiltonians(
        graph,
        penalizer, 
        constrained, 
        relaxation_type, 
        mixer_fns
        )

    circuit = ans.make_circuit(ans.qaoa_layer)

    dev = qml.device(device, wires=wires)

    cost_qnode, cost_function = qr.estimation_framework(
        wires,
        p, 
        dev, 
        circuit,
        cost_h, 
        mixer_fns,
        angles
        )

    sampling_qnode, probability_circuit = qr.sampling_framework(
        wires, 
        p, 
        dev, 
        sampler_shots, 
        circuit,
        cost_h, 
        mixer_fns, 
        angles
        )


    # ----------------  STEP 2:  Optimize parameters  ---------------- #

    cost_function_p = lambda p: partial(
        cost_qnode, circuit=circuit, wires=wires, p=p, 
        cost_h=cost_h, mixer_fns=mixer_fns, angles=angles
    )

    param_transfer_fn = PARAM_TRANSFER_REGISTRY[strategy["param_transfer_type"]].build

    map_mat = np.zeros((p, 10, 10))

    for q in range(1, p+1):
        for i in range(10):
            for j in range(10):
                strategy["init_param"] = [i * 2 * np.pi / 100, j * np.pi / 100]

                best_params, best_energies, best_energy_ps = param_transfer_fn(
                    cost_function_p, strategy, apparatus, silence=silence
                )


                if not silence:
                    energy_evolution(graph, best_energy_ps, penalizer)
                    print("Optimal Parameters:", best_params)

                # ----------------------  STEP 3: Sampling  ---------------------- #

                probs = probability_circuit(best_params)
                
                if not silence:
                    plt.style.use("seaborn-v0_8") 
                    plt.bar(range(2 ** len(wires)), probs)
                    plt.show()

                # ------------------  STEP 4: Extract solution  ------------------ #

                best_energy, approximation_ratio, success = qr.extract_solutions(
                    graph, 
                    probs, 
                    best_energies, 
                    penalizer, 
                    wires, 
                    silence
                    )

                map_mat[q, i, j] = approximation_ratio
                print(f"{(i,j)} done")
        map_mat[q, :, :] = map_mat[q, :, :].T
    return map_mat

def heatmap(data_mat):
    sns.heatmap(data_mat, annot=False, cmap="coolwarm", cbar=True)

    plt.xlabel("gamma")
    plt.ylabel("beta")
    plt.show()

# rng = np.random.default_rng()
# data_mat = rng.random((10, 10))
# heatmap(data_mat)