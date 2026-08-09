"""
data_analysis/param_init_test.py
"""

import pennylane as qml
from pennylane import numpy as np
from functools import partial
from matplotlib import pyplot as plt
import seaborn as sns

import qaoa_run as qr
import circuit.ansatz as ans
import circuit.warm_start as ws
import auxiliary.classical as clas
import auxiliary.graphs as gph

import circuit.ansatz as ans
from optimization.parameter_transfer import PARAM_TRANSFER_REGISTRY


def run_qaoa(problem, strategy, apparatus, num_samples, silence=True):

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

    map_mat = np.zeros((p, num_samples+1, num_samples+1))

    
    for i in range(num_samples+1):
        for j in range(num_samples+1):
            strategy["init_param"] = [i * 2 * np.pi / num_samples, j * np.pi / num_samples]

            best_params, best_energies, best_energy_ps = param_transfer_fn(
                cost_function_p, strategy, apparatus, silence=silence
            )


            if not silence:
                # plot.energy_evolution(graph, best_energy_ps, penalizer)
                print("Optimal Parameters:", best_params)

    # ----------------------  STEP 3: Sampling  ---------------------- #

            probs = probability_circuit(best_params)

            if not silence:
                # plot.bitstring_probas(wires, probs)
                pass

    # ------------------  STEP 4: Extract solution  ------------------ #

            theo_best_cost, theo_best_config = clas.best_config_branch_bound(graph)
            for q in range(1, p+1):
                energy = best_energy_ps[q-1][-1]
                approximation_ratio = qr.approx_ratio(graph, energy, penalizer, theo_best_cost)
                map_mat[q-1, i, j] = approximation_ratio
                print(f"{(i, j, q)} done")

    # ---------------------------  Output  --------------------------- #

    for q in range(1, p+1):
        map_mat[q-1, :, :] = map_mat[q-1, :, :].T

    return map_mat

def generate_mat(problem_config, strategy_config, apparatus_config, num_samples, id_name):
    data_mat = run_qaoa(
            problem=problem_config,
            strategy=strategy_config,
            apparatus=apparatus_config,
            num_samples=num_samples,
            silence=True
    )
    parameters = np.array(
        [problem_config, strategy_config, apparatus_config]
    )
    filename = f"data_mat_{id_name}"
    np.savez(filename, parameters=parameters, data_mat=data_mat)




N = 12
p = 5

problem_config = {
    "N": N,
    "graph": gph.randomGilbert(N, 0.25), # gph.randomDRegular(N, 3) | gph.randomGilbert(N, 0.25)
}

strategy_config = {
    "constrained": False,
    "relaxation_type": 'continuous',    #  None | 'continuous'
    "param_transfer_type": 'interp',    # 'given' | 'random' | 'interp' | 'fourier'
    "fourier_qR": (None, 5),
    "init_param": [0.55, 0.27, 0.11],
    "mixers": ["x", "y"],
}

apparatus_config = {
    "p": p,
    "device": "lightning.qubit",        # "lightning.qubit" | "lightning.amdgpu" 
    "estimator_shots": 10000,
    "sampler_shots": 10000,
    "optimizer": "L-BFGS-B",            # "L-BFGS-B" | "Adam"
    "opt_steps": 400,
}

for idx in range(10):
    print("Graph sample", idx+1)
    graph = gph.randomGilbert(N, 0.25)
    problem_config["graph"] = graph
    generate_mat(problem_config, strategy_config, apparatus_config, num_samples=20, id_name=f'{idx+1}')
