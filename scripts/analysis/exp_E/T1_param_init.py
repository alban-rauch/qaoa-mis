"""
data_analysis/param_init_test.py
"""

import pennylane as qml
from pennylane import numpy as np
from functools import partial
from pathlib import Path

import source.qaoa_run as qr
import source.circuit.ansatz as ans
import source.utils.classical as clas
import source.utils.graph_gen as gph

from source.optimization.parameter_transfer import PARAM_TRANSFER_REGISTRY
from source.paths import GRAPHS_DIR

def run_qaoa(problem, strategy, apparatus, grid_size, silence=True):

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

    counter = qr.CircuitCounter()

    # -----------------  STEP 1:  Build QAOA ansatz  ----------------- #

    cost_h, mixer_fns, angles, _ = qr.build_hamiltonians(
        graph,
        penalizer, 
        constrained, 
        relaxation_type, 
        mixer_fns
        )

    circuit = ans.make_circuit(ans.qaoa_layer)

    dev = qml.device(device, wires=wires)

    cost_qnode, _ = qr.estimation_framework(
        wires,
        p, 
        dev, 
        circuit,
        cost_h, 
        mixer_fns,
        angles,
        counter
        )

    # ----------------  STEP 2:  Optimize parameters  ---------------- #

    cost_function_p = lambda p: partial(
        cost_qnode, circuit=circuit, wires=wires, p=p, 
        cost_h=cost_h, mixer_fns=mixer_fns, angles=angles
    )

    param_transfer_fn = PARAM_TRANSFER_REGISTRY[strategy["param_transfer_type"]].build

    map_mat = np.zeros((p, grid_size+1, grid_size+1))

    theo_best_cost, _ = clas.best_config_branch_bound(graph)
    for i in range(grid_size+1):
        for j in range(grid_size+1):
            strategy["init_param"] = [i * 2 * np.pi / grid_size, j * np.pi / grid_size]

            _, _, best_energy_ps = param_transfer_fn(
                cost_function_p, strategy, apparatus, silence=silence
            )

    # ------------------  STEP 4: Extract solution  ------------------ #

            for q in range(1, p+1):
                energy = best_energy_ps[q-1][-1]
                approximation_ratio = qr.approx_ratio(graph, energy, penalizer, theo_best_cost)
                map_mat[q-1, i, j] = approximation_ratio
                print(f"{(i, j, q)} done")

    # ---------------------------  Output  --------------------------- #

    return np.transpose(map_mat, (0, 2, 1))

def generate_mat(problem_config, strategy_config, apparatus_config, grid_size, filename):
    data_mat = run_qaoa(
            problem=problem_config,
            strategy=strategy_config,
            apparatus=apparatus_config,
            grid_size=grid_size,
            silence=True
    )
    parameters = np.array(
        [problem_config, strategy_config, apparatus_config]
    )
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
    "init_param": [0.55, 0.27],
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

if __name__ == "__main__":
    foldername = Path("data/analysis_data/mapping")
    foldername.mkdir(parents=True, exist_ok=True)
    graphs_coll = gph.load_family(
        GRAPHS_DIR / f'Gilbert.npz', 
        "Gilbert", 
        params=[(12, 0.25)]
    )
    problem_config["N"] = 12
    for i in range(8):
        print(f"=== Graph sample {i+1}/8 ===")
        graph = gph.get_graph_from_edges(
            gph.get_sample(graphs_coll, (12, 0.25), s=i),
            N=12
        )
        problem_config["graph"] = graph
        
        filename = foldername / f'data_file_{i}.npz'
        generate_mat(problem_config, strategy_config, apparatus_config, grid_size=20, filename=filename)

    # foldername = Path("plots/T1/3reg-N8-p5-(20x20)")
    # foldername.mkdir(parents=True, exist_ok=True)
    # for idx in range(8):
    #     print(f"=== Graph sample {idx+1}/8 ===")
    #     problem_config["N"] = 8
    #     problem_config["graph"] = gph.randomDRegular(8, 3)
    #     filename = foldername / f'data_file_{idx}.npz'
    #     generate_mat(problem_config, strategy_config, apparatus_config, num_samples=20, filename=filename)
        

# for idx in range(1):
#     print("Graph sample", idx+1)
#     graph = gph.randomDRegular(16, 3)
#     problem_config["N"] = 16
#     problem_config["graph"] = graph
#     generate_mat(problem_config, strategy_config, apparatus_config, num_samples=20, id_name=f'N15d3')
#     graph = gph.randomDRegular(12, 5)
#     problem_config["N"] = 12
#     problem_config["graph"] = graph
#     generate_mat(problem_config, strategy_config, apparatus_config, num_samples=20, id_name=f'N12d5')
    
