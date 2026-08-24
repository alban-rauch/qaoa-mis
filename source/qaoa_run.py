"""
qaoa_run.py
===========
One full run of QAOA.
"""

import pennylane as qml
from pennylane import numpy as np

from functools import partial
import time

import source.utils.classical as clas
import source.circuit.ansatz as ans
import source.circuit.warm_start as ws
from source.circuit.mixers import MIXER_REGISTRY
from source.optimization.parameter_transfer import PARAM_TRANSFER_REGISTRY


class CircuitCounter:
    def __init__(self):
        self.cost_circuit_evals = 0
        self.sampling_circuit_evals = 0


def build_hamiltonians(graph, penalizer, constrained, relaxation_type, mixer_names):
    node_list = list(graph.nodes)
    edge_list = list(graph.edges)
    degrees = {i: graph.degree(i) for i in node_list}
    wires = range(len(node_list))
    
    cost_h = ans.cost_hamiltonian(node_list, edge_list, degrees, penalizer)

    x_d, _ = ws.relaxation(graph, wires)
    angles = ws.relaxation_angles(x_d, eps=0.25) if relaxation_type == 'continuous' else [0.5 * np.pi] * len(wires)
    mixer_fns = [MIXER_REGISTRY[name].build(graph, angles, constrained) for name in mixer_names]

    return cost_h, mixer_fns, angles, x_d


def estimation_framework(wires, p, dev, circuit, cost_h, mixer_fns, angles, counter):
    cost_qnode = qml.QNode(
        ans.estimator,
        device=dev,
        diff_method="adjoint",  # or "parameter-shift"
        shots=None
    )
    cost_function = partial(
        cost_qnode,
        circuit=circuit,
        wires=wires,
        p=p,
        cost_h=cost_h,
        mixer_fns=mixer_fns,
        angles=angles,
        counter=counter,
    )
    return cost_qnode, cost_function


def sampling_framework(wires, p, dev, sampler_shots, circuit, cost_h, mixer_fns, angles, counter):
    sampling_qnode = qml.QNode(
        ans.sampler,
        device=dev,
        shots=sampler_shots
    )
    probability_circuit = partial(
        sampling_qnode,
        circuit=circuit,
        wires=wires,
        p=p,
        cost_h=cost_h,
        mixer_fns=mixer_fns,
        angles=angles,
        counter=counter,
    )
    return sampling_qnode, probability_circuit

def approx_ratio(graph, best_energy, penalizer, theo_best_cost):
    node_list = list(graph.nodes)
    edge_list = list(graph.edges)
    best_cost = ans.energy_to_cost(best_energy, penalizer, node_list, edge_list)
    approximation_ratio = best_cost / theo_best_cost
    return approximation_ratio

def extract_solutions(graph, wires, probs, theo_best_config, silence):
    most_likely_idx = np.argmax(probs)
    most_likely_bin = [(most_likely_idx >> i) & 1 for i in reversed(range(len(wires)))]
    most_likely_bitstring = clas.list_to_string(most_likely_bin)

    if not silence:
        print("Optimal:", most_likely_bitstring)
        # plot.draw_select(graph, most_likely_bin)

    success = most_likely_bitstring in theo_best_config

    return most_likely_bitstring, success


def run_qaoa(problem, strategy, apparatus, silence=False):

    # ----------------------  Expand variables  ---------------------- #
    
    time0 = time.perf_counter()

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

    counter = CircuitCounter()

    # -----------------  STEP 1:  Build QAOA ansatz  ----------------- #

    time1 = time.perf_counter()

    cost_h, mixer_fns, angles, x_d = build_hamiltonians(
        graph,
        penalizer, 
        constrained, 
        relaxation_type, 
        mixer_fns
        )

    circuit = ans.make_circuit(ans.qaoa_layer)

    dev = qml.device(device, wires=wires)

    cost_qnode, cost_function = estimation_framework(
        wires,
        p, 
        dev, 
        circuit,
        cost_h, 
        mixer_fns,
        angles,
        counter
        )

    sampling_qnode, probability_circuit = sampling_framework(
        wires, 
        p, 
        dev, 
        sampler_shots, 
        circuit,
        cost_h, 
        mixer_fns, 
        angles,
        counter
        )


    # ----------------  STEP 2:  Optimize parameters  ---------------- #

    time2 = time.perf_counter()

    cost_function_p = lambda p: partial(
        cost_qnode, circuit=circuit, wires=wires, p=p, 
        cost_h=cost_h, mixer_fns=mixer_fns, angles=angles, counter=counter,
    )

    param_transfer_fn = PARAM_TRANSFER_REGISTRY[strategy["param_transfer_type"]].build

    best_params, param_hist_ps, best_energy_p, best_energy_ps = param_transfer_fn(
        cost_function_p, strategy, apparatus, silence=silence
    )


    if not silence:
        # plot.energy_evolution(graph, best_energy_ps, penalizer)
        print("Optimal Parameters:", best_params)

    # ----------------------  STEP 3: Sampling  ---------------------- #

    time3 = time.perf_counter()

    probs = probability_circuit(best_params)
    
    if not silence:
        # plot.bitstring_probas(wires, probs)
        pass

    # ------------------  STEP 4: Extract solution  ------------------ #

    time4 = time.perf_counter()

    theo_best_cost, theo_best_config = clas.best_config_branch_bound(graph)
    best_energy = best_energy_p[-1]
    approximation_ratio = approx_ratio(graph, best_energy, penalizer, theo_best_cost)
    most_likely_bitstring, success = extract_solutions(
        graph, wires, probs, theo_best_config, silence
    )

    
    # ---------------------------  Output  --------------------------- #

    time5 = time.perf_counter()

    times = (
        time1 - time0, 
        time2 - time1, 
        time3 - time2, 
        time4 - time3,
        time5 - time4
    )


    return {

        # Configuration
        "problem": problem,
        "strategy": strategy,
        "apparatus": apparatus,

        # Warm start
        "relax_values": x_d,
        "relax_angles": angles,

        # Optimization
        "cost_function": cost_function,
        "cost_hamiltonian": cost_h,

        # Optimization
        "best_params": best_params,
        "param_hist": param_hist_ps[-1],
        "param_hist_ps": param_hist_ps,
        "best_energy": best_energy,
        "best_energy_p": best_energy_p,
        "best_energy_ps": best_energy_ps,

        # Sampling
        "probs": probs,
        "most_likely_bitstring": most_likely_bitstring,
        "max_probability": float(np.max(probs)),
        "success": success,

        # Quality
        "approximation_ratio": approximation_ratio,
        "theoretical_best_cost": theo_best_cost,
        "theoretical_best_config": theo_best_config,

        # Performance
        "cost_circuit_evals": counter.cost_circuit_evals,
        "sampling_circuit_evals": counter.sampling_circuit_evals,
        "times": times,

    }

