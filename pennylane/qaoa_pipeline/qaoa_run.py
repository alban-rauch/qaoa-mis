"""
qaoa_run.py
===========
One full run of QAOA.
"""

import pennylane as qp
from pennylane import numpy as np

from functools import partial
from matplotlib import pyplot as plt

import graphs as gph
import qaoa_pipeline.warm_start as ws
import qaoa_pipeline.ansatz as qa
import qaoa_pipeline.optimization as opt
import classical as clas


# ======================================================================================== #
#                                         Standard                                         #
# ======================================================================================== #

def build_hamiltonians(graph, penalizer, constrained, relaxation_type):
    node_list = list(graph.nodes)
    edge_list = list(graph.edges)
    degrees = {i: graph.degree(i) for i in node_list}
    wires = range(len(node_list))
    
    # Cost Hamiltonian
    cost_h = qa.cost_hamiltonian(node_list, edge_list, degrees, penalizer)

    # Warm started mixer
    if relaxation_type is None:
        angles = [0.5 * np.pi] * len(wires)
    elif relaxation_type == 'continuous':
        angles = ws.relaxation_angles(graph, wires, eps=0.25)
    if not constrained:
        mixer_layer = lambda beta: qa.relaxed_mixer_layer(beta, graph, angles)
    else:
        mixer_layer = lambda beta: qa.cst_relaxed_mixer_layer(beta, graph, angles)

    return cost_h, mixer_layer, angles

def estimation_framework(wires, p, dev, circuit, cost_h, mixer_layer, angles):
    cost_qnode = qp.QNode(
        qa.estimator,
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
        mixer_layer=mixer_layer,
        angles=angles
    )
    return cost_qnode, cost_function

def sampling_framework(wires, p, dev, sampler_shots, circuit, cost_h, mixer_layer, angles):
    sampling_qnode = qp.QNode(
        qa.sampler,
        device=dev,
        shots=sampler_shots
    )
    probability_circuit = partial(
        sampling_qnode,
        circuit=circuit,
        wires=wires,
        p=p,
        cost_h=cost_h,
        mixer_layer=mixer_layer,
        angles=angles
    )
    return sampling_qnode, probability_circuit

def extract_solutions(graph, probs, best_energies, penalizer, wires, silence):
    node_list = list(graph.nodes)
    edge_list = list(graph.edges)

    most_likely_idx = np.argmax(probs)
    most_likely_bin = [(most_likely_idx >> i) & 1 for i in reversed(range(len(wires)))]
    most_likely_bitstring = clas.list_to_string(most_likely_bin)
    if not silence:
        print("Optimal:", most_likely_bin)
        gph.draw_select(graph, most_likely_bin)
        plt.show()

    best_energy = best_energies[-1]
    best_cost = qa.energy_to_cost(best_energy, penalizer, node_list, edge_list)

    theo_best_cost, theo_best_config = clas.best_config_branch_bound(graph)
    approximation_ratio = best_cost / theo_best_cost

    success = most_likely_bitstring in theo_best_config

    return best_energy, approximation_ratio, success


def standard_qaoa(problem, strategy, apparatus, silence=False):
    
    # ----------------------  Expand variables  ---------------------- #

    graph = problem["graph"]
    N = problem["N"]
    wires = range(N)

    constrained = strategy["constrained"]
    penalizer = 1.5 if not constrained else 0.0
    relaxation_type = strategy["relaxation_type"]
    param_transfer_type = strategy["param_transfer_type"]
    fourier_q, fourier_R = strategy["fourier_qR"]
    init_param = strategy["init_param"]

    p = apparatus["p"]
    device = apparatus["device"]
    estimator_shots = apparatus["estimator_shots"]
    sampler_shots = apparatus["sampler_shots"]
    optimizer = apparatus["optimizer"]
    opt_steps = apparatus["opt_steps"]
    

    # -----------------  STEP 1:  Build QAOA ansatz  ----------------- #

    cost_h, mixer_layer, angles = build_hamiltonians(
        graph,
        penalizer, 
        constrained, 
        relaxation_type, 
        )
    
    circuit = qa.make_circuit(qa.qaoa_layer)

    dev = qp.device(device, wires=wires)

    cost_qnode, cost_function = estimation_framework(
        wires,
        p, 
        dev, 
        circuit,
        cost_h, 
        mixer_layer, 
        angles
        )

    sampling_qnode, probability_circuit = sampling_framework(
        wires, 
        p, 
        dev, 
        sampler_shots, 
        circuit,
        cost_h, 
        mixer_layer, 
        angles
        )


    # ----------------  STEP 2:  Optimize parameters  ---------------- #

    if param_transfer_type == 'given':
        angles_given = ws.mixed_init_param(p, init_param)
        best_params, best_energies = opt.run_optimization(
                cost_function=cost_function,
                init_params=angles_given, 
                optimizer=optimizer, 
                precision=0.01,
                silence=silence
                )

    elif param_transfer_type == 'random':
        best_params, best_energies, best_energy_ps = opt.param_restarts(
                cost_function=cost_function, 
                n_restarts=p,
                p=p,
                optimizer=optimizer, 
                opt_steps=opt_steps, 
                silence=silence
                )

    elif param_transfer_type == 'interp':
        cost_function_p = lambda p: partial(
                cost_qnode,
                circuit=circuit,
                wires=wires,
                p=p,
                cost_h=cost_h,
                mixer_layer=mixer_layer,
                angles=angles,
            )
        best_params, best_energy_ps = opt.interp_params(
                cost_function_p, 
                init_param=init_param, 
                optimizer=optimizer, 
                opt_steps=opt_steps,
                q=p, 
                silence=True
            )
        best_energies = best_energy_ps[-1]
    

    elif param_transfer_type == 'fourier':
        cost_function_p = lambda p: partial(
                cost_qnode,
                circuit=circuit,
                wires=wires,
                p=p,
                cost_h=cost_h,
                mixer_layer=mixer_layer,
                angles=angles,
            )
        (uL, vL), (uB, vB), best_energy_ps = opt.fourier_params(
            cost_function_p=cost_function_p,   # same one you use for interp_params
            init_param=init_param,
            optimizer=optimizer,
            opt_steps=300,
            p_max=p,
            q_max=fourier_q,
            R=fourier_R,
            alp=0.6,
            silence=silence,
        )
        gamma, beta = opt.uv_to_gammabeta(uB, vB, p)
        best_params = np.array([gamma, beta])
        best_energies = best_energy_ps[-1]

    if not silence:
        plt.style.use("default")
        theo_best_cost, _ = clas.best_config_branch_bound(graph)
        costs = [[qa.energy_to_cost(energy_val, penalizer, graph.nodes, graph.edges) / theo_best_cost 
                    for energy_val in sublist] 
                    for sublist in best_energy_ps]
        length = len(costs)
        cmap = plt.colormaps['cividis']
        colors = [cmap(i / (length - 1)) for i in range(length)] if length > 1 else [cmap(0)]
        current_x = 0
        plt.figure(figsize=(10, 5))
        current_x = 0
        for i in range(length):
            segment = costs[i]
            x_coords = [current_x + j for j in range(len(segment))]
            plt.plot(x_coords, segment, color=colors[i], linewidth=2)
            current_x += len(segment) - 1

        plt.title("Energy optimization by step for interp")
        plt.xlabel("Step")
        plt.ylabel("Approximation Ratio")

        plt.gca().set_facecolor("white")
        plt.minorticks_on()

        plt.grid(True, which="major", linestyle="-", linewidth=0.6, color="#898989", alpha=0.8)
        plt.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#b9b9b9", alpha=0.7)

        plt.tight_layout()
        plt.show()

    if not silence: print("Optimal Parameters:", best_params)

    # ----------------------  STEP 3: Sampling  ---------------------- #

    probs = probability_circuit(best_params)
    
    if not silence:
        plt.style.use("seaborn-v0_8") 
        plt.bar(range(2 ** len(wires)), probs)
        plt.show()

    # ------------------  STEP 4: Extract solution  ------------------ #

    best_energy, approximation_ratio, success = extract_solutions(graph, probs, best_energies, penalizer, wires, silence)

    return {
        "cost_function": cost_function,
        "cost_hamiltonian": cost_h,
        "best_params": best_params,
        "best_energy": best_energy,
        "approximation_ratio": approximation_ratio,
        "success": success,
    }


# ======================================================================================== #
#                                     y - mixer layer                                      #
# ======================================================================================== #

def y_build_hamiltonians(graph, penalizer, constrained, relaxation_type):
    node_list = list(graph.nodes)
    edge_list = list(graph.edges)
    degrees = {i: graph.degree(i) for i in node_list}
    wires = range(len(node_list))
    
    # Cost Hamiltonian
    cost_h = qa.cost_hamiltonian(node_list, edge_list, degrees, penalizer)

    # Warm started mixer
    if relaxation_type is None:
        angles = [0.5 * np.pi] * len(wires)
    elif relaxation_type == 'continuous':
        angles = ws.relaxation_angles(graph, wires, eps=0.25)
    if not constrained:
        mixer_layer = lambda beta: qa.relaxed_mixer_layer(beta, graph, angles)
    else:
        mixer_layer = lambda beta: qa.cst_relaxed_mixer_layer(beta, graph, angles)
    
    y_mixer = lambda alpha: qa.y_mixer_layer(alpha, graph)

    return cost_h, mixer_layer, y_mixer, angles

def y_estimation_framework(wires, p, dev, circuit, cost_h, mixer_layer, y_mixer, angles):
    cost_qnode = qp.QNode(
        qa.expressive_estimator,
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
        mixer_layer=mixer_layer,
        y_mixer=y_mixer,
        angles=angles
    )
    return cost_qnode, cost_function

def y_sampling_framework(wires, p, dev, sampler_shots, circuit, cost_h, mixer_layer, y_mixer, angles):
    sampling_qnode = qp.QNode(
        qa.expressive_sampler,
        device=dev,
        shots=sampler_shots
    )
    probability_circuit = partial(
        sampling_qnode,
        circuit=circuit,
        wires=wires,
        p=p,
        cost_h=cost_h,
        mixer_layer=mixer_layer,
        y_mixer=y_mixer,
        angles=angles
    )
    return sampling_qnode, probability_circuit

def y_standard_qaoa(problem, strategy, apparatus, silence=False):
    
    # ----------------------  Expand variables  ---------------------- #

    graph = problem["graph"]
    N = problem["N"]
    wires = range(N)

    constrained = strategy["constrained"]
    penalizer = 1.5 if not constrained else 0.0
    relaxation_type = strategy["relaxation_type"]
    param_transfer_type = strategy["param_transfer_type"]
    fourier_q, fourier_R = strategy["fourier_qR"]

    p = apparatus["p"]
    device = apparatus["device"]
    estimator_shots = apparatus["estimator_shots"]
    sampler_shots = apparatus["sampler_shots"]
    optimizer = apparatus["optimizer"]
    opt_steps = apparatus["opt_steps"]
    

    # -----------------  STEP 1:  Build QAOA ansatz  ----------------- #

    cost_h, mixer_layer, y_mixer, angles = y_build_hamiltonians(
        graph,
        penalizer, 
        constrained, 
        relaxation_type, 
        )
    
    circuit = qa.make_expressive_circuit(qa.expressive_qaoa_layer)

    dev = qp.device(device, wires=wires)

    cost_qnode, cost_function = y_estimation_framework(
        wires,
        p, 
        dev, 
        circuit,
        cost_h, 
        mixer_layer,
        y_mixer,
        angles
        )

    sampling_qnode, probability_circuit = y_sampling_framework(
        wires, 
        p, 
        dev, 
        sampler_shots, 
        circuit,
        cost_h, 
        mixer_layer,
        y_mixer,
        angles
        )


    # ----------------  STEP 2:  Optimize parameters  ---------------- #

    if param_transfer_type == 'given':
        angles_given = [[0.5] * p] * 2
        best_params, best_energies = opt.run_optimization(
                cost_function=cost_function,
                init_params=angles_given, 
                optimizer=optimizer, 
                precision=0.01,
                silence=silence
                )

    elif param_transfer_type == 'random':
        best_params, best_energies, best_energy_ps = opt.param_restarts(
                cost_function=cost_function, 
                n_restarts=p,
                p=p, 
                optimizer=optimizer, 
                opt_steps=opt_steps, 
                silence=silence
                )

    elif param_transfer_type == 'interp':
        cost_function_p = lambda p: partial(
                cost_qnode,
                circuit=circuit,
                wires=wires,
                p=p,
                cost_h=cost_h,
                mixer_layer=mixer_layer,
                angles=angles
            )
        best_params, best_energy_ps = opt.interp_params(
            cost_function_p, 
            optimizer, 
            opt_steps, 
            q=p, 
            silence=True
            )
        best_energies = best_energy_ps[-1]
    

    elif param_transfer_type == 'fourier':
        cost_function_p = lambda p: partial(
                cost_qnode,
                circuit=circuit,
                wires=wires,
                p=p,
                cost_h=cost_h,
                mixer_layer=mixer_layer,
                angles=angles
            )
        (uL, vL), (uB, vB), best_energy_ps = opt.fourier_params(
            cost_function_p=cost_function_p,   # same one you use for interp_params
            optimizer=optimizer,
            opt_steps=300,
            p_max=p,
            q_max=fourier_q,
            R=fourier_R,
            alp=0.6,
            silence=silence,
        )
        gamma, beta = opt.uv_to_gammabeta(uB, vB, p)
        best_params = np.array([gamma, beta])
        best_energies = best_energy_ps[-1]

    if not silence:
        plt.style.use("default")
        theo_best_cost, _ = clas.best_config_branch_bound(graph)
        costs = [[qa.energy_to_cost(energy_val, penalizer, graph.nodes, graph.edges) / theo_best_cost 
                    for energy_val in sublist] 
                    for sublist in best_energy_ps]
        length = len(costs)
        cmap = plt.colormaps['cividis']
        colors = [cmap(i / (length - 1)) for i in range(length)] if length > 1 else [cmap(0)]
        current_x = 0
        plt.figure(figsize=(10, 5))
        current_x = 0
        for i in range(length):
            segment = costs[i]
            x_coords = [current_x + j for j in range(len(segment))]
            plt.plot(x_coords, segment, color=colors[i], linewidth=2)
            current_x += len(segment) - 1

        plt.title("Energy optimization by step for interp")
        plt.xlabel("Step")
        plt.ylabel("Approximation Ratio")

        plt.gca().set_facecolor("white")
        plt.minorticks_on()

        plt.grid(True, which="major", linestyle="-", linewidth=0.6, color="#898989", alpha=0.8)
        plt.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#b9b9b9", alpha=0.7)

        plt.tight_layout()
        plt.show()

    if not silence: print("Optimal Parameters:", best_params)

    # ----------------------  STEP 3: Sampling  ---------------------- #

    probs = probability_circuit(best_params)
    
    if not silence:
        plt.style.use("seaborn-v0_8") 
        plt.bar(range(2 ** len(wires)), probs)
        plt.show()

    # ------------------  STEP 4: Extract solution  ------------------ #

    best_energy, approximation_ratio, success = extract_solutions(graph, probs, best_energies, penalizer, wires, silence)

    return {
        "cost_function": cost_function,
        "cost_hamiltonian": cost_h,
        "best_params": best_params,
        "best_energy": best_energy,
        "approximation_ratio": approximation_ratio,
        "success": success,
    }

# ======================================================================================== #
#                                       Multi angle                                        #
# ======================================================================================== #


def ma_build_hamiltonians(graph, penalizer, constrained, relaxation_type):
    node_list = list(graph.nodes)
    edge_list = list(graph.edges)
    degrees = {i: graph.degree(i) for i in node_list}
    wires = range(len(node_list))

    cost_h = qa.cost_hamiltonian(node_list, edge_list, degrees, penalizer)
    num_gamma_terms = len(cost_h.ops)

    if relaxation_type is None:
        angles = [0.5 * np.pi] * len(wires)
    elif relaxation_type == 'continuous':
        angles = ws.relaxation_angles(graph, wires, eps=0.25)

    if not constrained:
        mixer_layer = lambda betas: qa.ma_relaxed_mixer_layer(betas, graph, angles)
    else:
        mixer_layer = lambda betas: qa.ma_cst_relaxed_mixer_layer(betas, graph, angles)

    return cost_h, mixer_layer, angles, num_gamma_terms


def ma_estimation_framework(wires, p, dev, cost_h, mixer_layer, angles):
    cost_qnode = qp.QNode(
        qa.ma_estimator,
        device=dev,
        diff_method="adjoint",  # or "parameter-shift"
        shots=None
    )
    cost_function = partial(
        cost_qnode,
        wires=wires,
        p=p,
        cost_h=cost_h,
        mixer_layer=mixer_layer,
        angles=angles
    )
    return cost_qnode, cost_function

def ma_sampling_framework(wires, p, dev, sampler_shots, cost_h, mixer_layer, angles):
    sampling_qnode = qp.QNode(
        qa.ma_sampler,
        device=dev,
        shots=sampler_shots
    )
    probability_circuit = partial(
        sampling_qnode,
        wires=wires,
        p=p,
        cost_h=cost_h,
        mixer_layer=mixer_layer,
        angles=angles
    )
    return sampling_qnode, probability_circuit


def multi_angle_qaoa(problem, strategy, apparatus, silence=False):
    
    # ----------------------  Expand variables  ---------------------- #

    graph = problem["graph"]
    N = problem["N"]
    wires = range(N)

    constrained = strategy["constrained"]
    penalizer = 1.5 if not constrained else 0.0
    relaxation_type = strategy["relaxation_type"]
    param_transfer_type = strategy["param_transfer_type"]
    fourier_q, fourier_R = strategy["fourier_qR"]

    p = apparatus["p"]
    device = apparatus["device"]
    estimator_shots = apparatus["estimator_shots"]
    sampler_shots = apparatus["sampler_shots"]
    optimizer = apparatus["optimizer"]
    opt_steps = apparatus["opt_steps"]
    

    # -----------------  STEP 1:  Build QAOA ansatz  ----------------- #

    cost_h, mixer_layer, angles, num_gamma_terms = ma_build_hamiltonians(
        graph,
        penalizer, 
        constrained, 
        relaxation_type, 
        )

    dev = qp.device(device, wires=wires)

    cost_qnode, cost_function = ma_estimation_framework(
        wires,
        p, 
        dev, 
        cost_h, 
        mixer_layer, 
        angles
        )

    sampling_qnode, probability_circuit = ma_sampling_framework(
        wires, 
        p, 
        dev, 
        sampler_shots, 
        cost_h, 
        mixer_layer, 
        angles
        )


    # ----------------  STEP 2:  Optimize parameters  ---------------- #

    if param_transfer_type == 'given':
        angles_given = [[0.5] * p] * 2
        best_params, best_energies = opt.run_optimization(
                cost_function=cost_function,
                init_params=angles_given, 
                optimizer=optimizer, 
                precision=0.01,
                silence=silence
                )

    elif param_transfer_type == 'random':
        best_params, best_energies, best_energy_ps = opt.param_restarts(
                cost_function=cost_function, 
                n_restarts=p,
                p=p, 
                optimizer=optimizer, 
                opt_steps=opt_steps, 
                silence=silence
                )

    if not silence:
        plt.style.use("default")
        theo_best_cost, _ = clas.best_config_branch_bound(graph)
        costs = [[qa.energy_to_cost(energy_val, penalizer, graph.nodes, graph.edges) / theo_best_cost 
                    for energy_val in sublist] 
                    for sublist in best_energy_ps]
        length = len(costs)
        cmap = plt.colormaps['cividis']
        colors = [cmap(i / (length - 1)) for i in range(length)] if length > 1 else [cmap(0)]
        current_x = 0
        plt.figure(figsize=(10, 5))
        current_x = 0
        for i in range(length):
            segment = costs[i]
            x_coords = [current_x + j for j in range(len(segment))]
            plt.plot(x_coords, segment, color=colors[i], linewidth=2)
            current_x += len(segment) - 1

        plt.title("Energy optimization by step for interp")
        plt.xlabel("Step")
        plt.ylabel("Approximation Ratio")

        plt.gca().set_facecolor("white")
        plt.minorticks_on()

        plt.grid(True, which="major", linestyle="-", linewidth=0.6, color="#898989", alpha=0.8)
        plt.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#b9b9b9", alpha=0.7)

        plt.tight_layout()
        plt.show()

    if not silence: print("Optimal Parameters:", best_params)

    # ----------------------  STEP 3: Sampling  ---------------------- #

    probs = probability_circuit(best_params)
    
    if not silence:
        plt.style.use("seaborn-v0_8") 
        plt.bar(range(2 ** len(wires)), probs)
        plt.show()

    # ------------------  STEP 4: Extract solution  ------------------ #

    best_energy, approximation_ratio, success = extract_solutions(graph, probs, best_energies, penalizer, wires, silence)

    return {
        "cost_function": cost_function,
        "cost_hamiltonian": cost_h,
        "best_params": best_params,
        "best_energy": best_energy,
        "approximation_ratio": approximation_ratio,
        "success": success,
    }
