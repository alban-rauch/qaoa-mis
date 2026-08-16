"""
circuit/ansatz.py
=========
Build the QAOA circuit.
"""

import pennylane as qml
from pennylane import qaoa


def energy_to_cost(energy, penalizer, node_list, edge_list):
    return 0.5 * (len(node_list) - 0.5 * penalizer * len(edge_list) - energy)

def cost_hamiltonian(node_list, edge_list, degrees, penalizer):
    coeffs = []
    ops = []
    for i in node_list:
        coeffs.append((1.0 - penalizer * degrees[i] / 2.0))
        ops.append(qml.PauliZ(i))
    coupling_coeff = penalizer / 2.0
    for i, j in edge_list:
        coeffs.append(coupling_coeff)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops)


def relaxed_mixer_layer(beta, graph, angles):
    for node in graph.nodes:
        theta = angles[node]
        qml.RY(-theta, wires=node)
        qml.RZ(-2 * beta, wires=node)
        qml.RY(theta, wires=node)

def y_mixer_layer(alpha, graph):
    for node in graph.nodes:
        qml.RY(-2 * alpha, wires=node)

def qaoa_layer(gamma, mixer_params, cost_h, mixer_fns):
    qaoa.cost_layer(gamma, cost_h)
    for fn, param in zip(mixer_fns, mixer_params):
        fn(param)

def make_circuit(layer_structure):
    def circuit(wires, p, params, cost_h, mixer_fns, angles):
        for w in wires:
            qml.RY(angles[w], wires=w)
        qml.layer(
            layer_structure, p, params[0], params[1:].T, cost_h=cost_h, mixer_fns=mixer_fns
            )
    return circuit


def estimator(params, circuit, wires, p, cost_h, mixer_fns, angles, counter=None):
    if counter is not None:
        counter.cost_circuit_evals += 1
    circuit(wires, p, params, cost_h, mixer_fns, angles)
    return qml.expval(cost_h)

def sampler(params, circuit, wires, p, cost_h, mixer_fns, angles, counter=None):
    if counter is not None:
        counter.sampling_circuit_evals += 1
    circuit(wires, p, params, cost_h, mixer_fns, angles)
    return qml.probs(wires=wires)



############ CONSTRAINED ############

def cst_relaxed_mixer_layer(beta, graph, angles):
    for node in graph.nodes:
        theta = angles[node]
        neighbors = list(graph.neighbors(node))

        if not neighbors:
            qml.RY(-theta, wires=node)
            qml.RZ(-2 * beta, wires=node)
            qml.RY(theta, wires=node)
            continue

        qml.ControlledSequence(
            qml.RY(-theta, wires=node),
            control_wires=neighbors
        )
        qml.ControlledSequence(
            qml.RZ(-2 * beta, wires=node),
            control_wires=neighbors
        )
        qml.ControlledSequence(
            qml.RY(theta, wires=node),
            control_wires=neighbors
        )

def cst_y_mixer_layer(alpha, graph):
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        
        if not neighbors:
            qml.RY(-2 * alpha, wires=node)
            continue

        qml.ControlledSequence(
            qml.RY(-2 * alpha, wires=node),
            control_wires=neighbors
        )