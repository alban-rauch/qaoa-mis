"""
ansatz.py
=========
Build the QAOA circuit.
"""

import pennylane as qp
from pennylane import qaoa


def energy_to_cost(energy, penalizer, node_list, edge_list):
    return 0.5 * (len(node_list) - 0.5 * penalizer * len(edge_list) - energy)

def cost_hamiltonian(node_list, edge_list, degrees, penalizer):
    coeffs = []
    ops = []
    for i in node_list:
        coeffs.append((1.0 - penalizer * degrees[i] / 2.0))
        ops.append(qp.PauliZ(i))
    coupling_coeff = penalizer / 2.0
    for i, j in edge_list:
        coeffs.append(coupling_coeff)
        ops.append(qp.PauliZ(i) @ qp.PauliZ(j))
    return qp.Hamiltonian(coeffs, ops)


def relaxed_mixer_layer(beta, graph, angles):
    for node in graph.nodes:
        theta = angles[node]
        qp.RY(-theta, wires=node)
        qp.RZ(-2 * beta, wires=node)
        qp.RY(theta, wires=node)

def y_mixer_layer(alpha, graph):
    for node in graph.nodes:
        qp.RY(-2 * alpha, wires=node)

def qaoa_layer(gamma, mixer_params, cost_h, mixer_fns):
    qaoa.cost_layer(gamma, cost_h)
    for fn, param in zip(mixer_fns, mixer_params):
        fn(param)

def make_circuit(layer_structure):
    def circuit(wires, p, params, cost_h, mixer_fns, angles):
        for w in wires:
            qp.RY(angles[w], wires=w)
        qp.layer(
            layer_structure, p, params[0], params[1:].T, cost_h=cost_h, mixer_fns=mixer_fns
            )
    return circuit


def estimator(circuit, params, wires, p, cost_h, mixer_fns, angles):
    circuit(wires, p, params, cost_h, mixer_fns, angles)
    return qp.expval(cost_h)

def sampler(circuit, params, wires, p, cost_h, mixer_fns, angles):
    circuit(wires, p, params, cost_h, mixer_fns, angles)
    return qp.probs(wires=wires)



############ CONSTRAINED ############

def cst_relaxed_mixer_layer(beta, graph, angles):
    for node in graph.nodes:
        theta = angles[node]
        neighbors = list(graph.neighbors(node))

        if not neighbors:
            qp.RY(-theta, wires=node)
            qp.RZ(-2 * beta, wires=node)
            qp.RY(theta, wires=node)
            continue

        qp.ControlledSequence(
            qp.RY(-theta, wires=node),
            control_wires=neighbors
        )
        qp.ControlledSequence(
            qp.RZ(-2 * beta, wires=node),
            control_wires=neighbors
        )
        qp.ControlledSequence(
            qp.RY(theta, wires=node),
            control_wires=neighbors
        )

def cst_y_mixer_layer(alpha, graph):
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        
        if not neighbors:
            qp.RY(-2 * alpha, wires=node)
            continue

        qp.ControlledSequence(
            qp.RY(-2 * alpha, wires=node),
            control_wires=neighbors
        )