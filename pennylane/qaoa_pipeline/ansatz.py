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

# ======================================================================================== #
#                                     Layerwise angle                                      #
# ======================================================================================== #

def relaxed_mixer_layer(beta, graph, angles):
    for node in graph.nodes:
        theta = angles[node]
        qp.RY(-theta, wires=node)
        qp.RZ(-2 * beta, wires=node)
        qp.RY(theta, wires=node)

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

def qaoa_layer(gamma, beta, cost_h, mixer_layer):
    qaoa.cost_layer(gamma, cost_h)
    mixer_layer(beta)

def make_circuit(layer_structure):
    def circuit(wires, p, params, cost_h, mixer_layer, angles):
        for w in wires:
            qp.RY(angles[w], wires=w)
        qp.layer(layer_structure, p, params[0], params[1], cost_h=cost_h, mixer_layer=mixer_layer)
    return circuit


# print(qp.draw(circuit)([[0.3, 0.4], [0.5, 0.1]]))
# qp.draw_mpl(circuit, layer='device')([[0.3, 0.4], [0.5, 0.1]])
# plt.show()

def estimator(params, circuit, wires, p, cost_h, mixer_layer, angles):
    circuit(wires, p, params, cost_h, mixer_layer, angles)
    return qp.expval(cost_h)

def sampler(params, circuit, wires, p, cost_h, mixer_layer, angles):
    circuit(wires, p, params, cost_h, mixer_layer, angles)
    return qp.probs(wires=wires)


# ======================================================================================== #
#                                     3rd layer QAOA                                       #
# ======================================================================================== #

def make_expressive_circuit(layer_structure):
    def circuit(wires, p, params, cost_h, mixer_layer, y_mixer, angles):
        for w in wires:
            qp.RY(angles[w], wires=w)
        qp.layer(layer_structure, p, params[0], params[1], params[2],
                  cost_h=cost_h, mixer_layer=mixer_layer, y_mixer=y_mixer)
    return circuit

def y_mixer_layer(alpha, graph):
    for node in graph.nodes:
        qp.RY(alpha, wires=node)

def expressive_qaoa_layer(gamma, beta, alpha, cost_h, mixer_layer, y_mixer):
    qaoa.cost_layer(gamma, cost_h)
    mixer_layer(beta)
    y_mixer(alpha)


def expressive_estimator(circuit, params, wires, p, cost_h, mixer_layer, y_mixer, angles):
    circuit(wires, p, params, cost_h, mixer_layer, y_mixer, angles)
    return qp.expval(cost_h)

def expressive_sampler(circuit, params, wires, p, cost_h, mixer_layer, y_mixer, angles):
    circuit(wires, p, params, cost_h, mixer_layer, y_mixer, angles)
    return qp.probs(wires=wires)


# ======================================================================================== #
#                                    Multi-angle QAOA                                      #
# ======================================================================================== #

def ma_cost_layer(gammas, cost_h):
    for gamma_k, coeff_k, op_k in zip(gammas, cost_h.coeffs, cost_h.ops):
        angle = 2 * gamma_k * coeff_k
        if len(op_k.wires) == 1:
            qp.RZ(angle, wires=op_k.wires)
        else:
            qp.MultiRZ(angle, wires=op_k.wires)

def ma_relaxed_mixer_layer(betas, graph, angles):
    for node, beta_i in zip(graph.nodes, betas):
        theta = angles[node]
        qp.RY(-theta, wires=node)
        qp.RZ(-2 * beta_i, wires=node)
        qp.RY(theta, wires=node)

def ma_qaoa_layer(gammas, betas, cost_h, mixer_layer):
    ma_cost_layer(gammas, cost_h)
    mixer_layer(betas)

def ma_circuit(wires, p, gammas, betas, cost_h, mixer_layer, angles):
    for w in wires:
        qp.RY(angles[w], wires=w)
    qp.layer(ma_qaoa_layer, p, gammas, betas, cost_h=cost_h, mixer_layer=mixer_layer)


def ma_estimator(gammas, betas, wires, p, cost_h, mixer_layer, angles):
    ma_circuit(wires, p, gammas, betas, cost_h, mixer_layer, angles)
    return qp.expval(cost_h)

def ma_sampler(gammas, betas, wires, p, cost_h, mixer_layer, angles):
    ma_circuit(wires, p, gammas, betas, cost_h, mixer_layer, angles)
    return qp.probs(wires=wires)