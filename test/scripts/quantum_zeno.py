import pennylane as qp
from pennylane import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

n_repeats = 30
qubits = 10
dts = np.linspace(0, np.pi, qubits)

dev = qp.device("lightning.amdgpu", wires=qubits)

@qp.qnode(dev)
def zeno_circuit(n_repeats, dts):
    measurements = []
    for _ in range(n_repeats):
        wires_measures = []
        for j in range(qubits):
            qp.RX(dts[j], wires=j)
            m = qp.measure(j)
            wires_measures.append(m)
        measurements.append(wires_measures)
    return [qp.sample(m) for step in measurements for m in step]

def zeno_circuit_plot(bit_measurements):
    fig, ax = plt.subplots(figsize=(10, 6))

    for j in range(qubits):
        ax.plot([0, n_repeats - 1], [j, j], color='lightgray', linestyle='-', linewidth=2, zorder=1)
        
        for s in range(n_repeats):
            flat_idx = s * qubits + j
            outcome = bit_measurements[flat_idx]        
            color = 'green' if outcome == 1 else 'red'
            ax.scatter(s, j, color=color, s=250, edgecolor='black', zorder=2)

    ax.set_yticks(range(qubits))
    ax.set_yticklabels([f"{j}" for j in range(qubits)])
    ax.set_xticks(range(n_repeats))
    ax.set_xticklabels([f"{s+1}" for s in range(n_repeats)])
    ax.set_xlabel("Layers / Repeats", fontsize=12)
    ax.set_ylabel("Wires (Qubits)", fontsize=12)
    ax.set_title("Single Quantum Zeno Trajectory (shots=1)", fontsize=14, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Outcome: 1 ($|1\\rangle$)', markerfacecolor='green', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Outcome: 0 ($|0\\rangle$)', markerfacecolor='red', markersize=12, markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))

    plt.tight_layout()
    plt.show()
    # plt.savefig('zeno_trajectory.png')

bit_measurements = zeno_circuit(n_repeats=n_repeats, dts=dts, shots=1)
zeno_circuit_plot(bit_measurements)

## Expected output:
##  All red for wire 0
##  Alternating half-half for wire 10


def ladder_zeno_layer(dt):
    qp.RX(dt, wires=0)
    for i in range(qubits - 1):
        m = qp.measure(i)
        qp.cond(m == 1, qp.RX)(dt, wires=i+1)

# pick n_layers >= qubits
@qp.qnode(dev, shots=100)
def ladder_zeno_circuit(n_layers, dt):
    for _ in range(n_layers):
        ladder_zeno_layer(dt)
    return [qp.expval(qp.PauliZ(i)) for i in range(qubits)]

# print(qp.draw(ladder_zeno_circuit)(n_layers=1, dt=0.1))

thetas = np.linspace(0, np.pi, 10)
all_samples = {}
for theta in thetas:
    raw_z = ladder_zeno_circuit(n_layers=5, dt=theta)
    raw_z = np.array(raw_z)
    qubit_states = (1.0 - raw_z) / 2.0
    clean_thetas = round(float(theta), 4)
    clean_states = [round(val, 4) for val in qubit_states.tolist()]
    all_samples[clean_thetas] = clean_states

print(all_samples)

## Expected output:
##  First thetas: all 0.0
##  Last thetas: all 1.0
