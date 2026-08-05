import pennylane as qp
import numpy as np

dev = qp.device("lightning.amdgpu", wires=3)

@qp.qnode(dev)
def teleportation_circuit(theta):
    
    # Prepare State Q
    qp.RY(theta, wires=0)

    # Prepare Entanglement State (A, B): Bell state |phi^+> ~ |00> + |11> 
    qp.Hadamard(wires=1)
    qp.CNOT(wires=[1, 2])

    # Teleportation protocol
    qp.CNOT(wires=[0, 1])
    qp.Hadamard(wires=0)
    
    # Measure and apply gates conditionally to B
    m0 = qp.measure(0)
    m1 = qp.measure(1)
    qp.cond(m1, qp.X)(wires=2)
    qp.cond(m0, qp.Z)(wires=2)

    return qp.density_matrix(wires=2)


## Expected output:
##  [[cos^2(theta/2), 0.5*sin(theta)],
##   [0.5*sin(theta), sin^2(theta/2)]]

theta = 0.2

result = teleportation_circuit(theta)

theory = [
    [np.cos(theta/2)**2, 0.5*np.sin(theta)], 
    [0.5*np.sin(theta), np.sin(theta/2)**2]
    ]

print(f"Result: {result}")

print(f"Theory: {theory}")