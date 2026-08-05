import pennylane as qp

dev = qp.device("lightning.amdgpu", wires=2)

@qp.qnode(dev)
def bell_state_circuit():
    qp.Hadamard(wires=0)
    qp.CNOT(wires=[0, 1])
    return qp.probs(wires=[0, 1])

print(bell_state_circuit())


## Expected output:
##  [0.5, 0.0, 0.0, 0.5]