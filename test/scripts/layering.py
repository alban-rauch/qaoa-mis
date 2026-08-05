import pennylane as qp
from pennylane import numpy as np

dev = qp.device("lightning.amdgpu", wires=10)

def circ_layer(t):
    qp.RX(t*np.pi, wires=0)
    qp.measure(0)

@qp.qnode(dev)
def circuit(params, n):
    qp.layer(circ_layer, n, params)
    return [qp.expval(qp.PauliZ(i)) for i in range(2)]

print(qp.draw(circuit)(params=[0.1, 0.5, 1.0], n=3))

## Expected output:
##  0: ──RX(0.63)──┤↗├──RX(1.57)──┤↗├──RX(3.14)──┤↗├─┤  <Z>
##  1: ───────────────────────────────────────────────┤  <Z>