import pennylane as qp
from pennylane import numpy as np

dev = qp.device("lightning.amdgpu", wires=1)

@qp.qnode(dev)
def rotate_qubit(phi1, phi2):
    qp.RX(phi1, wires=0)
    qp.RY(phi2, wires=0)
    return qp.expval(qp.PauliZ(0))

## State after gates:
##    [cos(phi1/2)cos(phi2/2) - i sin(phi1/2)sin(phi2/2)] |0>
##  + [cos(phi1/2)sin(phi2/2) - i sin(phi1/2)cos(phi2/2)] |1>

## Expectation value:
##   cos(phi1) cos(phi2)

opt = qp.GradientDescentOptimizer(stepsize=0.1)         # θ_{t+1} = θ_t - stepsize * grad f (θ_t)
# opt = qp.AdamOptimizer(stepsize=0.1)                      # Adaptative stepsize
params = np.array([0.011, 0.012], requires_grad=True)

for steps in range(100):
    params = opt.step(rotate_qubit, *params)

## We want to minimize the expectation value to -1,
##  which happens when (phi1, phi2) = (0, pi) or (pi, 0).

print(f"Optimized angle: {params}")
print(f"Final measurement: {rotate_qubit(*params)}")

## Expected output:
##    "Optimized angle: [0.0, 3.14]"
## or "Optimized angle: [3.14, 0.0]"
##    "Final measurement: -1"