## TOTAL SPATIAL METRICS
# Path length:          L = \sum |\Delta \theta_i|
# Net displacement:     D = |\theta_N - \theta_0|
# Straightness:         S = D / L
# Reversals:            R = number of sign flips of \Delta \theta, divided by N-1

## TIME DEPENDENCE
# Profile over iterations: L(t), D(t), S(t)
# Settling iteration: 


## Cross-volatility!!
## Correlate with energy changes - track energy / gradient along curve

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

PERIODS = np.array([2 * np.pi, np.pi]) 

def normalize_history(history, period=2*np.pi):
    return np.asarray(history, float) / period

# We do NOT wrap the values to remain in the [0, 2pi] interval. 
# If the parameter has evolved 0.1 to 2pi - 0.1, we should not 
# claim an evolution 0.1 to -0.1.
# Except D because that measures the net parameter change.

def movement_metrics(
    theta, 
    t1=0, t2=-1, 
    eps=1e-6, tol=1e-2, period=2*np.pi):

    theta = np.asarray(theta, float)[t1:t2]

    d = np.diff(theta)
    if d.size == 0:
        return dict(path_length=0.0, net_displacement=0.0, straightness=1.0,
                    reversals=0.0, settling=0)
    
    L = np.abs(d).sum()                             # path length
    D = abs(theta[-1] - theta[0]) % period          # net displacement
    S = D / L if L > eps else 1.0                   # straightness

    s = np.sign(d[np.abs(d) > eps])
    R = float(np.sum(s[1:] != s[:-1]) / max(len(s) - 1, 1))       # sign reversals

    far = np.where(np.abs(theta - theta[-1]) > tol)[0]
    settle = int(far[-1] + 1) if len(far) else 0

    return dict(
        path_length=L,
        net_displacement=D,
        straightness=S,
        reversals=R,
        settling=settle,
    )


def hessian_fd(energy_fn, x, h=1e-3):
    grad_fn = qml.grad(energy_fn)
    x = np.asarray(x, float); shape = x.shape; flat = x.ravel(); n = flat.size
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = h
        gp = np.asarray(grad_fn(pnp.array((flat + e).reshape(shape), requires_grad=True))).ravel()
        gm = np.asarray(grad_fn(pnp.array((flat - e).reshape(shape), requires_grad=True))).ravel()
        H[i] = (gp - gm) / (2 * h)
    return 0.5 * (H + H.T)