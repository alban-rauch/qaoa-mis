import numpy as np

import pennylane as qml
from pennylane import numpy as pnp

# --------------------------------------------------------------------------------- #
#  SCAN                                                                             #
# --------------------------------------------------------------------------------- #


def make_energy_fn(cost_function):
    def energy(params):
        return cost_function(params)
    return energy


def landscape_along_trajectory(energy_fn, angle_history, row, layer,
                               n_theta=120, max_iters=100, reverse_radius=True):
    
    history = np.asarray(angle_history, dtype=float)       # (n_iter, 2, p)
    n_iter = history.shape[0]
    k = 1.0 if row == 0 else 2.0                           # plotted angle = k * parameter

    its = np.unique(np.linspace(0, n_iter - 1, min(max_iters, n_iter)).round().astype(int))

    r_all = np.linspace(1.0, 0.2, n_iter) if reverse_radius else np.linspace(0.2, 1.0, n_iter)
    r = r_all[its]

    theta = np.linspace(0, 2 * np.pi, n_theta + 1)      # last point == first point (seam)
    grad_fn = qml.grad(energy_fn)

    G = np.empty((len(its), n_theta + 1))               # GRADIENT dE / d(parameter)
    E = np.empty_like(G)                                # ENERGY
    for a, t in enumerate(its):                         # One iteration (ring)
        for j in range(n_theta):                        # One angle (point on ring)
            ang = history[t].copy()
            ang[row, layer] = theta[j] / k
            g = np.asarray(grad_fn(pnp.array(ang, requires_grad=True)))
            E[a, j], G[a, j] = float(grad_fn.forward), g[row, layer]
        E[a, -1], G[a, -1] = E[a, 0], G[a, 0]
        print(f"\r[landscape row={row} layer={layer}] iteration {a + 1}/{len(its)}", end="")
    print()

    order = np.argsort(r)
    return dict(theta=theta, r=r[order], G=G[order], E=E[order], its=its[order])