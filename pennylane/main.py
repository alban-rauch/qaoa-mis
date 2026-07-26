"""
main.py
=======
Main file.
"""

import pennylane as qp
import numpy as np
from matplotlib import pyplot as plt

import graphs as gph
import qaoa_pipeline.qaoa_run as qr
import data_analysis.variance_lanscape as vl


# ================================================================ #
#                             VARIABLES                            #
# ================================================================ #

# ---------------------- Circuit variables ----------------------- #

N = 12                       # ⇢ Problem size (N)              #!!!#
# graph = gph.complete_graph(N)
graph = gph.randomGilbert(N, 0.25)
# graph = gph.randomDRegular(N, 3)
# graph = gph.paragon()
p = 3                       # ⇢ Layers/depth (p)              #!!!#

device = "lightning.qubit"  # "lightning.qubit" / "lightning.amdgpu" 

estimator_shots = 10000     # ⇢ Estimator shots
sampler_shots = 10000       # ⇢ Sampler shots


optimizer = "L-BFGS-B"      # or "Adam"
opt_steps = 400


# ================================================================ #
#                             QAOA TYPE                            #
# ================================================================ #

# -------------------------- constrained ------------------------- #

constrained = False

# -------------------------- warm  start ------------------------- #

relaxation_type = None              # None / 'continuous'
param_transfer_type = 'interp'     # 'given' / 'random' /  
                                    # 'interp' / 'fourier'
init_param = 0.5 * np.pi
(fourier_q, fourier_R) = (None, 5)


# ================================================================ #
#                             QAOA RUN                             #
# ================================================================ #

problem_config = {
    "N": N,
    "graph": graph,
}

strategy_config = {
    "constrained": constrained,
    "relaxation_type": relaxation_type,
    "param_transfer_type": param_transfer_type,
    "fourier_qR": (fourier_q, fourier_R),
    "init_param": init_param,
}

apparatus_config = {
    "p": p,
    "device": device,
    "estimator_shots": estimator_shots,
    "sampler_shots": sampler_shots,
    "optimizer": optimizer,
    "opt_steps": opt_steps,
}

strategy_config["relaxation_type"] = 'continuous'

one_qaoa_run = qr.standard_qaoa(
        problem=problem_config,
        strategy=strategy_config,
        apparatus=apparatus_config,
    )

print(one_qaoa_run)


params = np.linspace(0.0, 1.0, 100)
approx_ratios = []

for param in params:
    strategy_config["init_param"] = param
    one_qaoa_run = qr.standard_qaoa(
        problem=problem_config,
        strategy=strategy_config,
        apparatus=apparatus_config,
        silence=True
    )

    ar = one_qaoa_run["approximation_ratio"]
    approx_ratios.append(ar)
    print(f"{param} done")

plt.plot(params, np.array(approx_ratios))
plt.xlabel("Initial parameters")
plt.ylabel("Approximation ratio")
plt.grid(alpha=0.3)
plt.show()


# for p in range(1, 6):

#     apparatus_config["p"] = p

#     one_qaoa_run = qr.standard_qaoa(
#         problem=problem_config,
#         strategy=strategy_config,
#         apparatus=apparatus_config,
#         silence=True
#     )

#     print(one_qaoa_run)

#     cost_function = one_qaoa_run["cost_function"]
#     theta0 = one_qaoa_run["best_params"]
#     radii, variances = vl.full_energy_landscape_shell(cost_function, theta0, 30)
#     vl.plot_variance(radii, variances)
