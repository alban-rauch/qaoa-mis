"""
main.py
=======
Main file.
"""

import numpy as np
from matplotlib import pyplot as plt

import graphs as gph
import data_analysis.variance_landscape as vl
import qaoa_run as qr
import test as tst


# ================================================================ #
#                             QAOA RUN                             #
# ================================================================ #

N = 12
p = 3

problem_config = {
    "N": N,
    "graph": gph.randomGilbert(N, 0.25), # gph.randomDRegular(N, 3) | gph.randomGilbert(N, 0.25)
}

strategy_config = {
    "constrained": False,
    "relaxation_type": 'continuous',    #  None | 'continuous'
    "param_transfer_type": 'interp',    # 'random' | 'interp' | 'fourier'
    "fourier_qR": (None, 5),
    "init_param": [0.55, 0.27],
    "mixers": ["x"],
}

apparatus_config = {
    "p": p,
    "device": "lightning.qubit",        # "lightning.qubit" | "lightning.amdgpu" 
    "estimator_shots": 10000,
    "sampler_shots": 10000,
    "optimizer": "L-BFGS-B",            # "L-BFGS-B" | "Adam"
    "opt_steps": 400,
}

strategy_config["relaxation_type"] = 'continuous'

# one_qaoa_run = qr.run_qaoa(
#         problem=problem_config,
#         strategy=strategy_config,
#         apparatus=apparatus_config,
#         silence=False
#     )

# print(one_qaoa_run)


data_mat = tst.run_qaoa(
        problem=problem_config,
        strategy=strategy_config,
        apparatus=apparatus_config,
        silence=True
    )

tst.heatmap(data_mat)

# plt.plot(params, np.array(approx_ratios))
# plt.xlabel("Initial parameters")
# plt.ylabel("Approximation ratio")
# plt.grid(alpha=0.3)
# plt.show()


# for p in range(1, 6):

#     apparatus_config["p"] = p

#     one_qaoa_run = qaoa_func(
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
