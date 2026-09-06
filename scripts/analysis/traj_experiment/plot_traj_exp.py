"""
Compare the warm and cold for a given approximation ratio
"""

import numpy as np
import colorsys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LogNorm, FuncNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D

from scripts.analysis.param_traj import make_hue_cmap, create_polar_fig, add_traj, finalize_polar_fig
from scripts.dataset.collect_information import load_cache, load_sample, collect_run
from source.paths import CACHE_DIR

from source.utils.graph_gen import get_graph_from_edges
from source.utils.classical import best_config_branch_bound, approx_ratio


### GRAPH TO LOAD

cache_dir = CACHE_DIR
family = 'Gilbert'
axis_dict = {
    "N": 10,
    "q": 0.25,
}
p = 1
sample_idx = 0


### COLLECT RUN RESULTS

cache = load_cache(cache_dir, family, axis_dict)
sample = load_sample(cache, sample_idx)
cold_run = collect_run(sample, "standard_cold", p)
warm_run = collect_run(sample, "standard_warm", p)


### EXTRACT HIST OF ARATIO/PARAM

graph = get_graph_from_edges(sample["edges"], N=axis_dict["N"])
theo_best_cost, _ = best_config_branch_bound(graph)
cold_ratio_history = approx_ratio(graph, cold_run["best_energy_p"], 1.5, theo_best_cost)
warm_ratio_history = approx_ratio(graph, warm_run["best_energy_p"], 1.5, theo_best_cost)
approx_ratio_histories = [cold_ratio_history, warm_ratio_history]

param_histories = [cold_run["param_hist"], warm_run["param_hist"]]


LIGHTNESS_BOUNDS = np.array([
    0.0, 0.4,
    0.6, 0.7,
    0.8, 0.85, 
    0.9, 0.925, 
    0.95, 0.97,
    0.98, 0.99, 
    1.0,
])
n_lightness = len(LIGHTNESS_BOUNDS) - 1
norm = BoundaryNorm(LIGHTNESS_BOUNDS, n_lightness)


CONTRAST_PARAMS = {
    "s_range": (0.3, 1.0), 
    "v_range": (0.8, 0.1),
    "a_range": (0.5, 1.0),
}
n_runs = 2
run_colors = [
    (0.0, 0.0, 1.0, 1.0),  # blue
    (1.0, 0.0, 0.0, 1.0),  # red
#    (0.0, 0.6, 0.0, 1.0),  # green
#    (1.0, 0.8, 0.0, 1.0),  # yellow
]
run_cmaps = [make_hue_cmap(run_colors[i], n_lightness, **CONTRAST_PARAMS) for i in range(n_runs)]



fig, axes = create_polar_fig(p)


last_lc = None
final_params = [par[-1] for par in param_histories]
final_costs = [cost[-1] for cost in approx_ratio_histories]
run_labels = ["cold", "warm"]

for i in range(len(approx_ratio_histories)):

    param_history = param_histories[i]
    approx_ratio_history = approx_ratio_histories[i]
    cmap = run_cmaps[i]

    last_lc = add_traj(
        axes,
        param_history,
        approx_ratio_history,
        cmap=cmap,
        norm=norm,
        reverse_radius=True,
        smooth_window=1,
        show_endpoints=True,
    )


finalize_polar_fig(
    fig, axes, last_lc, LIGHTNESS_BOUNDS, CONTRAST_PARAMS, run_colors, run_labels, 
    title=f"QAOA angle trajectories |  Final AR: {max(final_costs):.2f}",
)

plt.show()