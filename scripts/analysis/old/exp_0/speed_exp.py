"""
speed_exp.py
============
Initial speed experiment
"""

import numpy as np

import source.qaoa_run as qr
from source.utils import cond_gen as cnd

from scripts.analysis.old.gen_sweep import run_sweep
from source.paths import DATA_DIR, COND_DIR



# ============================================================
# EDIT HERE - tunable part
# Variables:
#   x-axis: p, axes (N, q, etc.)
#   y-axis: times, evals, aratio
# ============================================================

p_values = [1, 2, 3, 5, 7, 10]

SWEEP_CONFIG = {

    'Gilbert': {
        'p_values': p_values,
        'num_samples': 10,
        'axes': {'N': np.arange(5, 16), 'q': [0.25]}
    }, 

    'DRegular': {
        'p_values': p_values,
        'num_samples': 10,
        'axes': {'N': np.arange(5, 16), 'd': [3]},
    },

    'complete': {
        'p_values': p_values,
        'num_samples': 1,
        'axes': {'N': np.arange(5, 21)},
    },

    'linear': {
            'p_values': p_values,
            'num_samples': 1,
            'axes': {'N': np.arange(5, 21)},
    },

    'circular': {
            'p_values': p_values,
            'num_samples': 1,
            'axes': {'N': np.arange(5, 21)},
    },

}

# ============================================================


def speed_metric_fn(
    family, N, p, axis_dict, sample_idx,
    problem_config, strategy_config, apparatus_config,
    extra_args,
):
    one_qaoa_run = qr.run_qaoa(
        problem=problem_config,
        strategy=strategy_config,
        apparatus=apparatus_config,
        silence=True,
    )

    times_val = sum(one_qaoa_run["times"])
    evals_val = one_qaoa_run["cost_circuit_evals"]
    aratio_val = one_qaoa_run["approximation_ratio"]

    return {
        "times": times_val, 
        "evals": evals_val, 
        "aratio": aratio_val,
    }

# cond2.json for INTERP + WS
exp_configs = cnd.load_condition(COND_DIR / "cond2.json")
outdir = DATA_DIR / "analysis_data/exp_0"
metrics = ["times", "evals", "aratio"]

run_sweep(
    "speed_exp", SWEEP_CONFIG, 
    exp_configs, outdir, 
    metrics, speed_metric_fn, 
    extra_args=None,
)