"""
csv_speed_exp.py
================
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

p_values = np.arange(1, 11)

SWEEP_CONFIG = {

    'Gilbert': {
        'p_values': p_values,
        'num_samples': 20,
        'axes': {'N': np.arange(5, 16), 'q': [0.1, 0.25, 0.5]}
    },

    'DRegular': {
        'p_values': p_values,
        'num_samples': 20,
        'axes': {'N': np.arange(5, 21), 'd': [2, 3]},
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
    family, N, p, 
    axis_dict, sample_idx, 
    graph, runs, extra_args, 
):

    cold_qaoa_run = runs["cold"]
    warm_qaoa_run = runs["warm"]

    times_val_cold = sum(cold_qaoa_run["times"])
    times_val_warm = sum(warm_qaoa_run["times"])
    evals_val_cold = cold_qaoa_run["cost_circuit_evals"]
    evals_val_warm = warm_qaoa_run["cost_circuit_evals"]
    aratio_val_cold = cold_qaoa_run["approximation_ratio"]
    aratio_val_warm = warm_qaoa_run["approximation_ratio"]

    return {
        "times_cold": times_val_cold, 
        "times_warm": times_val_warm, 
        "evals_cold": evals_val_cold, 
        "evals_warm": evals_val_warm, 
        "aratio_cold": aratio_val_cold, 
        "aratio_warm": aratio_val_warm, 
    }

outdir = DATA_DIR / "analysis_data/relax_exp"
cache_dir = DATA_DIR / "dataset"
metrics = [
    "times_cold", 
    "times_warm", 
    "evals_cold", 
    "evals_warm",
    "aratio_cold",
    "aratio_warm",
]
qaoa_names = ["cold", "warm"]
extra_args = None

run_sweep(
    exp_name="relax_exp", 
    SWEEP_CONFIG=SWEEP_CONFIG, 
    cache_dir=cache_dir, 
    outdir=outdir, 
    metrics=metrics, 
    metric_fn=speed_metric_fn, 
    qaoa_names=qaoa_names,
    extra_args=extra_args,
)