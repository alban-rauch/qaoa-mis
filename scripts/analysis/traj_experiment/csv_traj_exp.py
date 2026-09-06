"""
csv_relax_exp.py
================
"""
 
import numpy as np

from scripts.analysis.relax_experiment.csv_relax_pipeline import compare_pipeline
from scripts.analysis.exp_sweep import run_sweep
from source.paths import DATA_DIR



# ============================================================
# EDIT HERE - tunable part
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


def relax_metric_fn(
    family, N, p, 
    axis_dict, sample_idx, 
    graph, runs, extra_args, 
):
    cold_qaoa_run = runs["standard_cold"]
    warm_qaoa_run = runs["standard_warm"]
    relax_core_ratio, relax_rounded_ratio, opt_warm_ratio, opt_cold_ratio = compare_pipeline(
        warm_qaoa_run, 
        cold_qaoa_run, 
        graph=graph,
        N=N,
        seed=0
    )
    return {
        "relax_core_ratio": relax_core_ratio, 
        "relax_rounded_ratio": relax_rounded_ratio, 
        "opt_warm_ratio": opt_warm_ratio, 
        "opt_cold_ratio": opt_cold_ratio,
    }

outdir = DATA_DIR / "analysis_data/relax_exp"
cache_dir = DATA_DIR / "dataset"
metrics = ["relax_core_ratio", "relax_rounded_ratio", "opt_warm_ratio", "opt_cold_ratio"]
qaoa_names = ["standard_cold", "standard_warm"]
extra_args = None


run_sweep(
    exp_name="relax_exp", 
    SWEEP_CONFIG=SWEEP_CONFIG, 
    cache_dir=cache_dir, 
    outdir=outdir, 
    metrics=metrics, 
    metric_fn=relax_metric_fn, 
    qaoa_names=qaoa_names,
    extra_args=extra_args,
)

