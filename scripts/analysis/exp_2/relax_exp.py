"""
relax_exp.py
============
"""
 
import numpy as np

import source.qaoa_run as qr
from source.utils import cond_gen as cnd

from relax_pipeline import compare_pipeline
from scripts.analysis.gen_sweep import run_sweep
from source.paths import DATA_DIR, COND_DIR



# ============================================================
# EDIT HERE - tunable part
# ============================================================

p_values = [1, 2, 3, 5, 7, 10, 15]

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



def relax_metric_fn(
    family, N, p, axis_dict, sample_idx,
    problem_config, strategy_config, apparatus_config,
    extra_args,
):
    strategy_config_warm, strategy_config_cold = extra_args
    warm_qaoa_run = qr.run_qaoa(
        problem=problem_config,
        strategy=strategy_config_warm,
        apparatus=apparatus_config,
        silence=True,
    )
    cold_qaoa_run = qr.run_qaoa(
        problem=problem_config,
        strategy=strategy_config_cold,
        apparatus=apparatus_config,
        silence=True,
    )
    seed = abs(hash((family, p, tuple(sorted(axis_dict.items())), sample_idx))) % (2**32)
    relax_ratio, opt_warm_ratio, opt_cold_ratio = compare_pipeline(
                                                    warm_qaoa_run, cold_qaoa_run, seed=seed)
    return {
        "relax_ratio": relax_ratio, 
        "opt_warm_ratio": opt_warm_ratio, 
        "opt_cold_ratio": opt_cold_ratio,
    }

exp_configs = cnd.load_condition(COND_DIR / "cond2.json")
outdir = DATA_DIR / "analysis_data/exp_2"
metrics = ["relax_ratio", "opt_warm_ratio", "opt_cold_ratio"]

strategy_config_warm = dict(exp_configs[1])
strategy_config_cold = dict(exp_configs[1])
strategy_config_cold["relaxation_type"] = None

run_sweep(
    "relax_exp", SWEEP_CONFIG, 
    exp_configs, outdir, 
    metrics, relax_metric_fn, 
    extra_args=(strategy_config_warm, strategy_config_cold),
)
