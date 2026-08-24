import numpy as np

import source.qaoa_run as qr
from source.utils import cond_gen as cnd
from source.paths import DATA_DIR, COND_DIR

from generate_dataset import run_dataset


outdir = DATA_DIR / "dataset"
outdir.mkdir(parents=True, exist_ok=True)

exp_configs = cnd.load_condition(COND_DIR / "cond2.json")

strategy_config_warm = dict(exp_configs[1])
strategy_config_cold = dict(exp_configs[1])
strategy_config_cold["relaxation_type"] = None

qaoa_methods = {
    "standard_cold": {
        "runner": qr.run_qaoa,
        "config": strategy_config_cold,
    },

    "standard_warm": {
        "runner": qr.run_qaoa,
        "config": strategy_config_warm,
    },
}


p_values = np.arange(1, 11)

SWEEP_CONFIGS = {

    1: {
        'Gilbert': {
            'p_values': p_values,
            'num_samples': 20,
            'axes': {'N': np.arange(5, 21), 'q': [0.25]}
        }
    },

    2: {
        'Gilbert': {
            'p_values': p_values,
            'num_samples': 20,
            'axes': {'N': np.arange(5, 21), 'q': [0.5]}
        }
    },

    3: {
        'Gilbert': {
            'p_values': p_values,
            'num_samples': 20,
            'axes': {'N': np.arange(5, 21), 'q': [0.10]}
        }
    },

    4: {
        'DRegular': {
            'p_values': p_values,
            'num_samples': 20,
            'axes': {'N': np.arange(5, 21), 'd': [3]}
        }
    },

    5: {
        'DRegular': {
            'p_values': p_values,
            'num_samples': 20,
            'axes': {'N': np.arange(5, 21), 'd': [2]}
        }
    },

    6: {
        'complete': {
            'p_values': p_values,
            'num_samples': 1,
            'axes': {'N': np.arange(5, 21)}
        },
        'linear': {
            'p_values': p_values,
            'num_samples': 1,
            'axes': {'N': np.arange(5, 21)}
        },
        'circular': {
            'p_values': p_values,
            'num_samples': 1,
            'axes': {'N': np.arange(5, 21)}
        }
    }
}

run_dataset(
    SWEEP_CONFIGS[1],
    exp_configs,
    qaoa_methods,
    outdir,
    save_every=50,
)