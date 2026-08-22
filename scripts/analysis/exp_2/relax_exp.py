"""
relax_exp.py
============
"""

import numpy as np

from source.utils import cond_gen as cnd
from source.paths import DATA_DIR, GRAPHS_DIR, COND_DIR

import csv
import itertools
 
import numpy as np
 
from source.utils import graph_gen as gph
from source.utils import cond_gen as cnd
from relax_pipeline import compare_pipeline
 
import source.qaoa_run as qr
from source.paths import DATA_DIR, GRAPHS_DIR, COND_DIR


problem_config, strategy_config, apparatus_config = cnd.load_condition(COND_DIR / "cond2.json")
strategy_config_warm = dict(strategy_config)
strategy_config_cold = dict(strategy_config)
strategy_config_cold["relaxation_type"] = None


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

DETERM_FAMILIES = {'complete', 'linear', 'circular'}

random_graphs = {}
for family, cond in SWEEP_CONFIG.items():
    axis_names = list(cond['axes'].keys())
    axis_value_lists = [cond['axes'][name] for name in axis_names]
    combos = list(itertools.product(*axis_value_lists))

    if family in DETERM_FAMILIES:
        params_needed = [combo[axis_names.index('N')] for combo in combos]
    else:
        params_needed = combos

    random_graphs[family] = gph.load_family(
        GRAPHS_DIR / f'{family}.npz', 
        family, 
        params=params_needed
    )


#### Output ####

outdir = DATA_DIR / "analysis_data/exp_2"
outdir.mkdir(parents=True, exist_ok=True)
raw_path = outdir / "relax_data2208_raw.csv"
summary_path = outdir / "relax_data2208_summary.csv"

all_axis_names = sorted({name for cond in SWEEP_CONFIG.values() for name in cond['axes']})

raw_fields = ["family"] + all_axis_names + ["p", "sample_idx", "relax_ratio", "opt_warm_ratio", "opt_cold_ratio"]
summary_fields = ["family"] + all_axis_names + [
    "p",
    "relax_ratio_mean", "relax_ratio_stderr",
    "opt_warm_ratio_mean", "opt_warm_ratio_stderr",
    "opt_cold_ratio_mean", "opt_cold_ratio_stderr",
]


#### Run ####

with open(raw_path, "w", newline="") as raw_file, open(summary_path, "w", newline="") as summary_file:

    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fields, restval="")
    summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields, restval="")
    raw_writer.writeheader()
    summary_writer.writeheader()

    for family, cond in SWEEP_CONFIG.items():
        p_values = cond['p_values']
        num_samples = cond['num_samples']
        axis_names = list(cond['axes'].keys())
        axis_value_lists = [cond['axes'][name] for name in axis_names]
        combos = list(itertools.product(*axis_value_lists))

        for p in p_values:
            apparatus_config["p"] = p
            for combo in combos:
                axis_dict = dict(zip(axis_names, combo))
                key = axis_dict['N'] if family in DETERM_FAMILIES else combo
                N = axis_dict.get('N')
                problem_config["N"] = N

                relax_ratio_samples = np.zeros(num_samples)
                opt_warm_ratio_samples = np.zeros(num_samples)
                opt_cold_ratio_samples = np.zeros(num_samples)

                for i in range(num_samples):
                    problem_config["graph"] = gph.get_graph_from_edges(
                        gph.get_sample(random_graphs[family], key, s=i), N=N
                    )

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

                    seed = abs(hash((family, p, tuple(sorted(axis_dict.items())), i))) % (2**32)
                    relax_ratio, opt_warm_ratio, opt_cold_ratio = compare_pipeline(
                        warm_qaoa_run, cold_qaoa_run, seed=seed
                    )

                    relax_ratio_samples[i] = relax_ratio
                    opt_warm_ratio_samples[i] = opt_warm_ratio
                    opt_cold_ratio_samples[i] = opt_cold_ratio


                    raw_writer.writerow({
                        "family": family, **axis_dict, "p": p, "sample_idx": i,
                        "relax_ratio": relax_ratio, 
                        "opt_warm_ratio": opt_warm_ratio, 
                        "opt_cold_ratio": opt_cold_ratio,
                    })


                raw_file.flush()

                summary_writer.writerow({
                    "family": family, **axis_dict, "p": p,
                    "relax_ratio_mean": np.mean(relax_ratio_samples),
                    "relax_ratio_stderr": np.std(relax_ratio_samples, ddof=1) / np.sqrt(num_samples),
                    "opt_warm_ratio_mean": np.mean(opt_warm_ratio_samples),
                    "opt_warm_ratio_stderr": np.std(opt_warm_ratio_samples, ddof=1) / np.sqrt(num_samples),
                    "opt_cold_ratio_mean": np.mean(opt_cold_ratio_samples),
                    "opt_cold_ratio_stderr": np.std(opt_cold_ratio_samples, ddof=1) / np.sqrt(num_samples),
                })
                summary_file.flush()

                print((family, axis_dict, float(p)), "done")


print(f"Raw samples written to     {raw_path}")
print(f"Summary stats written to   {summary_path}")
