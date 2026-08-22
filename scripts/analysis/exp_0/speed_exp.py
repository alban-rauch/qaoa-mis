"""
speed_exp.py
============
Initial speed experiment
"""

import csv
import itertools

import numpy as np

from source.utils import graph_gen as gph
from source.utils import cond_gen as cnd

import source.qaoa_run as qr
from source.paths import DATA_DIR, GRAPHS_DIR, COND_DIR


#### Fixed variables ####

# Use cond2.yaml by default
problem_config, strategy_config, apparatus_config = cnd.load_condition(COND_DIR / "cond2.json")



# ============================================================
# EDIT HERE - tunable part
# Variables:
#   x-axis: p, axes (N, q, etc.)
#   y-axis: times, evals, aratio
# ============================================================
SWEEP_CONFIG = {
    'Gilbert': {
        'p_values': np.arange(1, 11),
        'num_samples': 10,
        'axes': {'N': np.arange(5, 16), 'q': [0.25]}
    }, 
    # 'DRegular': {
    #     'p_values': np.arange(1, 11),
    #     'num_samples': 10,
    #     'axes': {'N': np.arange(5, 16), 'd': [3]},
    # },
    # 'complete': {
    #     'p_values': np.arange(1, 11),
    #     'num_samples': 1,
    #     'axes': {'N': np.arange(5, 16)},
    # },
}
# ============================================================

DETERM_FAMILIES = {'complete', 'linear', 'circular'}


#### Random graphs prep ####

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

outdir = DATA_DIR / "analysis_data/exp_0"
outdir.mkdir(parents=True, exist_ok=True)
raw_path = outdir / "speed_data4_raw.csv"
summary_path = outdir / "speed_data4_summary.csv"

all_axis_names = sorted({name for cond in SWEEP_CONFIG.values() for name in cond['axes']})

raw_fields = ["family"] + all_axis_names + [
    "p", "sample_idx", 
    "times", "evals", "aratio"
]
summary_fields = ["family"] + all_axis_names + [
    "p",
    "times_mean", "times_stderr",
    "evals_mean", "evals_stderr",
    "aratio_mean", "aratio_stderr",
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

                times_samples = np.zeros(num_samples)
                evals_samples = np.zeros(num_samples)
                aratio_samples = np.zeros(num_samples)

                for i in range(num_samples):
                    problem_config["graph"] = gph.get_graph_from_edges(
                        gph.get_sample(random_graphs[family], key, s=i), N=N
                    )

                    one_qaoa_run = qr.run_qaoa(
                        problem=problem_config,
                        strategy=strategy_config,
                        apparatus=apparatus_config,
                        silence=True,
                    )

                    times_val = sum(one_qaoa_run["times"])
                    evals_val = one_qaoa_run["cost_circuit_evals"]
                    aratio_val = one_qaoa_run["approximation_ratio"]

                    times_samples[i] = times_val
                    evals_samples[i] = evals_val
                    aratio_samples[i] = aratio_val


                    raw_writer.writerow({
                        "family": family, **axis_dict, "p": p, "sample_idx": i,
                        "times": times_val, "evals": evals_val, "aratio": aratio_val,
                    })

                raw_file.flush()

                summary_writer.writerow({
                    "family": family, **axis_dict, "p": p,
                    "times_mean": np.mean(times_samples),
                    "times_stderr": np.std(times_samples, ddof=1) / np.sqrt(num_samples),
                    "evals_mean": np.mean(evals_samples),
                    "evals_stderr": np.std(evals_samples, ddof=1) / np.sqrt(num_samples),
                    "aratio_mean": np.mean(aratio_samples),
                    "aratio_stderr": np.std(aratio_samples, ddof=1) / np.sqrt(num_samples),
                })
                summary_file.flush()

                print((family, axis_dict, float(p)), "done")


print(f"Raw samples  written to    {raw_path}")
print(f"Summary stats written to   {summary_path}")
