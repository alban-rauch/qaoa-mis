"""
gen_sweep.py
============
"""

import numpy as np
import csv
import itertools

from source.paths import GRAPHS_DIR
from source.utils import graph_gen as gph


def run_sweep(
    exp_name, SWEEP_CONFIG, 
    exp_configs, outdir, 
    metrics, metric_fn, extra_args,
):

    problem_config, strategy_config, apparatus_config = exp_configs

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
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / f"{exp_name}_data_raw.csv"
    summary_path = outdir / f"{exp_name}_data_summary.csv"

    all_axis_names = sorted({name for cond in SWEEP_CONFIG.values() for name in cond['axes']})

    metrics_mstd = [[f'{y}_mean', f'{y}_stderr'] for y in metrics]
    metrics_mstd_flat = list(itertools.chain.from_iterable(metrics_mstd))

    raw_fields = ["family"] + all_axis_names + ["p", "sample_idx"] + metrics
    summary_fields = ["family"] + all_axis_names + ["p"] + metrics_mstd_flat


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

                    all_samples = {y: np.zeros(num_samples) for y in metrics}

                    for idx in range(num_samples):
                        problem_config["graph"] = gph.get_graph_from_edges(
                            gph.get_sample(random_graphs[family], key, s=idx), N=N
                        )

                        res = metric_fn(
                            family=family, N=N, p=p,
                            axis_dict=axis_dict, sample_idx=idx,
                            problem_config=problem_config,
                            strategy_config=strategy_config,
                            apparatus_config=apparatus_config,
                            extra_args=extra_args,

                        )

                        if set(res.keys()) != set(metrics):
                            raise ValueError(
                                f"metric_fn returned keys {sorted(res.keys())}, "
                                f"expected exactly {sorted(metrics)} (from `ys`). "
                            )

                        for y in metrics:
                            all_samples[y][idx] = res[y]

                        raw_writer.writerow({
                            "family": family, **axis_dict, "p": p, 
                            "sample_idx": idx, **res,
                        })


                    raw_file.flush()

                    summary_row = {"family": family, **axis_dict, "p": p}

                    for y in metrics:
                        s = all_samples[y]
                        summary_row[f"{y}_mean"] = np.mean(s)
                        summary_row[f"{y}_stderr"] = (
                            np.std(s, ddof=1) / np.sqrt(num_samples) 
                                if num_samples > 1 else np.nan
                        )

                    summary_writer.writerow(summary_row)
                    summary_file.flush()

                    print((family, axis_dict, float(p)), "done")


    print(f"Raw samples written to     {raw_path}")
    print(f"Summary stats written to   {summary_path}")
