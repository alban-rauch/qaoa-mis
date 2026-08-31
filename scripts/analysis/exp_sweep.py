"""
exp_sweep.py
============
Collect experimental results from the dataset.
"""

import csv
import itertools
import pickle

import numpy as np

from source.utils import graph_gen as gph


def make_graph_id(family, axis_dict):
    parts = [family]
    for name in sorted(axis_dict):
        parts.append(f"{name}{axis_dict[name]}")
    return "_".join(parts)


def run_sweep(
    exp_name, SWEEP_CONFIG,
    cache_dir, outdir,
    metrics, metric_fn, qaoa_names, extra_args,
):

    #### Output ####
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / f"{exp_name}_data_raw.csv"
    summary_path = outdir / f"{exp_name}_data_summary.csv"


    #### Axes ####
    all_axis_names = sorted({name for cond in SWEEP_CONFIG.values() for name in cond['axes']})

    metrics_mstd_flat = list(itertools.chain.from_iterable([f'{y}_mean', f'{y}_stderr'] for y in metrics))
        # e.g. aratio_mean, aratio_stderr
    raw_fields = ["family"] + all_axis_names + ["p", "sample_idx"] + metrics
    summary_fields = ["family"] + all_axis_names + ["p"] + metrics_mstd_flat


    #### Run ####
    with open(raw_path, "w", newline="") as raw_file, open(summary_path, "w", newline="") as summary_file:

        raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fields, restval="")
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields, restval="")
        raw_writer.writeheader()
        summary_writer.writeheader()

        # FAMILY e.g. Gilbert
        for family, cond in SWEEP_CONFIG.items():
            p_values = cond['p_values']
            num_samples = cond['num_samples']
            axis_names = list(cond['axes'].keys())
            axis_value_lists = [cond['axes'][name] for name in axis_names]
            combos = list(itertools.product(*axis_value_lists))  # e.g. (12, 0.25)

            # GRAPH TYPE e.g. (N12, q0.25)
            for combo in combos:

                axis_dict = dict(zip(axis_names, combo))
                N = axis_dict.get('N')

                # Get the graph type
                graph_id = make_graph_id(family, axis_dict)
                path = cache_dir / f"{graph_id}.pkl"

                if not path.exists():  # Missing graph file
                    print(f"MISSING graph {graph_id} (no cache file at {path}) -- skipped.")
                    continue

                with open(path, "rb") as file:
                    cache = pickle.load(file)


                # RUN (sample, p, method)
                for p in p_values:

                    # Prepare the batching
                    all_samples = {y: np.zeros(num_samples) for y in metrics}
                    n_found = 0

                    for sample_idx in range(num_samples):
                        sample = cache["samples"].get(sample_idx)
                        if sample is None:  # Missing sample
                            print(f"MISSING {graph_id} sample {sample_idx}: not in cache -- skipped.")
                            continue

                        runs = {}

                        # Verify if all samples have the required runs
                        missing_method = False

                        for qaoa_name in qaoa_names:
                            run = sample["runs"].get((qaoa_name, p))

                            if run is None: # Missing run in the sample
                                print(f"MISSING {graph_id} sample {sample_idx}: no ({qaoa_name}, p={p}) run cached -- skipped.")
                                missing_method = True
                                break

                            runs[qaoa_name] = run

                        if missing_method:
                            continue


                        graph = gph.get_graph_from_edges(sample["edges"], N=N)

                        res = metric_fn(
                            family=family, N=N, p=p, axis_dict=axis_dict, sample_idx=sample_idx,
                            graph=graph, runs=runs, extra_args=extra_args,
                        )

                        if set(res.keys()) != set(metrics):
                            raise ValueError(
                                f"metric_fn returned keys {sorted(res.keys())}, "
                                f"expected exactly {sorted(metrics)}."
                            )

                        for y in metrics:
                            all_samples[y][n_found] = res[y]
                        n_found += 1

                        raw_writer.writerow({
                            "family": family, **axis_dict, "p": p,
                            "sample_idx": sample_idx, **res,
                        })

                    raw_file.flush()

                    if n_found == 0:
                        print(f"SKIPPING summary for {graph_id}, p={p}: no complete samples found")
                        continue

                    summary_row = {"family": family, **axis_dict, "p": p}
                    for y in metrics:
                        s = all_samples[y][:n_found]
                        summary_row[f"{y}_mean"] = np.mean(s)
                        summary_row[f"{y}_stderr"] = np.std(s, ddof=1) / np.sqrt(n_found) if n_found > 1 else np.nan
                    summary_writer.writerow(summary_row)
                    summary_file.flush()

                    print((family, axis_dict, float(p)), f"done ({n_found}/{num_samples} samples)")

    print(f"Raw samples written to     {raw_path}")
    print(f"Summary stats written to   {summary_path}")