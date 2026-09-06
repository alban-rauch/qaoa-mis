"""
gen_plot.py
===========
Load and plot the CSV data produced by any experiment.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_csv(file_path):
    # If raw: One row per individual run: family, <axes...>, p, sample_idx, <ys...>
    # If summary: One row per (family, <axes...>, p): mean/stderr of each y in `ys`.
    return pd.read_csv(file_path)

def plot_metric(df, x, y, color, group_by=None, fixed=None, ax=None, title=None):
    # Summary-CSV plot: y_mean (+/- y_stderr) wrt x
    # df:       data
    # x:        column as input variable -- eg "p" / "N" / "q"
    # y:        metric base name -- eg "times" / "evals" / "aratio" / "relax_ratio"
    # group_by: column to draw one line per value -- eg "N" when x="p"
    # fixed:    dict of {column: value} to filter data -- eg {"family": "Gilbert", "q": 0.25}

    data = df
    if fixed:
        for col, val in fixed.items():
            data = data[data[col] == val]

    mean_col, stderr_col = f"{y}_mean", f"{y}_stderr"

    if mean_col not in data.columns:
        raise ValueError(
            f"'{y}' has no '{mean_col}' column -- available metrics: "
            f"{[c[:-5] for c in df.columns if c.endswith('_mean')]}"
        )

    group_cols = [x] + ([group_by] if group_by else [])
    dupe_mask = data.duplicated(subset=group_cols, keep=False)
    if dupe_mask.any():
        example = data.loc[dupe_mask, group_cols].iloc[0].to_dict()
        raise ValueError(
            f"Multiple rows share the same {group_cols} (e.g. {example}) after applying "
            f"`fixed`={fixed}. A column that varies in the data isn't pinned down in "
            f"`fixed` -- add it so each ({x}, {group_by}) point maps to exactly one row."
        )

    create_own_fig = ax is None
    if create_own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    if group_by is None:
        data = data.sort_values(x)
        ax.errorbar(data[x], data[mean_col], yerr=data[stderr_col], marker="o", markersize=3)
    else:
        group_values = sorted(data[group_by].unique())
        shade_values = np.linspace(0.35, 0.9, len(group_values))

        for group_val, shade in zip(group_values, shade_values):
            group = data[data[group_by] == group_val]
            group = group.sort_values(x)
            ax.errorbar(
                group[x], group[mean_col], yerr=group[stderr_col],
                marker="o", markersize=3, 
                color=plt.cm.Blues(shade),
                label=f"{group_by}={group_val}")

    ax.set_xlabel(x)
    ax.set_ylabel(f"mean {y}")
    ax.set_title(title or f"{y} vs {x}" + (f" (grouped by {group_by})" if group_by else ""))
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    if create_own_fig:
        fig.tight_layout()
    return ax


def plot_metrics_comparison(df, x, ys, fixed=None, group_by=None, group_metric_cmaps=None, ax=None, title=None):
    # Overlay several y metrics (each of the form "{y}_mean"/"{y}_stderr") on ONE plot 
    # wrt a single x-axis, at a single fixed slice.

    data = df
    if fixed:
        for col, val in fixed.items():
            data = data[data[col] == val]

    group_cols = [x] + ([group_by] if group_by else [])

    if data.duplicated(subset=group_cols, keep=False).any():
        raise ValueError(
            f"Multiple rows share the same '{x}' value after applying `fixed`={fixed}. "
            f"Pin down every other varying column in `fixed` so each {x} maps to one row."
        )

    create_own_fig = ax is None
    if create_own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    data = data.sort_values(x)
    if group_by is None:
        for y in ys:
            mean_col, stderr_col = f"{y}_mean", f"{y}_stderr"
            if mean_col not in data.columns:
                continue
            ax.errorbar(data[x], data[mean_col], yerr=data[stderr_col],
                        marker="o", markersize=3, label=y)

    else:
        group_values = sorted(data[group_by].unique())
        n_groups = len(group_values)

        if n_groups == 1:
            shade_values = [0.7]
        else:
            shade_values = np.linspace(0.35, 0.9, n_groups)

        for y in ys:
            mean_col = f"{y}_mean"
            stderr_col = f"{y}_stderr"

            if mean_col not in data.columns:
                continue

            cmap = group_metric_cmaps.get(y, plt.cm.viridis)

            for group_val, shade in zip(group_values, shade_values):
                group = data[data[group_by] == group_val]
                group = group.sort_values(x)
                ax.errorbar(
                    group[x],
                    group[mean_col],
                    yerr=group[stderr_col],
                    marker="o",
                    markersize=3,
                    color=cmap(shade),
                    label=f"{y}, {group_by}={group_val}",
                )

    ax.set_xlabel(x)
    ax.set_ylabel("value")
    ax.set_title(
        title or 
        f"{', '.join(ys)} vs {x}"
        + (f" (grouped by {group_by})" if group_by else "")
    )
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    if create_own_fig:
        fig.tight_layout()
    return ax