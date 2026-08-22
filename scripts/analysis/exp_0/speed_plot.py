"""
speed_plot.py
=============
Load and plot the CSV data produced by the speed experiment
(speed_data4_raw.csv / speed_data4_summary.csv).
"""

import matplotlib.pyplot as plt
import pandas as pd

from source.paths import DATA_DIR

RAW_PATH = DATA_DIR / "analysis_data/exp_0/speed_data4_raw.csv"
SUMMARY_PATH = DATA_DIR / "analysis_data/exp_0/speed_data4_summary.csv"


def load_raw(path=RAW_PATH):
    # One row per individual QAOA run: family, N, q, p, sample_idx, times, evals, aratio.
    return pd.read_csv(path)


def load_summary(path=SUMMARY_PATH):
    # One row per (family, N, q, p): mean/stderr of times, evals, aratio.
    return pd.read_csv(path)


def plot_metric(df, x, y, group_by=None, fixed=None, ax=None, title=None):
    # Summary-CSV plot: y_mean (+/- y_stderr) wrt x
    # x:        column as input variable -- eg "p" / "N" / "q"
    # y:        metric base name -- eg "times" / "evals" / "aratio"
    # group_by: column to draw one line per value -- eg "N" when x="p"
    # fixed:    dict of {column: value} to filter data -- eg {"family": "Gilbert", "q": 0.25}

    data = df
    if fixed:
        for col, val in fixed.items():
            data = data[data[col] == val]

    mean_col, stderr_col = f"{y}_mean", f"{y}_stderr"

    # Check if ...
    if mean_col not in data.columns:
        raise ValueError(f"'{y}' has no '{mean_col}' column -- available metrics: "
                            f"{[c[:-5] for c in df.columns if c.endswith('_mean')]}")

    # Check if fixed has pinned down every varying column
    group_cols = [x] + ([group_by] if group_by else [])
    dupe_mask = data.duplicated(subset=group_cols, keep=False)
    if dupe_mask.any():
        example = data.loc[dupe_mask, group_cols].iloc[0].to_dict()
        raise ValueError(
            f"Multiple rows share the same {group_cols} (e.g. {example}) after applying "
            f"`fixed`={fixed}. A column that varies  in the data isn't pinned down in "
            f"`fixed` -- add it so each ({x}, {group_by}) point maps to exactly one row."
        )

    # ax specified when making several plots on the same figure
    create_own_fig = ax is None
    if create_own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    if group_by is None:
        data = data.sort_values(x)
        ax.errorbar(data[x], data[mean_col], yerr=data[stderr_col], marker="o", markersize=3)
    else:
        for group_val, group in data.groupby(group_by):
            group = group.sort_values(x)
            ax.errorbar(group[x], group[mean_col], yerr=group[stderr_col],
                        marker="o", markersize=3, label=f"{group_by}={group_val}")
        ax.legend(fontsize=7, ncol=2)

    ax.set_xlabel(x)
    ax.set_ylabel(f"mean {y}")
    ax.set_title(title or f"{y} vs {x}" + (f" (grouped by {group_by})" if group_by else ""))
    if create_own_fig:
        fig.tight_layout()
    return ax

if __name__ == "__main__":

    raw = load_raw()
    summary = load_summary()

    # General info
    print(f"raw: {len(raw)} rows, columns: {list(raw.columns)}")
    print(f"summary: {len(summary)} rows, columns: {list(summary.columns)}\n")

    # Filtering for Gilbert, N=10, all p values [summary]
    n10_gilbert = summary[(summary["family"] == "Gilbert") & (summary["N"] == 10)]
    print("Gilbert, N=10, all p values:")
    print(n10_gilbert[["p", "times_mean", "times_stderr", "aratio_mean"]].to_string(index=False))

    # Spread of a single (N, p) combo across samples [raw data]
    one_combo = raw[(raw["family"] == "Gilbert") & (raw["N"] == 10) & (raw["p"] == 5)]
    print(f"\nGilbert, N=10, p=5 -- {len(one_combo)} individual samples:")
    print(one_combo[["sample_idx", "times", "evals", "aratio"]].to_string(index=False))

    # Data for specific plot (here pivot with Gilbert) [summary]
    pivot = summary[summary["family"] == "Gilbert"].pivot_table(
        index="N", columns="p", values="times_mean"
    )
    print("\ntimes_mean pivoted (rows=N, columns=p):")
    print(pivot)

    # Plotting for every x, y, grouping

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    plot_metric(summary, x="p", y="times", group_by="N",
                fixed={"family": "Gilbert", "q": 0.25}, ax=axes[0, 0])
    plot_metric(summary, x="N", y="times", group_by="p",
                fixed={"family": "Gilbert", "q": 0.25}, ax=axes[1, 0])

    plot_metric(summary, x="p", y="evals", group_by="N",
                fixed={"family": "Gilbert", "q": 0.25}, ax=axes[0, 1])
    plot_metric(summary, x="N", y="evals", group_by="p",
                fixed={"family": "Gilbert", "q": 0.25}, ax=axes[1, 1])

    plot_metric(summary, x="p", y="aratio", group_by="N",
                fixed={"family": "Gilbert", "q": 0.25}, ax=axes[0, 2])
    plot_metric(summary, x="N", y="aratio", group_by="p",
                fixed={"family": "Gilbert", "q": 0.25}, ax=axes[1, 2])

    fig.tight_layout()
    fig.savefig(DATA_DIR / "analysis_data/exp_0/grid_overview.png", dpi=150)
    print("\nSaved plot: grid_overview.png (times/evals/aratio, vs p and vs N)")

    # For one specific plot:
    #   ax = plot_metric(summary, x="p", y="times", group_by="N",
    #                     fixed={"family": "Gilbert", "q": 0.25})
    #   ax.figure.savefig("my_plot.png", dpi=150)