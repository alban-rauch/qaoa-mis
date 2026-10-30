
from source.paths import DATA_DIR
from scripts.analysis.exp_plot import load_csv, plot_metric, plot_metrics_comparison
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path


CONDITIONS = {
    # "default": DATA_DIR / "analysis_data/relax_exp/relax_exp_data_summary.csv",
    "00": DATA_DIR / "analysis_data/relax_exp/relax_exp_00_data_summary.csv",
    "cubic": DATA_DIR / "analysis_data/relax_exp/relax_exp_cubic_data_summary.csv",
    # "extzero": DATA_DIR / "analysis_data/relax_exp/relax_exp_extzero_data_summary.csv",
}

def load_combined(conditions=CONDITIONS):
    frames = []
    reference_cols = None
    for label, path in conditions.items():
        if not Path(path).exists():
            print(f"SKIPPING '{label}': no file at {path}")
            continue
        df = load_csv(path).copy()
        df["condition"] = label
        cols = set(df.columns) - {"condition"}
        if reference_cols is None:
            reference_cols = cols
        else:
            missing = reference_cols - cols
            extra = cols - reference_cols
            if missing or extra:
                print(f"WARNING: '{label}' columns differ -- missing={missing}, extra={extra}")
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No condition CSVs found -- check CONDITIONS paths.")
    return pd.concat(frames, ignore_index=True)


def plot_ratio_across_conditions(combined, y, x="p", fixed=None, ax=None):
    return plot_metric(
        combined, x=x, y=y, group_by="condition", fixed=fixed, ax=ax,
        title=f"{y} across parameter-transfer conditions",
    )


def table_at_slice(combined, family, N, p, axis_dict=None, metrics=None):
    fixed = {"family": family, "N": N, "p": p, **(axis_dict or {})}
    data = combined
    for col, val in fixed.items():
        data = data[data[col] == val]
    if metrics is None:
        metrics = [c[:-5] for c in combined.columns if c.endswith("_mean")]
    cols = ["condition"] + [f"{m}_mean" for m in metrics] + [f"{m}_stderr" for m in metrics]
    return data[cols].sort_values("condition").reset_index(drop=True)
 
 
if __name__ == "__main__":
    combined = load_combined()
    print(f"Loaded {len(combined)} rows across {combined['condition'].nunique()} conditions: "
          f"{sorted(combined['condition'].unique())}\n")
 
    metrics_to_show = ["relax_rounded_ratio", "opt_warm_ratio", "opt_cold_ratio"]
    fixed = {"family": "Gilbert", "N": 10, "q": 0.25}
 
    fig, axes = plt.subplots(1, 3,)
    for ax, y in zip(axes.flat, metrics_to_show):
        plot_ratio_across_conditions(combined, y=y, fixed=fixed, ax=ax)
        ax.get_legend().remove()
        ax.title.set_visible(False)
    fig.suptitle(f"{fixed}")
    fig.tight_layout()
    plt.show()
 
    # Example tabular lookup:
    print(table_at_slice(combined, family="Gilbert", N=10, p=3, axis_dict={"q": 0.25}))
