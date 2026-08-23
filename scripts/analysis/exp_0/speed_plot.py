"""
speed_plot.py
=============
Load and plot the CSV data produced by the speed experiment
(speed_data4_raw.csv / speed_data4_summary.csv).
"""

import matplotlib.pyplot as plt
import pandas as pd

from source.paths import DATA_DIR
from scripts.analysis.gen_plot import load_raw, load_summary, plot_metric, plot_metrics_comparison

RAW_PATH = DATA_DIR / "analysis_data/exp_0/speed_data4_raw.csv"
SUMMARY_PATH = DATA_DIR / "analysis_data/exp_0/speed_data4_summary.csv"


if __name__ == "__main__":

    summary = load_summary(DATA_DIR / "analysis_data/exp_0", "speed_exp2208")
    ax = plot_metric(summary, x="p", y="times", group_by="N", fixed={"family": "Gilbert", "q": 0.25})

    ax.figure.tight_layout()
    ax.figure.savefig(DATA_DIR / "analysis_data/exp_0/speed_exp2208_1.png", dpi=150)


    summary2 = load_summary(DATA_DIR / "analysis_data/exp_2", "relax_exp2208")
    ax2 = plot_metrics_comparison(summary2, x="p", ys=["relax_ratio", "opt_warm_ratio", "opt_cold_ratio"],
                         fixed={"family": "Gilbert", "N": 15, "q": 0.25})

    ax2.figure.tight_layout()
    ax2.figure.savefig(DATA_DIR / "analysis_data/exp_2/ratio_comparison.png", dpi=150)
