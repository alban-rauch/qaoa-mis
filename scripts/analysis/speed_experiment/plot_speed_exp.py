"""
plot_speed_exp.py
=================
"""

from source.paths import DATA_DIR
from scripts.analysis.exp_plot import load_csv, plot_metric, plot_metrics_comparison
from matplotlib import pyplot as plt

RAW_PATH = DATA_DIR / "analysis_data/speed_exp/speed_exp_data_raw.csv"
SUMMARY_PATH = DATA_DIR / "analysis_data/speed_exp/speed_exp_data_summary.csv"

METRIC_CMAPS = {
    "times_cold": plt.cm.Blues,
    "times_warm": plt.cm.Reds,
    "evals_cold": plt.cm.Blues,
    "evals_warm": plt.cm.Reds,
    "aratio_cold": plt.cm.Blues,
    "aratio_warm": plt.cm.Reds,
}

summary = load_csv(SUMMARY_PATH)
ax = plot_metrics_comparison(
    df=summary, 
    x="N", 
    ys=["times_cold", "times_warm"], 
    group_by="p",
    fixed={"family": "DRegular", "d": 3},
    group_metric_cmaps=METRIC_CMAPS,
)

ax.figure.tight_layout()
plt.show()