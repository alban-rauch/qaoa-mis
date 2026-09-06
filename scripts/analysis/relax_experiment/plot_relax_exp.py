"""
plot_relax_exp.py
=================
"""

from source.paths import DATA_DIR
from scripts.analysis.exp_plot import load_csv, plot_metric, plot_metrics_comparison
from matplotlib import pyplot as plt

RAW_PATH = DATA_DIR / "analysis_data/relax_exp/relax_exp_data_raw.csv"
SUMMARY_PATH = DATA_DIR / "analysis_data/relax_exp/relax_exp_data_summary.csv"

METRIC_CMAPS = {
    "relax_core_ratio": plt.cm.Greens,
    "relax_rounded_ratio": plt.cm.Purples,
    "opt_warm_ratio": plt.cm.Reds,
    "opt_cold_ratio": plt.cm.Blues,
}

summary = load_csv(SUMMARY_PATH)
ax = plot_metrics_comparison(
    df=summary, 
    x="N", 
    group_by=None,
    ys=["relax_rounded_ratio", "opt_warm_ratio", "opt_cold_ratio"],
    fixed={"family": "circular", "p": 10},
    group_metric_cmaps=METRIC_CMAPS,
)

ax.figure.tight_layout()
plt.show()
# ax.figure.savefig(DATA_DIR / "analysis_data/relax_exp/ratio_comparison.png", dpi=150)
