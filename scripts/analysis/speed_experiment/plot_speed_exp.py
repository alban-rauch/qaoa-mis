"""
plot_speed_exp.py
=================
"""

from source.paths import DATA_DIR
from scripts.analysis.exp_plot import load_csv, plot_metric, plot_metrics_comparison
from matplotlib import pyplot as plt

# RAW_PATH = DATA_DIR / "analysis_data/speed_exp/speed_exp_data_raw.csv"
# SUMMARY_PATH = DATA_DIR / "analysis_data/speed_exp/speed_exp_data_summary.csv"
RAW_PATH = DATA_DIR / "analysis_data/speed_exp/speed_exp_extzero_data_raw.csv"
SUMMARY_PATH = DATA_DIR / "analysis_data/speed_exp/speed_exp_extzero_data_summary.csv"

METRIC_CMAPS = {
    "times_cold": plt.cm.Blues,
    "times_warm": plt.cm.Reds,
    "evals_cold": plt.cm.Blues,
    "evals_warm": plt.cm.Reds,
    "aratio_cold": plt.cm.Blues,
    "aratio_warm": plt.cm.Reds,
}

summary = load_csv(SUMMARY_PATH)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
# ax1 = plot_metrics_comparison(
#     df=summary, 
#     x="N", 
#     ys=["aratio_cold"], 
#     group_by="p",
#     fixed={"family": "DRegular", "d": 3},
#     group_metric_cmaps=METRIC_CMAPS,
#     ax=ax1
# )
# ax2 = plot_metrics_comparison(
#     df=summary, 
#     x="N", 
#     ys=["aratio_warm"], 
#     group_by="p",
#     fixed={"family": "DRegular", "d": 3},
#     group_metric_cmaps=METRIC_CMAPS,
#     ax=ax2,
# )
# ax1.get_legend().remove()
# ax2.get_legend().remove()

# plt.show()

for stud in ["times", "evals"]:
    fam = "DRegular"
    for d in [2, 3]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        ax1 = plot_metrics_comparison(
            df=summary, 
            x="N", 
            ys=[f"{stud}_cold"], 
            group_by="p",
            fixed={"family": fam, "d": d},
            group_metric_cmaps=METRIC_CMAPS,
            ax=ax1
        )
        ax2 = plot_metrics_comparison(
            df=summary, 
            x="N", 
            ys=[f"{stud}_warm"], 
            group_by="p",
            fixed={"family": fam, "d": d},
            group_metric_cmaps=METRIC_CMAPS,
            ax=ax2,
        )
        ax1.get_legend().remove()
        # ax1.set_ylim(top=1.05)
        ax2.get_legend().remove()
        # ax2.set_ylim(top=1.05)

        fig.tight_layout()
        # fig.savefig(DATA_DIR / f"analysis_data/speed_exp/{stud}_{fam}_d{d}.png", dpi=150)
        fig.savefig(DATA_DIR / f"analysis_data/speed_exp/{stud}/{stud}_{fam}_d{d}_extzero.png", dpi=150)