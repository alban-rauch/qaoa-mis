"""
spec_plot.py
============
Load and plot the CSV data produced by specific experiments
(speed_data4_raw.csv / speed_data4_summary.csv).
"""

from source.paths import DATA_DIR
from scripts.analysis.old.gen_plot import load_raw, load_summary, plot_metric, plot_metrics_comparison

RAW_PATH = DATA_DIR / "analysis_data/exp_0/speed_data4_raw.csv"
SUMMARY_PATH = DATA_DIR / "analysis_data/exp_0/speed_data4_summary.csv"


if __name__ == "__main__":

    summary = load_summary(DATA_DIR / "analysis_data/exp_0", "speed2208")
    for variable in ["times", "evals", "aratio"]:
        for family in ["linear", "circular", "complete"]:
            ax = plot_metric(summary, x="N", y=variable, group_by="p", fixed={"family": family})
            ax.figure.tight_layout()
            ax.figure.savefig(DATA_DIR / f"analysis_data/exp_0/speed_exp2208_{variable}_{family}.png", dpi=150)
        ax = plot_metric(summary, x="N", y=variable, group_by="p", fixed={"family": "Gilbert", "q": 0.25})
        ax.figure.tight_layout()
        ax.figure.savefig(DATA_DIR / f"analysis_data/exp_0/speed_exp2208_{variable}_Gilbert.png", dpi=150)

    summary2 = load_summary(DATA_DIR / "analysis_data/exp_2", "relax2208")
    # for variable in ["times", "evals", "aratio"]:
    #     for family in ["linear", "circular", "complete"]:
    #         ax = plot_metric(summary, x="N", y=variable, group_by="p", fixed={"family": family})
    #         ax.figure.tight_layout()
    #         ax.figure.savefig(DATA_DIR / f"analysis_data/exp_0/speed_exp2208_{variable}_{family}.png", dpi=150)
    #     ax = plot_metric(summary, x="N", y=variable, group_by="p", fixed={"family": "Gilbert", "q": 0.25})
    #     ax.figure.tight_layout()
    #     ax.figure.savefig(DATA_DIR / f"analysis_data/exp_0/speed_exp2208_{variable}_Gilbert.png", dpi=150)
    # ax2 = plot_metrics_comparison(summary2, x="p", ys=["relax_ratio", "opt_warm_ratio", "opt_cold_ratio"],
    #                      fixed={"family": "Gilbert", "N": 13, "q": 0.25})

    # ax2.figure.tight_layout()
    # ax2.figure.savefig(DATA_DIR / "analysis_data/exp_2/ratio_comparison.png", dpi=150)
