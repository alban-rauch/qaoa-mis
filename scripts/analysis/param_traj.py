import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LogNorm, FuncNorm

def moving_average(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window) / window     # [1/w, ..., 1/w]
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")
 
def format_as_pi_fraction(theta):
    # e.g. 1.5708 -> "0.50π"
    return f"{theta / np.pi:.2f}\u03c0"

def plot_angle_polar_trajectories(
    angle_history,
    cost_history,
    cmap,
    norm,
    color_bounds,
    reverse_radius=True,        # Trajectory from edge to center
    smooth_window=1,
    linewidth=2.0,
    show_final_values=True,     # final angle, final metric
):
    history = np.asarray(angle_history)
    cost = np.asarray(cost_history, dtype=float)
    n_iter, _, p = history.shape
    cost_smooth = moving_average(cost, smooth_window)

    gamma = history[:, 0, :]  # (n_iter, p)
    beta = history[:, 1, :]   # (n_iter, p)
    theta_gamma = np.mod(gamma, 2 * np.pi)
    theta_beta = np.mod(2 * beta, 2 * np.pi)

    if reverse_radius:
        r = n_iter - 1 - np.arange(n_iter)  # edge -> center
    else:
        r = np.arange(n_iter)               # center -> edge
    fig, axes = plt.subplots(
        2, p, figsize=(3.4*p, 6.8),
        subplot_kw={"projection": "polar"}
    )

    if p == 1:
        axes = axes.reshape(2, 1)


    def draw_rays(ax, theta, r, c):         # Rays btw consecutive points
        points = np.array([theta, r]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = 0.5 * (c[:-1] + c[1:])
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidth=linewidth,
        )
        lc.set_array(seg_colors)
        ax.add_collection(lc)
        return lc
    r_max_global = r.max() * 1.05

    rad_ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    rad_labels = [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', 
                  r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
    
    for i in range(p):

        # gamma subplot
        ax = axes[1, i]
        lc = draw_rays(ax, theta_gamma[:, i], r, cost_smooth)

        ax.plot(
            [theta_gamma[0, i], theta_gamma[0, i]],
            [0, r_max_global],
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

        ax.plot(
            [theta_gamma[-1, i], theta_gamma[-1, i]],
            [0, r_max_global],
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

        ax.text(
            theta_gamma[0, i],
            r_max_global * 1.1,
            f"{format_as_pi_fraction(theta_gamma[0, i])}",
            fontsize=7,
            ha="center",
            va="bottom",
        )

        ax.text(
            theta_gamma[-1, i],
            r_max_global * 1.1,
            f"{format_as_pi_fraction(theta_gamma[-1, i])}",
            fontsize=7,
            ha="center",
            va="bottom",
        )

        ax.scatter(theta_gamma[-1, i], r[-1], color="red", s=40, zorder=5)
        title = f"\u03b3_{i}"
        if show_final_values:
            title += f"\n {format_as_pi_fraction(theta_gamma[0, i])} → {format_as_pi_fraction(theta_gamma[-1, i])}"
        ax.set_title(title, fontsize=9)
        ax.set_yticklabels([])
        ax.set_ylim(0, r_max_global)
        ax.set_xticks(rad_ticks)
        ax.set_xticklabels(rad_labels)
        ax.grid(True)

        # beta subplot
        ax = axes[0, i]
        lc = draw_rays(ax, theta_beta[:, i], r, cost_smooth)
        ax.plot(
            [theta_beta[0, i], theta_beta[0, i]],
            [0, r_max_global],
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

        ax.plot(
            [theta_beta[-1, i], theta_beta[-1, i]],
            [0, r_max_global],
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

        ax.text(
            theta_beta[0, i],
            r_max_global * 1.1,
            f"{format_as_pi_fraction(theta_beta[0, i])}",
            fontsize=7,
            ha="center",
            va="bottom",
        )

        ax.text(
            theta_beta[-1, i],
            r_max_global * 1.1,
            f"{format_as_pi_fraction(theta_beta[-1, i])}",
            fontsize=7,
            ha="center",
            va="bottom",
        )

        ax.scatter(theta_beta[-1, i], r[-1], color="red", s=40, zorder=5, label="final")
        title = f"\u03b2_{i}"
        if show_final_values:
            title += f"\n {format_as_pi_fraction(theta_beta[0, i])} → {format_as_pi_fraction(theta_beta[-1, i])}"
        ax.set_title(title, fontsize=9)
        ax.set_yticklabels([])
        ax.set_ylim(0, r_max_global)
        ax.set_xticks(rad_ticks)
        ax.set_xticklabels(rad_labels)
        ax.grid(True)

    cbar = fig.colorbar(
        lc,
        ax=axes,
        boundaries=color_bounds,
        spacing="uniform",
        orientation="vertical",
        fraction=0.02,
        pad=0.08,
    )  

    ticks = [
        0.0,
        0.5,
        0.8,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00
    ]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(
        [f"{x:.2f}" for x in ticks]
    )
    cbar.set_label("Approx. ratio")


    suptitle = f"QAOA angle trajectories"
    if show_final_values:
        suptitle += f"  |  Final: {cost[-1]:.4f}"
    fig.suptitle(suptitle)
    plt.show()


if __name__ == "__main__":

    import source.qaoa_run as qr
    from source.utils import graph_gen as gph
    from source.utils.classical import best_config_branch_bound
    from source.utils.cond_gen import load_condition
    from source.paths import DATA_DIR, COND_DIR, GRAPHS_DIR

    color_bounds = np.concatenate([
        np.linspace(0.0, 0.4, 5),
        np.linspace(0.5, 0.70, 5),
        np.linspace(0.75, 0.99, 25),
        [1.0],
    ])

    cmap = plt.get_cmap("magma", len(color_bounds) - 1)
    norm = BoundaryNorm(color_bounds, cmap.N)
    
    for i in range(10):
        exp_configs = load_condition(COND_DIR / f"cond2.json")

        exp_configs[1]["init_param"] = [i / 10, 0.33]

        family = "Gilbert"
        N = 12
        param = (12, 0.25)
        graphs_loaded = {}
        graphs_loaded[family] = gph.load_family(
            GRAPHS_DIR / f'{family}.npz', 
            family, 
            params=[param]
        )
        graph = gph.get_graph_from_edges(
            gph.get_sample(
                graphs_loaded[family], 
                param, 
                s=1), 
            N=N,
        )

        exp_configs[0]["N"] = N
        exp_configs[0]["graph"] = graph
        exp_configs[2]["p"] = 2
        exp_configs[2]["device"] = "lightning.qubit"

        one_qaoa_run = qr.run_qaoa(
            problem=exp_configs[0],
            strategy=exp_configs[1],
            apparatus=exp_configs[2],
            silence=False
        )

        theo_best_cost, _ = best_config_branch_bound(graph)
        approx_ratio_history = qr.approx_ratio(
            graph,
            one_qaoa_run["best_energy_p"],
            1.5,
            theo_best_cost,
        )

        plot_angle_polar_trajectories(
            one_qaoa_run["param_hist"],
            approx_ratio_history,
            cmap=cmap,
            norm=norm,
            color_bounds=color_bounds,
            reverse_radius=True,
            smooth_window=1,
            linewidth=2.0,
        )