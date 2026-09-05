import numpy as np
import colorsys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LogNorm, FuncNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D


### Helpers ###
 
def decimal_to_pi_frac(theta):
    # e.g. 1.5708 -> "0.50π"
    return f"{theta / np.pi:.2f}\u03c0"

def make_hue_cmap(
        base_rgb, 
        N, 
        s_range=(0.3, 1.0), 
        v_range=(0.8, 0.2),
        a_range=(0.5, 1.0),
    ):
    h, _, _ = colorsys.rgb_to_hsv(*base_rgb[:3])
    s_lo, s_hi = s_range
    v_lo, v_hi = v_range
    a_lo, a_hi = a_range
    ts = np.linspace(0, 1, N)
    colors = [
        colorsys.hsv_to_rgb(
            h, 
            s_lo + t * (s_hi - s_lo), 
            v_lo + t * (v_hi - v_lo),
        ) +(a_lo + t * (a_hi - a_lo),)
        for t in ts
    ]
    return LinearSegmentedColormap.from_list(f"hue{h:.2f}", colors, N=N)

def load_trajs(p, init_angle_considered):

    approx_ratio_histories = []
    param_histories = []

    for pair in init_angle_considered: 
        (i, j) = pair
        exp_configs = load_condition(COND_DIR / f"cond2.json")
        exp_configs[1]["init_param"] = [i, j]

        family = "Gilbert"
        N = 12
        param = (12, 0.25)
        graphs_loaded = {}
        graphs_loaded[family] = gph.load_family(
            GRAPHS_DIR / f'{family}.npz', 
            family, 
            params=[param],
        )
        graph = gph.get_graph_from_edges(
            gph.get_sample(graphs_loaded[family], param, s=2), 
            N=N,
        )

        exp_configs[0]["N"] = N
        exp_configs[0]["graph"] = graph
        exp_configs[2]["p"] = p
        exp_configs[1]["param_transfer_type"] = 'interp'
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

        param_history = one_qaoa_run["param_hist"]

        approx_ratio_histories.append(approx_ratio_history)
        param_histories.append(param_history)

    return approx_ratio_histories, param_histories



### RADAR PLOT ###

def create_polar_fig(p):
    fig, axes = plt.subplots(
        2, p, figsize=(3.4 * p, 9.5),
        subplot_kw={"projection": "polar"}
    )

    if p == 1:
        axes = axes.reshape(2, 1)

    FIELD_LINE = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    FIELD_LINE_LABELS = [
        r'$0$', 
        '',
        r'$\pi / 2$', 
        '',
        r'$\pi$', 
        '',
        r'$3\pi / 2$', 
        '',
    ]

    theta = np.linspace(0, 2 * np.pi, 200)
    equipot_r = np.arange(0.2, 1.01, 0.2)
    segments = [np.column_stack([theta, np.full_like(theta, r)]) for r in equipot_r]
    colors = [
        (0.6, 0.5, 0.5, 0.8),    # Dark gray
        (0.5, 0.5, 0.5, 0.15),   # Light gray
        (0.5, 0.5, 0.5, 0.4),    # Medium gray
        (0.5, 0.5, 0.5, 0.15),   # Light gray
        (0.6, 0.5, 0.5, 0.8),    # Dark gray
    ]
    linewidths = [0.8, 0.8, 0.8, 0.8, 0.8]

    for i in range(p):
        for row, label in ((0, r'\gamma'), (1, r'2\beta')):
            ax = axes[row, i]
            ax.text(3 * np.pi / 4, 1.4, fr"${label}_{i}$", fontsize=12, ha='center', va='center')
            ax.set_xticks(FIELD_LINE)
            ax.set_xticklabels(FIELD_LINE_LABELS, fontsize=9)
            ax.set_yticks([])
            ax.grid(False)
            ax.xaxis.grid(True, color='gray', alpha=0.3)

            lc = LineCollection(segments, colors=colors, linewidths=linewidths)
            ax.add_collection(lc)
            ax.set_rlim(0, 1.0)

    fig.subplots_adjust(hspace=0.5, top=0.90)
    return fig, axes

def finalize_polar_fig(fig, axes, lc, color_bounds, contrast_params, run_colors, run_labels, title="QAOA angle trajectories"):
    def make_alpha_grey_cmap(N, h=0.83, contrast_params=contrast_params):
        s_lo, s_hi = contrast_params["s_range"]
        v_lo, v_hi = contrast_params["v_range"]
        a_lo, a_hi = contrast_params["a_range"]
        ts = np.linspace(0, 1, N)
        colors = [
            colorsys.hsv_to_rgb(
                h, 
                s_lo + t * (s_hi - s_lo), 
                v_lo + t * (v_hi - v_lo),
            ) +(a_lo + t * (a_hi - a_lo),) 
            for t in ts
        ]
        return LinearSegmentedColormap.from_list("alpha_grey", colors, N=N)
    grey_sm = plt.cm.ScalarMappable(cmap=make_alpha_grey_cmap(lc.cmap.N), norm=lc.norm)
    grey_sm.set_array([])
    cbar = fig.colorbar(
        grey_sm,
        ax=axes,
        boundaries=color_bounds,
        spacing="proportional",
        orientation="vertical",
        fraction=0.02,
        pad=0.08,
    )
    TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cbar.set_ticks(TICKS)
    cbar.set_ticklabels([f"{x:.2f}" for x in TICKS])
    cbar.set_label("Approx. ratio", rotation=270, labelpad=15)
 
    fig.suptitle(title)

    handles = [
        Line2D([0], [0], marker='o', linestyle='', color=run_colors[i], label=lbl)
        for i, lbl in enumerate(run_labels)
    ]

    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
               fontsize=8, title="Run (init)", frameon=False)


### RAY TRAJECTORIES ###

def moving_average(x, window):
    # Smooth out the curve
    if window <= 1:
        return x
    kernel = np.ones(window) / window     # [1/w, ..., 1/w]
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")

def add_traj(
    axes,
    angle_history,
    cost_history,
    cmap,
    norm,
    reverse_radius=True,
    smooth_window=1,
    show_endpoints=True,        # final angle
    endpoint_kwargs=None,
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
        r = np.linspace(1.0, 0.2, n_iter)   # edge -> center
    else:
        r = np.linspace(0.2, 1.0, n_iter)   # center -> edge

    r_max_global = r.max() * 1.05

    def draw_rays(ax, theta, r, c):         # Rays btw consecutive points
        points = np.array([theta, r]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = 0.5 * (c[:-1] + c[1:])
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidth=1.4,
        )
        lc.set_array(seg_colors)
        ax.add_collection(lc)
        ax.set_ylim(0, max(ax.get_ylim()[1], r_max_global))
        return lc

    final_cost = cost[-1]
    final_color = cmap(norm(final_cost))

    ek = dict(s=18, zorder=5, color=final_color)
    if endpoint_kwargs:
        ek.update(endpoint_kwargs)


    lc = None
    for i in range(p):
        ax = axes[0, i]
        lc = draw_rays(ax, theta_gamma[:, i], r, cost_smooth)
        if show_endpoints:
            ax.scatter(theta_gamma[-1, i], r[-1], **ek)
 
        ax = axes[1, i]
        lc = draw_rays(ax, theta_beta[:, i], r, cost_smooth)
        if show_endpoints:
            ax.scatter(theta_beta[-1, i], r[-1], **ek)
 
    return lc



if __name__ == "__main__":

    import source.qaoa_run as qr
    from source.utils import graph_gen as gph
    from source.utils.classical import best_config_branch_bound
    from source.utils.cond_gen import load_condition
    from source.paths import DATA_DIR, COND_DIR, GRAPHS_DIR

    init_angle_considered = [(i / 3, j / 3) for i in range(3) for j in range(3)]
    p = 1

    LIGHTNESS_BOUNDS = np.array([
        0.0, 0.4,
        0.6, 0.7, 0.75,
        0.8, 0.85, 0.9, 0.95, 1.0,
    ])

    CONTRAST_PARAMS = {
        "s_range": (0.3, 1.0), 
        "v_range": (0.8, 0.0),
        "a_range": (0.5, 1.0),
    }

    n_lightness = len(LIGHTNESS_BOUNDS) - 1
    norm = BoundaryNorm(LIGHTNESS_BOUNDS, n_lightness)

    n_runs = len(init_angle_considered)
    run_colors = plt.cm.tab10(np.linspace(0, 1, n_runs))
    run_cmaps = [make_hue_cmap(run_colors[i], n_lightness, **CONTRAST_PARAMS) for i in range(n_runs)]


    fig, axes = create_polar_fig(p)


    approx_ratio_histories, param_histories = load_trajs(p, init_angle_considered)

    last_lc = None
    final_params = [par[-1] for par in param_histories]
    final_costs = [cost[-1] for cost in approx_ratio_histories]
    run_labels = [f"init={i/n_runs:.1f}" for i in range(len(approx_ratio_histories))]

    for i in range(len(approx_ratio_histories)):

        param_history = param_histories[i]
        approx_ratio_history = approx_ratio_histories[i]
        cmap = run_cmaps[i]

        last_lc = add_traj(
            axes,
            param_history,
            approx_ratio_history,
            cmap=cmap,
            norm=norm,
            reverse_radius=True,
            smooth_window=1,
            show_endpoints=True,
        )


    finalize_polar_fig(
        fig, axes, last_lc, LIGHTNESS_BOUNDS, CONTRAST_PARAMS, run_colors, run_labels, 
        title=f"QAOA angle trajectories |  Final AR: {max(final_costs):.2f}",
    )

    plt.show()