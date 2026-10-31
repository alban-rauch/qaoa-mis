import numpy as np
import colorsys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import (
    BoundaryNorm, LogNorm, FuncNorm, SymLogNorm, 
    Normalize, LinearSegmentedColormap,
)
from matplotlib.lines import Line2D


# --------------------------------------------------------------------------------- #
#  HELPERS                                                                          #
# --------------------------------------------------------------------------------- #

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



# --------------------------------------------------------------------------------- #
#  RADAR PLOT                                                                       #
# --------------------------------------------------------------------------------- #

def create_polar_fig(p, cbar_included=2.7):
    fig_width = 3.4 * p + cbar_included
    fig, axes = plt.subplots(
        2, p, figsize=(fig_width, 7.0),
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
            ax.tick_params(axis='x', pad=-1)
            ax.set_yticks([])
            ax.grid(False)
            ax.xaxis.grid(True, color='gray', alpha=0.3)

            lc = LineCollection(segments, colors=colors, linewidths=linewidths)
            ax.add_collection(lc)
            ax.set_rlim(0, 1.0)

    fig.subplots_adjust(
        left=0.03, right=1 - cbar_included / fig_width,
        top=0.90, bottom=0.04,
        hspace=0.25, wspace=0.3,
    )

    return fig, axes

def make_cbar_axes(fig, axes, n=2, width_in=0.18, gap_in=1.0, pad_in=0.8, height_frac=0.5):
    fig.canvas.draw()
    fig_width = fig.get_size_inches()[0]
    boxes = [ax.get_position() for ax in axes.ravel()]
    x0 = max(b.x1 for b in boxes) + pad_in / fig_width
    y_min = min(b.y0 for b in boxes)
    y_max = max(b.y1 for b in boxes)

    h = (y_max - y_min) * height_frac
    y0 = y_min + ((y_max - y_min) - h) / 2

    return [
        fig.add_axes([x0 + k * (width_in + gap_in) / fig_width, y0, width_in / fig_width, h])
        for k in range(n)
    ]

def finalize_polar_fig(
        fig, axes, lc, 
        color_bounds, contrast_params, 
        run_colors, run_labels, run_dashed, run_markers,
        title="QAOA angle trajectories", cax=None):
    
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
        cax=cax,
        ax=axes if cax is None else None,
        boundaries=color_bounds,
        spacing="proportional",
        orientation="vertical",
    )
    TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cbar.set_ticks(TICKS)
    cbar.set_ticklabels([f"{x:.2f}" for x in TICKS])
    cbar.set_label("Approx. ratio", rotation=270, labelpad=15)

    fig.suptitle(title)

    handles = [
        Line2D([0], [0], color=run_colors[i], linestyle="--" if run_dashed[i] else "-",
               marker=run_markers[i], label=lbl)
        for i, lbl in enumerate(run_labels)
    ]

    # fig.legend(
    #     handles=handles,
    #     loc="center right",
    #     bbox_to_anchor=(0.99, 0.5),
    #     fontsize=8,
    #     title="Run (cond)",
    #     frameon=False,
    # )


# --------------------------------------------------------------------------------- #
#  RAY TRAJECTORIES                                                                 #
# --------------------------------------------------------------------------------- #

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
    dashed=False, dash=(2, 2),
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

    def draw_rays(ax, theta, r, c, n_sub=16):         # Rays btw consecutive points (shortest angular step)
        dtheta = (np.diff(theta) + np.pi) % (2 * np.pi) - np.pi
        dr = np.diff(r)
        seg_colors = 0.5 * (c[:-1] + c[1:])

        if dashed:
            on, off = dash
            keep = (np.arange(len(dtheta)) % (on + off)) < on
            theta, r = theta[:-1][keep], r[:-1][keep]
            dtheta, dr, seg_colors = dtheta[keep], dr[keep], seg_colors[keep]
        else:
            theta, r = theta[:-1], r[:-1]

        t = np.linspace(0, 1, n_sub + 1)
        th_pts = theta[:, None] + dtheta[:, None] * t[None, :]
        r_pts = r[:, None] + dr[:, None] * t[None, :]
        segments = np.stack([th_pts, r_pts], axis=-1)      # (n_seg, n_sub + 1, 2)

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



# --------------------------------------------------------------------------------- #
#  LANDSCAPE BACKGROUND                                                             #
# --------------------------------------------------------------------------------- #

def make_grad_norm(vmax, linthresh_frac=0.01):
    return SymLogNorm(linthresh=vmax * linthresh_frac, vmin=-vmax, vmax=vmax)

def add_grad_colorbar(fig, axes, vmax, linthresh_frac=0.01, cmap="RdBu_r", alpha=None, cax=None):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=make_grad_norm(vmax, linthresh_frac))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, ax=axes if cax is None else None,
                        orientation="vertical", alpha=alpha)
    cbar.set_label(r"$\partial E / \partial \theta$", rotation=270, labelpad=18)
    return cbar

def add_landscape_background(ax, land, quantity="grad", cmap=None, alpha=0.12,
                             vmax=None, linthresh_frac=0.01, zero_contour=True, norm=None):
    if quantity == "grad":
        C = land["G"]
        vmax = vmax or np.percentile(np.abs(C), 98)
        norm = make_grad_norm(vmax, linthresh_frac)
        cmap = cmap or "RdBu_r"                            # red: dE/d(theta) > 0, blue: < 0
    else:
        C = land["E"]
        norm = norm or Normalize(C.min(), C.max())
        cmap = cmap or "viridis"

    pcm = ax.pcolormesh(
        land["theta"], land["r"], C, cmap=cmap, norm=norm,
        shading="nearest", alpha=alpha, zorder=0,
        edgecolors="none", linewidth=0, antialiased=False,
    )
    
    if quantity == "grad" and zero_contour:
        ax.contour(land["theta"], land["r"], C, levels=[0.0],
                   colors="k", linewidths=0.8, linestyles=":", zorder=1)
    ax.set_rlim(0, 1.05)
    return pcm