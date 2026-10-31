import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from source.paths import DATA_DIR, RESULTS_DIR
from radar_plot import (
    make_hue_cmap, create_polar_fig, 
    finalize_polar_fig, add_traj,
    make_cbar_axes,
    add_landscape_background,
    add_grad_colorbar
)
from data_io import (
    create_subfold_name, 
    load_meta, 
    load_histories, 
    load_landscape,
) 

def full_plotting(
    graph_type, param_transfer_type, p, 
    strat_cond, qty, scan_run_row_layer):

    subfolder_name = create_subfold_name(graph_type, param_transfer_type)
    run_dir = DATA_DIR / "trajectories/raw" / subfolder_name


    #                                - 1 -                               #
    # ======================== Load trajectories ======================= #

    meta = load_meta(run_dir)
    stored_idx = [meta["strat_cond"].index(list(c)) for c in strat_cond]

    data = [load_histories(run_dir, i) for i in stored_idx]
    all_p_aratios = [d["aratios"][p - 1] for d in data]
    all_p_params = [d["params"][p - 1] for d in data]

    all_final_cost = [cost[-1] for cost in all_p_aratios]
    all_final_param = [par[-1] for par in all_p_params]

    run_labels = [
        f"{'WS' if relax else 'CS'}, init={init}" 
        for relax, init in strat_cond
    ]

    #                                - 2 -                               #
    # ============ Rank the conditions by best achieved cost =========== #

    ranking = np.argsort(all_final_cost)[::-1]
    ranked_cond = [strat_cond[i] for i in ranking]
    ranked_final_costs = [all_final_cost[i] for i in ranking]
    for cond, cost in zip(ranked_cond, ranked_final_costs):
        print(cond, cost)

    #                                - 3 -                               #
    # ======================== Style variables ========================= #

    LIGHTNESS_BOUNDS = np.array([
        0.0, 0.4, 0.5, 0.6,
        0.7, 0.8, 0.9, 0.95, 1.0,
    ])
    # LIGHTNESS_BOUNDS = np.array([
    #     0.0, 0.4,
    #     0.6, 0.7,
    #     0.8, 0.85, 
    #     0.9, 0.925, 
    #     0.95, 0.97, 
    #     0.98, 0.99, 
    #     1.0, 
    # ])
    n_lightness = len(LIGHTNESS_BOUNDS) - 1
    norm = BoundaryNorm(LIGHTNESS_BOUNDS, n_lightness)


    CONTRAST_PARAMS = { 
        "s_range": (0.3, 1.0), 
        "v_range": (0.8, 0.1), 
        "a_range": (0.7, 1.0), 
    } 

    BASE_COLORS = [
        (0.0, 0.0, 1.0, 1.0),  # blue
        (1.0, 0.0, 0.0, 1.0),  # red
        (0.0, 0.6, 0.0, 1.0),  # green
        (1.0, 0.5, 0.0, 1.0),  # orange
    ]

    LAND_ALPHA = 0.15

    relax_list = list(dict.fromkeys(relax for relax, _ in strat_cond))
    init_list = []
    for _, init in strat_cond:
        if init not in init_list:
            init_list.append(init)
    assert len(init_list) <= len(BASE_COLORS), "More init_param than colors in BASE_COLORS"
    
    run_colors = [
        BASE_COLORS[init_list.index(init)] 
        for _, init in strat_cond
    ]
    run_dashed  = [
        relax_list.index(relax) == 1 
        for relax, _ in strat_cond
    ]
    run_markers = ["s" if d else "o" for d in run_dashed]
    run_cmaps = [
        make_hue_cmap(run_colors[i], n_lightness, **CONTRAST_PARAMS) 
        for i in range(len(strat_cond))
    ]


    #                                - 4 -                               #
    # ========================= Create figure ========================== #

    fig, axes = create_polar_fig(p)
    cax_ar, cax_second = make_cbar_axes(fig, axes, n=2)
    if qty != "grad":
        cax_second.set_visible(False)

    for i in range(len(strat_cond)):
        last_lc = add_traj(
            axes,
            all_p_params[i],
            all_p_aratios[i],
            cmap=run_cmaps[i],
            norm=norm,
            reverse_radius=True,
            smooth_window=1,
            show_endpoints=True,
            dashed=run_dashed[i],
            endpoint_kwargs={"marker": run_markers[i]},
        )


    #                                - 4 -                               #
    # ========================= Sweep lanscape ========================= #

    lands = [
        load_landscape(run_dir, stored_idx[scan_run], p, scan_row, scan_layer)
        for scan_run, scan_row, scan_layer in scan_run_row_layer
    ]

    if qty == "grad":
        all_abs = np.concatenate([np.abs(l["G"]).ravel() for l in lands])
        grad_vmax = np.percentile(all_abs, 98)
        add_grad_colorbar(fig, axes, grad_vmax, alpha=LAND_ALPHA, cax=cax_second)


    for (scan_run, scan_row, scan_layer), land in zip(scan_run_row_layer, lands):
        ax = axes[scan_row, scan_layer]

        if qty == "grad":
            add_landscape_background(ax, land, quantity="grad", vmax=grad_vmax, alpha=LAND_ALPHA)

        else:
            add_landscape_background(
                ax, land, quantity="energy",
                cmap=run_cmaps[scan_run],   # same hue as that run's trajectory
                norm=norm,                  # same BoundaryNorm (LIGHTNESS_BOUNDS)
                alpha=LAND_ALPHA,
            )

    scan_run = scan_run_row_layer[0][0]
    title = (f"QAOA angle trajectories | {run_labels[scan_run]} | "
         f"Final AR: {all_final_cost[scan_run]:.2f}")

    finalize_polar_fig(
        fig, axes, last_lc, LIGHTNESS_BOUNDS, CONTRAST_PARAMS, 
        run_colors, run_labels, run_dashed, run_markers, 
        title=title, cax=cax_ar,
    )

    out_dir = RESULTS_DIR / "trajectories/plots" / subfolder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / f"radar_c{scan_run_row_layer[0][0]}_p{p}_{qty}.png", dpi=200, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":

    # ================================================================== #
    # * . * . * . * . * . * .  EDITABLE SECTION  . * . * . * . * . * . * #
    # ------------------------------------------------------------------ #

    param_transfer_type = "interp"

    graph_type = {
        "family": "Gilbert",
        "param": (14, 0.25),
        "s": 0, 
    }

    # ! Maximum 4 colors
    strat_cond = [
        ("continuous", [0.1, 0.0]),         # 0
        ("continuous", [0.1, 0.2]),         # 1 | Super winner
        ("continuous", [0.3, -0.4]),        # 2
        ("continuous", [0.5, -0.5]),        # 3
    ]

    for p in range(1, 4):
        for cond in range(4):

            scan_run_row_layer = [
                # (0, 1, 0),              # cond0, beta_0
                (cond, i, j) for i in (0, 1) for j in range(p)
            ]       

            for qty in ["energy", "grad"]:

                # ------------------------------------------------------------------ #
                # ° o ° o ° o °   That's it for the editable section   ° o ° o ° o ° #
                # ================================================================== #

                full_plotting(graph_type, param_transfer_type, p, strat_cond, qty, scan_run_row_layer)

                #TODO METRICS