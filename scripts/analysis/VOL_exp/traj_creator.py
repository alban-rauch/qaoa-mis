import numpy as np
from matplotlib.colors import BoundaryNorm

from source.qaoa_run import build_cost_function_p
from source.paths import DATA_DIR
from calculate_trajectories import load_graph, make_base_config, compute_trajs

from gradient_scan import (
    make_energy_fn, 
    landscape_along_trajectory, 
)
from data_io import (
    create_subfold_name, make_run_dir, 
    save_meta, 
    save_histories, load_histories, 
    save_landscape,
) 

###                    !!! IMPORTANT !!!                    ###
# Change maxiter / gtol in source for more or less iterations #


def prepare_directory(graph_type, param_transfer_type, p, strat_cond):

    #                                - 1 -                               #
    # =================== Prepare graph and directory ================== #

    subfolder_name = create_subfold_name(graph_type, param_transfer_type)
    run_dir = make_run_dir(DATA_DIR / "trajectories/raw", subfolder_name)


    #                                - 2 -                               #
    # ================== Look for missing trajectories ================= #

    meta = {
        "p": p,
        "graph_type": graph_type,
        "param_transfer_type": param_transfer_type,
        "strat_cond": strat_cond,
    }
    new_pos, stored_idx = save_meta(run_dir, meta)

    new_conds = [strat_cond[i] for i in new_pos]
    new_idx = [stored_idx[i] for i in new_pos]

    return run_dir, graph_type, new_conds, new_idx


def calculate_history(run_dir, graph_type, param_transfer_type, p, new_conds, new_idx):

    graph, N = load_graph(graph_type)
    base_config = make_base_config(param_transfer_type)
    
    all_energy_ps, all_aratio_ps, all_param_ps, cost_functions = compute_trajs(
        graph, N, base_config, p, new_conds,
    )

    save_histories(run_dir, new_idx, all_aratio_ps, all_param_ps, all_energy_ps)

    return all_energy_ps, all_aratio_ps, all_param_ps, cost_functions

    # MISSING SAVE OF COST FUNCTIONS? BTW IS IT DIFFERENT DEPENDING ON INIT COND OR ONLY ON GRAPH?


def calculate_cost_fns(graph_type):
    graph, N = load_graph(graph_type)

    base_config_cold = make_base_config(None)    
    cost_function_cold = build_cost_function_p(graph, N, base_config_cold[1], base_config_cold[2])

    base_config_warm = make_base_config("continuous")
    cost_function_warm = build_cost_function_p(graph, N, base_config_warm[1], base_config_warm[2])

    return cost_function_cold, cost_function_warm


def calculate_landscapes(run_dir, p, new_conds, new_idx, cost_functions):

    cost_function_cold, cost_function_warm = cost_functions

    for i in range(len(new_conds)):
        for q in range(1, p+1):
            cost_function = cost_function_cold if new_conds[i][0] is None else cost_function_warm
            energy_fn = make_energy_fn(cost_function(q))
            hist_q = load_histories(run_dir, new_idx[i])["params"][q-1]
            for row in (0, 1):
                for layer in range(q):
                    land = landscape_along_trajectory(energy_fn, hist_q, row, layer)
                    save_landscape(run_dir, new_idx[i], q, row, layer, land)






if __name__ == "__main__":

    # ================================================================== #
    # * . * . * . * . * . * .  EDITABLE SECTION  . * . * . * . * . * . * #
    # ------------------------------------------------------------------ #

    param_transfer_type = "interp"
    p = 3

    graph_type = {
        "family": "Gilbert",
        "param": (14, 0.25),
        "s": 1, 
    }

    # ! Maximum 4 colors
    strat_cond = [
        ("continuous", [0.1, 0.0]),
        ("continuous", [0.1, 0.2]),       # Super winner
        # and (0.1, 0.1),
        # [0.1, 0.3],
        ("continuous", [0.3, -0.4]),
        ("continuous", [0.5, -0.5]), 
        # and (0.1, 0.0) / (0.1, 1.0) and (0.0, 0.2) / (0.0, 0.8),
        # [0.3, 0.7],
        # [0.67, 0.33],
        # [0.8, 0.4],
    ]

    scan_run, scan_row, scan_layer = 0, 1, 0       # cond0, beta_0

    # ------------------------------------------------------------------ #
    # ° o ° o ° o °   That's it for the editable section   ° o ° o ° o ° #
    # ================================================================== #

    run_dir, graph_type, new_conds, new_idx = prepare_directory(
        graph_type, param_transfer_type, p, strat_cond)

    all_energy_ps, all_aratio_ps, all_param_ps, cost_functions = calculate_history(
        run_dir, graph_type, param_transfer_type, p, new_conds, new_idx)

    calculate_landscapes(run_dir, p, new_conds, new_idx, calculate_cost_fns(graph_type))
