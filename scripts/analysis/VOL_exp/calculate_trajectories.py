from source.qaoa_run import run_qaoa
from source.utils.classical import approx_ratio
from source.utils import graph_gen as gph
from source.utils.classical import best_config_branch_bound
from source.utils.cond_gen import load_condition
from source.paths import DATA_DIR, COND_DIR, GRAPHS_DIR


# --------------------------------------------------------------------------------- #
#  LOAD TRAJECTORIES                                                                #
# --------------------------------------------------------------------------------- #


# graph_type = {
#    "family": "Gilbert",
#    "param": (14, 0.4),
#    "s": 0, 
# }

def load_graph(graph_type):
    family = graph_type["family"]
    param = graph_type["param"]
    N = param[0]
    graphs_loaded = {}
    graphs_loaded[family] = gph.load_family(
        GRAPHS_DIR / f'{family}.npz', 
        family, 
        params=[param], 
    )
    graph = gph.get_graph_from_edges(
        gph.get_sample(graphs_loaded[family], param, s=graph_type["s"]), 
        N=N,
    )
    return graph, N

def make_base_config(param_transfer_type):
    base_config = load_condition(COND_DIR / f"cond2.json")
    base_config[1]["param_transfer_type"] = param_transfer_type
    base_config[2]["device"] = "lightning.qubit"
    return base_config


def compute_trajs(graph, N, base_config, p, cond_diff):
    all_energy_ps, all_aratio_ps = [], []
    all_param_ps, cost_functions = [], []

    exp_config = base_config
    exp_config[0]["N"] = N
    exp_config[0]["graph"] = graph
    exp_config[2]["p"] = p

    theo_best_cost, _ = best_config_branch_bound(graph)

    for cond in cond_diff: 
        exp_config[1]["relaxation_type"] = cond[0]
        exp_config[1]["init_param"] = cond[1]

        one_qaoa_run = run_qaoa(
            problem=exp_config[0],
            strategy=exp_config[1],
            apparatus=exp_config[2],
            silence=False
        )

        approx_ratio_history_ps = [
            approx_ratio(graph, e, 1.5, theo_best_cost)
            for e in one_qaoa_run["best_energy_ps"]
        ]

        all_energy_ps.append(one_qaoa_run["best_energy_ps"])
        all_aratio_ps.append(approx_ratio_history_ps)
        all_param_ps.append(one_qaoa_run["param_hist_ps"])
        cost_functions.append(one_qaoa_run["cost_function_p"])

    return all_energy_ps, all_aratio_ps, all_param_ps, cost_functions

# Return in the form of nested list with:
## d1: conditions
## d2: layer q <= p
## d3: iterations

