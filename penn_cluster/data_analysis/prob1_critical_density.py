import numpy as np
import auxiliary.graphs as gph
from qaoa_run import run_qaoa

N_range = np.array([8, 12, 16])

density = 10
density_range = np.range(0.0, 1.0, density+1)
density_range = density_range[1:-1]

num_graph_samples = 20
#TODO init_condition = !!!! size of mixers

def critical_density_test(problem_config, strategy_config, apparatus_config, N_range, density_range):
    data = np.zeros(
        (
            len(N_range),
            len(density_range),
            # -> Graph type: randomGilbert(N, dens)
            num_graph_samples,
            # -> One specific graph
            len(init_params),
            # -> Different initializations
        )
    )
    for N in N_range:
        problem_config["N"] = N
        for dens in density_range:
            for i in range(num_graph_samples):
                graph = gph.randomGilbert(N, dens)
                problem_config["graph"] = graph
                for init_param in init_params:
                    strategy_config["init_param"] = init_param
                    qaoa_run = run_qaoa(
                        problem=problem_config,
                        strategy=strategy_config,
                        apparatus=apparatus_config,
                        silence=False
                    )
                    approximation_ratio = qaoa_run["approximation_ratio"]
                    cost_circuit_evals = qaoa_run["cost_circuit_evals"]
                    sampling_circuit_evals = qaoa_run["sampling_circuit_evals"]
                    data[N, dens, i, init_param] = (approximation_ratio, cost_circuit_evals, sampling_circuit_evals)