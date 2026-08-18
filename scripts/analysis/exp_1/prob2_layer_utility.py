import numpy as np
import source.utils.graph_gen as gph
from source.qaoa_run import run_qaoa


def critical_density_test(
        problem_config, strategy_config, apparatus_config, 
        N_range, density_range, p_range, num_graph_samples, init_params,
    ):

    data = np.zeros(
        (
            len(N_range),
            len(density_range),
            # -> Graph type: randomGilbert(N, dens)
            num_graph_samples,
            # -> One specific graph
            len(p_range),
            len(init_params),
            # -> Different initializations
        )
    )

    for N in N_range:
        problem_config["N"] = N

        for dens in density_range:

            if dens == 0.0:
                continue

            elif dens == 1.0:
                data[N, dens, :, :] = None

                graph = gph.complete_graph(N)
                problem_config["graph"] = graph

                for p in p_range:
                    problem_config["p"] = p

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

                        data[N, dens, 0, init_param] = (approximation_ratio, cost_circuit_evals, sampling_circuit_evals)

            else:
                for i in range(num_graph_samples):
                    graph = gph.randomGilbert(N, dens)
                    problem_config["graph"] = graph

                    for p in p_range:
                        problem_config["p"] = p

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

    return data

###############################
##### Constants variables #####
###############################

N = None
p = None

problem_config = {
    "N": N,
    "graph": None, # gph.randomDRegular(N, 3) | gph.randomGilbert(N, 0.25)
}

strategy_config = {
    "constrained": False,
    "relaxation_type": 'continuous',    #  None | 'continuous'
    "param_transfer_type": 'interp',    # 'given' | 'random' | 'interp' | 'fourier'
    "fourier_qR": (None, 5),
    "init_param": None,
    "mixers": ["x"],
}

apparatus_config = {
    "p": None,
    "device": "lightning.amdgpu",        # "lightning.qubit" | "lightning.amdgpu" 
    "estimator_shots": 10000,
    "sampler_shots": 10000,
    "optimizer": "L-BFGS-B",            # "L-BFGS-B" | "Adam"
    "opt_steps": 400,
}

###########################
##### Study variables #####
###########################

N_range = np.array([8, 12, 16])
p_range = np.array([2, 5, 10, 25, 50])

density = 10
density_range = np.range(0.0, 1.0, density+1)

num_graph_samples = 20
#TODO init_params = !!!! size of mixers

data_mat = critical_density_test(
        problem_config, strategy_config, apparatus_config, 
        N_range, density_range, p_range, num_graph_samples, init_params,
    )

filename = f"critical_density_test"
np.savez(filename, data_mat=data_mat)