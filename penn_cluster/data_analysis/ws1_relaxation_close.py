import numpy as np
import qaoa_run as qr
import auxiliary.classical as clas
import circuit.warm_start as ws

def random_gen_from_relax(d, N):
    bitstring = np.zeros(N)
    for i in range(N):
        if np.random.rand() < d[i]:
            bitstring[i] = -1
        else:
            bitstring[i] = 1
    return bitstring

def energy_from_bitstring(bitstring, graph, penalizer):
    N = graph.size
    relax_energy = sum((1 - graph.degree(i) * penalizer / 2) * bitstring[i] for i in range(N))
    for edge in graph.edges:
        relax_energy += penalizer / 2 * bitstring[edge[0]] * bitstring[edge[1]]
    return relax_energy

def qaoa_energy(problem_config, strategy_config, apparatus_config):
    qaoa_run = qr.run_qaoa(
        problem=problem_config,
        strategy=strategy_config,
        apparatus=apparatus_config,
        silence=True
    )
    final_energy = qaoa_run["best_energy"]
    return final_energy

def extract_ratios(graph, relax_energy, final_energy, penalizer):
    theo_best_cost, _ = clas.best_config_branch_bound(graph)
    relax_ratio = qr.approx_ratio(graph, relax_energy, penalizer, theo_best_cost)
    final_ratio = qr.approx_ratio(graph, final_energy, penalizer, theo_best_cost)
    return relax_ratio, final_ratio

def compare_pipeline(problem_config, strategy_config, apparatus_config):
    N = problem_config["N"]
    graph = problem_config["graph"]
    penalizer = 1.5

    d, _ = ws.relaxation(graph, range(N))
    bitstring = random_gen_from_relax(d, N)
    relax_energy = energy_from_bitstring(bitstring, graph, penalizer)

    final_energy = qaoa_energy(problem_config, strategy_config, apparatus_config)
    relax_ratio, final_ratio = extract_ratios(graph, relax_energy, final_energy, penalizer)

    return relax_ratio, final_ratio

def main_test():
    for N in N_values:
        pass

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