"""
Initial speed experiment
"""

import numpy as np

import source.utils.graphs as gph
import source.qaoa_run as qr
from source.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR


#### Fixed variables ####

problem_config = {
    "N": None,
    "graph": None, # gph.randomDRegular(N, 3) | gph.randomGilbert(N, 0.25)
}

strategy_config = {
    "constrained": False,
    "relaxation_type": 'continuous',    #  None | 'continuous'
    "param_transfer_type": 'interp',    # 'given' | 'random' | 'interp' | 'fourier'
    "fourier_qR": (None, 5),
    "init_param": [0.55, 0.27],
    "mixers": ["x"],
}

apparatus_config = {
    "p": None,
    "device": "lightning.qubit",        # "lightning.qubit" | "lightning.amdgpu" 
    "estimator_shots": 10000,
    "sampler_shots": 10000,
    "optimizer": "L-BFGS-B",            # "L-BFGS-B" | "Adam"
    "opt_steps": 1000,
}



#### (N, p, samples) considered ####

N_values = np.array(list(range(5, 10)))
N_size = len(N_values)
p_values = np.array(list(range(1, 5)))
p_size = len(p_values)
num_samples = 10



#### Random graphs prep ####

random_graphs = np.empty((N_size, num_samples), dtype=object)
for N_idx, N in enumerate(N_values):
    for i in range(num_samples):
        random_graphs[N_idx, i] = gph.randomGilbert(N, 0.25)


#### Run ####

times_mean = np.zeros((N_size, p_size))
times_stderr = np.zeros((N_size, p_size))

evals_mean = np.zeros((N_size, p_size))
evals_stderr = np.zeros((N_size, p_size))

aratio_mean = np.zeros((N_size, p_size))
aratio_stderr = np.zeros((N_size, p_size))


for N_idx, N in enumerate(N_values):
    for p_idx, p in enumerate(p_values):
        times_samples = np.zeros(num_samples)
        aratio_samples = np.zeros(num_samples)
        evals_samples = np.zeros(num_samples)

        for i in range(num_samples):
            problem_config["N"] = N
            apparatus_config["p"] = p
            problem_config["graph"] = random_graphs[N_idx, i]
            one_qaoa_run = qr.run_qaoa(
                    problem=problem_config,
                    strategy=strategy_config,
                    apparatus=apparatus_config,
                    silence=True
                )

            times = sum(one_qaoa_run["times"])
            evals = one_qaoa_run["cost_circuit_evals"]
            aratio = one_qaoa_run["approximation_ratio"]

            evals_samples[i] = evals
            times_samples[i] = times 
            aratio_samples[i] = aratio

        print((float(N), float(p)), "done")

        times_mean[N_idx, p_idx] = (np.mean(times_samples))
        times_stderr[N_idx, p_idx] = (np.std(times_samples, ddof=1) / np.sqrt(num_samples))
        evals_mean[N_idx, p_idx] = (np.mean(evals_samples))
        evals_stderr[N_idx, p_idx] = (np.std(evals_samples, ddof=1) / np.sqrt(num_samples))
        aratio_mean[N_idx, p_idx] = (np.mean(aratio_samples))
        aratio_stderr[N_idx, p_idx] = (np.std(aratio_samples, ddof=1) / np.sqrt(num_samples))


outfile = DATA_DIR / "analysis_data/exp_0/speed_data2.npz"

np.savez(
    outfile, 
    N_values=N_values,
    p_values=p_values,
    graphs=random_graphs,
    times_mean=times_mean, 
    times_stderr=times_stderr, 
    evals_mean=evals_mean, 
    evals_stderr=evals_stderr, 
    aratio_mean=aratio_mean,
    aratio_stderr=aratio_stderr,
)

