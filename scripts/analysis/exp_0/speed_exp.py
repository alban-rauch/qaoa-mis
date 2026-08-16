"""
Initial speed experiment
"""

import numpy as np
import time
from pathlib import Path

import source.utils.graphs as gph
import source.qaoa_run as qr
from source.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR


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


N_values = np.array(list(range(5, 10)))
p_values = np.array(list(range(1, 5)))
num_samples = 10

speed_mean = np.zeros((len(N_values), len(p_values)))
speed_stderr = np.zeros((len(N_values), len(p_values)))

evals_mean = np.zeros((len(N_values), len(p_values)))
evals_stderr = np.zeros((len(N_values), len(p_values)))

aratio_mean = np.zeros((len(N_values), len(p_values)))
aratio_stderr = np.zeros((len(N_values), len(p_values)))

random_graphs = 


for N_idx, N in enumerate(N_values):
    for p_idx, p in enumerate(p_values):
        speed_samples = np.zeros(num_samples)
        aratio_samples = np.zeros(num_samples)
        evals_samples = np.zeros(num_samples)

        for i in range(num_samples):
            problem_config["N"] = N
            apparatus_config["p"] = p
            problem_config["graph"] = gph.randomGilbert(N, 0.25)
            start_time = time.perf_counter()
            one_qaoa_run = qr.run_qaoa(
                    problem=problem_config,
                    strategy=strategy_config,
                    apparatus=apparatus_config,
                    silence=True
                )
            end_time = time.perf_counter()

            speed = end_time - start_time
            evals = one_qaoa_run["cost_circuit_evals"]
            aratio = one_qaoa_run["approximation_ratio"]

            evals_samples[i] = evals
            speed_samples[i] = speed
            aratio_samples[i] = aratio

        print((N, p), "done")

        speed_mean[N_idx, p_idx] = (np.mean(speed_samples))
        speed_stderr[N_idx, p_idx] = (np.std(speed_samples, ddof=1) / np.sqrt(num_samples))
        evals_mean[N_idx, p_idx] = (np.mean(evals_samples))
        evals_stderr[N_idx, p_idx] = (np.std(evals_samples, ddof=1) / np.sqrt(num_samples))
        aratio_mean[N_idx, p_idx] = (np.mean(aratio_samples))
        aratio_stderr[N_idx, p_idx] = (np.std(aratio_samples, ddof=1) / np.sqrt(num_samples))


outfile = DATA_DIR / "analysis_data/exp_0/speed_data2.npz"

np.savez(
    outfile, 
    N_values=N_values,
    p_values=p_values,
    speed_mean=speed_mean, 
    speed_stderr=speed_stderr, 
    evals_mean=evals_mean, 
    evals_stderr=evals_stderr, 
    aratio_mean=aratio_mean,
    aratio_stderr=aratio_stderr,
)

