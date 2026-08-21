"""
conditions.py
=============
Save/load reusable fixed experiment conditions (strategy_config,
apparatus_config) as YAML files under data/conditions/.
"""

from pathlib import Path
import json

from source.paths import COND_DIR


def save_condition(path, strategy_config, apparatus_config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    data["strategy_config"] = strategy_config
    data["apparatus_config"] = apparatus_config
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_condition(path):
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    problem_config = {"N": None, "graph": None}    
    strategy_config = data["strategy_config"]
    apparatus_config = data["apparatus_config"]

    return problem_config, strategy_config, apparatus_config

if __name__ == "__main__":
    
    problem_config = {
    "N": None,
    "graph": None,
    }

    strategy_config = {
        "constrained": False,
        "relaxation_type": 'continuous',
        "param_transfer_type": 'interp',
        "fourier_qR": [5, 10],
        "init_param": [0.67, 0.33],
        "mixers": ["x"],
    }

    apparatus_config = {
        "p": None,
        "device": "lightning.amdgpu",
        "estimator_shots": 10000,
        "sampler_shots": 10000,
        "optimizer": "L-BFGS-B",
        "opt_steps": 1000,
    }

    # cond1
    strategy_config["relaxation_type"] = None
    strategy_config["param_transfer_type"] = 'interp'
    strategy_config["mixers"] = ["x"]
    strategy_config["init_param"] = [0.67, 0.33]
    path = COND_DIR / "cond1.json"
    save_condition(path, strategy_config, apparatus_config)

    # cond2
    strategy_config["relaxation_type"] = 'continuous'
    strategy_config["param_transfer_type"] = 'interp'
    strategy_config["mixers"] = ["x"]
    strategy_config["init_param"] = [0.67, 0.33]
    path = COND_DIR / "cond2.json"
    save_condition(path, strategy_config, apparatus_config)

    # cond3
    strategy_config["relaxation_type"] = 'continuous'
    strategy_config["param_transfer_type"] = 'fourier'
    strategy_config["fourier_qR"] = [5, 10]
    strategy_config["mixers"] = ["x"]
    strategy_config["init_param"] = [0.67, 0.33]
    path = COND_DIR / "cond3.json"
    save_condition(path, strategy_config, apparatus_config)

    # cond4
    strategy_config["relaxation_type"] = 'continuous'
    strategy_config["param_transfer_type"] = 'interp'
    strategy_config["mixers"] = ["x", "y"]
    strategy_config["init_param"] = [0.67, 0.33, 0.33]
    path = COND_DIR / "cond4.json"
    save_condition(path, strategy_config, apparatus_config)