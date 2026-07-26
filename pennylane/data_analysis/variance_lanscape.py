"""
variance_landscape.py
=====================
File to plot the energy variance lanscape around an optimal parameter value theta_0.
"""

import pennylane as qp
from pennylane import numpy as np
import matplotlib.pyplot as plt

def measure_energies(cost_function, parameter_samples):
    energies = np.array([cost_function(p) for p in parameter_samples])
    print("Energies measured")
    return energies

def variance(energies):
    return np.var(energies)


def lhs_shell_samples(n, flat_theta0, r_inner, r_outer, num_samples):
    bins = (np.arange(num_samples) + np.random.uniform(0, 1, num_samples)) / num_samples
    radii = (r_inner**n + bins * (r_outer**n - r_inner**n)) ** (1.0 / n)
    np.random.shuffle(radii)
 
    directions = np.random.normal(size=(num_samples, n))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
 
    samples = flat_theta0 + directions * radii[:, None]
    return samples.reshape(num_samples, 2, n // 2)

    
def full_energy_landscape_shell(cost_function, theta0, num_samples):
    flat_theta0 = np.concatenate(theta0)
    n = flat_theta0.size
    
    radii = np.linspace(0.1, np.pi, 51)
    
    total_samples = []
    idx = 0
    indices = [0]
    for i in range(len(radii) - 1):
        r_samples = lhs_shell_samples(n, flat_theta0, radii[i], radii[i+1], num_samples=num_samples)
        total_samples.extend(r_samples)
        idx = len(total_samples)
        indices.append(idx)
    
    energies = measure_energies(cost_function, total_samples)
    variances = []
    for i in range(len(radii)-1):
        select_energies = energies[indices[i]:indices[i+1]]
        var_r = variance(select_energies)
        variances.append(var_r)

    return radii[1:] / np.pi, variances


def plot_variance(radii, variances):
    fig, ax = plt.subplots()
    
    ax.plot(radii, variances, '-o', 
             markersize=2,
             markerfacecolor=(1, 1, 1, 0.4),
             markeredgecolor='blue',
             alpha=0.7)
    ax.set_xlabel(r"$r / \pi$")
    ax.set_ylabel(r"Var($E$)")
    ax.set_title("Energy variance lanscape around optimized angle")
    
    ax.set_xlim(left=-0.01, right=None)
    ax.set_ylim(bottom=-0.1, top=None)
    
    ax.minorticks_on()
    ax.grid(visible=True, which='major', linestyle='-', linewidth=0.7, color='darkgray')
    ax.grid(visible=True, which='minor', linestyle=':', linewidth=0.5, color='gray')
    
    ax.set_axisbelow(True)
    
    plt.show()