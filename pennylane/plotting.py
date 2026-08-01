
from matplotlib import pyplot as plt
import classical as clas
import circuit.ansatz as qa

def plot_energies(energies):
    plt.plot(energies)
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy vs. Optimization Step")
    plt.grid(alpha=0.3)
    plt.show()

def energy_evolution(graph, best_energy_ps, penalizer):
    plt.style.use("default")
    theo_best_cost, _ = clas.best_config_branch_bound(graph)
    costs = [[qa.energy_to_cost(energy_val, penalizer, graph.nodes, graph.edges) / theo_best_cost 
                for energy_val in sublist] 
                for sublist in best_energy_ps]
    length = len(costs)
    cmap = plt.colormaps['cividis']
    colors = [cmap(i / (length - 1)) for i in range(length)] if length > 1 else [cmap(0)]
    current_x = 0
    plt.figure(figsize=(10, 5))
    current_x = 0
    for i in range(length):
        segment = costs[i]
        x_coords = [current_x + j for j in range(len(segment))]
        plt.plot(x_coords, segment, color=colors[i], linewidth=2)
        current_x += len(segment) - 1

    plt.title("Energy optimization by step for interp")
    plt.xlabel("Step")
    plt.ylabel("Approximation Ratio")

    plt.gca().set_facecolor("white")
    plt.minorticks_on()

    plt.grid(True, which="major", linestyle="-", linewidth=0.6, color="#898989", alpha=0.8)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#b9b9b9", alpha=0.7)

    plt.tight_layout()
    plt.show()    