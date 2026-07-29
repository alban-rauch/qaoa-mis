from dataclasses import dataclass
from typing import Callable

from . import ansatz as qa


@dataclass
class Mixer:
    name: str
    build: Callable

def x_mixer_builder(graph, angles, constrained):
    if not constrained:
        return lambda beta: qa.relaxed_mixer_layer(beta, graph, angles)
    else:
        return lambda beta: qa.cst_relaxed_mixer_layer(beta, graph, angles)

def y_mixer_builder(graph, angles, constrained):
    if not constrained:
        return lambda alpha: qa.y_mixer_layer(alpha, graph)
    else:
        return lambda alpha: qa.cst_y_mixer_layer(alpha, graph)


MIXER_REGISTRY = {
    "x": Mixer("x", x_mixer_builder),
    "y": Mixer("y", y_mixer_builder),
}