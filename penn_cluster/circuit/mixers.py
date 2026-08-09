"""
circuit/mixers.py
=========
Registry to access the desired mixer version.
"""

from dataclasses import dataclass
from typing import Callable

from .ansatz import relaxed_mixer_layer, cst_relaxed_mixer_layer, y_mixer_layer, cst_y_mixer_layer


@dataclass
class Mixer:
    name: str
    build: Callable

def x_mixer_builder(graph, angles, constrained):
    if not constrained:
        return lambda beta: relaxed_mixer_layer(beta, graph, angles)
    else:
        return lambda beta: cst_relaxed_mixer_layer(beta, graph, angles)

def y_mixer_builder(graph, angles, constrained):
    if not constrained:
        return lambda alpha: y_mixer_layer(alpha, graph)
    else:
        return lambda alpha: cst_y_mixer_layer(alpha, graph)


MIXER_REGISTRY = {
    "x": Mixer("x", x_mixer_builder),
    "y": Mixer("y", y_mixer_builder),
}