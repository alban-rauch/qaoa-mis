"""
optimization/parameter_transfer.py
===============
Registry to access the desired parameter transfer framework.
"""

from dataclasses import dataclass
from typing import Callable

from . import pt_fourier, pt_interp, pt_random

@dataclass
class ParamTransfer:
    name: str
    build: Callable


PARAM_TRANSFER_REGISTRY = {
    "random": ParamTransfer("random", pt_random.random_pt),
    "interp": ParamTransfer("interp", pt_interp.interp_pt),
    "fourier": ParamTransfer("fourier", pt_fourier.fourier_pt),
}