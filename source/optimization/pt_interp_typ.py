"""
pt_interp_typ.py
===============
Different interpolation pattern that serve the "INTERP" framework.
"""

# ======================================================================================== #
#                                         'interp'                                         #
# ======================================================================================== #


from pennylane import numpy as np
from scipy.interpolate import make_interp_spline
from .pt_interp import interp_pt

### LINEAR

def linear_interpolation(params_list, p):
    # Convert from p-dimensional gamma and beta to (p+1)-dimensional
    params_arr = np.array(params_list)
    new_params = np.zeros((params_arr.shape[0], p+1))
    new_params[:, 0] = params_arr[:, 0]
    new_params[:, -1] = params_arr[:, -1]
    for i in range(1, p):
        new_params[:, i] = (i / p) * params_arr[:, i-1] + ((p-i) / p) * params_arr[:, i]
    return new_params.tolist()

def general_linear_interpolation(params_list, p, q):
    # Convert from p-dimensional gamma and beta to (p+1)-dimensional
    params_arr = np.array(params_list)
    if q == 1:
        return np.mean(params_arr, axis=1, keepdims=True).tolist()
    positions = np.linspace(0, p-1, q)
    floors = np.floor(positions).astype(int)
    ceilings = np.minimum(floors + 1, p-1)
    distances = positions - floors
    new_params = (1 - distances) * params_arr[:, floors] + distances * params_arr[:, ceilings]
    return new_params.tolist()

def interp_linear_pt(cost_function_p, strategy, apparatus, silence=True):
    return interp_pt(cost_function_p, strategy, apparatus, linear_interpolation, silence)


### CUBIC

def cubic_interpolation(params_list, p):
    # Convert from p-dimensional gamma and beta to (p+1)-dimensional
    params_arr = np.array(params_list)
    x_old = np.linspace(0, 1, p)
    x_new = np.linspace(0, 1, p+1)

    k = min(3, p - 1) if p > 1 else 0    # Cubic splines need >= 4 pts

    new_params = np.zeros((params_arr.shape[0], p + 1))
    for row_idx, row in enumerate(params_arr):
        if p == 1:
            new_params[row_idx] = row[0]
        else:
            if k == 3:
                spline = make_interp_spline(x_old, row, k=k, bc_type="clamped")
            else:
                spline = make_interp_spline(x_old, row, k=k)
            new_params[row_idx] = spline(x_new)
    return new_params.tolist()

def interp_cubic_pt(cost_function_p, strategy, apparatus, silence=True):
    return interp_pt(cost_function_p, strategy, apparatus, cubic_interpolation, silence)


### EXTEND ZERO

def extend_zero_interpolation(params_list, p):
    params_arr = np.array(params_list)
    new_params = np.zeros((params_arr.shape[0], p + 1))
    new_params[:, :-1] = params_arr
    new_params[:, -1] = [0.1, 0.2]  # Only work for X
    return new_params.tolist()

def interp_extzero_pt(cost_function_p, strategy, apparatus, silence=True):
    return interp_pt(cost_function_p, strategy, apparatus, extend_zero_interpolation, silence)