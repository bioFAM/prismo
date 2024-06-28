import torch
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment


def match(
    fixed,
    variable,
    dim: int = -1,
) -> torch.tensor:
    # convert all tensors to numpy arrays
    if isinstance(fixed, torch.Tensor):
        fixed = fixed.numpy()
    if isinstance(variable, torch.Tensor):
        variable = variable.numpy()

    # move the assignment dimension to the end
    fixed = np.moveaxis(fixed, dim, -1)
    variable = np.moveaxis(variable, dim, -1)

    # compute the correlation matrix between fixed and variable
    correlation = np.zeros([fixed.shape[-1], variable.shape[-1]])
    for i in range(fixed.shape[-1]):
        for j in range(variable.shape[-1]):
            correlation[i, j] = pearsonr(
                fixed[..., i].flatten(), variable[..., j].flatten()
            )[0]
    correlation = np.nan_to_num(correlation, 0)

    # find the permutation that maximizes the correlation
    row_ind, permutation = linear_sum_assignment(-1 * np.abs(correlation))
    signs = np.ones_like(permutation)

    # if correlation is negative, flip the sign of the corresponding column
    for k in range(signs.shape[0]):
        if correlation[row_ind, permutation][k] < 0:
            signs[k] *= -1

    return permutation, signs
