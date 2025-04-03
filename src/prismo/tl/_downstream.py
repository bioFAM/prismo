import logging

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

_logger = logging.getLogger(__name__)


def match(
    reference: torch.Tensor | np.ndarray, permutable: torch.Tensor | np.ndarray, dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find optimal permutation and signs to match two tensors along specified dimension.

    Finds the permutation and sign of permutable along one dimension to maximize
    correlation with reference. Useful for comparing ground truth factor scores/loadings
    with inferred values where factor order and sign is arbitrary.

    Args:
        reference: Reference tensor to match against.
        permutable: Tensor to be permuted and sign-adjusted.
        dim: Dimension along which to perform matching.

    Returns:
        tuple:
            - np.ndarray: Optimal permutation indices
            - np.ndarray: Optimal signs (+1 or -1) for each permuted element

    Notes:
        - Handles various input types (torch.Tensor, np.ndarray, pd.DataFrame)
        - Special handling for non-negative tensors
        - Uses linear sum assignment to find optimal matching
    """
    # convert all tensors to numpy arrays
    if isinstance(reference, torch.Tensor):
        reference = reference.numpy()
    if isinstance(permutable, torch.Tensor):
        permutable = permutable.numpy()
    if isinstance(reference, pd.DataFrame):
        reference = reference.to_numpy()
    if isinstance(permutable, pd.DataFrame):
        permutable = permutable.to_numpy()

    nonnegative = False
    if np.all(reference >= 0) and np.all(permutable >= 0):
        nonnegative = True

    # move the assignment dimension to the end
    reference = np.moveaxis(reference, dim, -1)
    permutable = np.moveaxis(permutable, dim, -1)

    # compute the correlation matrix between reference and permutable
    correlation = np.zeros([reference.shape[-1], permutable.shape[-1]])
    for i in range(reference.shape[-1]):
        for j in range(permutable.shape[-1]):
            correlation[i, j] = pearsonr(reference[..., i].flatten(), permutable[..., j].flatten())[0]
    correlation = np.nan_to_num(correlation, 0)

    # find the permutation that maximizes the correlation
    reference_ind, permutable_ind = linear_sum_assignment(-1 * np.abs(correlation))
    signs = np.ones_like(permutable_ind)

    # if correlation is negative, flip the sign of the corresponding column
    for k in range(signs.shape[0]):
        if correlation[reference_ind, permutable_ind][k] < 0 and not nonnegative:
            signs[k] *= -1

    return reference_ind, permutable_ind, signs
