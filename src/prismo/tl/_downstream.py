import logging

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from mudata import MuData
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

from .._core import PRISMO, PrismoDataset, pcgse_test

_logger = logging.getLogger(__name__)


def test_annotation_significance(
    model: PRISMO,
    annotations: dict[str, pd.DataFrame],
    data: MuData | dict[str, dict[str, AnnData]] | PrismoDataset | None = None,
    corr_adjust: bool = True,
    p_adj_method: str = "fdr_bh",
    min_size: int = 10,
    subsample: int = 1000,
) -> dict[str, pd.DataFrame]:
    """Test feature sets for significant associations with model factors.

    This is an implementation of PCGSE :cite:p:`pmid26300978`.

    Args:
        model: The PRISMo model.
        annotations: Boolean dataframe with feature sets in each row for each view.
        data: The data that the model was trained on. Only required if `corr_adjust=True`.
        corr_adjust: Whether to adjust for correlations between features.
        p_adj_method: Method for multiple testing adjustment.
        min_size: Minimum size threshold for feature sets.
        subsample: Work with a random subsample of the data to speed up testing. Set to 0 to use
            all data (may use excessive amounts of memory). Only relevant if `corr_adjust=True`.

    Returns:
        PCGSE results for each view.
    """
    if corr_adjust and data is None:
        raise ValueError("`data` cannot be `None` if `corr_adjust=True`.")

    if data is not None and not isinstance(data, PrismoDataset):
        data = model._prismodataset(data)
    annotations = {
        view_name: annot.loc[:, features].astype(bool)
        for view_name, annot in annotations.items()
        if view_name in model.view_names
        and (features := annot.columns.intersection(model.feature_names[view_name])).size > 0
    }

    if len(annotations) > 0:
        return pcgse_test(
            data,
            model._model_opts.nonnegative_weights,
            annotations,
            model.get_weights("pandas"),
            corr_adjust=corr_adjust,
            p_adj_method=p_adj_method,
            min_size=min_size,
            subsample=subsample,
        )
    else:
        return {}


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
