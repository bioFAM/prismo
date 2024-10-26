import logging
from typing import Union

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from statsmodels.stats import multitest
from tqdm import tqdm

logger = logging.getLogger(__name__)

Index = Union[int, str, list[int], list[str], np.ndarray, pd.Index]


def test(
    model,
    view_name: str,
    feature_sets: pd.DataFrame = None,
    sign: str = "all",
    corr_adjust: bool = True,
    p_adj_method: str = "fdr_bh",
    min_size: int = 10,
):
    """Perform significance test of factor loadings against feature sets.

    Parameters
    ----------
    model : PRISMO
        A PRISMO model
    view_name : str
        View name
    feature_sets : pd.DataFrame, optional
        Boolean dataframe with feature sets in each row, by default None
    sign : str, optional
        Two sided ("all") or one-sided ("neg" or "pos"), by default "all"
    corr_adjust : bool, optional
        Whether to adjust for multiple testing, by default True
    p_adj_method : str, optional
        Adjustment method for multiple testing, by default "fdr_bh"
    min_size : int, optional
        Lower size limit for feature sets to be considered, by default 10

    Returns
    -------
    dict
        Dictionary of test results with "t", "p" and "p_adj" keys
        and pd.DataFrame values with factor_names as indices,
        and feature_sets as columns
    """
    use_prior_mask = feature_sets is None
    adjust_p = p_adj_method is not None

    if not isinstance(view_name, (str, int)) and view_name != "all":
        raise IndexError(f"Invalid `view_name`, `{view_name}` must be a string or an integer.")
    if view_name not in model.view_names:
        raise IndexError(f"`{view_name}` not found in the view names.")

    if use_prior_mask and not model.annotations is not None:
        raise ValueError("`feature_sets` is None, no feature sets provided for uninformed model.")

    sign = sign.lower().strip()
    allowed_signs = ["all", "pos", "neg"]
    if sign not in allowed_signs:
        raise ValueError(f"sign `{sign}` must be one of `{', '.join(allowed_signs)}`.")

    if use_prior_mask:
        logger.warning("No feature sets provided, extracting feature sets from prior mask.")
        feature_sets = model.annotations[view_name]
        if not feature_sets.any(axis=None):
            raise ValueError(f"Empty `feature_sets`, view `{view_name}` " "has not been informed prior to training.")

    feature_sets = feature_sets.astype(bool)
    if not feature_sets.any(axis=None):
        raise ValueError("Empty `feature_sets`.")
    feature_sets = feature_sets.loc[feature_sets.sum(axis=1) >= min_size, :]

    if not feature_sets.any(axis=None):
        raise ValueError("Empty `feature_sets` after filtering feature sets " f"of fewer than {min_size} features.")

    feature_sets = feature_sets.loc[~(feature_sets.all(axis=1)), feature_sets.any()]
    if not feature_sets.any(axis=None):
        raise ValueError("Empty `feature_sets` after filtering feature sets " f"of fewer than {min_size} features.")

    # subset available features only
    feature_intersection = feature_sets.columns.intersection(model.feature_names[view_name])
    feature_sets = feature_sets.loc[:, feature_intersection]

    if not feature_sets.any(axis=None):
        raise ValueError("Empty `feature_sets` after feature intersection with the observations.")

    y = pd.concat(
        [group_data[view_name].to_df().loc[:, feature_intersection].copy() for _, group_data in model.data.items()],
        axis=0,
    )
    factor_loadings = model.get_weights(return_type="anndata")[view_name].to_df().loc[:, feature_intersection].copy()
    factor_loadings /= np.max(np.abs(factor_loadings.to_numpy()))

    if "pos" in sign:
        factor_loadings[factor_loadings < 0] = 0.0
    if "neg" in sign:
        factor_loadings[factor_loadings > 0] = 0.0
    factor_loadings = factor_loadings.abs()

    factor_names = factor_loadings.index

    t_stat_dict = {}
    prob_dict = {}
    i = 0
    for feature_set in tqdm(feature_sets.index.tolist()):
        i += 1
        fs_features = feature_sets.loc[feature_set, :]

        features_in = factor_loadings.loc[:, fs_features]
        features_out = factor_loadings.loc[:, ~fs_features]

        n_in = features_in.shape[1]
        n_out = features_out.shape[1]

        df = n_in + n_out - 2.0
        mean_diff = features_in.mean(axis=1) - features_out.mean(axis=1)
        # why divide here by df and not denom later?
        svar = ((n_in - 1) * features_in.var(axis=1) + (n_out - 1) * features_out.var(axis=1)) / df

        vif = 1.0
        if corr_adjust:
            corr_df = y.loc[:, fs_features].corr()
            mean_corr = (np.nansum(corr_df.to_numpy()) - n_in) / (n_in * (n_in - 1))
            vif = 1 + (n_in - 1) * mean_corr
            df = y.shape[0] - 2
        denom = np.sqrt(svar * (vif / n_in + 1.0 / n_out))

        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.divide(mean_diff, denom)
        prob = t_stat.apply(lambda t: scipy.stats.t.sf(np.abs(t), df) * 2)  # noqa: B023

        t_stat_dict[feature_set] = t_stat
        prob_dict[feature_set] = prob

    t_stat_df = pd.DataFrame(t_stat_dict, index=factor_names)
    prob_df = pd.DataFrame(prob_dict, index=factor_names)
    t_stat_df.fillna(0.0, inplace=True)
    prob_df.fillna(1.0, inplace=True)
    if adjust_p:
        prob_adj_df = prob_df.apply(
            lambda p: multitest.multipletests(p, method=p_adj_method)[1], axis=1, result_type="broadcast"
        )

    if "all" not in sign:
        prob_df[t_stat_df < 0.0] = 1.0
        if adjust_p:
            prob_adj_df[t_stat_df < 0.0] = 1.0
        t_stat_df[t_stat_df < 0.0] = 0.0

    result = {"t": t_stat_df, "p": prob_df}
    if adjust_p:
        result["p_adj"] = prob_adj_df

    return result


def match(
    reference: torch.Tensor | np.ndarray, permutable: torch.Tensor | np.ndarray, dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find the permutation and sign of permutable along one dimension to maximize correlation with reference.

    This is useful for comparing ground truth factor scores / loadings with inferred values because the factor order
    and sign is arbitrary.

    Parameters
    ----------
    reference : torch.Tensor | np.ndarray
        The reference tensor.
    permutable : torch.Tensor | np.ndarray
        The permutable tensor.
    dim : int
        The dimension along which to permute the tensor.

    Returns
    -------
    permutation : torch.tensor
        The permutation of permutable that maximizes the correlation with reference.
    signs : torch.tensor
        The sign of permutable that maximizes the correlation with reference.
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

    if nonnegative:
        correlation = np.clip(correlation, a_min=0.0, a_max=None)

    # find the permutation that maximizes the correlation
    row_ind, permutation = linear_sum_assignment(-1 * np.abs(correlation))
    signs = np.ones_like(permutation)

    # if correlation is negative, flip the sign of the corresponding column
    for k in range(signs.shape[0]):
        if correlation[row_ind, permutation][k] < 0:
            signs[k] *= -1

    return permutation, signs
