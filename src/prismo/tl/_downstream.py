import logging

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from statsmodels.stats import multitest
from tqdm import tqdm

logger = logging.getLogger(__name__)

Index = int | str | list[int] | list[str] | np.ndarray | pd.Index


def _test_single_view(
    model,
    view_name: str,
    feature_sets: pd.DataFrame = None,
    sign: str = "all",
    corr_adjust: bool = True,
    p_adj_method: str = "fdr_bh",
    min_size: int = 10,
):
    """Perform significance test of factor loadings against feature sets.

    Args:
        model: A PRISMO model instance.
        view_name: Name of the view to test.
        feature_sets: Boolean dataframe with feature sets in each row. If None, uses model annotations.
        sign: Test direction - "all" for two-sided, "neg" or "pos" for one-sided tests.
        corr_adjust: Whether to adjust for correlations between features.
        p_adj_method: Method for multiple testing adjustment (e.g. "fdr_bh").
        min_size: Minimum size threshold for feature sets.

    Returns:
        dict: Test results containing:
            - "t": DataFrame of t-statistics
            - "p": DataFrame of p-values
            - "p_adj": DataFrame of adjusted p-values (if p_adj_method is not None)
    """
    use_prior_mask = feature_sets is None
    adjust_p = p_adj_method is not None

    if not isinstance(view_name, str):
        raise IndexError(f"Invalid `view_name`, `{view_name}` must be a string.")
    if view_name not in model.view_names:
        raise IndexError(f"`{view_name}` not found in the view names.")

    informed = model.annotations is not None and len(model.annotations) > 0
    if use_prior_mask and not informed:
        raise ValueError("`feature_sets` is None, no feature sets provided for uninformed model.")

    sign = sign.lower().strip()
    allowed_signs = ["all", "pos", "neg"]
    if sign not in allowed_signs:
        raise ValueError(f"sign `{sign}` must be one of `{', '.join(allowed_signs)}`.")

    if use_prior_mask:
        logger.warning("No feature sets provided, extracting feature sets from prior mask.")
        feature_sets = model.get_annotations("pandas")[view_name]
        if not feature_sets.any(axis=None):
            raise ValueError(f"Empty `feature_sets`, view `{view_name}` has not been informed prior to training.")

    feature_sets = feature_sets.astype(bool)
    if not feature_sets.any(axis=None):
        raise ValueError("Empty `feature_sets`.")
    feature_sets = feature_sets.loc[feature_sets.sum(axis=1) >= min_size, :]

    if not feature_sets.any(axis=None):
        raise ValueError(f"Empty `feature_sets` after filtering feature sets of fewer than {min_size} features.")

    feature_sets = feature_sets.loc[~(feature_sets.all(axis=1)), feature_sets.any()]
    if not feature_sets.any(axis=None):
        raise ValueError(f"Empty `feature_sets` after filtering feature sets of fewer than {min_size} features.")

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

    for feature_set in tqdm(feature_sets.index.tolist()):
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


def test(
    model, feature_sets: pd.DataFrame = None, corr_adjust: bool = True, p_adj_method: str = "fdr_bh", min_size: int = 10
):
    """Perform significance testing across multiple views and sign directions.

    Args:
        model: A PRISMO model instance.
        feature_sets: Boolean dataframe with feature sets in each row. If None, uses model annotations.
        corr_adjust: Whether to adjust for correlations between features.
        p_adj_method: Method for multiple testing adjustment.
        min_size: Minimum size threshold for feature sets.

    Returns:
        dict: Nested dictionary with structure:
            {sign: {view_name: test_results}}
            where test_results contains t-statistics, p-values, and adjusted p-values.
    """
    use_prior_mask = feature_sets is None
    if use_prior_mask:
        view_names = [vi for vi in model.view_names if vi in model.annotations]

    if len(view_names) == 0:
        if use_prior_mask:
            raise ValueError("`feature_sets` is None, and none of the selected views are informed.")
        raise ValueError("No valid views.")

    signs = ["neg", "pos"]

    results = {}

    for sign in signs:
        results[sign] = {}
        for view_name in view_names:
            try:
                results[sign][view_name] = _test_single_view(
                    model,
                    view_name=view_name,
                    feature_sets=feature_sets,
                    sign=sign,
                    corr_adjust=corr_adjust,
                    p_adj_method=p_adj_method,
                    min_size=min_size,
                )
            except ValueError as e:
                logger.warning(e)
                results[sign][view_name] = {"t": pd.DataFrame(), "p": pd.DataFrame()}
                if p_adj_method is not None:
                    results[sign][view_name]["p_adj"] = pd.DataFrame()
                continue
    return results


def match(reference: NDArray, permutable: NDArray, axis: int) -> tuple[NDArray[int], NDArray[int], NDArray[np.uint8]]:
    """Find optimal permutation and signs to match two tensors along specified axis.

    Finds the permutation and sign of permutable along one axis to maximize
    correlation with reference. Useful for comparing ground truth factor scores/loadings
    with inferred values where factor order and sign is arbitrary.

    Args:
        reference: Reference array to match against.
        permutable: Array to be permuted and sign-adjusted.
        axis: Axis along which to perform matching.

    Returns:
        A tuple with optimal permutation indices and optimal signs (+1 or -1) for each
        permuted element.

    Notes:
        - Special handling for non-negative arrays
        - Uses linear sum assignment to find optimal matching
    """
    nonnegative = np.all(reference >= 0) and np.all(permutable >= 0)
    one_d = reference.ndim == 1 or np.all(np.delete(reference.shape, axis) == 1)

    reference = np.moveaxis(reference, axis, -1).reshape(-1, reference.shape[axis]).T
    permutable = np.moveaxis(permutable, axis, -1).reshape(-1, permutable.shape[axis]).T

    signs = np.ones(shape=permutable.shape[0], dtype=np.int8)
    if not one_d:
        correlation = 1 - cdist(reference, permutable, metric="correlation")
        correlation = np.nan_to_num(correlation, 0)

        reference_ind, permutable_ind = linear_sum_assignment(-1 * np.abs(correlation))

        # if correlation is negative, flip the sign of the corresponding column
        for k in range(signs.shape[0]):
            if correlation[reference_ind, permutable_ind][k] < 0 and not nonnegative:
                signs[k] *= -1
    else:
        difference = cdist(reference, permutable, metric="euclidean")
        reference_ind, permutable_ind = linear_sum_assignment(difference)

    return reference_ind, permutable_ind, signs
