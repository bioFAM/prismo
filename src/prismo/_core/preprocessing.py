import logging
import warnings
from collections import defaultdict
from functools import reduce

import anndata as ad
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from mudata import MuData
from numpy.typing import NDArray
from scipy.sparse import issparse
from scipy.sparse._csr import csr_matrix

from .datasets import Preprocessor
from .utils import Likelihood, ViewStatistics

_logger = logging.getLogger(__name__)


class PrismoPreprocessor(Preprocessor):
    def __init__(
        self,
        likelihoods: dict[str, Likelihood],
        nonnegative_weights: dict[str, bool],
        nonnegative_factors: dict[str, bool],
        scale_per_group: bool = True,
        constant_feature_var_threshold: float = 1e-16,
    ):
        super().__init__()

        self._scale_per_group = scale_per_group
        self._constant_feature_var_threshold = constant_feature_var_threshold
        self._views_to_scale = {view_name for view_name, likelihood in likelihoods.items() if likelihood == "Normal"}
        self._nonnegative_weights = {k for k, v in nonnegative_weights.items() if v}
        self._nonnegative_factors = {k for k, v in nonnegative_factors.items() if v}

    def _initialize(self):
        self._nonconstantfeatures = {}
        for view_name, viewstats in self._viewstats.items():
            # Storing a boolean mask is probably more memory-efficient than storing indices: indices are int64 (4 bytes), while
            # booleans are 1 byte. As long as we keep more than 1/ of the features this uses less memory.
            nonconst = viewstats.var > self._constant_feature_var_threshold
            _logger.debug(f"Removing {viewstats.var.size - nonconst.sum()} features from view {view_name}.")
            self._nonconstantfeatures[view_name] = nonconst

            self._viewstats[view_name] = ViewStatistics(*(stat[nonconst] for stat in viewstats))
            for view in self._viewstats_per_group.values():
                view[view_name] = ViewStatistics(*(stat[nonconst] for stat in view[view_name]))

    @property
    def feature_means(self) -> dict[str, dict[str, float]]:
        return {
            group_name: {view_name: stats.mean for view_name, stats in group.items()}
            for group_name, group in self._viewstats_per_group.items()
        }

    @property
    def sample_means(self) -> dict[str, dict[str, float]]:
        return {
            group_name: {view_name: stats.mean for view_name, stats in group.items()}
            for group_name, group in self._samplestats.items()
        }

    def __call__(self, arr: NDArray, group: str, view: str):
        # remove constant features
        arr = arr[..., self._nonconstantfeatures[view]]

        if view in self._views_to_scale:
            viewstats = self._viewstats[view]

            # scale to unit variance
            arr /= np.sqrt(self._viewstats_per_group[group][view].var if self._scale_per_group else viewstats.var)

            # center
            if view in self._nonnegative_weights and group in self._nonnegative_factors:
                arr -= viewstats.min
            else:
                arr -= viewstats.mean
        return arr


def infer_likelihood(view: AnnData, *args) -> Likelihood:
    """Infer the likelihood for a view based on the data distribution."""
    data = view.X.data if issparse(view.X) else view.X
    if np.all(np.isclose(data, 0) | np.isclose(data, 1)):  # TODO: set correct atol value
        return "Bernoulli"
    elif np.allclose(data, np.round(data)) and data.min() >= 0:
        return "GammaPoisson"
    else:
        return "Normal"


def validate_likelihood(view: AnnData, group_name: str, view_name: str, likelihood: Likelihood) -> str | None:
    """Validate the likelihood for a view based on the data distribution."""
    data = view.X.data if issparse(view.X) else view.X
    if likelihood == "Bernoulli" and not np.all(
        np.isclose(data, 0) | np.isclose(data, 1)
    ):  # TODO: set correct atol value
        raise ValueError(f"Bernoulli likelihood in view {view_name} must be used with binary data.")
    elif likelihood == "GammaPoisson" and not np.allclose(data, np.round(data)) and data.min() >= 0:
        raise ValueError(
            f"GammaPoisson likelihood in view {view_name} must be used with count (non-negative integer) data."
        )


def cast_data(
    data: dict | MuData, group_by: str | list[str] | dict[str] | dict[list[str]] | None, copy: bool = False
) -> dict[dict[str, AnnData]]:
    """Convert data to a nested dictionary of AnnData objects (first level: groups; second level: views).

    Args:
        data: Input data. Allowed input structures are:
            - Adata object (single group, single view)
            - MuData object (single group, multiple views)
            - dict with view names as keys and AnnData objects as values (single group, multiple views)
            - dict with view names as keys and torch.Tensor objects as values (single group, multiple views)
            - dict with group names as keys and MuData objects as values (multiple groups, multiple views)
            - Nested dict with group names as keys, view names as subkeys and AnnData objects as values (multiple groups, multiple views)
            - Nested dict with group names as keys, view names as subkeys and torch.Tensor objects as values (multiple groups, multiple views)
        group_by: Key in obs to group the data by. If provided, the data will be split into groups based on this key.
        copy: If `True`, the data will always be copied, even if it's in the correct format already

    Returns:
        dict: Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    """
    # single group, single view case
    if isinstance(data, AnnData):
        data = {"group_1": {"view_1": data.copy()}}

    # single group cases
    elif isinstance(data, MuData):
        data = data.copy()
        if group_by is None:
            data.push_obs()
            data = {"group_1": data.mod}
        else:
            data = {
                name: {modname: mod.copy() for modname, mod in data[idx].mod.items()}
                for name, idx in data.obs.groupby(group_by).indices.items()
            }

    elif isinstance(data, dict) and all(isinstance(v, AnnData) for v in data.values()):
        if group_by is None:
            data = {"group_1": data}
        else:
            data = defaultdict(dict)
            for viewname, view in data.items():
                if isinstance(group_by, dict):
                    for name, idx in view.obs.groupby(group_by[viewname]).indices.items():
                        data[name][viewname] = view[idx].copy()
                else:
                    for name, idx in view.obs.groupby(group_by).indices.items():
                        data[name][viewname] = view[idx].copy()

    elif isinstance(data, dict) and all(isinstance(v, torch.Tensor) for v in data.values()):
        if group_by is not None:
            raise ValueError("`data` is dict of tensors but `group_by` is not `None`.")
        data = {"group_1": {k: AnnData(X=v.numpy()) for k, v in data.items()}}

    # multiple groups cases
    elif isinstance(data, dict) and all(isinstance(v, MuData) for v in data.values()):
        if group_by is not None:
            raise ValueError("`data` is dict of MuDatas but `group_by` is not `None`.")
        data = {k: {mod: v[mod].copy() if copy else v[mod] for mod in v.mod} for k, v in data.items()}

    elif (
        isinstance(data, dict)
        and all(isinstance(v, dict) for v in data.values())
        and all(all(isinstance(vv, AnnData) for vv in v.values()) for v in data.values())
    ):
        if group_by is not None:
            raise ValueError("`data` is nested dict of AnnDatas but `group_by` is not `None`.")
        if copy:
            data = {gname: {vname: adata.copy() for vname, adata in group.items()} for gname, group in data.items()}

    elif (
        isinstance(data, dict)
        and all(isinstance(v, dict) for v in data.values())
        and all(all(isinstance(vv, torch.Tensor) for vv in v.values()) for v in data.values())
    ):
        if group_by is not None:
            raise ValueError("`data` is nested dict of tensors but `group_by` is not `None`.")
        data = {k: {kk: AnnData(X=vv.numpy()) for kk, vv in v.items()} for k, v in data.items()}

    else:
        raise ValueError("Input data structure not recognized. Please refer to the documentation for allowed formats.")

    return data


def anndata_to_dense(data: dict) -> dict:
    """Convert sparse arrays in AnnData objects to dense.

    This function takes a nested dictionary of AnnData objects and converts
    any sparse arrays within them to dense arrays.

    Args:
        data: Nested dictionary of AnnData objects with group names as keys
            and view names as subkeys.

    Returns:
        dict: Nested dictionary of AnnData objects with group names as keys
            and view names as subkeys, where all sparse arrays have been
            converted to dense arrays.
    """
    for _, v_groups in data.items():
        for _, adata in v_groups.items():
            if isinstance(adata.X, csr_matrix):
                adata.X = adata.X.toarray()
            for obsm_key in adata.obsm.keys():
                if isinstance(adata.obsm[obsm_key], csr_matrix):
                    adata.obsm[obsm_key] = adata.obsm[obsm_key].toarray()
            for layer_key in adata.layers.keys():
                if isinstance(adata.layers[layer_key], csr_matrix):
                    adata.layers[layer_key] = adata.layers[layer_key].toarray()
            for varm_key in adata.varm.keys():
                if isinstance(adata.varm[varm_key], csr_matrix):
                    adata.varm[varm_key] = adata.varm[varm_key].toarray()

    return data


def infer_likelihoods(data: dict) -> dict:
    """Infer likelihoods for each view based on the data distribution.

    This function analyzes the data distribution in each view and infers the
    appropriate likelihood model. The possible likelihoods are "Normal",
    "Bernoulli" and "GammaPoisson".

    Args:
        data: Dictionary with view names as keys and AnnData objects as values.

    Returns:
        dict: Dictionary with view names as keys and inferred likelihoods as values.
    """
    likelihoods = {}

    for k, v in data.items():
        v_X = torch.tensor(v.X, dtype=torch.float32).clone().detach()
        v_X = torch.nan_to_num(v_X, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            # check if all values are close to 0 or 1
            if torch.all(torch.isclose(v_X, torch.zeros_like(v_X)) | torch.isclose(v_X, torch.ones_like(v_X))):
                likelihoods[k] = "Bernoulli"

            # check if all values are positive integers
            elif torch.all(torch.isclose(v_X, torch.round(v_X))) and torch.all(v_X >= 0.0):
                # check if every variable name exists twice with different suffixes
                var_names_base = v.var_names.str.rsplit("_", n=1, expand=True).to_frame().reset_index(drop=True)[0]
                if var_names_base.nunique() * 2 != len(var_names_base):
                    likelihoods[k] = "GammaPoisson"

            else:
                likelihoods[k] = "Normal"

        except Exception as e:
            logging.error(e)
            raise

    return likelihoods


def validate_likelihoods(data: dict, likelihoods: dict) -> dict:
    """Validate likelihoods for each view based on the data distribution.

    This function validates the specified likelihoods ("Normal", "Bernoulli", "GammaPoisson")
    for each view by comparing them against the actual data distribution.

    Args:
        data: Dictionary with view names as keys and AnnData objects as values.
        likelihoods: Dictionary with view names as keys and likelihoods as values.

    Returns:
        dict: Dictionary with view names as keys and validated likelihoods as values.

    """
    for k, v in data.items():
        v_X = torch.tensor(v.X, dtype=torch.float32).clone().detach()
        v_X = torch.nan_to_num(v_X, nan=0.0, posinf=0.0, neginf=0.0)

        if likelihoods[k] == "Bernoulli":
            # check if all values are close to 0 or 1
            if not torch.all(torch.isclose(v_X, torch.zeros_like(v_X)) | torch.isclose(v_X, torch.ones_like(v_X))):
                raise ValueError(f"Bernoulli likelihood in view {k} must be used with binary data.")

        if likelihoods[k] == "GammaPoisson":
            # check if all values are positive integers
            if not (torch.all(torch.isclose(v_X, torch.round(v_X))) and torch.all(v_X >= 0.0)):
                raise ValueError(
                    f"{likelihoods[k]} likelihood in view {k} must be used with (integer, non-negative) count data."
                )


def remove_constant_features(data: dict, likelihoods: dict) -> dict:
    """Remove features with constant values.

    This function removes features that have constant values across all samples
    from each AnnData object in the nested dictionary.

    Args:
        data: Nested dictionary of AnnData objects with group names as keys
            and view names as subkeys.
        likelihoods: Dictionary with view names as keys and likelihoods as values.

    Returns:
        dict: Nested dictionary of AnnData objects with group names as keys and
            view names as subkeys, with constant features removed.

    TODO: Raise error if all features are constant across all groups and views.
    """
    mask_keep_var = {}
    for view_name in likelihoods.keys():
        adata_view = ad.concat(
            [group_dict[view_name] for group_dict in data.values() if view_name in group_dict],
            join="outer",
            fill_value=0.0,
        )

        variances = np.nanvar(adata_view.X, axis=0)
        mask_keep_var[view_name] = variances > 1e-16
        n_removed_features = np.sum(~mask_keep_var[view_name])
        _logger.debug(f"Removing {n_removed_features} constant features from view {view_name}")

        if n_removed_features > 0:
            for group_name, group_dict in data.items():
                if view_name in group_dict.keys():
                    new_vars = np.intersect1d(
                        adata_view.var_names[mask_keep_var[view_name]], group_dict[view_name].var_names
                    )
                    data[group_name][view_name] = group_dict[view_name][:, new_vars].copy()

    return data


def get_data_mean(data: dict, how="feature") -> dict:
    """Compute the mean of each feature across all observations in a group.

    This function calculates the mean of each feature for all observations within a group.

    Args:
        data: Nested dictionary of AnnData objects with group names as keys
            and view names as subkeys.
        how: Specifies whether to compute the mean across features or samples.

    Returns:
        dict: Nested dictionary of np.ndarray objects with group names as keys
            and view names as subkeys, representing the feature means in the
            respective group.
    """
    if how not in ["feature", "sample"]:
        raise ValueError("how must be either 'feature' or 'sample'.")

    means = {}
    for k_groups, v_groups in data.items():
        means[k_groups] = {}
        for k_views, v_views in v_groups.items():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # In some views all values of a sample might be nan, so we need to ignore the warning
                means[k_groups][k_views] = np.nanmean(v_views.X, axis=0 if how == "feature" else 1)

    return means


def center_data(data: dict, likelihoods: dict, nonnegative_weights: dict, nonnegative_factors: dict) -> dict:
    """Center features to have zero mean in each group individually.

    This function centers the features within each group by subtracting the group-specific
    mean from each feature, ensuring zero mean across observations within each group.

    Args:
        data: Nested dictionary of AnnData objects with group names as keys
            and view names as subkeys.
        likelihoods: Dictionary with view names as keys and likelihoods as values.
        nonnegative_weights: Dictionary with view names as keys and boolean values
            indicating whether weights should be non-negative.
        nonnegative_factors: Dictionary with view names as keys and boolean values
            indicating whether factors should be non-negative.

    Returns:
        dict: Nested dictionary of AnnData objects with group names as keys and
            view names as subkeys, containing the centered feature data.
    """
    for k_groups, v_groups in data.items():
        for k_views, v_views in v_groups.items():
            if likelihoods[k_views] == "Normal":
                # only if both weights and factors are non-negative, data may not be negative
                if nonnegative_weights[k_views] and nonnegative_factors[k_groups]:
                    _logger.debug(f"- Anchoring {k_groups}/{k_views}...")
                    min_values = np.nanmin(v_views.X, axis=0)
                    v_views.X -= min_values
                else:
                    _logger.debug(f"- Centering {k_groups}/{k_views}...")
                    mean_values = np.nanmean(v_views.X, axis=0)
                    v_views.X -= mean_values

    data.update()
    return data


def scale_data(data: dict, likelihoods: dict, scale_per_group: bool = True) -> dict:
    """Scale each view to have unit variance in every group or across all groups.

    This function scales the data in each view to achieve unit variance. The scaling
    can be performed either independently for each group or globally across all groups.

    Args:
        data: Nested dictionary of AnnData objects with group names as keys
            and view names as subkeys.
        likelihoods: Dictionary with view names as keys and likelihoods as values.
        scale_per_group: If True, scales to unit variance independently for
            each group. If False, scales to unit variance across all groups combined.

    Returns:
        dict: Nested dictionary of AnnData objects with group names as keys and
            view names as subkeys, containing the scaled data.
    """
    if scale_per_group:
        for v_groups in data.values():
            for k_views, v_views in v_groups.items():
                if likelihoods[k_views] == "Normal":
                    # scale by inverse std across all observations and features within the group
                    std = np.nanstd(v_views.X)
                    v_views.X /= std

    else:
        for k_views in likelihoods.keys():
            if likelihoods[k_views] == "Normal":
                X_all_groups = np.concatenate([data[k_groups][k_views].X for k_groups in data.keys()])
                # scale by inverse std across all observations and features across all groups
                std = np.nanstd(X_all_groups)
                for k_groups in data.keys():
                    data[k_groups][k_views].X /= std

    data.update()
    return data


def align_obs(data: dict, use_obs: str = "union", cov_key: str = "x") -> dict:
    """Align observations across views.

    This function aligns observations across different views, ensuring each view has
    the same set of observations. Missing values are filled with nans. After alignment,
    observations are sorted alphabetically by their obs_names.

    Args:
        data: Nested dictionary of AnnData objects with group names as keys
            and view names as subkeys.
        use_obs: Strategy for observation alignment. Must be either:
            - "union": Include all observations from all views
            - "intersection": Include only observations present in all views
        cov_key: Key in obsm to merge covariate values across views.

    Returns:
        dict: Nested dictionary of AnnData objects with group names as keys and
            view names as subkeys, containing aligned observations.
    """
    if use_obs not in ["union", "intersection"]:
        raise ValueError("use_obs must be either 'union' or 'intersection'.")

    data_aligned = {}

    for k_group in data.keys():
        data_aligned[k_group] = {}

        if use_obs == "intersection":
            # series of sorted obs_names that are present in all views
            obs_names_intersection = pd.Series(
                reduce(np.intersect1d, [v.obs_names for v in data[k_group].values()])
            ).sort_values()
            # subset obs to intersection and sort by obs_names
            for k_view, v_view in data[k_group].items():
                data_aligned[k_group][k_view] = v_view[obs_names_intersection, :].copy()

        if use_obs == "union":
            # series of sorted obs_names that are present in any view
            obs_names_union = pd.Series(reduce(np.union1d, [v.obs_names for v in data[k_group].values()])).sort_values()

            obsm_cov = None
            for k_view, v_view in data[k_group].items():
                # use pandas data frames indices to expand obs
                orig = v_view.obs.loc[:, []].copy()
                expanded = orig.reindex(obs_names_union)
                expanded["ix"] = range(len(obs_names_union))
                ix = expanded.join(orig, how="right").ix.values

                # expand data matrix
                expanded_X = np.ones(shape=(len(obs_names_union), v_view.shape[1])) * np.nan
                expanded_X[ix, :] = v_view.X

                # expand obs data frame
                expanded_obs = v_view.obs.reindex(obs_names_union)

                # expand obsm matrices
                expanded_obsm = {}
                for obsm_k in v_view.obsm.keys():
                    if obsm_k != cov_key:
                        expanded_obsm[obsm_k] = (
                            np.ones(shape=(len(obs_names_union), v_view.obsm[obsm_k].shape[1])) * np.nan
                        )
                        expanded_obsm[obsm_k][ix, :] = v_view.obsm[obsm_k]
                    else:
                        # if this is the covariate, we don't want nan values but instead just merge all covariate values across views
                        if obsm_cov is None:
                            obsm_cov = np.ones(shape=(len(obs_names_union), v_view.obsm[obsm_k].shape[1])) * np.nan
                        obsm_cov[ix, :] = v_view.obsm[obsm_k]

                # expand layers matrices
                expanded_layers = {}
                for layer_k in v_view.layers.keys():
                    expanded_layers[layer_k] = (
                        np.ones(shape=(len(obs_names_union), v_view.layers[layer_k].shape[1])) * np.nan
                    )
                    expanded_layers[layer_k][ix, :] = v_view.layers[layer_k]

                # create new AnnData with expanded data
                data_aligned[k_group][k_view] = AnnData(
                    X=expanded_X,
                    obs=expanded_obs,
                    var=v_view.var,
                    obsm=expanded_obsm,
                    varm=v_view.varm,
                    layers=expanded_layers,
                    uns=v_view.uns,
                )

            # set the merged covariate values across views
            for k_view, v_view in data[k_group].items():
                for obsm_k in v_view.obsm.keys():
                    if obsm_k == cov_key:
                        data_aligned[k_group][k_view].obsm[cov_key] = obsm_cov

    return data_aligned


def extract_obs(data: dict) -> dict:
    """Extract obs DataFrames from AnnData objects.

    This function extracts the observation (obs) DataFrames from each AnnData object
    in the nested dictionary structure.

    Args:
        data: Nested dictionary of AnnData objects with group names as keys
            and view names as subkeys.

    Returns:
        dict: Nested dictionary of pandas.DataFrame objects with group names as keys
            and view names as subkeys, containing the extracted observation data.
    """
    metadata = {}

    for k_groups, v_groups in data.items():
        metadata[k_groups] = {}
        for k_views, v_views in v_groups.items():
            metadata[k_groups][k_views] = v_views.obs.copy()

    return metadata


def align_var(data: dict, use_var: str = "intersection") -> dict:
    """Align features across groups.

    This function aligns features across different groups in the data. The alignment
    can be done using either the union or intersection of features across groups,
    as specified by the input parameters.

    Args:
        data: Nested dictionary of AnnData objects with group names as keys
            and view names as subkeys.
        use_var: Strategy for feature alignment. Must be either:
            - "union": Include all features from all groups
            - "intersection": Include only features present in all groups

    Returns:
        dict: Nested dictionary of AnnData objects with group names as keys and
            view names as subkeys, containing the aligned features.
    """
    if use_var not in ["union", "intersection"]:
        raise ValueError("use_var must be either 'union' or 'intersection'.")

    group_names = data.keys()
    view_names = reduce(np.union1d, [list(v_groups.keys()) for v_groups in data.values()])

    data_aligned = {k_groups: {} for k_groups in group_names}

    for k_views in view_names:
        if use_var == "intersection":
            var_names_intersection = pd.Series(
                reduce(np.intersect1d, [data[k_groups][k_views].var_names for k_groups in group_names])
            ).sort_values()

            for k_groups in group_names:
                data_aligned[k_groups][k_views] = data[k_groups][k_views][:, var_names_intersection].copy()

        if use_var == "union":
            var_names_union = pd.Series(
                reduce(np.union1d, [data[k_groups][k_views].var_names for k_groups in group_names])
            ).sort_values()

            for k_groups in group_names:
                # use pandas data frames indices to expand var
                orig = data[k_groups][k_views].var.loc[:, []].copy()
                expanded = orig.reindex(var_names_union)
                expanded["ix"] = range(len(var_names_union))
                ix = expanded.join(orig, how="right").ix.values

                # expand data matrix
                expanded_X = np.ones(shape=(data[k_groups][k_views].shape[0], len(var_names_union))) * np.nan
                expanded_X[:, ix] = data[k_groups][k_views].X

                # expand var data frame
                expanded_var = pd.DataFrame(index=var_names_union)
                # expanded_var = data[k_groups][k_views].var.reindex(var_names_union)

                # expand varm matrices
                expanded_varm = {}
                for varm_k in data[k_groups][k_views].varm.keys():
                    varm_v = data[k_groups][k_views].varm[varm_k]
                    columns = varm_v.columns if isinstance(varm_v, pd.DataFrame) else range(varm_v.shape[1])
                    expanded_varm[varm_k] = np.ones(shape=(len(var_names_union), varm_v.shape[1])) * np.nan
                    expanded_varm[varm_k][ix, :] = varm_v
                    expanded_varm[varm_k] = pd.DataFrame(expanded_varm[varm_k], index=var_names_union, columns=columns)

                # expand layer matrices
                expanded_layers = {}
                for layer_k in data[k_groups][k_views].layers.keys():
                    expanded_layers[layer_k] = (
                        np.ones(shape=(data[k_groups][k_views].layers[layer_k].shape[0], len(var_names_union))) * np.nan
                    )
                    expanded_layers[layer_k][:, ix] = data[k_groups][k_views].layers[layer_k]

                # create new AnnData with expanded data
                data_aligned[k_groups][k_views] = AnnData(
                    X=expanded_X,
                    obs=data[k_groups][k_views].obs,
                    var=expanded_var,
                    obsm=data[k_groups][k_views].obsm,
                    varm=expanded_varm,
                )

    return data_aligned


def extract_covariate(
    data: dict, covariates_obs_key: dict = None, covariates_obsm_key: dict = None
) -> dict[str, torch.Tensor] | None:
    """Extract covariate data from AnnData objects.

    This function extracts covariate data from either the obs or obsm attributes
    of AnnData objects based on the provided keys.

    Returns:
    -------
    dict | None
        Dictionary of Tensors with group names as keys or None if no covariate keys provided.
    """
    # ensure that covariates_obs_key and covariates_obsm_key are dictionaries
    if isinstance(covariates_obs_key, str) or covariates_obs_key is None:
        covariates_obs_key = {k: covariates_obs_key for k in data.keys()}
    if isinstance(covariates_obsm_key, str) or covariates_obsm_key is None:
        covariates_obsm_key = {k: covariates_obsm_key for k in data.keys()}

    # dictionaries to store covariate data and names
    covariates = {}
    covariates_names = {}

    # iterate over groups to extract covariate data and names
    for group_name, group_dict in data.items():
        # ensure that the group_name is in the covariates_obs_kets and covariates_obsm_keys dictionaries
        if group_name not in covariates_obs_key.keys():
            covariates_obs_key[group_name] = None
        if group_name not in covariates_obsm_key.keys():
            covariates_obsm_key[group_name] = None

        # lists to store covariate data and names for current group
        covariates_group = []

        # check if both covariates_obs_key and covariates_obsm_key are provided and not None for current group (not allowed)
        if covariates_obs_key[group_name] is not None and covariates_obsm_key[group_name] is not None:
            raise ValueError(
                f"Provide either covariates_obs_key or covariates_obsm_key for group {group_name}, not both."
            )

        # check if no covariate keys are provided for current group (skip group)
        if covariates_obs_key[group_name] is None and covariates_obsm_key[group_name] is None:
            continue

        # extract covariate data and names from obs attribute
        obs_key = covariates_obs_key[group_name]
        if obs_key is not None:
            for view_adata in group_dict.values():
                if obs_key in view_adata.obs.columns:
                    covariates_group.append(torch.tensor(view_adata.obs[obs_key], dtype=torch.float32).unsqueeze(-1))
            if len(covariates_group) > 0:
                covariates_names[group_name] = obs_key
            else:
                warnings.warn(f"No covariate data found in obs attribute for group {group_name}.", stacklevel=2)

        # extract covariate data and names from obsm attribute
        obsm_key = covariates_obsm_key[group_name]
        if obsm_key is not None:
            covar_dim = []
            for view_adata in group_dict.values():
                if obsm_key in view_adata.obsm.keys():
                    covar_dim.append(view_adata.obsm[obsm_key].shape[1])
                    covariates_group.append(torch.tensor(view_adata.obsm[obsm_key], dtype=torch.float32))

            if len(set(covar_dim)) > 1:
                raise ValueError(f"Number of covariate dimensions in group {group_name} must be the same across views.")

            if len(covariates_group) > 0:
                if isinstance(view_adata.obsm[obsm_key], pd.DataFrame):
                    covariates_names[group_name] = view_adata.obsm[obsm_key].columns.to_numpy()
                if isinstance(view_adata.obsm[obsm_key], pd.Series):
                    covariates_names[group_name] = np.asarray(view_adata.obsm[obsm_key].name, dtype=object)

        covariates[group_name] = torch.stack(covariates_group, dim=0).nanmean(dim=0)

    return covariates, covariates_names
