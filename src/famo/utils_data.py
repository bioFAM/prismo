import logging
from collections import defaultdict
from functools import reduce
import warnings
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from mudata import MuData
from scipy.sparse._csr import csr_matrix

logger = logging.getLogger(__name__)


def cast_data(data: dict | MuData, group_by: str | list[str] | dict[str] | dict[list[str]] | None) -> dict:
    """Convert data to a nested dictionary of AnnData objects (first level: groups; second level: views).

    Parameters
    ----------
    data: dict or MuData
        Allowed input structures are:
        - Adata object (single group, single view)
        - MuData object (single group, multiple views)
        - dict with view names as keys and AnnData objects as values (single group, multiple views)
        - dict with view names as keys and torch.Tensor objects as values (single group, multiple views)
        - dict with group names as keys and MuData objects as values (multiple groups, multiple views)
        - Nested dict with group names as keys, view names as subkeys and AnnData objects as values (multiple groups, multiple views)
        - Nested dict with group names as keys, view names as subkeys and torch.Tensor objects as values (multiple groups, multiple views)

    Returns
    -------
    dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    """
    # single group, single view case
    if isinstance(data, AnnData):
        data = {"group_1": {"view_1": data.copy()}}

    # single group cases
    if isinstance(data, MuData):
        if group_by is None:
            data = {"group_1": data.mod}
        else:
            data = {
                name: {modname: mod.copy()}
                for name, idx in data.obs.groupby(group_by).indices.items()
                for modname, mod in data[idx].mod.items()
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
        data = {k: {mod: v[mod] for mod in v.mod} for k, v in data.items()}

    elif (
        isinstance(data, dict)
        and all(isinstance(v, dict) for v in data.values())
        and all(all(isinstance(vv, AnnData) for vv in v.values()) for v in data.values())
    ):
        if group_by is not None:
            raise ValueError("`data` is nested dict of AnnDatas but `group_by` is not `None`.")

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

    Parameters
    ----------
    data: dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.

    Returns
    -------
    dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
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
    """Infer likelihoods ("Normal", "Bernoulli", "BetaBinomial", "GammaPoisson") for each view based on the data distribution.

    Parameters
    ----------
    data: dict
        Dictionary with view names as keys and AnnData objects as values.

    Returns
    -------
    dict
        Dictionary with view names as keys and likelihoods as values.
    """
    likelihoods = {}

    for k, v in data.items():
        v_X = torch.tensor(v.X, dtype=torch.float).clone().detach()
        v_X = torch.nan_to_num(v_X, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            # check if all values are close to 0 or 1
            if torch.all(torch.isclose(v_X, torch.zeros_like(v_X)) | torch.isclose(v_X, torch.ones_like(v_X))):
                likelihoods[k] = "Bernoulli"

            # check if all values are positive integers
            elif torch.all(torch.isclose(v_X, torch.round(v_X))) and torch.all(v_X >= 0.0):
                # check if every variable name exists twice with different suffixes
                var_names_base = v.var_names.str.rsplit("_", n=1, expand=True).to_frame().reset_index(drop=True)[0]
                if var_names_base.nunique() * 2 == len(var_names_base):
                    likelihoods[k] = "BetaBinomial"

                else:
                    likelihoods[k] = "GammaPoisson"

            else:
                likelihoods[k] = "Normal"

        except Exception as e:
            logging.error(e)
            raise

    return likelihoods


def validate_likelihoods(data: dict, likelihoods: dict) -> dict:
    """Validate likelihoods ("Normal", "Bernoulli", "BetaBinomial", "GammaPoisson") for each view based on the data distribution.

    Parameters
    ----------
    data: dict
        Dictionary with view names as keys and AnnData objects as values.
    likelihoods: dict
        Dictionary with view names as keys and likelihoods as values.
    """
    for k, v in data.items():
        v_X = torch.tensor(v.X, dtype=torch.float).clone().detach()
        v_X = torch.nan_to_num(v_X, nan=0.0, posinf=0.0, neginf=0.0)

        if likelihoods[k] == "Bernoulli":
            # check if all values are close to 0 or 1
            if not torch.all(torch.isclose(v_X, torch.zeros_like(v_X)) | torch.isclose(v_X, torch.ones_like(v_X))):
                raise ValueError(f"Bernoulli likelihood in view {k} must be used with binary data.")

        elif likelihoods[k] in ["GammaPoisson", "BetaBinomial"]:
            # check if all values are positive integers
            if not (torch.all(torch.isclose(v_X, torch.round(v_X))) and torch.all(v_X >= 0.0)):
                raise ValueError(
                    f"{likelihoods[k]} likelihood in view {k} must be used with (integer, non-negative) count data."
                )

            if likelihoods[k] == "BetaBinomial":
                var_names_base = v.var_names.str.rsplit("_", n=1, expand=True).to_frame().reset_index(drop=True)[0]
                # check if all var_names appear twice with different suffixes
                if (not var_names_base.nunique() * 2 == len(var_names_base)) or v.var_names.duplicated().any():
                    raise ValueError(
                        f"BetaBinomial likelihood in view {k} requires every var_name to exist twice with different suffixes."
                    )


def remove_constant_features(data: dict, likelihoods: dict) -> dict:
    """Remove features with constant values.

    Parameters
    ----------
    data: dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    likelihoods: dict
        Dictionary with view names as keys and likelihoods as values.

    Returns
    -------
    dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    """
    # For each view, we collect the variances of all features and remove those with zero variance
    mask_keep_variable = {}

    for k_groups, v_groups in data.items():
        for k_views, v_views in v_groups.items():
            # Start with an array of all False and if feature must not be dropped, set to True
            if k_views not in mask_keep_variable:
                mask_keep_variable[k_views] = np.full(v_views.X.shape[1], False)

            # compute variance of each feature
            variances = np.var(v_views.X, axis=0)
            variances[variances < 1e-16] = 0

            # create mask that is True where a feature is constant (has zero variance)
            # Combine with or operator to keep track of all features that are constant across all groups
            mask_keep_variable[k_views] = np.logical_or(mask_keep_variable[k_views], (variances != 0))

            # the BetaBinomial likelihood is a special case because every feature occurs twice with different
            # suffixes and removal of one should also lead to removal of the other
            if likelihoods[k_views] == "BetaBinomial":
                # create DataFrame with indices of first and second occurence (columns) of every feature (rows)
                logger.warning("Not removing constants features in BetaBinomial.")
                # split_var_names = pd.Series(v_views.var_names).str.rsplit("_", n=1, expand=True)
                # split_var_names.columns = ["base", "suffix"]
                # suffix_indices = split_var_names.groupby("base").apply(
                #     lambda x: pd.Series(x.index.values), include_groups=False
                # )

                # # if any of the two occurences of a feature is masked, mask the other one as well
                # for var_name_base in suffix_indices.index:
                #     ix = suffix_indices.loc[var_name_base].values
                #     if mask_keep_variable[ix].any():
                #         mask_keep_variable[ix] = True

    for k_view in likelihoods.keys():
        if not mask_keep_variable[k_view].all():
            # We can reuse the last v_groups object to find the feature names
            dropped_features_names = list(v_groups[k_view].var_names[~mask_keep_variable[k_view]])

            for k_groups, v_groups in data.items():
                data[k_groups][k_views] = data[k_groups][k_views][:, mask_keep_variable[k_view]].copy()

            logger.debug(f"- Removing constant features in {k_groups}/{k_views}.")
            logger.debug(f"  - Removed {len(dropped_features_names)} features: {','.join(dropped_features_names)}")

    data.update()

    return data


def get_data_mean(data: dict, likelihoods: dict, how="feature") -> dict:
    """Compute the mean of each feature across all observations in a group. For BetaBinomial data, the mean is computed from the ratio of the first occurence of a feature to the sum of both occurences.

    Parameters
    ----------
    data: dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    likelihoods: dict
        Dictionary with view names as keys and likelihoods as values.

    Returns
    -------
    dict
        Nested dictionary of torch.Tensor objects with group names as keys and view names as subkeys,
        representing the feature means in the respective group.
    """
    if how not in ["feature", "sample"]:
        raise ValueError("how must be either 'feature' or 'sample'.")

    means = {}
    for k_groups, v_groups in data.items():
        means[k_groups] = {}
        for k_views, v_views in v_groups.items():
            if likelihoods[k_views] in ["Normal", "Bernoulli", "GammaPoisson"]:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    # In some views all values of a sample might be nan, so we need to ignore the warning 
                    means[k_groups][k_views] = np.nanmean(v_views.X, axis=0 if how == "feature" else 1)

            if likelihoods[k_views] == "BetaBinomial":
                # create DataFrame with indices of first and second occurence (columns) of every feature (rows)
                split_var_names = pd.Series(v_views.var_names).str.rsplit("_", n=1, expand=True)
                split_var_names.columns = ["base", "suffix"]
                suffix_indices = split_var_names.groupby("base").apply(
                    lambda x: pd.Series(x.index.values), include_groups=False
                )

                # get values of first and second occurence of every feature
                x_0 = v_views.X[:, suffix_indices.loc[split_var_names["base"]].values[:, 0]]
                x_1 = v_views.X[:, suffix_indices.loc[split_var_names["base"]].values[:, 1]]

                # compute mean of the ratio of the first occurence to the sum of both occurences
                ratio = x_0 / (x_0 + x_1 + 1e-6)
                means[k_groups][k_views] = np.nanmean(ratio, axis=0 if how == "feature" else 1)

    return means


def center_data(data: dict, likelihoods: dict, nonnegative_weights: dict, nonnegative_factors: dict) -> dict:
    """Center features to have zero mean in each group individually.

    Parameters
    ----------
    data: dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    likelihoods: dict
        Dictionary with view names as keys and likelihoods as values.
    nonnegative_weights: dict
        Dictionary with view names as keys and boolean values indicating whether weights should be non-negative.
    nonnegative_factors: dict
        Dictionary with view names as keys and boolean values indicating whether factors should be non-negative.

    Returns
    -------
    dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    """
    for k_groups, v_groups in data.items():
        for k_views, v_views in v_groups.items():
            if likelihoods[k_views] == "Normal":
                # only if both weights and factors are non-negative, data may not be negative
                if nonnegative_weights[k_views] and nonnegative_factors[k_groups]:
                    logger.debug(f"- Anchoring {k_groups}/{k_views}...")
                    min_values = np.nanmin(v_views.X, axis=0)
                    v_views.X -= min_values
                else:
                    logger.debug(f"- Centering {k_groups}/{k_views}...")
                    mean_values = np.nanmean(v_views.X, axis=0)
                    v_views.X -= mean_values

    data.update()
    return data


def scale_data(data: dict, likelihoods: dict, scale_per_group: bool = True) -> dict:
    """Scale each view to have unit variance in every group or across all groups.

    Parameters
    ----------
    data: dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    likelihoods: dict
        Dictionary with view names as keys and likelihoods as values.
    scale_per_group: bool
        If True, scale to unit variance per group. Otherwise across all groups.

    Returns
    -------
    dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
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
    """Align observations across views. After alignment, each view will have the same set of observations (with nans for missings) and observations will be sorted alphabetically according to their obs_names.

    Parameters
    ----------
    data: dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    use_obs: str
        If "union", use the union of observations across views. If "intersection", use the intersection.

    Returns
    -------
    dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
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

    Parameters
    ----------
    data: dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.

    Returns
    -------
    dict
        Nested dictionary of pandas.DataFrame objects with group names as keys and view names as subkeys.
    """
    metadata = {}

    for k_groups, v_groups in data.items():
        metadata[k_groups] = {}
        for k_views, v_views in v_groups.items():
            metadata[k_groups][k_views] = v_views.obs.copy()

    return metadata


def align_var(data: dict, likelihoods: dict, use_var: str = "intersection") -> dict:
    """Align features across groups.

    Parameters
    ----------
    data: dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    likelihoods: dict
        Dictionary with view names as keys and likelihoods as values.
    use_var_union: bool
        If True, use the union of features across groups.
    use_var_intersection: bool
        If True, use the intersection of features across groups.

    Returns
    -------
    dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
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

            if likelihoods[k_views] == "BetaBinomial":
                # keep only var names that have a base that occurs twice with different suffixes
                var_names_intersection_base = var_names_intersection.str.rsplit("_", n=1, expand=True)[0]
                duplicated = var_names_intersection_base.duplicated(keep=False)
                var_names_intersection = var_names_intersection[duplicated]

            for k_groups in group_names:
                data_aligned[k_groups][k_views] = data[k_groups][k_views][:, var_names_intersection].copy()

        if use_var == "union":
            var_names_union = pd.Series(
                reduce(np.union1d, [data[k_groups][k_views].var_names for k_groups in group_names])
            ).sort_values()

            if likelihoods[k_views] == "BetaBinomial":
                # keep only var names that have a base that occurs twice with different suffixes
                var_names_union_base = var_names_union.str.rsplit("_", n=1, expand=True)[0]
                duplicated = var_names_union_base.duplicated(keep=False)
                var_names_union = np.sort(var_names_union[duplicated])

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
                    expanded_varm[varm_k] = (
                        np.ones(shape=(len(var_names_union), data[k_groups][k_views].varm[varm_k].shape[1])) * np.nan
                    )
                    expanded_varm[varm_k][ix] = data[k_groups][k_views].varm[varm_k]

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
                    # varm=expanded_varm,
                )

    return data_aligned


def extract_covariate(data: dict, covariates_obs_key: dict = None, covariates_obsm_key: dict = None) -> dict | None:
    """Extract covariate data from AnnData objects.

    Parameters
    ----------
    data: dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    covariates_obs_key: dict
        Dictionary with group names as keys and covariate obs keys as values.
    covariates_obsm_key: dict
        Dictionary with group names as keys and covariate obsm keys as values.

    Returns
    -------
    dict | None
        Dictionary of Tensors with group names as keys or None if no covariate keys provided.
    """
    if covariates_obs_key is None and covariates_obsm_key is None:
        return None

    if covariates_obs_key is not None and covariates_obsm_key is not None:
        raise ValueError("Please provide either covariates_obs_key or covariates_obsm_key, not both.")

    else:
        covariates = {}

        for group_name, group_dict in data.items():
            group_covariates = []
            for view_adata in group_dict.values():
                if isinstance(covariates_obs_key, dict) and covariates_obs_key[group_name] is not None:
                    group_covariates.append(
                        torch.tensor(view_adata.obs[covariates_obs_key[group_name]], dtype=torch.float).unsqueeze(-1)
                    )

                elif isinstance(covariates_obsm_key, dict) and covariates_obsm_key[group_name] is not None:
                    group_covariates.append(
                        torch.tensor(view_adata.obsm[covariates_obsm_key[group_name]], dtype=torch.float)
                    )

            if len(group_covariates) == 0:
                continue
            covariates[group_name] = torch.stack(group_covariates, dim=0).nanmean(dim=0)

        return covariates
