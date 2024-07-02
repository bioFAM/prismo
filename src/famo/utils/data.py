import logging
from functools import reduce

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from mudata import MuData


def cast_data(data):
    """Convert data to a nested dictionary of AnnData objects with group names as keys and views as subkeys.

    Parameters
    ----------
    data: dict | MuData
        Allowed input structures are:
        - MuData object (single group)
        - dict with view names as keys and AnnData objects as values (single group)
        - dict with view names as keys and torch.Tensor objects as values (single group)
        - dict with group names as keys and MuData objects as values (multiple groups)
        - Nested dict with group names as keys, view names as subkeys and AnnData objects as values (multiple groups)
        - Nested dict with group names as keys, view names as subkeys and torch.Tensor objects as values (multiple groups)

    Returns
    -------
    dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.
    """
    # single group cases
    if isinstance(data, MuData):
        data = {"group_1": {mod: data[mod] for mod in data.mod.keys()}}

    elif isinstance(data, dict) and all([isinstance(v, AnnData) for v in data.values()]):
        data = {"group_1": data.copy()}

    elif isinstance(data, dict) and all([isinstance(v, torch.Tensor) for v in data.values()]):
        data = {"group_1": {k: AnnData(X=v.numpy()) for k, v in data.items()}}

    # multiple groups cases
    elif isinstance(data, dict) and all([isinstance(v, MuData) for v in data.values()]):
        data = {k: {mod: v[mod] for mod in v.mod.keys()} for k, v in data.items()}

    elif (
        isinstance(data, dict)
        and all([isinstance(v, dict) for v in data.values()])
        and all([all([isinstance(vv, AnnData) for vv in v.values()]) for v in data.values()])
    ):
        pass

    elif (
        isinstance(data, dict)
        and all([isinstance(v, dict) for v in data.values()])
        and all([all([isinstance(vv, torch.Tensor) for vv in v.values()]) for v in data.values()])
    ):
        data = {k: {kk: AnnData(X=vv.numpy()) for kk, vv in v.items()} for k, v in data.items()}

    else:
        raise ValueError("Input data structure not recognized. Please refer to the documentation for allowed formats.")

    return data


def infer_likelihoods(data):
    """Infer likelihoods ("Normal", "Bernoulli", "BetaBinomial", "GammaPoisson") for each view
    based on the data distribution.

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


def validate_likelihoods(data, likelihoods):
    """Validate likelihoods ("Normal", "Bernoulli", "BetaBinomial", "GammaPoisson") for each view
    based on the data distribution.

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


def remove_constant_features(data, likelihoods):
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
    for k_groups, v_groups in data.items():
        for k_views, v_views in v_groups.items():
            v_views.X = v_views.X.astype(float)

            # compute variance of each feature
            variances = np.var(v_views.X, axis=0)
            variances[variances < 1e-16] = 0

            # create mask that is True where a feature is constant (has zero variance)
            drop_mask = variances == 0

            # the BetaBinomial likelihood is a special case because every feature occurs twice with different
            # suffixes and removal of one should also lead to removal of the other
            if likelihoods[k_views] == "BetaBinomial":
                # create DataFrame with indices of first and second occurence (columns) of every feature (rows)
                split_var_names = pd.Series(v_views.var_names).str.rsplit("_", n=1, expand=True)
                split_var_names.columns = ["base", "suffix"]
                suffix_indices = split_var_names.groupby("base").apply(
                    lambda x: pd.Series(x.index.values), include_groups=False
                )

                # if any of the two occurences of a feature is masked, mask the other one as well

                for var_name_base in suffix_indices.index:
                    ix = suffix_indices.loc[var_name_base].values
                    if drop_mask[ix].any():
                        drop_mask[ix] = True

            if drop_mask.any():
                dropped_features_names = list(v_views.var_names[drop_mask])
                data[k_groups][k_views] = v_views[:, ~drop_mask]

                print(f"- Removing constant features in {k_groups}/{k_views}.")
                print(f"  - Removed {len(dropped_features_names)} features: {','.join(dropped_features_names)}")

    return data


def get_feature_mean(data, likelihoods):
    """Compute the mean of each feature across all observations in a group. For BetaBinomial data, the mean is
    computed from the ratio of the first occurence of a feature to the sum of both occurences.

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
    means = {}
    for k_groups, v_groups in data.items():
        means[k_groups] = {}
        for k_views, v_views in v_groups.items():
            if likelihoods[k_views] in ["Normal", "Bernoulli", "GammaPoisson"]:
                means[k_groups][k_views] = np.nanmean(v_views.X, axis=0)

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
                means[k_groups][k_views] = np.nanmean(ratio, axis=0)

    return means


def center_data(data, likelihoods):
    """Center features to have zero mean in each group individually.

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
    for k_groups, v_groups in data.items():
        for k_views, v_views in v_groups.items():
            if likelihoods[k_views] == "Normal":
                print(f"- Centering {k_groups}/{k_views}...")
                mean_values = np.nanmean(v_views.X, axis=0)
                v_views.X -= mean_values

    return data


def scale_data(data, likelihoods, scale_per_group=True):
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
        for _, v_groups in data.items():
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

    return data


def align_obs(data, use_obs="union"):
    """Align observations across views. After alignment, each view will have the same set of observations
    (with nans for missings) and observations will be sorted alphabetically according to their obs_names.

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
            obs_names_intersection = pd.Series(reduce(np.intersect1d, [v.obs_names for v in data[k_group].values()]))
            # subset obs to intersection and sort by obs_names
            for k_view, v_view in data[k_group].items():
                data_aligned[k_group][k_view] = v_view[obs_names_intersection, :].copy()

        if use_obs == "union":
            # series of sorted obs_names that are present in any view
            obs_names_union = pd.Series(reduce(np.union1d, [v.obs_names for v in data[k_group].values()]))

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
                    expanded_obsm[obsm_k] = np.ones(shape=(len(obs_names_union), v_view.obsm[obsm_k].shape[1])) * np.nan
                    expanded_obsm[obsm_k][ix, :] = v_view.obsm[obsm_k]

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

    return data_aligned


def align_var(data, likelihoods, use_var="intersection"):
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
            )

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
            )

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
                    varm=expanded_varm,
                )

    return data_aligned


def get_sizes_and_names(data):
    """Validate alignment of observations and features across views and groups and
    get sizes and names of obs and vars.

    Parameters
    ----------
    data: dict
        Nested dictionary of AnnData objects with group names as keys and view names as subkeys.

    Returns
    -------
    group_names: list
        List of group names.
    n_groups: int
        Number of groups.
    view_names: list
        List of view names.
    n_views: int
        Number of views.
    feature_names: dict
        Dictionary with view names as keys and lists of feature names as values.
    n_features: dict
        Dictionary with view names as keys and number of features as values.
    sample_names: dict
        Dictionary with group names as keys and lists of sample names as values.
    n_samples: dict
        Dictionary with group names as keys and number of samples as values.
    """
    group_names = []
    view_names = []
    feature_names = {}
    n_features = {}
    sample_names = {}
    n_samples = {}

    for k_groups, v_groups in data.items():
        group_names.append(k_groups)
        group_sample_names = None
        for k_views, v_views in v_groups.items():
            view_names.append(k_views)

            if group_sample_names is None:
                group_sample_names = v_views.obs_names.tolist()
            if group_sample_names != v_views.obs_names.tolist():
                raise ValueError("Error in preprocessing. Views do not have the same samples.")

            if k_views not in feature_names:
                feature_names[k_views] = v_views.var_names.tolist()
            if feature_names[k_views] != v_views.var_names.tolist():
                raise ValueError("Error in preprocessing. Groups do not have the same features.")

            # check whether every feature appears twice with different suffix
            var_names_base = pd.Series(v_views.var_names).str.rsplit("_", n=1, expand=True)[0]

            if var_names_base.nunique() * 2 == len(var_names_base):
                n_features[k_views] = len(feature_names[k_views]) // 2
                feature_names[k_views] = var_names_base.unique()

            else:
                n_features[k_views] = len(feature_names[k_views])

        sample_names[k_groups] = group_sample_names
        n_samples[k_groups] = len(group_sample_names)

        n_groups = len(group_names)
        n_views = len(view_names)

    return group_names, n_groups, view_names, n_views, feature_names, n_features, sample_names, n_samples
