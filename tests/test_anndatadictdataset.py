from functools import reduce

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from prismo._core.datasets import AnnDataDictDataset, PrismoDataset


@pytest.fixture(scope="module")
def anndata_dict(random_adata, rng):
    big_adata = random_adata("Normal", 500, 100)
    permuted = rng.permutation(range(500))
    group1_size = rng.choice(500)
    group_idxs = (permuted[:group1_size], permuted[group1_size:])

    permuted = rng.permutation(range(100))
    view1_size = rng.choice(100)
    view_idxs = permuted[:view1_size], permuted[view1_size:]

    adata_dict = {}

    for group_name, group_idx in enumerate(group_idxs):
        group = {}
        for view_name, view_idx in enumerate(view_idxs):
            cgroup_idx = rng.choice(group_idx, size=int(0.8 * group_idx.size), replace=False)
            cview_idx = rng.choice(view_idx, size=int(0.8 * view_idx.size), replace=False)

            group[f"view_{view_name}"] = big_adata[cgroup_idx, cview_idx].copy()
        adata_dict[f"group_{group_name}"] = group

    adata_dict["group_0"]["view_0"].X = sparse.csr_array(adata_dict["group_0"]["view_0"].X)
    adata_dict["group_1"]["view_1"].X = sparse.csc_array(adata_dict["group_1"]["view_1"].X)

    return adata_dict


@pytest.fixture(scope="module", params=("union", "intersection"))
def use_obs(request):
    return request.param


@pytest.fixture(scope="module", params=("union", "intersection"))
def use_var(request):
    return request.param


@pytest.fixture(scope="module")
def dataset(anndata_dict, use_obs, use_var):
    return PrismoDataset(anndata_dict, use_obs=use_obs, use_var=use_var, cast_to=np.float32)


def test_dataset(dataset):
    assert isinstance(dataset, AnnDataDictDataset)


def test_properties(anndata_dict, use_obs, use_var, dataset):
    obs_func = (lambda x, y: x.union(y)) if use_obs == "union" else lambda x, y: x.intersection(y)
    var_func = (lambda x, y: x.union(y)) if use_var == "union" else lambda x, y: x.intersection(y)

    obs_names = {
        group_name: reduce(obs_func, (view.obs_names for view in group.values()))
        for group_name, group in anndata_dict.items()
    }

    var_names = {}
    for group in anndata_dict.values():
        for view_name, view in group.items():
            if view_name not in var_names:
                var_names[view_name] = view.var_names
            else:
                var_names[view_name] = var_func(var_names[view_name], view.var_names)

    for group_name, group_obs in obs_names.items():
        assert dataset.n_samples[group_name] == group_obs.size
        assert np.all(np.sort(dataset.sample_names[group_name]) == group_obs.sort_values().to_numpy())

    for view_name, view_vars in var_names.items():
        assert dataset.n_features[view_name] == view_vars.size
        assert np.all(np.sort(dataset.feature_names[view_name]) == view_vars.sort_values().to_numpy())


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_alignment(anndata_dict, dataset, rng, axis):
    ndim = 3

    arr_shape = np.asarray([2] * ndim)

    for group_name, group_samples in dataset.sample_names.items():
        arr_shape[axis] = group_samples.size
        global_arr = rng.random(size=arr_shape)
        for view_name in dataset.view_names:
            local_arr = dataset.align_global_array_to_local(
                global_arr, group_name, view_name, align_to="samples", axis=axis
            )
            new_global_arr = dataset.align_local_array_to_global(
                local_arr, group_name, view_name, align_to="samples", axis=axis, fill_value=np.nan
            )
            new_local_arr = dataset.align_global_array_to_local(
                new_global_arr, group_name, view_name, align_to="samples", axis=axis
            )

            assert new_global_arr.shape == global_arr.shape
            assert new_local_arr.shape == local_arr.shape
            assert np.all(new_local_arr == local_arr)

            local_obsnames = anndata_dict[group_name][view_name].obs_names.intersection(group_samples, sort=False)
            idx = pd.Index(group_samples).get_indexer(local_obsnames)
            assert np.all(local_arr == np.take(global_arr, idx, axis=axis))

            idx = np.isin(group_samples, local_obsnames)
            assert np.all(np.isnan(np.compress(~idx, new_global_arr, axis=axis)))
            assert np.all(np.compress(idx, new_global_arr, axis=axis) == np.compress(idx, global_arr, axis=axis))

    for view_name, view_features in dataset.feature_names.items():
        arr_shape[axis] = view_features.size
        global_arr = rng.random(size=arr_shape)

        for group_name in dataset.group_names:
            local_arr = dataset.align_global_array_to_local(
                global_arr, group_name, view_name, align_to="features", axis=axis
            )
            new_global_arr = dataset.align_local_array_to_global(
                local_arr, group_name, view_name, align_to="features", axis=axis, fill_value=np.nan
            )
            new_local_arr = dataset.align_global_array_to_local(
                new_global_arr, group_name, view_name, align_to="featurs", axis=axis
            )

            assert new_local_arr.shape == local_arr.shape
            assert np.all(new_local_arr == local_arr)
            assert np.isnan(new_global_arr).sum() == global_arr.size - local_arr.size

            local_varnames = anndata_dict[group_name][view_name].var_names.intersection(view_features, sort=False)
            idx = pd.Index(view_features).get_indexer(local_varnames)
            assert np.all(local_arr == np.take(global_arr, idx, axis=axis))

            idx = np.isin(view_features, local_varnames)
            assert np.all(np.isnan(np.compress(~idx, new_global_arr, axis=axis)))
            assert np.all(np.compress(idx, new_global_arr, axis=axis) == np.compress(idx, global_arr, axis=axis))


def test_index_mapping(anndata_dict, dataset, rng):
    for group_name, group_samples in dataset.sample_names.items():
        global_idx = rng.choice(group_samples.size, size=int(0.3 * group_samples.size), replace=True)
        for view_name in dataset.view_names:
            local_idx = dataset.map_global_indices_to_local(global_idx, group_name, view_name, align_to="samples")

            local_obsnames = anndata_dict[group_name][view_name].obs_names.intersection(group_samples, sort=False)
            assert np.all(group_samples[global_idx][local_idx >= 0] == local_obsnames[local_idx[local_idx >= 0]])
            assert np.all(~np.isin(group_samples[global_idx][local_idx < 0], local_obsnames))

            new_global_idx = dataset.map_local_indices_to_global(
                local_idx[local_idx >= 0], group_name, view_name, align_to="samples"
            )
            assert np.all(global_idx[local_idx >= 0] == new_global_idx)

    for view_name, view_features in dataset.feature_names.items():
        global_idx = rng.choice(view_features.size, size=int(0.3 * view_features.size), replace=True)
        for group_name in dataset.group_names:
            local_idx = dataset.map_global_indices_to_local(global_idx, group_name, view_name, align_to="features")

            local_varnames = anndata_dict[group_name][view_name].var_names.intersection(view_features, sort=False)
            assert np.all(view_features[global_idx][local_idx >= 0] == local_varnames[local_idx[local_idx >= 0]])
            assert np.all(~np.isin(view_features[global_idx][local_idx < 0], local_varnames))

            new_global_idx = dataset.map_local_indices_to_global(
                local_idx[local_idx >= 0], group_name, view_name, align_to="features"
            )
            assert np.all(global_idx[local_idx >= 0] == new_global_idx)


def test_getitems(anndata_dict, dataset, rng):
    idx = {
        group_name: rng.choice(sample_names.size, size=sample_names.size // 3, replace=False)
        for group_name, sample_names in dataset.sample_names.items()
    }

    items = dataset.__getitems__(idx)
    for group_name, group in items["data"].items():
        sample_names = dataset.sample_names[group_name][idx[group_name]]
        assert np.all(items["sample_idx"][group_name] == idx[group_name])
        for view_name, view in group.items():
            assert type(view) is np.ndarray
            assert view.dtype == np.float32

            feature_names = dataset.feature_names[view_name]
            cadata = anndata_dict[group_name][view_name]

            cobsidx = np.isin(sample_names, cadata.obs_names)
            cvaridx = np.isin(feature_names, cadata.var_names)

            cobs = sample_names[cobsidx]
            cvar = feature_names[cvaridx]
            assert np.all(cadata[cobs, cvar].X == view)

            cnonmissing_obs = np.nonzero(cobsidx)[0]
            cnonmissing_var = np.nonzero(cvaridx)[0]
            assert np.all(items["nonmissing_samples"][group_name][view_name] == cnonmissing_obs)
            assert np.all(items["nonmissing_features"][group_name][view_name] == cnonmissing_var)


def test_apply_by_group_view(anndata_dict, dataset):
    def applyfun(adata, group_name, view_name, ref_adata, ref_sample_names, ref_feature_names):
        assert np.all(adata.obs_names == ref_adata.obs_names.intersection(ref_sample_names))
        assert np.all(adata.var_names == ref_adata.var_names.intersection(ref_feature_names))

    dataset.apply(
        applyfun,
        group_kwargs={"ref_sample_names": dataset.sample_names},
        view_kwargs={"ref_feature_names": dataset.feature_names},
        group_view_kwargs={"ref_adata": anndata_dict},
    )


def test_apply_by_view(anndata_dict, dataset):
    def applyfun(adata, group_name, view_name):
        view_obs = reduce(lambda x, y: x.union(y), (group[view_name].obs_names for group in anndata_dict.values()))
        view_obs = view_obs.intersection(np.concatenate(list(dataset.sample_names.values())))

        assert np.all(np.sort(adata.obs_names) == view_obs.sort_values())
        assert np.all(adata.var_names == dataset.feature_names[view_name])

    dataset.apply(applyfun, by_group=False)


def test_apply_by_group(anndata_dict, dataset):
    def applyfun(adata, group_name, view_name):
        group_var = reduce(lambda x, y: x.union(y), (view.var_names for view in anndata_dict[group_name].values()))
        group_var = group_var.intersection(np.concatenate(list(dataset.feature_names.values())))

        assert np.all(adata.obs_names == dataset.sample_names[group_name])
        assert np.all(np.sort(adata.var_names) == group_var.sort_values())

    dataset.apply(applyfun, by_view=False)


def test_get_covariates_from_obs(anndata_dict, dataset):
    covars, covar_names = dataset.get_covariates(obs_key=dict.fromkeys(dataset.group_names, "covar"))

    for group_name, group in anndata_dict.items():
        assert covar_names[group_name] == "covar"
        for view_name, view in group.items():
            sample_names = dataset.sample_names[group_name]
            globalidx = np.isin(sample_names, view.obs_names)
            localidx = view.obs_names.get_indexer(sample_names)
            localidx = localidx[localidx >= 0]

            assert np.all(covars[group_name][view_name][globalidx].squeeze() == view.obs["covar"].to_numpy()[localidx])
            assert np.all(np.isnan(covars[group_name][view_name][~globalidx]))


def test_get_covariates_from_obsm(anndata_dict, dataset):
    covars, covar_names = dataset.get_covariates(obsm_key=dict.fromkeys(dataset.group_names, "covar"))

    for group_name, group in anndata_dict.items():
        assert np.all(covar_names[group_name] == ["a", "b", "c"])
        for view_name, view in group.items():
            sample_names = dataset.sample_names[group_name]
            globalidx = np.isin(sample_names, view.obs_names)
            localidx = view.obs_names.get_indexer(sample_names)
            localidx = localidx[localidx >= 0]

            assert np.all(covars[group_name][view_name][globalidx, :] == view.obsm["covar"].to_numpy()[localidx, :])
            assert np.all(np.isnan(covars[group_name][view_name][~globalidx, :]))


def test_get_annotations(anndata_dict, dataset):
    annot, annot_names = dataset.get_annotations(varm_key=dict.fromkeys(dataset.view_names, "annot"))

    for view_name in dataset.view_names:
        assert np.all(annot_names[view_name] == [f"annot_{i}" for i in range(10)])
        feature_names = dataset.feature_names[view_name]
        for group in anndata_dict.values():
            view = group[view_name]
            globalidx = np.isin(feature_names, view.var_names)
            localidx = view.var_names.get_indexer(feature_names)
            localidx = localidx[localidx >= 0]

            assert np.all(annot[view_name][:, globalidx] == view.varm["annot"].to_numpy()[localidx, :].T)


def test_get_missing_obs(anndata_dict, dataset):
    missing = dataset.get_missing_obs()
    for (group_name, view_name), df in missing.groupby(["group", "view"]):
        view = anndata_dict[group_name][view_name]
        cmissing = ~np.isin(dataset.sample_names[group_name], view.obs_names)
        df = df.set_index("obs_name")
        assert np.all(df.loc[dataset.sample_names[group_name], "missing"][cmissing])
        assert np.all(~df.loc[dataset.sample_names[group_name], "missing"][~cmissing])
