import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from scipy import sparse

from prismo._core.datasets import MuDataDataset, PrismoDataset


@pytest.fixture(scope="module")
def mdata(rng):
    nobs = 500
    nvar_per_mod = 20
    ngroups = 4

    obs_names = [f"cell_{i}" for i in range(nobs)]
    adatas = {}
    for view in range(3):
        cobs_names = rng.choice(obs_names, size=int(0.8 * nobs), replace=False)
        adata = ad.AnnData(
            X=rng.poisson(0.5, size=(len(cobs_names), nvar_per_mod)),
            obs=pd.DataFrame(index=cobs_names),
            var=pd.DataFrame(index=[f"mod_{view}_feature_{i}" for i in range(nvar_per_mod)]),
        )
        if view < 2:
            adata.obs["covar"] = rng.random(size=len(cobs_names))
            adata.obsm["covar"] = pd.DataFrame(
                rng.random(size=(len(cobs_names), 3)), columns=["a", "b", "c"], index=adata.obs_names
            )
            adata.varm["annot"] = pd.DataFrame(
                rng.random(size=(nvar_per_mod, 10)),
                columns=[f"annot_view_{view}_{i}" for i in range(10)],
                index=adata.var_names,
            )
        adatas[f"view_{view}"] = adata

    adatas["view_0"].X = sparse.csr_array(adatas["view_0"].X)
    adatas["view_2"].X = sparse.csc_array(adatas["view_2"].X)

    mdata = md.MuData(adatas)

    mdata.obs["batch"] = pd.Categorical(rng.choice(ngroups, size=mdata.n_obs).astype(str))

    global_covar = rng.random(size=mdata.n_obs)
    global_covar[~mdata.obs_names.isin(adatas["view_2"].obs_names)] = np.nan
    mdata.obs["covar"] = global_covar

    global_covar = rng.random(size=(mdata.n_obs, 3))
    global_covar[~mdata.obs_names.isin(adatas["view_2"].obs_names)] = np.nan
    mdata.obsm["covar"] = pd.DataFrame(global_covar, columns=["a", "b", "c"], index=mdata.obs_names)

    global_annot = rng.random(size=(mdata.n_vars, 10))
    global_annot[~mdata.var_names.isin(adatas["view_2"].var_names)] = np.nan
    mdata.varm["annot"] = pd.DataFrame(
        global_annot, columns=[f"global_annot_{i}" for i in range(10)], index=mdata.var_names
    )

    return mdata


@pytest.fixture(scope="module")
def dataset(mdata):
    return PrismoDataset(mdata, group_by="batch", cast_to=np.float32)


def test_instance(dataset):
    assert isinstance(dataset, MuDataDataset)


def test_properties(mdata, dataset):
    for group_name, sample_names in dataset.sample_names.items():
        cmdata = mdata[mdata.obs["batch"] == group_name, :]
        assert np.all(np.sort(sample_names) == cmdata.obs_names.sort_values().to_numpy())
        assert dataset.n_samples[group_name] == cmdata.n_obs

    for view_name, view_names in dataset.feature_names.items():
        assert np.all(view_names == mdata[view_name].var_names)
        assert dataset.n_features[view_name] == mdata[view_name].n_vars


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_alignment(mdata, dataset, rng, axis):
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

            mod = mdata[mdata.obs["batch"] == group_name, :][view_name]
            local_obsnames = group_samples[np.isin(group_samples, mod.obs_names)]
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
                new_global_arr, group_name, view_name, align_to="features", axis=axis
            )

            assert new_global_arr.shape == global_arr.shape
            assert new_local_arr.shape == local_arr.shape
            assert np.all(new_local_arr == local_arr)

            mod = mdata[mdata.obs["batch"] == group_name, :][view_name]
            local_varnames = view_features[np.isin(view_features, mod.var_names)]
            idx = pd.Index(view_features).get_indexer(local_varnames)
            assert np.all(local_arr == np.take(global_arr, idx, axis=axis))

            idx = np.isin(view_features, local_varnames)
            assert np.all(np.isnan(np.compress(~idx, new_global_arr, axis=axis)))
            assert np.all(np.compress(idx, new_global_arr, axis=axis) == np.compress(idx, global_arr, axis=axis))


def test_getitems(mdata, dataset, rng):
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
            cdata = mdata[sample_names, feature_names][view_name]
            assert np.all(cdata.X == view)

            cnonmissing_obs = np.nonzero(np.isin(sample_names, cdata.obs_names))[0]
            assert np.all(items["nonmissing_samples"][group_name][view_name] == cnonmissing_obs)
            assert items["nonmissing_features"][group_name][view_name] == slice(None)


def test_apply_by_group_view(mdata, dataset):
    def applyfun(adata, group_name, view_name, ref_sample_names, ref_feature_names):
        assert np.all(
            adata.obs_names == pd.Index(ref_sample_names).intersection(mdata[view_name].obs_names, sort=False)
        )
        assert np.all(adata.var_names == ref_feature_names)

    dataset.apply(
        applyfun,
        group_kwargs={"ref_sample_names": dataset.sample_names},
        view_kwargs={"ref_feature_names": dataset.feature_names},
    )


def test_apply_by_view(mdata, dataset):
    def applyfun(adata, group_name, view_name):
        assert np.all(adata.obs_names.sort_values() == mdata[view_name].obs_names.sort_values())
        assert np.all(adata.var_names == mdata[view_name].var_names)

    dataset.apply(applyfun, by_group=False)


@pytest.mark.xfail(
    Version(ad.__version__) < Version("0.11.4"), reason="anndata bug: https://github.com/scverse/anndata/pull/1911"
)
def test_apply_by_group(mdata, dataset):
    def applyfun(adata, group_name, view_name):
        assert np.all(adata.obs_names == dataset.sample_names[group_name])
        assert np.all(adata.var_names == mdata.var_names)

    dataset.apply(applyfun, by_view=False)


def test_get_covariates_from_obs(mdata, dataset):
    covars, covar_names = dataset.get_covariates(obs_key=dict.fromkeys(dataset.group_names, "covar"))

    for group_name, group_covar in covars.items():
        sample_names = pd.Index(dataset.sample_names[group_name])
        assert covar_names[group_name] == "covar"
        for view_name, view_covar in group_covar.items():
            if view_name != "view_2":
                view = mdata[mdata.obs["batch"] == group_name][view_name]
                covar = view.obs["covar"]
                idx = sample_names.get_indexer(view.obs_names)
                assert np.all(covar == view_covar[idx].squeeze())
                assert np.all(np.isnan(view_covar[~sample_names.isin(view.obs_names)]))
            else:
                covar = mdata.obs.loc[mdata.obs["batch"] == group_name, "covar"]
                nanidx = np.isnan(covar)
                idx = sample_names.get_indexer(covar.index)
                assert np.all(covar[~nanidx] == view_covar[idx][~nanidx].squeeze())
                assert np.all(np.isnan(view_covar[nanidx]))


def test_get_covariates_from_obsm(mdata, dataset):
    covars, covar_names = dataset.get_covariates(obsm_key=dict.fromkeys(dataset.group_names, "covar"))

    for group_name, group_covar in covars.items():
        assert np.all(covar_names[group_name] == ["a", "b", "c"])
        sample_names = pd.Index(dataset.sample_names[group_name])
        for view_name, view_covar in group_covar.items():
            if view_name != "view_2":
                view = mdata[mdata.obs["batch"] == group_name][view_name]
                covar = view.obsm["covar"]
                idx = sample_names.get_indexer(view.obs_names)
                assert np.all(covar == view_covar[idx].squeeze())
                assert np.all(np.isnan(view_covar[~sample_names.isin(view.obs_names)]))
            else:
                covar = mdata.obsm["covar"][mdata.obs["batch"] == group_name]
                idx = sample_names.get_indexer(covar.index)
                covar = covar.to_numpy()
                nanidx = np.isnan(covar)
                assert np.all(covar[~nanidx] == view_covar[idx][~nanidx].squeeze())
                assert np.all(np.isnan(view_covar[nanidx]))


def test_get_annotations(mdata, dataset):
    annot, annot_names = dataset.get_annotations(varm_key=dict.fromkeys(dataset.view_names, "annot"))

    for view_name in dataset.view_names:
        if view_name != "view_2":
            assert np.all(annot[view_name] == mdata[view_name].varm["annot"].to_numpy().T)
            assert np.all(annot_names[view_name] == [f"annot_{view_name}_{i}" for i in range(10)])
        else:
            assert np.all(annot[view_name] == mdata.varm["annot"].loc[dataset.feature_names[view_name], :].to_numpy().T)
            assert np.all(annot_names[view_name] == [f"global_annot_{i}" for i in range(10)])


def test_get_missing_obs(mdata, dataset):
    missing = dataset.get_missing_obs()
    for (group_name, view_name), df in missing.groupby(["group", "view"]):
        view = mdata[mdata.obs["batch"] == group_name, :][view_name]
        cmissing = ~np.isin(dataset.sample_names[group_name], view.obs_names)
        df = df.set_index("obs_name")
        assert np.all(df.loc[dataset.sample_names[group_name], "missing"][cmissing])
        assert np.all(~df.loc[dataset.sample_names[group_name], "missing"][~cmissing])
