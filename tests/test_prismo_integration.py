# integration tests: only testing if the code runs without errors
import warnings

import numpy as np
import pytest
from scipy.sparse import SparseEfficiencyWarning, csc_array, csc_matrix, csr_array, csr_matrix, issparse

from prismo import PRISMO, DataOptions, ModelOptions, SmoothOptions, TrainingOptions


@pytest.fixture(scope="module")
def anndata_dict(random_adata, rng):
    big_adatas = (
        random_adata("Normal", 500, 100, var_names=[f"normal_var_{i}" for i in range(100)]),
        random_adata("Bernoulli", 400, 200, var_names=[f"bernoulli_var_{i}" for i in range(200)]),
        random_adata("GammaPoisson", 600, 90, var_names=[f"gammapoisson_var_{i}" for i in range(90)]),
    )

    group_idxs = []
    for adata in big_adatas:
        permuted = rng.permutation(range(adata.n_obs))
        group_size = rng.choice(adata.n_obs)
        group_idxs.append((permuted[:group_size], permuted[group_size:]))

    adata_dict = {"group_1": {}, "group_2": {}}
    for view_name, (view_idx, view) in zip(
        ("view_normal", "view_bernoulli", "view_gammapoisson"), enumerate(big_adatas), strict=False
    ):
        for group_idx, group in enumerate(adata_dict.values()):
            idx = rng.choice(adata.n_vars, size=int(0.9 * adata.n_vars), replace=False)
            group[view_name] = view[group_idxs[view_idx][group_idx], idx].copy()

    adata_dict["group_1"]["view_bernoulli"].X = csr_array(adata_dict["group_1"]["view_bernoulli"].X)
    adata_dict["group_1"]["view_gammapoisson"].X = csc_array(adata_dict["group_1"]["view_gammapoisson"].X)
    adata_dict["group_2"]["view_bernoulli"].X = csr_matrix(adata_dict["group_2"]["view_bernoulli"].X)
    adata_dict["group_2"]["view_gammapoisson"].X = csc_matrix(adata_dict["group_2"]["view_gammapoisson"].X)

    return adata_dict


@pytest.mark.parametrize(
    "attrname,attrvalue",
    [
        ("scale_per_group", False),
        ("scale_per_group", True),
        ("annotations_varm_key", None),
        ("annotations_varm_key", "annot"),
        ("covariates_obs_key", None),
        ("covariates_obs_key", "covar"),
        ("covariates_obsm_key", None),
        ("covariates_obsm_key", "covar"),
        ("use_obs", "union"),
        ("use_obs", "intersection"),
        ("use_var", "union"),
        ("use_var", "intersection"),
        ("remove_constant_features", True),
        ("remove_constant_features", False),
        ("weight_prior", "Normal"),
        ("weight_prior", "Laplace"),
        ("weight_prior", "Horseshoe"),
        ("weight_prior", "SnS"),
        ("factor_prior", "Normal"),
        ("factor_prior", "Laplace"),
        ("factor_prior", "Horseshoe"),
        ("factor_prior", "SnS"),
        ("nonnegative_weights", False),
        ("nonnegative_weights", True),
        ("nonnegative_factors", False),
        ("nonnegative_factors", True),
        ("init_factors", "random"),
        ("init_factors", "orthogonal"),
        ("init_factors", "pca"),
        ("save_path", None),
        ("save_path", "test.h5"),
        ("batch_size", 0),
        ("batch_size", 100),
    ],
)
def test_integration(anndata_dict, tmp_path, attrname, attrvalue):
    opts = (
        DataOptions(plot_data_overview=False, annotations_varm_key="annot"),
        ModelOptions(n_factors=5),
        TrainingOptions(max_epochs=2, seed=42, save_path=False),
    )
    if attrname == "save_path" and isinstance(attrvalue, str):
        attrvalue = str(tmp_path / attrvalue)
    for opt in opts:
        if hasattr(opt, attrname):
            setattr(opt, attrname, attrvalue)

    model = PRISMO(anndata_dict, *opts)  # noqa F841


@pytest.mark.parametrize(
    "attrname,attrvalue",
    [
        ("kernel", "RBF"),
        ("kernel", "Matern"),
        ("mefisto_kernel", True),
        ("mefisto_kernel", False),
        ("warp_groups", []),
        ("warp_groups", ["group_1"]),
    ],
)
def test_integration_gp(anndata_dict, attrname, attrvalue):
    opts = (
        DataOptions(covariates_obs_key="covar", plot_data_overview=False),
        ModelOptions(n_factors=5, factor_prior="GP"),
        TrainingOptions(max_epochs=2, seed=42, save_path=False),
    )
    smooth_opts = SmoothOptions(n_inducing=20, warp_interval=1)
    setattr(smooth_opts, attrname, attrvalue)

    model = PRISMO(anndata_dict, *opts, smooth_opts)  # noqa F841


def test_imputation(rng, anndata_dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SparseEfficiencyWarning)

        nanidx = {}
        for group_name, group in anndata_dict.items():
            del group["view_gammapoisson"]
            cnanidx = {}
            for view_name, view in group.items():
                n_nans = rng.choice(int(0.05 * view.n_obs * view.n_vars))
                rowidx = rng.choice(view.n_obs, size=n_nans)
                colidx = rng.choice(view.n_vars, size=n_nans)

                view.X[rowidx, colidx] = np.nan
                cnanidx[view_name] = (rowidx, colidx)
            nanidx[group_name] = cnanidx

    model = PRISMO(
        anndata_dict,
        DataOptions(plot_data_overview=False),
        ModelOptions(n_factors=5),
        TrainingOptions(max_epochs=2, seed=42, save_path=False),
    )

    imputed = model.impute_data(anndata_dict, missing_only=False)
    for group in imputed.values():
        for view in group.values():
            assert np.isnan(view.X if not issparse(view.X) else view.X.data).sum() == 0

    imputed = model.impute_data(anndata_dict, missing_only=True)
    preprocessor = model._prismodataset(anndata_dict).preprocessor
    for group_name, group in imputed.items():
        for view_name, view in group.items():
            assert np.isnan(view.X if not issparse(view.X) else view.X.data).sum() == 0

            orig_data = anndata_dict[group_name][view_name]
            new_X = view[orig_data.obs_names, orig_data.var_names].X
            orig_X = orig_data.X
            if issparse(orig_X):
                orig_X = orig_X.toarray()
            if issparse(new_X):
                new_X = new_X.toarray()
            nonnan = ~np.isnan(orig_X)
            assert np.allclose(
                preprocessor(orig_X, slice(None), slice(None), group_name, view_name)[0][nonnan], new_X[nonnan]
            )
