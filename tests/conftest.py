import warnings
from pathlib import Path

import mudata as md
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import mofaflex as mfl


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def random_array(rng):
    def _arr(likelihood, shape):
        match likelihood:
            case "Normal":
                arr = rng.normal(size=shape)
            case "Bernoulli":
                arr = rng.binomial(1, 0.5, size=shape)
            case "NegativeBinomial":
                arr = rng.negative_binomial(10, 0.9, size=shape)
        return arr

    return _arr


@pytest.fixture(scope="session")
def create_adata():
    def _adata(X, var_names=None, obs_names=None):
        if var_names is None:
            var_names = [f"var{i}" for i in range(X.shape[1])]
        if obs_names is None:
            obs_names = [f"obs{i}" for i in range(X.shape[0])]
        return AnnData(X, var=pd.DataFrame(index=var_names), obs=pd.DataFrame(index=obs_names))

    return _adata


@pytest.fixture(scope="module")
def random_adata(rng, random_array):
    def _adata(likelihood, nobs, nvar, var_names=None, obs_names=None):
        adata = AnnData(
            X=random_array(likelihood, (nobs, nvar)).astype(np.float32),
            obs=pd.DataFrame(
                {"covar": rng.random(size=nobs)},
                index=obs_names if obs_names is not None else [f"cell_{i}" for i in range(nobs)],
            ),
            var=pd.DataFrame(index=var_names if var_names is not None else [f"gene_{i}" for i in range(nvar)]),
        )
        adata.obsm["covar"] = pd.DataFrame(rng.random(size=(nobs, 3)), columns=["a", "b", "c"], index=adata.obs_names)
        adata.obs["gvar_normal"] = pd.DataFrame(rng.random(size=(nobs)), index=adata.obs_names)
        adata.obs["gvar_bernoulli"] = pd.DataFrame(rng.binomial(1, 0.5, size=(nobs)), index=adata.obs_names)
        adata.obs["gvar_categorical"] = pd.DataFrame(rng.choice(["A", "B", "C"], size=(nobs)), index=adata.obs_names)
        adata.varm["annot"] = pd.DataFrame(
            rng.choice([False, True], size=(nvar, 10)), columns=[f"annot_{i}" for i in range(10)], index=adata.var_names
        )
        return adata

    return _adata


@pytest.fixture(scope="session")
def cll_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return md.read_h5mu(Path(__file__).parent / "data" / "cll.h5mu")


@pytest.fixture(scope="session")
def cll_model():
    return mfl.MOFAFLEX.load(Path(__file__).parent / "data" / "cll_model.h5", map_location="cpu")


@pytest.fixture(scope="session")
def mousebrain_model():
    return mfl.MOFAFLEX.load(Path(__file__).parent / "data" / "mousebrain_model.h5", map_location="cpu")
