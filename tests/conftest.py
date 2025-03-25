import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def random_array(rng):
    def _arr(likelihood, shape):
        match likelihood:
            case "Normal":
                arr = rng.normal(size=(100, 30))
            case "Bernoulli":
                arr = rng.binomial(1, 0.5, size=(100, 30))
            case "GammaPoisson":
                arr = rng.negative_binomial(10, 0.9, size=(100, 30))
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
