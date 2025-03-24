import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def create_adata():
    def _adata(X, var_names=None, obs_names=None):
        if var_names is None:
            var_names = [f"var{i}" for i in range(X.shape[1])]
        if obs_names is None:
            obs_names = [f"obs{i}" for i in range(X.shape[0])]
        return AnnData(X, var=pd.DataFrame(index=var_names), obs=pd.DataFrame(index=obs_names))

    return _adata
