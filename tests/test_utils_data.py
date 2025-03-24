from math import isclose

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from .utils import preprocess


def create_adata(X, var_names=None, obs_names=None):
    if var_names is None:
        var_names = [f"var{i}" for i in range(X.shape[1])]
    if obs_names is None:
        obs_names = [f"obs{i}" for i in range(X.shape[0])]
    return AnnData(X, var=pd.DataFrame(index=var_names), obs=pd.DataFrame(index=obs_names))


@pytest.mark.filterwarnings("ignore:Observation names are not unique.+:UserWarning")
def test_get_data_mean():
    data = {
        "group1": {
            "view1": create_adata(np.array([[1.5, -5.2], [4.4, 1.1], [70.0, 9.1]])),
            "view2": create_adata(np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])),
            "view3": create_adata(np.array([[1.0, 10.0], [8.0, 2.0], [18.0, 12.0]])),
        },
        "group2": {
            "view1": create_adata(np.array([[1.5, -5.2], [4.4, 1.1], [70.0, 9.1]])),
            "view2": create_adata(np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])),
            "view3": create_adata(np.array([[1.0, 10.0], [8.0, 2.0], [18.0, 12.0]])),
        },
    }

    preprocessor = preprocess(data, {vn: "Normal" for vn in data["group1"].keys()})[0]

    feature_mean = preprocessor.feature_means

    assert isclose(feature_mean["group1"]["view1"][1], (-5.2 + 1.1 + 9.1) / 3)
    assert isclose(feature_mean["group2"]["view1"][1], (-5.2 + 1.1 + 9.1) / 3)
    assert isclose(feature_mean["group1"]["view2"][0], (1 + 0 + 0) / 3)
    assert isclose(feature_mean["group2"]["view2"][0], (1 + 0 + 0) / 3)
    assert isclose(feature_mean["group1"]["view3"][1], (10 + 2 + 12) / 3)
    assert isclose(feature_mean["group2"]["view3"][1], (10 + 2 + 12) / 3)


@pytest.mark.filterwarnings("ignore:Observation names are not unique.+:UserWarning")
def test_center_data():
    # create sample data set
    data = {
        "group1": {
            "view1": create_adata(np.random.randn(5, 3)),
            "view2": create_adata(np.random.randint(0, 1, (5, 3))),
            "view3": create_adata(np.random.randint(0, 1e4, (5, 3))),
        },
        "group2": {
            "view1": create_adata(np.random.randn(8, 3)),
            "view2": create_adata(np.random.randint(0, 1, (7, 5))),
            "view3": create_adata(np.random.randint(0, 1e4, (2, 8))),
        },
    }
    likelihoods = {"view1": "Normal", "view2": "Bernoulli", "view3": "GammaPoisson"}

    result = preprocess(data, likelihoods, cast_to=None)[1]

    assert np.allclose(result["group1"]["view1"].mean(axis=0), 0)
    assert np.allclose(result["group2"]["view1"].mean(axis=0), 0)


@pytest.mark.filterwarnings("ignore:Observation names are not unique.+:UserWarning")
def test_scale_data():
    data = {
        "group1": {
            "view1": create_adata(np.random.randn(5, 3)),
            "view2": create_adata(np.random.randint(0, 1, (5, 3))),
            "view3": create_adata(np.random.randint(0, 1e4, (5, 3))),
        },
        "group2": {
            "view1": create_adata(np.random.randn(5, 3)),
            "view2": create_adata(np.random.randint(0, 1, (5, 5))),
            "view3": create_adata(np.random.randint(0, 1e4, (5, 8))),
        },
    }

    likelihoods = {"view1": "Normal", "view2": "Bernoulli", "view3": "GammaPoisson"}

    result = preprocess(data, likelihoods, scale_per_group=True, cast_to=None)[1]

    assert np.allclose(result["group1"]["view1"].std(), 1)
    assert np.allclose(result["group2"]["view1"].std(), 1)

    result = preprocess(data, likelihoods, scale_per_group=False, cast_to=None)[1]

    combined_view1 = np.concatenate([result["group1"]["view1"], result["group2"]["view1"]], axis=0)
    assert np.allclose(combined_view1.std(), 1)
