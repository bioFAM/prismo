import anndata as ad
import numpy as np
import pandas as pd
import pytest

from .utils import preprocess


@pytest.mark.filterwarnings("ignore:Observation names are not unique.+:UserWarning")
def test_remove_constant_features_across_groups():
    X1 = np.array(
        [
            [1.0, 2.0, 3.0, 1.0],  # Second, and Last feature is constant
            [3.0, 2.0, 2.0, 1.0],
            [2.0, 2.0, 4.0, 1.0],
        ]
    )
    X2 = np.array(
        [
            [2.0, 2.0, 1.0, 5.0],  # Second feature is constant
            [3.0, 2.0, 2.0, 4.0],
            [1.0, 2.0, 3.0, 6.0],
        ]
    )

    adata1 = ad.AnnData(X1, var=pd.DataFrame(index=[f"gene{i}" for i in range(4)]))
    adata2 = ad.AnnData(X2, var=pd.DataFrame(index=[f"gene{i}" for i in range(4)]))
    data = {"group1": {"view1": adata1}, "group2": {"view1": adata2}}
    likelihoods = {"view1": "GammaPoisson"}  # turn off scaling

    result = preprocess(data, likelihoods)[1]

    # Test that constant features were removed
    assert result["group1"]["view1"].shape[1] == 3  # One constant feature should be removed
    assert result["group2"]["view1"].shape[1] == 3  # One constant feature should be removed

    # Test that the correct features were kept
    assert not np.allclose(result["group1"]["view1"][:, 0], 1.0)  # Non-constant feature
    assert not np.allclose(result["group1"]["view1"][:, 1], 2.0)  # Non-constant feature
    assert not np.allclose(result["group1"]["view1"][:, 2], 3.0)  # Non-constant feature


def test_remove_constant_features_all_varying():
    # Test case where no features are constant
    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

    adata = ad.AnnData(X, var=pd.DataFrame(index=[f"gene{i}" for i in range(3)]))
    data = {"group1": {"view1": adata}}
    likelihoods = {"view1": "GammaPoisson"}  # turn off scaling

    result = preprocess(data, likelihoods)[1]

    # Test that no features were removed
    assert result["group1"]["view1"].shape[1] == 3
    assert np.array_equal(result["group1"]["view1"], X)


@pytest.mark.filterwarnings("ignore:Degrees of freedom.+:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in( scalar)? divide:RuntimeWarning")
def test_remove_constant_features_all_constant():
    # Test case where all features are constant
    X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    adata = ad.AnnData(X, var=pd.DataFrame(index=[f"gene{i}" for i in range(3)]))
    data = {"group1": {"view1": adata}}
    likelihoods = {"view1": "Normal"}

    result = preprocess(data, likelihoods)[1]

    # Test that all features were removed
    assert result["group1"]["view1"].shape[1] == 0


def test_remove_constant_features_multiple_views():
    # Test case with multiple views per group
    X1 = np.array(
        [
            [1.0, 2.0],  # First feature constant
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    X2 = np.array(
        [
            [1.0, 2.0],  # Second feature constant
            [2.0, 2.0],
            [3.0, 2.0],
        ]
    )

    adata1 = ad.AnnData(X1, var=pd.DataFrame(index=[f"gene{i}" for i in range(2)]))
    adata2 = ad.AnnData(X2, var=pd.DataFrame(index=[f"gene{i}" for i in range(2)]))

    data = {"group1": {"view1": adata1, "view2": adata2}}
    likelihoods = {"view1": "GammaPoisson", "view2": "GammaPoisson"}  # turn off scaling

    result = preprocess(data, likelihoods)[1]

    # Test that constant features were removed from each view
    assert result["group1"]["view1"].shape[1] == 1
    assert result["group1"]["view2"].shape[1] == 1

    # Test that the correct features were kept
    assert not np.allclose(result["group1"]["view1"][:, 0], 2.0)
    assert not np.allclose(result["group1"]["view2"][:, 0], 1.0)
