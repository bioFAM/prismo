from collections import Counter, defaultdict
from functools import reduce
from math import isclose

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from prismo._core import PrismoDataset, preprocessing

from .utils import preprocess


def create_adata(X, var_names=None, obs_names=None):
    if var_names is None:
        var_names = [f"var{i}" for i in range(X.shape[1])]
    if obs_names is None:
        obs_names = [f"obs{i}" for i in range(X.shape[0])]
    return AnnData(X, var=pd.DataFrame(index=var_names), obs=pd.DataFrame(index=obs_names))


def test_infer_likelihoods():
    data = {
        "view1": create_adata(np.random.randn(2, 2)),
        "view2": create_adata(np.random.randint(0, 1, (2, 2))),
        "view3": create_adata(np.random.randint(0, 100, (2, 2))),
    }

    likelihoods = {vn: preprocessing.infer_likelihood(v) for vn, v in data.items()}

    assert likelihoods["view1"] == "Normal"
    assert likelihoods["view2"] == "Bernoulli"
    assert likelihoods["view3"] == "GammaPoisson"


def test_validate_likelihoods():
    data = {
        "view1": create_adata(np.random.randn(2, 2)),
        "view2": create_adata(np.random.randint(0, 1, (2, 2))),
        "view3": create_adata(np.random.randint(0, 100, (2, 2))),
    }

    likelihoods = {"view1": "Normal", "view2": "Bernoulli", "view3": "GammaPoisson"}

    for view_name in data.keys():
        preprocessing.validate_likelihood(data[view_name], None, view_name, likelihoods[view_name])


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


def test_align_obs_var():
    obs_names = {
        "group1": {"view1": ["b", "c", "d", "e"], "view2": ["d", "e", "f"], "view3": ["a", "b", "c", "d"]},
        "group2": {"view1": ["h", "i", "j", "k"], "view2": ["j", "k", "l"], "view3": ["k", "l", "m"]},
    }

    var_names = {
        "view1": {"group1": ["g1", "g2"], "group2": ["g1", "g2", "g3"]},
        "view2": {"group1": ["g3", "g4"], "group2": ["g3", "g4", "g5"]},
        "view3": {"group1": ["g5", "g6"], "group2": ["g6", "g7", "g8"]},
    }

    data = {}
    obs_histogram = {}
    var_histogram = defaultdict(Counter)
    for group_name, gobs_names in obs_names.items():
        gdata = {}
        ghist = Counter()
        for view_name, vvar_names in var_names.items():
            cobsnames = gobs_names[view_name]
            cvarnames = vvar_names[group_name]
            gdata[view_name] = create_adata(
                np.random.randn(len(cobsnames), len(cvarnames)), obs_names=cobsnames, var_names=cvarnames
            )

            ghist.update(cobsnames)
            var_histogram[view_name].update(cvarnames)
        data[group_name] = gdata
        obs_histogram[group_name] = ghist

    dataset = PrismoDataset(data, use_obs="union", use_var="union")

    nobs_histogram = {group_name: Counter(ghist.values()) for group_name, ghist in obs_histogram.items()}
    nvar_histogram = {view_name: Counter(vhist.values()) for view_name, vhist in var_histogram.items()}

    aligned_obs = {
        group_name: np.zeros(
            len(reduce(lambda x, y: x | y, (set(names) for names in group.values()), set())), dtype=int
        )
        for group_name, group in obs_names.items()
    }
    aligned_var = {
        view_name: np.zeros(len(reduce(lambda x, y: x | y, (set(names) for names in view.values()), set())), dtype=int)
        for view_name, view in var_names.items()
    }

    obs_mapping = {}
    for group_name, group in obs_names.items():
        galigned = aligned_obs[group_name]
        cobsmapping = {}
        for view_name, view in group.items():
            arr = np.ones(len(view), dtype=int)
            aligned = dataset.align_local_array_to_global(
                arr, group_name, view_name, align_to="samples", axis=0, fill_value=0
            )
            cobsmapping[view_name] = np.nonzero(aligned)[0]
            galigned += aligned
        obs_mapping[group_name] = cobsmapping

    for group_name, galigned in aligned_obs.items():
        assert np.sum(galigned == 0) == 0
        for n, count in nobs_histogram[group_name].items():
            assert np.sum(galigned == n) == count

    for group_name, group in obs_names.items():
        arr = np.arange(len(obs_histogram[group_name]))
        for view_name, view in group.items():
            aligned_arr = dataset.align_global_array_to_local(arr, group_name, view_name, align_to="samples", axis=0)
            assert aligned_arr.size == len(view)
            assert np.all(aligned_arr == arr[obs_mapping[group_name][view_name]])

    var_mapping = {}
    for view_name, view in var_names.items():
        valigned = aligned_var[view_name]
        cvarmapping = {}
        for group_name, group in view.items():
            arr = np.ones(len(group), dtype=int)
            aligned = dataset.align_local_array_to_global(
                arr, group_name, view_name, align_to="features", axis=0, fill_value=0
            )
            cvarmapping[group_name] = np.nonzero(aligned)[0]
            valigned += aligned
        var_mapping[view_name] = cvarmapping

    for view_name, valigned in aligned_var.items():
        assert np.sum(valigned == 0) == 0
        for n, count in nvar_histogram[view_name].items():
            assert np.sum(valigned == n) == count

    for view_name, view in var_names.items():
        arr = np.arange(len(var_histogram[view_name]))
        for group_name, group in view.items():
            aligned_arr = dataset.align_global_array_to_local(arr, group_name, view_name, align_to="features", axis=0)
            assert aligned_arr.size == len(group)
            assert np.all(aligned_arr == arr[var_mapping[view_name][group_name]])
