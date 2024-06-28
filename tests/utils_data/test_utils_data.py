from math import isclose

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from mudata import MuData

from famo import utils_data


def create_adata(X, var_names=None, obs_names=None):
    if var_names is None:
        var_names = [f"var{i}" for i in range(X.shape[1])]
    if obs_names is None:
        obs_names = [f"obs{i}" for i in range(X.shape[0])]
    return AnnData(
        X, var=pd.DataFrame(index=var_names), obs=pd.DataFrame(index=obs_names)
    )


def test_cast_data():
    datasets = []

    datasets.append(
        MuData(
            {
                "view1": create_adata(np.random.randn(2, 2), var_names=["g1", "g2"]),
                "view2": create_adata(np.random.randn(2, 2), var_names=["g3", "g4"]),
            }
        )
    )

    datasets.append(
        {
            "view1": create_adata(np.random.randn(2, 2)),
            "view2": create_adata(np.random.randn(2, 2)),
        }
    )

    datasets.append({"view1": torch.randn([2, 2]), "view2": torch.randn([2, 2])})

    datasets.append(
        {
            "group1": MuData(
                {
                    "view1": create_adata(
                        np.random.randn(2, 2), var_names=["g1", "g2"]
                    ),
                    "view2": create_adata(
                        np.random.randn(2, 2), var_names=["g3", "g4"]
                    ),
                }
            ),
            "group2": MuData(
                {
                    "view1": create_adata(
                        np.random.randn(2, 2), var_names=["g1", "g2"]
                    ),
                    "view2": create_adata(
                        np.random.randn(2, 2), var_names=["g3", "g4"]
                    ),
                }
            ),
        }
    )

    datasets.append(
        {
            "group1": {
                "view1": create_adata(np.random.randn(2, 2)),
                "view2": create_adata(np.random.randn(2, 2)),
            },
            "group2": {
                "view1": create_adata(np.random.randn(2, 2)),
                "view2": create_adata(np.random.randn(2, 2)),
            },
        }
    )

    datasets.append(
        {
            "group1": {"view1": torch.randn(5, 3), "view2": torch.randn(10, 4)},
            "group2": {"view1": torch.randn(6, 2), "view2": torch.randn(11, 4)},
        }
    )

    for data in datasets:
        data_c = utils_data.cast_data(data)
        assert isinstance(data_c, dict)
        assert all([isinstance(v, dict) for v in data_c.values()])
        assert all(
            [
                all([isinstance(vv, AnnData) for vv in v.values()])
                for v in data_c.values()
            ]
        )


def test_infer_likelihoods():
    data = {
        "view1": create_adata(np.random.randn(2, 2)),
        "view2": create_adata(np.random.randint(0, 1, (2, 2))),
        "view3": create_adata(np.random.randint(0, 100, (2, 2))),
        "view4": create_adata(
            np.random.randint(0, 100, (2, 6)),
            var_names=["g3_b", "g1_a", "g2_a", "g1_b", "g2_b", "g3_a"],
        ),
    }

    likelihoods = utils_data.infer_likelihoods(data)

    assert likelihoods["view1"] == "Normal"
    assert likelihoods["view2"] == "Bernoulli"
    assert likelihoods["view3"] == "GammaPoisson"
    assert likelihoods["view4"] == "BetaBinomial"


def test_validate_likelihoods():
    data = {
        "view1": create_adata(np.random.randn(2, 2)),
        "view2": create_adata(np.random.randint(0, 1, (2, 2))),
        "view3": create_adata(np.random.randint(0, 100, (2, 2))),
        "view4": create_adata(
            np.random.randint(0, 100, (2, 6)),
            var_names=["g3_b", "g1_a", "g2_a", "g1_b", "g2_b", "g3_a"],
        ),
    }

    likelihoods = {
        "view1": "Normal",
        "view2": "Bernoulli",
        "view3": "GammaPoisson",
        "view4": "BetaBinomial",
    }

    utils_data.validate_likelihoods(data, likelihoods)


def test_remove_constant_features():
    var_names_g1_v4 = ["g3_b", "g1_a", "g4_a", "g2_a", "g1_b", "g2_b", "g3_a", "g4_b"]
    var_names_g2_v4 = ["g5_b", "g1_a", "g4_a", "g2_a", "g1_b", "g2_b", "g5_a", "g4_b"]

    data = {
        "group1": {
            "view1": create_adata(np.random.randn(500, 5)),
            "view2": create_adata(np.random.randint(0, 1, (500, 5))),
            "view3": create_adata(np.random.randint(0, 100, (500, 5))),
            "view4": create_adata(
                np.random.randint(0, 100, (500, 8)), var_names=var_names_g1_v4
            ),
        },
        "group2": {
            "view1": create_adata(np.random.randn(500, 5)),
            "view2": create_adata(np.random.randint(0, 1, (500, 5))),
            "view3": create_adata(np.random.randint(0, 100, (500, 5))),
            "view4": create_adata(
                np.random.randint(0, 100, (500, 8)), var_names=var_names_g2_v4
            ),
        },
    }

    likelihoods = {
        "view1": "Normal",
        "view2": "Bernoulli",
        "view3": "GammaPoisson",
        "view4": "BetaBinomial",
    }

    data["group1"]["view1"][:, "var3"] = 2.2
    data["group2"]["view2"][:, "var2"] = 1
    data["group1"]["view3"][:, "var1"] = 8
    data["group2"]["view4"][:, "g4_a"] = 3

    data_c = utils_data.remove_constant_features(data, likelihoods)

    assert "var3" not in data_c["group1"]["view1"].var_names
    assert "var2" not in data_c["group2"]["view2"].var_names
    assert "var1" not in data_c["group1"]["view3"].var_names
    assert "g4_a" not in data_c["group2"]["view4"].var_names
    assert "g4_b" not in data_c["group2"]["view4"].var_names


def test_get_feature_mean():
    data = {
        "group1": {
            "view1": create_adata(np.array([[1.5, -5.2], [4.4, 1.1], [70.0, 9.1]])),
            "view2": create_adata(np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])),
            "view3": create_adata(np.array([[1.0, 10.0], [8.0, 2.0], [18.0, 12.0]])),
            "view4": create_adata(
                np.array(
                    [
                        [11.0, 5.0, 2.0, 2.0],
                        [3.0, 12.0, 18.0, 9.0],
                        [9.0, 8.0, 7.0, 6.0],
                    ]
                ),
                var_names=["g1_a", "g1_b", "g2_a", "g2_b"],
            ),
        },
        "group2": {
            "view1": create_adata(np.array([[1.5, -5.2], [4.4, 1.1], [70.0, 9.1]])),
            "view2": create_adata(np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])),
            "view3": create_adata(np.array([[1.0, 10.0], [8.0, 2.0], [18.0, 12.0]])),
            "view4": create_adata(
                np.array(
                    [
                        [11.0, 5.0, 2.0, 2.0],
                        [3.0, 12.0, 18.0, 9.0],
                        [9.0, 8.0, 7.0, 6.0],
                    ]
                ),
                var_names=["g1_a", "g1_b", "g2_a", "g2_b"],
            ),
        },
    }

    likelihoods = {
        "view1": "Normal",
        "view2": "Bernoulli",
        "view3": "GammaPoisson",
        "view4": "BetaBinomial",
    }

    feature_mean = utils_data.get_feature_mean(data, likelihoods)

    assert isclose(feature_mean["group1"]["view1"][1], (-5.2 + 1.1 + 9.1) / 3)
    assert isclose(feature_mean["group2"]["view1"][1], (-5.2 + 1.1 + 9.1) / 3)
    assert isclose(feature_mean["group1"]["view2"][0], (1 + 0 + 0) / 3)
    assert isclose(feature_mean["group2"]["view2"][0], (1 + 0 + 0) / 3)
    assert isclose(feature_mean["group1"]["view3"][1], (10 + 2 + 12) / 3)
    assert isclose(feature_mean["group2"]["view3"][1], (10 + 2 + 12) / 3)
    assert isclose(
        feature_mean["group1"]["view4"][0],
        ((11 / (5 + 11)) + (3 / (12 + 3)) + (9 / (8 + 9))) / 3,
        abs_tol=1e-3,
    )


def test_center_data():
    # create sample data set
    data = {
        "group1": {
            "view1": create_adata(np.random.randn(5, 3)),
            "view2": create_adata(np.random.randint(0, 1, (5, 3))),
            "view3": create_adata(np.random.randint(0, 1e4, (5, 3))),
            "view4": create_adata(
                np.random.randint(0, 1e3, (5, 6)),
                var_names=["g3_b", "g1_a", "g2_a", "g1_b", "g2_b", "g3_a"],
            ),
        },
        "group2": {
            "view1": create_adata(np.random.randn(8, 3)),
            "view2": create_adata(np.random.randint(0, 1, (7, 5))),
            "view3": create_adata(np.random.randint(0, 1e4, (2, 8))),
            "view4": create_adata(
                np.random.randint(0, 1e3, (9, 4)),
                var_names=["g1_a", "g2_a", "g1_b", "g2_b"],
            ),
        },
    }

    likelihoods = {
        "view1": "Normal",
        "view2": "Bernoulli",
        "view3": "GammaPoisson",
        "view4": "BetaBinomial",
    }

    data_c = utils_data.center_data(
        data, likelihoods, nmf={k: False for k in likelihoods}
    )

    assert np.allclose(data_c["group1"]["view1"].X.mean(axis=0), 0)
    assert np.allclose(data_c["group2"]["view1"].X.mean(axis=0), 0)


def test_scale_data():
    data = {
        "group1": {
            "view1": create_adata(np.random.randn(5, 3)),
            "view2": create_adata(np.random.randint(0, 1, (5, 3))),
            "view3": create_adata(np.random.randint(0, 1e4, (5, 3))),
            "view4": create_adata(
                np.random.randint(0, 1e3, (5, 6)),
                var_names=["g3_b", "g1_a", "g2_a", "g1_b", "g2_b", "g3_a"],
            ),
        },
        "group2": {
            "view1": create_adata(np.random.randn(5, 3)),
            "view2": create_adata(np.random.randint(0, 1, (5, 5))),
            "view3": create_adata(np.random.randint(0, 1e4, (5, 8))),
            "view4": create_adata(
                np.random.randint(0, 1e3, (5, 4)),
                var_names=["g1_a", "g2_a", "g1_b", "g2_b"],
            ),
        },
    }

    likelihoods = {
        "view1": "Normal",
        "view2": "Bernoulli",
        "view3": "GammaPoisson",
        "view4": "BetaBinomial",
    }

    data_c = utils_data.scale_data(data, scale_per_group=True, likelihoods=likelihoods)

    assert np.allclose(data_c["group1"]["view1"].X.std(), 1)
    assert np.allclose(data_c["group2"]["view1"].X.std(), 1)

    data_c = utils_data.scale_data(data, scale_per_group=False, likelihoods=likelihoods)

    combined_view1 = np.concatenate(
        [data_c["group1"]["view1"].X, data_c["group2"]["view1"].X], axis=0
    )
    assert np.allclose(combined_view1.std(), 1)


def test_align_obs_var():
    g1_v1_obs_names = ["b", "c", "d", "e"]
    g1_v2_obs_names = ["d", "e", "f"]
    g1_v3_obs_names = ["a", "b", "c", "d"]
    g1_v4_obs_names = ["c", "d"]

    g2_v1_obs_names = ["h", "i", "j", "k"]
    g2_v2_obs_names = ["j", "k", "l"]
    g2_v3_obs_names = ["k", "l", "m"]
    g2_v4_obs_names = ["h", "i"]

    g1_v1_var_names = ["g1", "g2"]
    g2_v1_var_names = ["g1", "g2", "g3"]

    g1_v2_var_names = ["g3", "g4"]
    g2_v2_var_names = ["g3", "g4", "g5"]

    g1_v3_var_names = ["g5", "g6"]
    g2_v3_var_names = ["g6", "g7", "g8"]

    g1_v4_var_names = ["g7_a", "g8_a", "g7_b", "g8_b"]
    g2_v4_var_names = ["g7_a", "g7_b", "g8_a", "g8_b", "g9_a", "g9_b"]

    data = {
        "group1": {
            "view1": create_adata(
                np.random.randn(
                    len(g1_v1_obs_names),
                    len(g1_v1_var_names),
                ),
                obs_names=g1_v1_obs_names,
                var_names=g1_v1_var_names,
            ),
            "view2": create_adata(
                np.random.randint(0, 1, (len(g1_v2_obs_names), len(g1_v2_var_names))),
                obs_names=g1_v2_obs_names,
                var_names=g1_v2_var_names,
            ),
            "view3": create_adata(
                np.random.randint(0, 100, (len(g1_v3_obs_names), len(g1_v3_var_names))),
                obs_names=g1_v3_obs_names,
                var_names=g1_v3_var_names,
            ),
            "view4": create_adata(
                np.random.randint(0, 100, (len(g1_v4_obs_names), len(g1_v4_var_names))),
                obs_names=g1_v4_obs_names,
                var_names=g1_v4_var_names,
            ),
        },
        "group2": {
            "view1": create_adata(
                np.random.randn(len(g2_v1_obs_names), len(g2_v1_var_names)),
                obs_names=g2_v1_obs_names,
                var_names=g2_v1_var_names,
            ),
            "view2": create_adata(
                np.random.randint(0, 1, (len(g2_v2_obs_names), len(g2_v2_var_names))),
                obs_names=g2_v2_obs_names,
                var_names=g2_v2_var_names,
            ),
            "view3": create_adata(
                np.random.randint(0, 100, (len(g2_v3_obs_names), len(g2_v3_var_names))),
                obs_names=g2_v3_obs_names,
                var_names=g2_v3_var_names,
            ),
            "view4": create_adata(
                np.random.randint(0, 100, (len(g2_v4_obs_names), len(g2_v4_var_names))),
                obs_names=g2_v4_obs_names,
                var_names=g2_v4_var_names,
            ),
        },
    }

    likelihoods = {
        "view1": "Normal",
        "view2": "Bernoulli",
        "view3": "GammaPoisson",
        "view4": "BetaBinomial",
    }

    data_c = utils_data.align_obs(data, use_obs="union")
    data_c = utils_data.align_var(data_c, use_var="union", likelihoods=likelihoods)

    feature_names = {}
    for k_groups, v_groups in data_c.items():
        group_obs_names = None
        for k_views, v_views in v_groups.items():
            if group_obs_names is None:
                group_obs_names = v_views.obs_names
            else:
                assert np.all(group_obs_names == v_views.obs_names.tolist())

            if k_views not in feature_names:
                feature_names[k_views] = v_views.var_names
            else:
                print(
                    k_groups,
                    k_views,
                    feature_names[k_views],
                    v_views.var_names.tolist(),
                )
                assert np.all(feature_names[k_views] == v_views.var_names.tolist())
