import string

import numpy as np
import pandas as pd
import pytest

from prismo import FeatureSet, FeatureSets


def random_strings(n, rng, min_len=2, max_len=20):
    alphabet = np.asarray(list(string.ascii_letters + string.digits + string.punctuation))
    lengths = rng.integers(min_len, max_len + 1, size=n)
    return ["".join(rng.choice(alphabet, size=length)) for length in lengths]


def test_featureset(rng):
    set1 = random_strings(10, rng)
    set2 = random_strings(7, rng)

    fset1 = FeatureSet(set1, name="fset1")
    fset2 = FeatureSet(set2, name="fset2")

    assert len(fset1) == len(set1)
    assert not fset1.empty
    assert fset1 == fset1
    assert fset1 != fset2

    union = fset1 | fset2
    assert fset1 + fset2 == union
    assert len(union) == len(fset1) + len(fset2)
    assert sorted(union.features) == sorted(fset1.features | fset2.features)

    assert (fset1 & fset2).empty

    subset = fset1.subset(set1[:3])
    assert len(subset) == 3
    assert sorted(subset.features) == sorted(set1[:3])


@pytest.fixture(scope="module")
def all_featuresets(rng):
    return [
        FeatureSet(random_strings(nfeatures, rng), name=f"fset_{i}")
        for i, nfeatures in enumerate(rng.integers(5, 100, size=20))
    ]


@pytest.fixture(scope="module")
def featurelist1(all_featuresets):
    return all_featuresets[:11]


@pytest.fixture(scope="module")
def featurelist2(all_featuresets):
    return all_featuresets[11:]


@pytest.fixture(scope="module")
def featuresets1(featurelist1):
    return FeatureSets(featurelist1)


@pytest.fixture(scope="module")
def featuresets2(featurelist2):
    return FeatureSets(featurelist2)


def test_featuresets_basic_ops(all_featuresets, featurelist1, featurelist2, featuresets1, featuresets2):
    assert len(featuresets1) == len(featuresets1)
    assert featuresets1.median_size == int(np.median([len(fset) for fset in featuresets1]))
    assert sorted(featuresets1.features) == sorted(set().union(*(fset.features for fset in featuresets1)))

    assert len(FeatureSets(featurelist1 + [FeatureSet([], name="")], remove_empty=True)) == len(featuresets1)
    assert len(FeatureSets(featurelist1 + [FeatureSet([], name="")], remove_empty=False)) == len(featuresets1) + 1

    fset = featuresets1.feature_set_by_name
    assert fset.keys() == {fset.name for fset in featuresets1}
    for i in range(len(featurelist1)):
        name = f"fset_{i}"
        assert fset[name] == featuresets1[name]
        assert fset[name] == featurelist1[i]

    assert featuresets1 == featuresets1
    assert featuresets1 != featuresets2

    union = featuresets1 | featuresets2
    assert featuresets1 + featuresets2 == union
    assert len(union) == len(all_featuresets)
    assert union.feature_set_by_name.keys() == {fset.name for fset in all_featuresets}

    assert (featuresets1 & featuresets2).empty


def test_find(featuresets1):
    assert len(featuresets1.find("fset")) == len(featuresets1)
    assert len(featuresets1.find("_0")) == 1


def test_remove(featuresets1):
    fsets = featuresets1.remove(["fset_1", "fset_7"])
    assert len(fsets) == len(featuresets1) - 2
    assert "fset_1" not in fsets.feature_set_by_name
    assert "fset_7" not in fsets.feature_set_by_name


def test_keep(featuresets1):
    fsets = featuresets1.keep(["fset_1", "fset_7"])
    assert len(fsets) == 2
    assert fsets["fset_1"] == featuresets1["fset_1"]
    assert fsets["fset_7"] == featuresets1["fset_7"]


@pytest.mark.parametrize("min_count,max_count", ((1, None), (5, None), (5, 1), (5, 70), (20, None), (150, None)))
def test_trim(featurelist1, featuresets1, min_count, max_count):
    fsets = featuresets1.trim(min_count, max_count)
    trimmed = [
        fset
        for fset in featurelist1
        if min_count <= len(fset) <= (max_count if max_count is not None else float("inf"))
    ]

    assert len(trimmed) == len(fsets)
    fsets_by_name = fsets.feature_set_by_name
    for fset in trimmed:
        assert fset.name in fsets_by_name
    assert FeatureSets(trimmed) == fsets


def test_subset(featurelist1, featuresets1):
    features = (
        [list(featurelist1[0].features)[1]] + [list(featurelist1[0].features)[3]] + list(featurelist1[3].features)[:4]
    )
    fsets = featuresets1.subset(features)
    assert len(fsets) == 2
    assert fsets["fset_0"].features == set(features[:2])
    assert fsets["fset_3"].features == set(features[2:])


@pytest.mark.parametrize("min_fraction", [0.5, 0.7])
@pytest.mark.parametrize("min_count", [1, 3])
@pytest.mark.parametrize("max_count", [None, 1, 20])
@pytest.mark.parametrize("keep", [None, ["fset_1"], ["fset_1", "fset_7"]])
@pytest.mark.parametrize("subset", [True, False])
def test_filter(featurelist1, featuresets1, min_fraction, min_count, max_count, keep, subset):
    features1 = [list(featurelist1[0].features)[1]] + [list(featurelist1[0].features)[3]]
    features2 = list(featurelist1[3].features)[: int(0.6 * len(featurelist1[3]))]

    fsets = featuresets1.filter(
        features1 + features2,
        min_fraction=min_fraction,
        min_count=min_count,
        max_count=max_count,
        keep=keep,
        subset=subset,
    )

    nsets = 0
    if (
        len(features1) >= min_count
        and len(features1) >= min_fraction * len(featurelist1[0])
        and (max_count is None or len(features1) <= max_count)
    ):
        nsets += 1
        assert "fset_0" in fsets.feature_set_by_name
        if subset:
            assert fsets["fset_0"].features == set(features1)
        else:
            assert fsets["fset_0"] == featuresets1["fset_0"]
    if (
        len(features2) >= min_count
        and len(features2) >= min_fraction * len(featurelist1[3])
        and (max_count is None or len(features2) <= max_count)
    ):
        nsets += 1
        assert "fset_3" in fsets.feature_set_by_name
        if subset:
            assert fsets["fset_3"].features == set(features2)
        else:
            assert fsets["fset_3"] == featuresets1["fset_3"]
    if keep is not None:
        nsets += len(keep)
        assert all(fset in fsets.feature_set_by_name for fset in keep)
        assert all(fsets[fset] == featuresets1[fset] for fset in keep)

    assert len(fsets) == nsets


def test_to_mask(featuresets1):
    mask = featuresets1.to_mask()
    assert mask.shape[0] == len(featuresets1)
    assert mask.shape[1] == len(featuresets1.features)

    assert np.all(mask.index.isin(featuresets1.feature_set_by_name))
    assert np.all(mask.columns.isin(featuresets1.features))

    assert np.all(mask.sum(axis=0) == 1)


def test_similarity_to_feature_sets(featurelist1, featuresets1):
    sim = featuresets1.similarity_to_feature_sets(featuresets1).to_numpy()
    assert np.all(sim == np.eye(sim.shape[0], dtype=sim.dtype))

    flist = featurelist1 + [
        FeatureSet(list(featurelist1[0].features)[:1] + list(featurelist1[3].features)[:4], name="nonunique")
    ]
    fsets = FeatureSets(flist)
    sim = featuresets1.similarity_to_feature_sets(fsets)
    assert sim.shape[0] == len(featuresets1)
    assert sim.shape[1] == len(fsets)

    assert np.all(sim.to_numpy()[:, : sim.shape[0]] == np.eye(sim.shape[0], dtype=sim.iloc[:, 0].dtype))
    assert 0 < sim.loc["fset_0", "nonunique"] < 1
    assert 0 < sim.loc["fset_3", "nonunique"] < 1
    assert np.all(sim.drop(index=["fset_0", "fset_3"])["nonunique"] == 0)


def test_similarity_to_observations(rng, featurelist1, featuresets1):
    features = list(featuresets1.features)
    cols = [feature for feature in features if feature not in featurelist1[0].features]
    cols = cols + [f"randomcol_{i}" for i in range(len(features) - len(cols))]

    obs = pd.DataFrame(rng.normal(size=(500, len(features))), columns=cols)
    sim = featuresets1.similarity_to_observations(obs)
    assert sim.shape[0] == sim.shape[1] == len(featuresets1)

    assert np.all(np.isnan(sim.loc[:, "fset_0"]))
    assert np.all(np.isnan(sim.loc["fset_0", :]))

    sim.drop(index="fset_0", columns="fset_0", inplace=True)
    assert np.all(sim >= -1)
    assert np.all(sim <= 1)


def test_find_similar_pairs(featurelist1):
    flist = featurelist1 + [
        FeatureSet(
            list(featurelist1[0].features)[: int(0.9 * len(featurelist1[0]))] + list(featurelist1[3].features)[:4],
            name="nonunique",
        )
    ]
    fsets = FeatureSets(flist)

    pairs = fsets.find_similar_pairs()
    assert len(pairs) == 1

    pair = next(iter(pairs))
    assert pair[0] == "fset_0" and pair[1] == "nonunique" or pair[0] == "nonunique" and pair[1] == "fset_0"


def test_merge_pairs(featuresets1):
    merged = featuresets1.merge_pairs([("fset_0", "fset_3"), ("fset_5", "fset_6")])
    assert len(merged) == len(featuresets1) - 2
    assert "fset_0" not in merged.feature_set_by_name
    assert "fset_3" not in merged.feature_set_by_name
    assert "fset_5" not in merged.feature_set_by_name
    assert "fset_6" not in merged.feature_set_by_name

    assert (
        merged.feature_set_by_name["fset_0|fset_3"]
        == featuresets1.feature_set_by_name["fset_0"] | featuresets1.feature_set_by_name["fset_3"]
    )
    assert (
        merged.feature_set_by_name["fset_5|fset_6"]
        == featuresets1.feature_set_by_name["fset_5"] | featuresets1.feature_set_by_name["fset_6"]
    )


def test_merge_similar(featurelist1):
    flist = featurelist1 + [
        FeatureSet(
            list(featurelist1[0].features)[: int(0.9 * len(featurelist1[0]))] + list(featurelist1[3].features)[:4],
            name="nonunique",
        )
    ]
    fsets = FeatureSets(flist)
    merged = fsets.merge_similar()

    assert len(merged) == len(fsets) - 1
    assert "fset_0" not in merged.feature_set_by_name
    assert "nonunique" not in merged.feature_set_by_name
    assert merged.feature_set_by_name["fset_0|nonunique"] == flist[0] | flist[-1]


def test_gmt(featuresets1, tmp_path):
    fpath = tmp_path / "test.gmt"

    featuresets1.to_gmt(fpath)
    assert FeatureSets.from_gmt(fpath) == featuresets1


def test_dict(featuresets1):
    dct = featuresets1.to_dict()
    assert FeatureSets.from_dict(dct) == featuresets1


def test_dataframe(featuresets1):
    df = []
    for fset in featuresets1.feature_sets:
        df.append({"nm": fset.name, "ftrs": list(fset.features)})
    df = pd.DataFrame(df)

    assert FeatureSets.from_dataframe(df, name_col="nm", features_col="ftrs") == featuresets1
