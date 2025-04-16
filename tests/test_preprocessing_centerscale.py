import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, issparse

from mofaflex._core import MofaFlexDataset
from mofaflex._core.preprocessing import MofaFlexPreprocessor
from mofaflex._core.utils import sample_all_data_as_one_batch

_sparse_arr = [csc_array, csc_matrix, csr_array, csr_matrix]

_ngroups = 2
_nviews = 3
_lklhds = ["Normal", "Bernoulli", "NegativeBinomial"]


def combination_from_idx(idx, n1, n2):
    combidx = [0] * n1
    for i in range(n1 - 1):
        offset = n2 ** (n1 - (i + 1))
        cidx = idx // offset
        combidx[i] = cidx
        idx -= cidx * offset
    combidx[-1] = idx
    return combidx


@pytest.fixture(
    scope="module",
    params=np.arange(len(_lklhds) ** _nviews),
    ids=lambda idx: "-".join(_lklhds[i] for i in combination_from_idx(idx, _nviews, len(_lklhds))),
)
def likelihoods(request):
    return {
        f"view_{i}": _lklhds[idx] for i, idx in enumerate(combination_from_idx(request.param, _nviews, len(_lklhds)))
    }


# full combinatorics results in approx. 1e6 tests
@pytest.fixture(scope="module")
def sparse_arr(likelihoods):
    i = 0
    fundict = {}
    for group in range(_ngroups):
        cdict = {}
        for view in range(_nviews):
            view_name = f"view_{view}"
            if likelihoods[view_name] == "Normal":
                cdict[view_name] = np.asarray
            else:
                cdict[view_name] = _sparse_arr[i % len(_sparse_arr)]
                i += 1
        fundict[f"group_{group}"] = cdict
    return fundict


@pytest.fixture(scope="module")
def adata_dict(rng, create_adata, random_array, sparse_arr, likelihoods):
    data = {}
    for group_name, group_sparse in sparse_arr.items():
        cdata = {}
        for view_name, view_sparse in group_sparse.items():
            arr = random_array(likelihoods[view_name], (100, 30))
            cdata[view_name] = create_adata(
                view_sparse(arr), obs_names=[f"{group_name}_{i}" for i in range(arr.shape[0])]
            )
        data[group_name] = cdata
    return data


@pytest.fixture(scope="module", params=[True, False])
def nonnegative_weights(likelihoods, request):
    return {f"view_{i}": request.param for i in range(_nviews)}


@pytest.fixture(scope="module", params=[True, False])
def nonnegative_factors(likelihoods, request):
    return {f"group_{i}": request.param for i in range(_ngroups)}


@pytest.fixture(scope="module", params=[True, False])
def scale_per_group(request):
    return request.param


@pytest.fixture(scope="module")
def dataset(adata_dict, likelihoods, nonnegative_weights, nonnegative_factors, scale_per_group):
    dataset = MofaFlexDataset(adata_dict, cast_to=None)
    preprocessor = MofaFlexPreprocessor(
        dataset,
        likelihoods,
        nonnegative_weights,
        nonnegative_factors,
        scale_per_group=scale_per_group,
        remove_constant_features=False,
    )
    dataset.preprocessor = preprocessor
    return dataset


def test_get_sample_means(adata_dict, dataset):
    for group_name, group in adata_dict.items():
        for view_name, view in group.items():
            arr = view.X.toarray() if issparse(view.X) else view.X
            assert np.all(np.nanmean(arr, axis=1) == dataset.preprocessor.sample_means[group_name][view_name])


def test_get_feature_means(adata_dict, dataset):
    for group_name, group in adata_dict.items():
        for view_name, view in group.items():
            arr = view.X.toarray() if issparse(view.X) else view.X
            assert np.all(np.nanmean(arr, axis=0) == dataset.preprocessor.feature_means[group_name][view_name])


def test_center_data(likelihoods, dataset, nonnegative_weights, nonnegative_factors):
    result = dataset.__getitems__(sample_all_data_as_one_batch(dataset))["data"]
    for group_name, group in result.items():
        for view_name, view in group.items():
            if likelihoods[view_name] == "Normal":
                if nonnegative_weights[view_name] and nonnegative_factors[group_name]:
                    assert np.allclose(np.nanmin(view, axis=0), 0)
                else:
                    assert np.allclose(view.mean(axis=0), 0)


def test_scale_data(likelihoods, dataset, scale_per_group):
    result = dataset.__getitems__(sample_all_data_as_one_batch(dataset))["data"]
    if scale_per_group:
        for group in result.values():
            for view_name, view in group.items():
                if likelihoods[view_name] == "Normal":
                    assert np.allclose(view.var(), 1)
    else:
        for view_name in dataset.view_names:
            if likelihoods[view_name] == "Normal":
                concat = np.concat([group[view_name] for group in result.values() if view_name in group], axis=0)
                assert np.allclose(concat.var(), 1)
