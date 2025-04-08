import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from prismo._core import utils


@pytest.fixture(scope="module")
def array(rng):
    arr = rng.normal(size=(100, 70))
    zero_mask = rng.choice([False, True], size=arr.size, p=[0.7, 0.3])
    arr[zero_mask.reshape(arr.shape)] = 0
    return arr


@pytest.fixture(scope="module", params=[csr_array, csc_array, csr_matrix, csc_matrix])
def sparse_array(array, request):
    return request.param(array)


@pytest.fixture(scope="module")
def nan_array(rng, array):
    arr = array.copy()
    zero_mask = rng.choice([False, True], size=arr.size, p=[0.8, 0.2])
    arr[zero_mask.reshape(arr.shape)] = np.nan
    return arr


@pytest.fixture(scope="module", params=[csr_array, csc_array, csr_matrix, csc_matrix])
def sparse_nan_array(nan_array, request):
    return request.param(nan_array)


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("method", ["mean", "var", "min", "max"])
def test_op(array, sparse_array, axis, keepdims, method):
    arr_result = getattr(np, method)(array, axis=axis, keepdims=keepdims)
    utils_result = getattr(utils, method)(array, axis=axis, keepdims=keepdims)
    sparse_result = getattr(utils, method)(sparse_array, axis=axis, keepdims=keepdims)

    assert np.allclose(arr_result, utils_result)
    assert np.allclose(arr_result, sparse_result)

    assert np.all(arr_result.shape == utils_result.shape)
    assert np.all(arr_result.shape == sparse_result.shape)


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("method", ["nanmean", "nanvar", "nanmin", "nanmax"])
def test_nan_op(nan_array, sparse_nan_array, axis, keepdims, method):
    arr_result = getattr(np, method)(nan_array, axis=axis, keepdims=keepdims)
    utils_result = getattr(utils, method)(nan_array, axis=axis, keepdims=keepdims)
    sparse_result = getattr(utils, method)(sparse_nan_array, axis=axis, keepdims=keepdims)

    assert np.allclose(arr_result, utils_result)
    assert np.allclose(arr_result, sparse_result)

    assert np.all(arr_result.shape == utils_result.shape)
    assert np.all(arr_result.shape == sparse_result.shape)


def test_wherenan(nan_array, sparse_nan_array):
    arr_result = np.nonzero(np.isnan(nan_array))
    utils_result = utils.wherenan(nan_array)
    sparse_result = utils.wherenan(sparse_nan_array)

    assert np.all(arr_result[0] == utils_result[0])
    assert np.all(arr_result[1] == utils_result[1])

    assert np.all(arr_result[0] == sparse_result[0])
    assert np.all(arr_result[1] == sparse_result[1])
