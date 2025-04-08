import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from mofaflex._core import preprocessing


@pytest.fixture(scope="module", params=["Normal", "Bernoulli", "GammaPoisson"])
def likelihood(request):
    return request.param


@pytest.fixture(scope="module", params=[np.asarray, csc_array, csc_matrix, csr_array, csr_matrix])
def adata(rng, create_adata, likelihood, random_array, request):
    return create_adata(request.param(random_array(likelihood, (20, 5))))


def test_infer_likelihoods(adata, likelihood):
    inferred = preprocessing.infer_likelihood(adata)
    assert likelihood == inferred


def test_validate_likelihoods(adata, likelihood):
    preprocessing.validate_likelihood(adata, None, None, likelihood)
