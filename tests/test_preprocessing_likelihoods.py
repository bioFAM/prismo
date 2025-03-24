import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from prismo._core import preprocessing


@pytest.fixture(scope="module", params=["Normal", "Bernoulli", "GammaPoisson"])
def likelihood(request):
    return request.param


@pytest.fixture(scope="module", params=[np.asarray, csc_array, csc_matrix, csr_array, csr_matrix])
def adata(rng, create_adata, likelihood, request):
    match likelihood:
        case "Normal":
            arr = rng.normal(size=(20, 5))
        case "Bernoulli":
            arr = rng.binomial(1, 0.5, size=(20, 5))
        case "GammaPoisson":
            arr = rng.negative_binomial(10, 0.9, (20, 5))

    return create_adata(request.param(arr))


def test_infer_likelihoods(adata, likelihood):
    inferred = preprocessing.infer_likelihood(adata)
    assert likelihood == inferred


def test_validate_likelihoods(adata, likelihood):
    preprocessing.validate_likelihood(adata, None, None, likelihood)
