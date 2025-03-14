import numpy as np
import pytest


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)
