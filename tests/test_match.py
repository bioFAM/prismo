import numpy as np
import pytest

from prismo.tl import match


@pytest.mark.parametrize(
    "size,axis",
    [
        ((50,), 0),
        ((50,), -1),
        ((50, 50, 50, 50), 0),
        ((50, 50, 50, 50), 1),
        ((50, 50, 50, 50), 3),
        ((50, 50, 50, 50), -1),
        ((50, 1, 50, 50), 2),
    ],
)
def test_match(rng, size, axis):
    reference = rng.normal(scale=2, size=size)
    permutation = rng.permutation(size[axis])

    reference_ind, permutable_ind, _ = match(reference, np.take(reference, permutation, axis=axis), axis)
    assert np.all(reference_ind == permutation[permutable_ind])

    if reference.ndim > 1:
        permutable = reference + rng.normal(scale=0.1, size=size)
        reference_ind, permutable_ind, _ = match(reference, np.take(permutable, permutation, axis=axis), axis)
        assert np.all(reference_ind == permutation[permutable_ind])
