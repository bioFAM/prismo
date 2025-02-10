from collections import namedtuple
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

WeightPrior: TypeAlias = Literal["Normal", "Laplace", "Horseshoe", "SnS", "GP"]
FactorPrior: TypeAlias = Literal["Normal", "Laplace", "Horseshoe", "SnS"]
Likelihood: TypeAlias = Literal["Normal", "GammaPoisson", "Bernoulli"]
PossiblySparseArray: TypeAlias = NDArray | sparse.spmatrix | sparse.sparray

MeanStd = namedtuple("MeanStd", ["mean", "std"])


def mean(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    if sparse.issparse(arr):
        mean = np.asarray(arr.mean(axis=axis))
        if not keepdims and axis is not None:
            mean = mean.squeeze(axis)
        elif keepdims and axis is None:
            mean = np.expand_dims(mean, tuple(range(arr.ndim)))
    else:
        mean = arr.mean(axis=axis, keepdims=keepdims)
    return mean


def var(arr: PossiblySparseArray, axis: int | None = None):
    if sparse.issparse(arr):
        _mean = mean(arr, axis=axis, keepdims=True)
        var = (np.asarray(arr - _mean) ** 2).sum(axis=axis) / (
            arr.shape[axis] if axis is not None else np.prod(arr.shape)
        )
    else:
        var = np.var(arr, axis=axis)
    return var


def min(arr: PossiblySparseArray, axis: int | None = None):
    return _minmax(arr, method="min", axis=axis)


def max(arr: PossiblySparseArray, axis: int | None = None):
    return _minmax(arr, method="max", axis=axis)


def _minmax(arr: PossiblySparseArray, method: Literal["min", "max"], axis: int | None = None):
    res = getattr(arr, method)(axis=axis)
    if sparse.issparse(res):
        res = res.todense()
    if isinstance(res, np.matrix):
        res = np.asarray(res).squeeze(axis)
    return res
