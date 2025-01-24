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
ViewStatistics = namedtuple("ViewStatistics", ["mean", "var", "min", "max"])


def mean(arr: PossiblySparseArray, axis: int = 0):
    mean = arr.mean(axis=axis)
    if isinstance(mean, np.matrix):
        mean = np.asarray(mean).squeeze(axis)
    return mean


def var(arr: PossiblySparseArray, axis: int = 0):
    if sparse.issparse(arr):
        _mean = mean(arr, axis=axis)
        mean2 = mean(arr.power(2), axis=axis)
        var = mean2 - _mean**2
    else:
        var = np.var(arr, axis=axis)
    return var


def min(arr: PossiblySparseArray, axis: int = 0):
    return _minmax(arr, method="min", axis=axis)


def max(arr: PossiblySparseArray, axis: int = 0):
    return _minmax(arr, method="max", axis=axis)


def _minmax(arr: PossiblySparseArray, method: Literal["min", "max"], axis: int = 0):
    res = getattr(arr, method)(axis=axis)
    if sparse.issparse(res):
        res = res.todense()
    if isinstance(res, np.matrix):
        res = np.asarray(res).squeeze(axis)
    return res
