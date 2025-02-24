from collections import namedtuple
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy import sparse
from scipy.special import expit

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
        var = arr.var(axis=axis)
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


def _imputation_link(imputation: NDArray, likelihood: Likelihood):
    if likelihood == "Bernoulli":
        return expit(imputation)
    elif likelihood != "Normal":
        raise NotImplementedError(f"Imputation for {likelihood} not implemented.")


def impute(
    data: AnnData, group_name, view_name, factors, weights, sample_names, feature_names, likelihood, missingonly
):
    havemissing = data.n_obs < factors.shape[0] or data.n_vars < weights.shape[1]
    if missingonly and not havemissing:
        return data
    elif not missingonly:
        imputation = _imputation_link(factors @ weights, likelihood)
    else:
        missing_obs = align_local_array_to_global(  # noqa F821
            np.broadcast_to(False, (data.n_obs,)), group_name, view_name, fill_value=True, align_to="samples"
        )
        missing_var = align_local_array_to_global(  # noqa F821
            np.broadcast_to(False, (data.n_vars,)), group_name, view_name, fill_value=True, align_to="features"
        )

        if sparse.issparse(data.X):
            imputation = sparse.lil_array((factors.shape[0], weights.shape[1]))
        else:
            imputation = np.empty((sample_names.size, feature_names.size), dtype=data.X.dtype)

        imputation[np.ix_(~missing_obs, ~missing_var)] = data.X

        if sparse.issparse(data.X):
            for row in np.nonzero(missing_obs)[0]:
                imputation[row, :] = _imputation_link(factors[row, :] @ weights, likelihood)
            imputation = imputation.T  # slow column slicing for lil arrays
            for col in np.nonzero(missing_var)[0]:
                imputation[col, :] = _imputation_link(factors @ weights[:, col], likelihood).T
            imputation = imputation.tocsr().T
        else:
            imputation[np.ix_(missing_obs, missing_var)] = _imputation_link(
                factors[missing_obs, :] @ weights[:, missing_var]
            )

        return AnnData(X=imputation, obs=pd.DataFrame(index=sample_names), var=pd.DataFrame(index=feature_names))
