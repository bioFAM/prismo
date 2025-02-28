from collections import namedtuple
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, issparse, lil_array, sparray, spmatrix
from scipy.special import expit
from torch.utils.data import BatchSampler, SequentialSampler

from .datasets import PrismoDataset

WeightPrior: TypeAlias = Literal["Normal", "Laplace", "Horseshoe", "SnS", "GP"]
FactorPrior: TypeAlias = Literal["Normal", "Laplace", "Horseshoe", "SnS"]
Likelihood: TypeAlias = Literal["Normal", "GammaPoisson", "Bernoulli"]
PossiblySparseArray: TypeAlias = NDArray | spmatrix | sparray

MeanStd = namedtuple("MeanStd", ["mean", "std"])


def sample_all_data_as_one_batch(data: PrismoDataset) -> dict[str, list[int]]:
    return {
        k: next(
            iter(BatchSampler(SequentialSampler(range(nsamples)), batch_size=data.n_samples_total, drop_last=False))
        )
        for k, nsamples in data.n_samples.items()
    }


def mean(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    if issparse(arr):
        mean = np.asarray(arr.mean(axis=axis))
        if not keepdims and axis is not None:
            mean = mean.squeeze(axis)
        elif keepdims and axis is None:
            mean = np.expand_dims(mean, tuple(range(arr.ndim)))
    else:
        mean = arr.mean(axis=axis, keepdims=keepdims)
    return mean


# TODO: use numba for this?
def _nanmean_cs_aligned(arr: csr_array | csr_matrix | csc_array | csc_matrix):
    axis = 0 if isinstance(arr, csr_array | csr_matrix) else 1
    out = np.empty(arr.shape[axis], dtype=np.float64 if np.issubdtype(arr.dtype, np.integer) else arr.dtype)
    for r in range(arr.shape[axis]):
        data = arr.data[arr.indptr[r] : arr.indptr[r + 1]]
        mask = np.isnan(data)
        out[r] = data[~mask].sum() / (arr.shape[1 - axis] - mask.sum())
    return out


# TODO: use numba for this?
def _nanmean_cs_nonaligned(arr: csr_array | csr_matrix | csc_array | csc_matrix):
    axis = 1 if isinstance(arr, csr_array | csr_matrix) else 0
    out = np.zeros(arr.shape[axis], dtype=np.float64 if np.issubdtype(arr.dtype, np.integer) else arr.dtype)
    n = np.full(arr.shape[axis], fill_value=arr.shape[1 - axis], dtype=np.uint32)
    for r in range(arr.shape[axis]):
        idx = arr.indices[arr.indptr[r] : arr.indptr[r + 1]]
        data = arr.data[arr.indptr[r] : arr.indptr[r + 1]]
        mask = np.isnan(data)
        out[idx[~mask]] += data[mask]
        n[idx[mask]] -= 1
    out /= n
    return out


def nanmean(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    if issparse(arr):
        if axis is None:
            mean = np.nanmean(arr.data)
            if keepdims:
                mean = mean[None, None]
        else:
            if (
                axis == 0
                and isinstance(arr, csr_array | csr_matrix)
                or axis == 1
                and isinstance(arr, csc_array | csc_matrix)
            ):
                mean = _nanmean_cs_aligned(arr)
            elif (
                axis == 1
                and isinstance(arr, csr_array | csr_matrix)
                or axis == 0
                and isinstance(arr, csc_array | csc_matrix)
            ):
                mean = _nanmean_cs_nonaligned(arr)
            else:
                raise NotImplementedError(f"Unsupported sparse matrix type {type(arr)}.")
            if keepdims:
                mean = mean.expand_dims(axis)
    else:
        mean = np.nanmean(arr, axis=axis, keepdims=keepdims)
    return mean


def var(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    if issparse(arr):
        _mean = mean(arr, axis=axis, keepdims=True)
        var = (np.asarray(arr - _mean) ** 2).mean(axis=axis, keepdims=keepdims)
    else:
        var = arr.var(axis=axis, keepdims=keepdims)
    return var


def nanvar(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    if issparse(arr):
        _mean = nanmean(arr, axis=axis, keepdims=True)
        var = np.nanmean(np.asarray(arr - _mean) ** 2, axis=axis, keepdims=keepdims)
    else:
        var = np.nanvar(arr, axis=axis, keepdims=keepdims)
    return var


def min(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    return _minmax(arr, method="min", axis=axis, keepdims=keepdims)


def max(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    return _minmax(arr, method="max", axis=axis, keepdims=keepdims)


def nanmin(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    return _minmax(arr, method="nanmin", axis=axis, keepdims=keepdims)


def nanmax(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    return _minmax(arr, method="nanmax", axis=axis, keepdims=keepdims)


def _minmax(
    arr: PossiblySparseArray, method: Literal["min", "max", "nanmin", "nanmax"], axis: int | None = None, keepdims=False
):
    if hasattr(arr, method):
        res = getattr(arr, method)(axis=axis)
    else:
        res = getattr(np, method)(arr, axis=axis)
    if issparse(res):
        res = res.toarray()
    if keepdims:
        res = np.expand_dims(res, axis if axis is not None else tuple(range(arr.ndim)))
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

        if issparse(data.X):
            imputation = lil_array((factors.shape[0], weights.shape[1]))
        else:
            imputation = np.empty((sample_names.size, feature_names.size), dtype=data.X.dtype)

        imputation[np.ix_(~missing_obs, ~missing_var)] = data.X

        if issparse(data.X):
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
