from collections import namedtuple
from collections.abc import Callable, Mapping, Sequence, Set
from importlib.util import find_spec
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import (
    coo_array,
    coo_matrix,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
    issparse,
    sparray,
    spmatrix,
)

from ..settings import settings

AlignmentMap = namedtuple("AlignmentMap", ["l2g", "g2l"])


def have_dask():
    return find_spec("dask") is not None and find_spec("sparse") is not None


def array_to_dask(arr: NDArray | spmatrix | sparray | pd.DataFrame):
    import os

    os.environ["SPARSE_AUTO_DENSIFY"] = "1"  # https://github.com/pydata/sparse/issues/842
    import dask.array as da
    import dask.dataframe as dd
    import sparse

    if isinstance(arr, pd.DataFrame):
        return dd.from_pandas(arr, sort=False)

    elemsize = arr.dtype.itemsize

    chunksize = settings.dask_chunksize_mb * 1024 * 1024
    if issparse(arr):
        if isinstance(arr, csr_array | csr_matrix):
            arr.sort_indices()
            arr = sparse.GCXS((arr.data, arr.indices, arr.indptr), shape=arr.shape, compressed_axes=(0,))

            chunks = (chunksize // (arr.shape[1] * elemsize), -1)
        elif isinstance(arr, csc_array | csc_matrix):
            arr.sort_indices()
            arr = sparse.GCXS((arr.data, arr.indices, arr.indptr), shape=arr.shape, compressed_axes=(1,))

            chunks = (-1, chunksize // (arr.shape[0] * elemsize))
        elif isinstance(arr, coo_array):
            arr = sparse.COO(arr.coords, arr.data, shape=arr.shape)
            chunks = (-1, -1)
        elif isinstance(arr, coo_matrix):
            arr = sparse.COO(np.stack((arr.row, arr.col), axis=0), arr.data, shape=arr.shape)
            chunks = (-1, -1)
        else:
            arr = sparse.asarray(arr, format="csr")
            chunks = (chunksize // (arr.shape[1] * elemsize), -1)
    else:
        chunks = (-1, -1)
    return da.from_array(arr, chunks=chunks)


def from_dask(arr, convert_coo=True):
    if type(arr).__module__.startswith("dask."):
        arr = arr.compute()
    if type(arr).__module__.startswith("sparse."):
        import os

        os.environ["SPARSE_AUTO_DENSIFY"] = "1"  # https://github.com/pydata/sparse/issues/842

        if arr.ndim == 2:
            arr = arr.to_scipy_sparse()
            if convert_coo and isinstance(arr, coo_array | coo_matrix):
                arr = arr.tocsr()
        else:
            arr = arr.todense()
    return arr


def apply_to_nested(data, func: Callable[[Any], Any]):
    if isinstance(data, Mapping):
        return type(data)({k: apply_to_nested(v, func) for k, v in data.items()})
    elif isinstance(data, tuple):
        args = (apply_to_nested(v, func) for v in data)
        if hasattr(data, "_fields"):  # namedtuple
            return type(data)(*args)
        else:
            return type(data)(args)
    elif isinstance(data, Sequence | Set) and not isinstance(data, str | bytes):
        return type(data)(apply_to_nested(v, func) for v in data)
    else:
        return func(data)


def anndata_to_dask(adata: AnnData):
    dask_adata = AnnData(
        X=array_to_dask(adata.X), var=adata.var, obs=adata.obs
    )  # AnnData does not support Dask DataFrames for var and obs
    for attrname in ("obsm", "obsp", "varm", "varp", "layers"):
        attr = getattr(adata, attrname)
        dask_attr = getattr(dask_adata, attrname)
        for k, v in attr.items():
            dask_attr._data[k] = array_to_dask(v)
    return dask_adata
