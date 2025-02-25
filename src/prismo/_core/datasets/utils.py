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

AlignmentMap = namedtuple("AlignmentMap", ["l2g", "g2l"])


def have_dask():
    return find_spec("dask") is not None and find_spec("sparse") is not None


def array_to_dask(arr: NDArray | spmatrix | sparray | pd.DataFrame):
    import os

    os.environ["SPARSE_AUTO_DENSIFY"] = "1"  # https://github.com/pydata/sparse/issues/842
    import dask.array as da
    import dask.dataframe as dd
    import sparse

    if issparse(arr):
        if isinstance(arr, csr_array | csr_matrix):
            arr.sort_indices()
            arr = sparse.GCXS((arr.data, arr.indices, arr.indptr), shape=arr.shape, compressed_axes=(0,))
        elif isinstance(arr, csc_array | csc_matrix):
            arr.sort_indices()
            arr = sparse.GCXS((arr.data, arr.indices, arr.indptr), shape=arr.shape, compressed_axes=(1,))
        elif isinstance(arr, coo_array):
            arr = sparse.COO(arr.coords, arr.data, shape=arr.shape)
        elif isinstance(arr, coo_matrix):
            arr = sparse.COO(np.stack((arr.row, arr.col), axis=0), arr.data, shape=arr.shape)
        else:
            arr = sparse.asarray(arr, format="csr")
    if isinstance(arr, pd.DataFrame):
        return dd.from_pandas(arr, sort=False)
    else:
        return da.from_array(arr, chunks=np.full(len(arr.shape), fill_value=-1))


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
        return type(data)({k: apply_to_nested(data[k], func) for k in data})
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
    dadata = AnnData(
        X=array_to_dask(adata.X), var=adata.var, obs=adata.obs
    )  # AnnData does not support Dask DataFrames for var and obs
    for attr in ("obsm", "obsp", "varm", "varp", "layers"):
        aattr = getattr(adata, attr)
        dattr = getattr(dadata, attr)
        for k, v in aattr.items():
            dattr[k] = array_to_dask(v)
    return dadata
