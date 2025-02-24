from collections import namedtuple
from collections.abc import Callable, Mapping, Sequence, Set
from importlib.util import find_spec
from typing import Any

import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import coo_array, csc_array, csr_array, issparse, sparray, spmatrix

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
        arr = sparse.asarray(arr)
    if isinstance(arr, pd.DataFrame):
        return dd.from_pandas(arr, sort=False)
    else:
        return da.from_array(arr)


def from_dask(arr):
    if type(arr).__module__.startswith("dask."):
        arr = arr.compute()
    if type(arr).__module__.startswith("sparse."):
        import os

        os.environ["SPARSE_AUTO_DENSIFY"] = "1"  # https://github.com/pydata/sparse/issues/842
        import sparse

        if isinstance(arr, sparse.GCXS):
            if arr.compressed_axes == (0,):
                arr = csr_array((arr.data, arr.indices, arr.indptr), shape=arr.shape)
            elif arr.compressed_axes == (1,):
                arr = csc_array((arr.data, arr.indices, arr.indptr), shape=arr.shape)
            else:
                raise ValueError(f"Unsupported compressed axes {arr.compressed_axes} in sparse array.")
        elif isinstance(arr, sparse.COO):
            arr = coo_array((arr.data, arr.coords), shape=arr.shape)
        else:
            raise ValueError(f"Unsupported sparse array format {type(arr)}")
    return arr


def apply_to_nested(data, func: Callable[[Any], Any]):
    if isinstance(data, Mapping):
        return type(data)({k: apply_to_nested(data[k], func) for k in data})
    elif isinstance(data, tuple):
        return type(data)(*(apply_to_nested(v, func) for v in data))
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
