import logging
from functools import reduce
from typing import Any, Literal, TypeVar

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy import sparse

from .base import ApplyCallable, Preprocessor, PrismoDataset

T = TypeVar("T")
_logger = logging.getLogger(__name__)


class AnnDataDictDataset(PrismoDataset):
    def __init__(
        self,
        data: dict[str, dict[str, AnnData]],
        *,
        use_obs: Literal["union", "intersection"] = "union",
        use_var: Literal["union", "intersection"] = "union",
        preprocessor: Preprocessor | None = None,
        cast_to: np.ScalarType = np.float32,
        **kwargs,
    ):
        super().__init__(data, preprocessor=preprocessor, cast_to=cast_to)

        self._aligned_obs = {}
        self._aligned_var = {}
        self._obsmap = {}
        self._varmap = {}

        obsfunc = (lambda x, y: x.intersection(y)) if use_obs == "intersection" else lambda x, y: x.union(y)
        varfunc = (lambda x, y: x.intersection(y)) if use_var == "intersection" else lambda x, y: x.union(y)

        for view_name in self.view_names:
            self._aligned_var[view_name] = reduce(
                varfunc, (group[view_name].var_names for group in self._data.values() if view_name in group)
            )

        for group_name, group in self._data.items():
            aligned_obs = reduce(obsfunc, (view.obs_names for view in group.values()))
            self._obsmap[group_name] = {
                view_name: aligned_obs.get_indexer(view.obs_names) for view_name, view in group.items()
            }
            self._varmap[group_name] = {
                view_name: self._aligned_var[view_name].get_indexer(view.var_names) for view_name, view in group.items()
            }
            self._aligned_obs[group_name] = aligned_obs

    @staticmethod
    def _accepts_input(data):
        return isinstance(data, dict) and all(
            isinstance(group, dict) and all(isinstance(view, AnnData) for view in group.values())
            for group in data.values()
        )

    @property
    def n_features(self) -> dict[str, int]:
        return {view_name: var.size for view_name, var in self._aligned_var.items()}

    @property
    def n_samples(self) -> dict[str, int]:
        return {group_name: obs.size for group_name, obs in self._aligned_obs.items()}

    @property
    def view_names(self) -> NDArray[str]:
        return np.asarray(tuple(reduce(lambda x, y: x | y, (group.keys() for group in self._data.values()))))

    @property
    def group_names(self) -> NDArray[str]:
        return np.asarray(tuple(self._data.keys()))

    @property
    def sample_names(self) -> dict[str, NDArray[str]]:
        return {group_name: obs.to_numpy() for group_name, obs in self._aligned_obs.items()}

    @property
    def feature_names(self) -> dict[str, NDArray[str]]:
        return {view_name: var.to_numpy() for view_name, var in self._aligned_var.items()}

    def __getitem__(self, idx: dict[str, int]) -> tuple[dict[str, dict[str, NDArray]], dict[str, int]]:
        ret = {}
        for group_name, group_idx in idx.items():
            group = {}
            gobsmap = self._obsmap[group_name]
            for view_name, view in self._data[group_name].items():
                obsmap = gobsmap[view_name][group_idx]
                obsidx = obsmap >= 0

                arr = view.X[obsmap[obsidx], :]
                obsmap[obsidx] = np.arange(arr.shape[0])
                # have to align before preprocessing because preprocessor may depend (and probably does)
                # on var order
                if sparse.issparse(arr):
                    arr = arr.toarray()
                arr = self._align_array_to_samples(arr, group_name, view_name, axis=(0, 1), obsmap=obsmap)
                group[view_name] = self.preprocessor(arr, group_name, view_name).astype(self.cast_to)
                ret[group_name] = group
                idx[group_name] = np.asarray(group_idx)
        return {"data": ret, "sample_idx": idx}

    __getitems__ = __getitem__

    def _align_sparse_array_to_var(
        self, arr: sparse.sparray | sparse.spmatrix, group_name: str, view_name: str, fill_value: np.ScalarType = np.nan
    ):
        varmap = self._varmap[group_name][view_name]
        n_newcols = self._aligned_var[view_name].size - arr.shape[1]
        n_new_elems = arr.shape[0] * n_newcols
        if isinstance(arr.csr_matrix | sparse.csr_array):
            newnnzsize = arr.data.size + n_new_elems

            newcolidx = np.nonzero(varmap < 0)[0]
            oldcolidx = np.argsort(varmap)[newcolidx.size :]
            newindptr = arr.indptr.copy()
            newindptr[1:] += np.cumsum(np.repeat(n_newcols, arr.shape[0]))
            newindices = np.empty(newnnzsize, dtype=arr.indices.dtype)
            newdata = np.full(newnnzsize, fill_value=fill_value, dtype=arr.data.dtype)
            for row in range(arr.shape[0]):  # TODO: use numba for this
                cindices = oldcolidx[arr.indices[arr.indptr[row] : arr.indptr[row + 1]]]
                newindices[newindptr[row] : newindptr[row] + cindices.size] = cindices
                newindices[newindptr[row] + cindices.size : newindptr[row + 1]] = newcolidx
                newdata[newindptr[row] : newindptr[row] + arr.indptr[row + 1] - arr.indptr[row]] = arr.data[
                    arr.indptr[row] : arr.indptr[row + 1]
                ]

            ret = sparse.csr_array(
                (newdata, newindices, newindptr), shape=(arr.shape[0], arr.shape[1] + n_newcols)
            ).sort_indices()
        elif isinstance(arr.csc_matrix | sparse.csc_array):
            newnnzsize = arr.data.size + n_new_elems

            newdata = arr.data.resize(newnnzsize)
            newdata[arr.data.size :] = fill_value

            newindptr = np.empty(arr.indptr.size + n_newcols, dtype=arr.indptr.dtype)
            newindices = np.empty(newnnzsize, dtype=arr.indices.dtype)
            newdata = np.emtpy(newnnzsize, dtype=arr.data.dtype)
            newindptr[0] = 0
            for newcol, oldcol in enumerate(varmap):  # TODO: use numba for this
                if oldcol == -1:
                    newindptr[newcol + 1] = newindptr[newcol] + arr.shape[0]
                    newindices[newindptr[newcol] : newindptr[newcol + 1]] = np.arange(arr.shape[0])
                    newdata[newindptr[newcol] : newindptr[newcol] + 1] = fill_value
                else:
                    newindptr[newcol + 1] = newindptr[newcol] + arr.indptr[oldcol + 1] - arr.indptr[oldcol]
                    newindices[newindptr[newcol] : newindptr[newcol + 1]] = arr.indices[
                        arr.indptr[oldcol] : arr.indptr[oldcol + 1]
                    ]
                    newdata[newindptr[newcol] : newindptr[newcol + 1]] = arr.data[
                        arr.indptr[oldcol] : arr.indptr[oldcol + 1]
                    ]

            ret = sparse.csc_array((newdata, newindices, newindptr), shape=(arr.shape[0], arr.shape[1] + n_newcols))
        elif isinstance(arr.coo_matrix | sparse.coo_array):
            newcolidx = np.nonzero(varmap < 0)[0]
            newdata = arr.data.resize(arr.data.size + n_new_elems)
            newdata[arr.data.size :] = fill_value
            newcoords = tuple(np.concatenate(oldrow, newcolidx) for oldrow in arr.coords)
            ret = sparse.coo_array((newdata, newcoords), shape=(arr.shape[0], arr.shape[1] + n_newcols))
        else:
            raise NotImplementedError("unsupported sparse matrix format")
        return ret

    def _align_array_to_samples(
        self,
        arr: NDArray[T],
        group_name: str,
        view_name: str,
        axis: int | tuple[int, int] = 0,
        align_to: Literal["obs", "var"] = "obs",
        fill_value: np.ScalarType = np.nan,
        obsmap: NDArray[int] | None = None,
        varmap: NDArray[int] | None = None,
    ) -> NDArray[T]:
        """Align an array corresponding to a view with potentially missing observations to global samples by inserting filler values for missing samples.

        Args:
            arr: The array to align.
            group_name: Group name.
            view_name: View name.
            axis: The axis to align along. If a single integer, that axis will be aligned to observations. If a tuple, the first element will be aligned
                to observations and the second to features.
            align_to: What to align to. Only relevant if `axis` is a single integer.
            fill_value: The value to insert for missing samples.
            obsmap: Array mapping global observations to local observations, with -1 indicating missing observations. If `None`, will use the
                global obsmap in `self._obsmap[group_name][view_name]`. This is useful for aligning a subsetted array.
            varmap: Array mapping global features to local features, with -1 indicating missing features. If `None`, will use the
                global varmap in `self._varmap[group_name][view_name]`. This is useful for aligning a subsetted array.
        """
        if isinstance(axis, int):
            axis = [axis]
            align_to_both = False
        else:
            align_to_both = True

        if obsmap is None:
            obsmap = self._obsmap[group_name][view_name]
        if varmap is None:
            varmap = self._varmap[group_name][view_name]

        if (
            not align_to_both
            and (
                align_to == "obs"
                and arr.shape[axis[0]] == obsmap.size
                and np.all(np.diff(obsmap) == 1)
                or align_to == "var"
                and arr.shape[axis[0]] == varmap.size
                and np.all(np.diff(varmap) == 1)
            )
            or align_to_both
            and arr.shape[axis[0]] == obsmap.size
            and arr.shape[axis[1]] == varmap.size
            and np.all(np.diff(obsmap) == 1)
            and np.all(np.diff(varmap) == 1)
        ):
            return arr

        if align_to_both:
            outshape = [obsmap.size, varmap.size]
        elif align_to == "obs":
            outshape = [obsmap.size]
            idxtouse = obsmap
        else:
            outshape = [varmap.size]
            idxtouse = varmap
        ax1, ax2 = min(axis), max(axis)
        outshape.extend(arr.shape[:ax1])
        if align_to_both:
            outshape.extend(arr.shape[ax1 + 1 : ax2])
            outshape.extend(arr.shape[ax2 + 1 :])
        else:
            outshape.extend(arr.shape[ax1 + 1 :])
        if align_to_both:
            obsnnz = obsmap >= 0
            varnnz = varmap >= 0
            outidx = [obsnnz, varnnz]
            inidx = [obsmap[obsnnz], varmap[varnnz]]
            arr = np.moveaxis(arr, axis[0], 0)
            arr = np.moveaxis(arr, axis[1], 1)
        else:
            nnz = idxtouse >= 0
            outidx = [nnz]
            inidx = [idxtouse[nnz]]
            arr = np.moveaxis(arr, axis[0], 0)
        outidx.append(...)
        inidx.append(...)
        out = np.full(outshape, fill_value=fill_value, dtype=arr.dtype, order="C")
        out[*outidx] = arr[*inidx]

        if align_to_both:
            out = np.moveaxis(out, 1, axis[1] + 1 if axis[1] > axis[0] else axis[1])
        return np.moveaxis(out, 0, axis[0])

    def align_array_to_samples(
        self, arr: NDArray[T], group_name: str, view_name: str, axis: int = 0, fill_value: np.ScalarType = np.nan
    ) -> NDArray[T]:
        return self._align_array_to_samples(arr, group_name, view_name, axis=axis, fill_value=fill_value)

    def align_array_to_data(self, arr: NDArray[T], group_name: str, view_name: str, axis: int = 0) -> NDArray[T]:
        idx = self._obsmap[group_name][view_name]
        return np.take(arr, np.argsort(idx)[(idx < 0).sum() :], axis=axis)

    def _get_attr(self, attr: Literal["obs", "var"]) -> dict[str, pd.DataFrame]:
        return {
            group_name: {
                view_name: getattr(view, attr)
                .reindex(
                    getattr(self, f"_aligned_{attr}")[group_name if attr == "obs" else view_name], fill_value=pd.NA
                )
                .apply(lambda x: x.astype("string") if x.dtype == "O" else x, axis=1)
                for view_name, view in group.items()
            }
            for group_name, group in self._data.items()
        }

    def get_obs(self) -> dict[str, pd.DataFrame]:
        return self._get_attr("obs")

    def get_missing_obs(self) -> pd.DataFrame:
        dfs = []
        for group_name, group in self._data.items():
            for view_name, view in group.items():
                if sparse.issparse(view.X):
                    viewmissing = view.X.copy()
                    viewmissing.data = np.isnan(viewmissing.data)
                    viewmissing = np.asarray(viewmissing.sum(axis=1)).squeeze() > 0
                else:
                    viewmissing = np.isnan(view.X).any(axis=1)
                viewmissing = self._align_array_to_samples(viewmissing, group_name, view_name, fill_value=1)
                dfs.append(
                    pd.DataFrame(
                        {
                            "view": view_name,
                            "group": group_name,
                            "obs_name": self._aligned_obs[group_name],
                            "missing": viewmissing,
                        }
                    )
                )
        return pd.concat(dfs, axis=0, ignore_index=True)

    def get_covariates(
        self, covariates_obs_key: dict[str, str] | None = None, covariates_obsm_key: dict[str, str] | None = None
    ) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
        covariates, covariates_names = {}, {}
        if covariates_obs_key is None:
            covariates_obs_key = {}
        if covariates_obsm_key is None:
            covariates_obsm_key = {}
        for group_name, group in self._data.items():
            obskey = covariates_obs_key.get(group_name, None)
            obsmkey = covariates_obsm_key.get(group_name, None)
            if obskey is None and obsmkey is None:
                continue
            if obskey and obsmkey:
                raise ValueError(
                    f"Provide either covariates_obs_key or covariates_obsm_key for group {group_name}, not both."
                )

            ccovs = {}
            if obskey is not None:
                for view_name, view in group.items():
                    if obskey in view.obs.columns:
                        ccovs[view_name] = self._align_array_to_samples(
                            view.obs[obskey].to_numpy(), group_name, view_name
                        )[:, None]
                if len(ccovs):
                    covariates_names[group_name] = obskey
                else:
                    _logger.warn(f"No covariate data found in obs attribute for group {group_name}.")
            elif obsmkey is not None:
                covar_dim = []
                for view_name, view in group.items():
                    if obsmkey in view.obsm:
                        covar = view.obsm[obsmkey]
                        if isinstance(covar, pd.DataFrame):
                            covariates_names[group_name] = covar.columns.to_numpy()
                        elif isinstance(covar, pd.Series):
                            covariates_names[group_name] = np.asarray(covar.name, dtype=object)

                        covar = np.asarray(covar)
                        if covar.ndim == 1:
                            covar = covar[..., None]
                        covar_dim.append(covar.shape[1])
                        ccovs[view_name] = self._align_array_to_samples(covar, group_name, view_name)
                if len(set(covar_dim)) > 1:
                    raise ValueError(
                        f"Number of covariate dimensions in group {group_name} must be the same across views."
                    )

            covariates[group_name] = ccovs
        return covariates, covariates_names

    def get_annotations(self, varm_key: dict[str, str]) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
        annotations, annotations_names = {}, {}
        if varm_key is not None:
            for view_name, key in varm_key.items():
                for group in self._data.values():
                    if key in group[view_name].varm:
                        annot = group[view_name].varm[key]
                        if isinstance(annot, pd.DataFrame):
                            annotations_names[view_name] = annot.columns.to_list()
                            annotations[view_name] = annot.to_numpy().T
                        else:
                            annotations[view_name] = annot.T
                        break
        return annotations, annotations_names

    def apply(
        self,
        func: ApplyCallable[T],
        by_group: bool = True,
        by_view: bool = True,
        view_kwargs: dict[str, dict[str, Any]] | None = None,
        group_kwargs: dict[str, dict[str, Any]] | None = None,
        group_view_kwargs: dict[str, dict[str, dict[str, Any]]] | None = None,
        **kwargs,
    ) -> dict[str, dict[str, T]]:
        if not by_view:
            raise NotImplementedError("by_view must be True.")

        if view_kwargs is None:
            view_kwargs = {}

        if group_kwargs is None:
            group_kwargs = {}
        elif not by_group:
            raise ValueError("You cannot specify group_kwargs with by_group=False.")

        if group_view_kwargs is None:
            group_view_kwargs = {}
        elif not by_group:
            raise ValueError("You cannot specify group_view_kwargs with by_group=False.")

        ret = {}
        if by_group:
            newvar = self._get_attr("var")
            for group_name, group in self._data.items():
                cgroup_kwargs = {
                    argname: kwargs[group_name] if group_name in kwargs else None
                    for argname, kwargs in group_kwargs.items()
                }
                cgroup_view_kwargs = {
                    argname: kwargs[group_name] if group_name in kwargs else None
                    for argname, kwargs in group_view_kwargs.items()
                }

                cret = {}
                for view_name, view in group.items():
                    cview_kwargs = {
                        argname: kwargs[view_name] if view_name in kwargs else None
                        for argname, kwargs in view_kwargs.items()
                    }
                    cview_kwargs.update(
                        {
                            argname: kwargs[view_name] if kwargs is not None and view_name in kwargs else None
                            for argname, kwargs in cgroup_view_kwargs.items()
                        }
                    )

                    if sparse.issparse(view.X):
                        cX = self._align_sparse_array_to_var(view.X, group_name, view_name)
                    else:
                        cX = self._align_array_to_samples(view.X, group_name, view_name, axis=1, align_to="var")

                    adata = AnnData(X=cX, obs=view.obs, var=newvar[group_name][view_name])

                    cret[view_name] = func(adata, group_name, view_name, **kwargs, **cgroup_kwargs, **cview_kwargs)
                ret[group_name] = cret
        else:
            for view_name in self.view_names:
                cview_kwargs = {
                    argname: kwargs[view_name] if view_name in kwargs else None
                    for argname, kwargs in view_kwargs.items()
                }

                new_obs = []
                newX = []
                var = pd.DataFrame(index=self._aligned_var[view_name])
                for group_name, group in self._data.items():
                    if view_name in group:
                        view = group[view_name]
                        if sparse.issparse(view.X):
                            cX = self._align_sparse_array_to_var(view.X, group_name, view_name)
                        else:
                            cX = self._align_array_to_samples(view.X, group_name, view_name, axis=1, align_to="var")
                        newX.append(cX)
                        new_obs.append(view.obs)
                        var = var.join(view.var.drop(columns=view.var.columns, errors="ignore"), how="left", sort=False)

                if all(sparse.issparse(X) for X in newX):
                    newX = sparse.vstack(newX)
                elif all(not sparse.issparse(X) for X in newX):
                    newX = np.concatenate(newX, axis=0)
                else:
                    nelem_dense = sum([np.prod(X.shape) for X in newX])

                    nelem_sparse = sum(
                        [2 * (np.prod(X.shape) if not sparse.issparse(X) else X.nnz) + X.shape[0] + 1 for X in newX]
                    )  # assume CSR
                    if nelem_dense <= nelem_sparse:
                        newX = [np.asarray(X) for X in newX]
                        newX = np.concatenate(newX, axis=0)
                    else:
                        newX = [sparse.csr_array(X) if not np.issparse(X) else X for X in newX]

                new_obs = pd.concat(new_obs, axis=0)
                adata = AnnData(X=newX, obs=new_obs, var=var)
                ret[view_name] = func(adata, group_name, view_name, **kwargs, **cview_kwargs)

        return ret
