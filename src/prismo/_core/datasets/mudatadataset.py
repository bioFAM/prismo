import logging
from typing import Any, Literal, TypeVar

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from numpy.typing import NDArray
from scipy import sparse

from .base import ApplyCallable, Preprocessor, PrismoDataset

T = TypeVar("T")
_logger = logging.getLogger(__name__)


class MuDataDataset(PrismoDataset):
    def __init__(
        self,
        mudata: MuData,
        *,
        group_by: str | list[str] | None = None,
        preprocessor: Preprocessor | None = None,
        cast_to: np.ScalarType = np.float32,
        **kwargs,
    ):
        super().__init__(mudata, preprocessor=preprocessor, cast_to=cast_to)
        self._groups = self._data.obs.groupby(group_by if group_by is not None else lambda x: "group_1").indices
        self._needs_alignment = {}
        for group_name, group_idx in self._groups.items():
            gneeds_align = {}
            for view_name, obsmap in self._data.obsmap.items():
                obsmap = obsmap[group_idx]
                gneeds_align[view_name] = np.any(obsmap == 0) or not np.any(np.diff(obsmap) != 1)
            self._needs_alignment[group_name] = gneeds_align

    @staticmethod
    def _accepts_input(data):
        return isinstance(data, MuData)

    @property
    def n_features(self) -> dict[str, int]:
        return {modname: mod.n_vars for modname, mod in self._data.mod.items()}

    @property
    def n_samples(self) -> dict[str, int]:
        return {groupname: len(groupidx) for groupname, groupidx in self._groups.items()}

    @property
    def n_samples_total(self) -> int:
        return self._data.n_obs

    @property
    def view_names(self) -> NDArray[str]:
        return np.asarray(tuple(self._data.mod.keys()))

    @property
    def group_names(self) -> NDArray[str]:
        return np.asarray(tuple(self._groups.keys()))

    @property
    def sample_names(self) -> dict[str, NDArray[str]]:
        return {groupname: self._data[groupidx, :].obs_names.to_numpy() for groupname, groupidx in self._groups.items()}

    @property
    def feature_names(self) -> dict[str, NDArray[str]]:
        return {viewname: mod.var_names.to_numpy() for viewname, mod in self._data.mod.items()}

    def _get_minibatch(self, idx: dict[str, int | list[int]], return_nonmissing: bool = True) -> dict[str, dict]:
        data = {}
        if return_nonmissing:
            nonmissing_obs = {}
            nonmissing_var = {}
        for group_name, group_idx in idx.items():
            group = {}
            if return_nonmissing:
                gnonmissing_obs = {}
                gnonmissing_var = {}
            glabel = self._groups[group_name][group_idx]
            subdata = self._data[glabel, :]
            for modname, mod in subdata.mod.items():
                gnonmissing_var[modname] = slice(None)
                arr = mod.X
                arr = self.preprocessor(arr, group_name, modname).astype(self._cast_to)
                if sparse.issparse(arr):
                    arr = arr.toarray()
                if return_nonmissing:
                    group[modname] = arr
                    gnonmissing_obs[modname] = (
                        np.nonzero(subdata.obsmap[modname] > 0)[0]
                        if self._needs_alignment[group_name][modname]
                        else slice(None)
                    )
                else:
                    group[modname] = self._align_array_to_samples(arr, modname, subdata=subdata)
            data[group_name] = group
            idx[group_name] = np.asarray(group_idx)
            if return_nonmissing:
                nonmissing_obs[group_name] = gnonmissing_obs
                nonmissing_var[group_name] = gnonmissing_var
        ret = {"data": data, "sample_idx": idx}
        if return_nonmissing:
            ret["nonmissing_samples"] = nonmissing_obs
            ret["nonmissing_features"] = nonmissing_var
        return ret

    def __getitem__(self, idx: dict[str, int]) -> dict[str, dict]:
        return self._get_minibatch(idx, return_nonmissing=False)

    def __getitems__(self, idx: dict[str, int | list[int]]) -> dict[str, dict]:
        return self._get_minibatch(idx)

    def _align_array_to_samples(
        self,
        arr: NDArray[T],
        view_name: str,
        subdata: MuData | None = None,
        group_name: str | None = None,
        axis: int = 0,
        fill_value: np.ScalarType = np.nan,
    ) -> NDArray[T]:
        if subdata is None:
            if group_name is None:
                raise ValueError("Need either subdata or group_name, but both are None.")
            if not self._needs_alignment[group_name][view_name]:
                return arr
            subdata = self._data[self._groups[group_name], :]

        viewidx = subdata.obsmap[view_name]
        nnz = viewidx > 0

        outshape = [subdata.n_obs] + list(arr.shape[:axis]) + list(arr.shape[axis + 1 :])

        out = np.full(outshape, fill_value=fill_value, dtype=arr.dtype, order="C")
        out[nnz, ...] = np.moveaxis(arr, axis, 0)[viewidx[nnz] - 1, ...]
        return np.moveaxis(out, 0, axis)

    def align_local_array_to_global(
        self,
        arr: NDArray[T],
        group_name: str,
        view_name: str,
        align_to: Literal["samples", "features"],
        axis: int = 0,
        fill_value: np.ScalarType = np.nan,
    ):
        if align_to == "samples":
            return self._align_array_to_samples(arr, view_name, group_name=group_name, axis=axis, fill_value=fill_value)
        else:
            return arr

    def align_global_array_to_local(
        self, arr: NDArray[T], group_name: str, view_name: str, align_to: Literal["samples", "features"], axis: int = 0
    ) -> NDArray[T]:
        if align_to == "samples":
            subdata = self._data[self._groups[group_name], :]
            idx = subdata.obsmap[view_name]
            return np.take(arr, np.argsort(idx)[(idx == 0).sum() :], axis=axis)
        else:
            return arr

    def get_obs(self) -> dict[str, pd.DataFrame]:
        # We don't want to duplicate MuData's push_obs logic, but at the same time
        # we don't want to modify the data object. So we create a temporary fake
        # MuData object with the same metadata, but no actual data
        fakeadatas = {
            modname: AnnData(X=sparse.csr_array(mod.X.shape), obs=mod.obs, var=mod.var)
            for modname, mod in self._data.mod.items()
        }

        # need to pass obs in the constructor to make shape validation for obsmap work
        fakemudata = MuData(fakeadatas, obs=self._data.obs, obsmap=self._data.obsmap)
        # need to replace obs since the constructor runs update(), which breaks push_obs()
        fakemudata.obs = self._data.obs
        fakemudata.push_obs()
        return {
            group_name: {
                modname: mod.obs.reindex(self._data[group_idx, :].obs_names, fill_value=pd.NA).apply(
                    lambda x: x.astype("string") if x.dtype == "O" else x, axis=1
                )
                for modname, mod in fakemudata.mod.items()
            }
            for group_name, group_idx in self._groups.items()
        }

    def get_missing_obs(self) -> pd.DataFrame:
        dfs = []
        for group_name, group_idx in self._groups.items():
            subdata = self._data[group_idx, :]
            for modname, mod in subdata.mod.items():
                if sparse.issparse(mod.X):
                    modmissing = mod.X.copy()
                    modmissing.data = np.isnan(modmissing.data)
                    modmissing = np.asarray(modmissing.sum(axis=1)).squeeze() > 0
                else:
                    modmissing = np.isnan(mod.X).any(axis=1)
                modmissing = self._align_array_to_samples(modmissing, modname, subdata, fill_value=1)
                dfs.append(
                    pd.DataFrame(
                        {"view": modname, "group": group_name, "obs_name": subdata.obs_names, "missing": modmissing}
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
        for group_name, group_idx in self._groups.items():
            obskey = covariates_obs_key.get(group_name, None)
            obsmkey = covariates_obsm_key.get(group_name, None)
            if obskey is None and obsmkey is None:
                continue
            if obskey and obsmkey:
                raise ValueError(
                    f"Provide either covariates_obs_key or covariates_obsm_key for group {group_name}, not both."
                )

            ccovs = {}
            subdata = self._data[group_idx, :]
            if obskey is not None:
                for modname, mod in subdata.mod.items():
                    ccov = None
                    if obskey in mod.obs.columns:
                        ccov = mod.obs[obskey].to_numpy()
                    elif obskey in subdata.obs.columns:
                        ccov = subdata.obs[obskey].to_numpy()

                    if ccov is not None:
                        ccovs[modname] = self._align_array_to_samples(ccov, modname, subdata)[:, None]

                if len(ccovs):
                    covariates_names[group_name] = obskey
                else:
                    _logger.warn(f"No covariate data found in obs attribute for group {group_name}.")
            elif obsmkey is not None:
                covar_dim = []
                for modname, mod in subdata.mod.items():
                    covar = None
                    if obsmkey in mod.obsm:
                        covar = mod.obsm[obsmkey]
                    elif obsmkey in subdata.obsm:
                        covar = subdata.obsm[obsmkey]
                    if covar is not None:
                        if isinstance(covar, pd.DataFrame):
                            covariates_names[group_name] = covar.columns.to_numpy()
                            covar = covar.to_numpy()
                        elif isinstance(covar, pd.Series):
                            covariates_names[group_name] = np.asarray(covar.name, dtype=object)
                            covar = covar.to_numpy()
                        elif sparse.issparse(covar):
                            covar = covar.todense()
                        if covar.ndim == 1:
                            covar = covar[..., None]
                        covar_dim.append(covar.shape[1])

                        ccovs[modname] = self._align_array_to_samples(covar, modname, subdata)
                if len(covar_dim) > 1:
                    raise ValueError(
                        f"Number of covariate dimensions in group {group_name} must be the same across views."
                    )

            covariates[group_name] = ccovs
        return covariates, covariates_names

    def get_annotations(self, varm_key: dict[str, str]) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
        annotations, annotations_names = {}, {}
        if varm_key is not None:
            for modname, key in varm_key.items():
                if key in self._data[modname].varm:
                    annot = self._data[modname].varm[key]
                    if isinstance(annot, pd.DataFrame):
                        annotations_names[modname] = annot.columns.to_list()
                        annotations[modname] = annot.to_numpy().T
                    else:
                        annotations[modname] = annot.T
                elif key in self._data.varm:
                    annot = self._data.varm[key]
                    varidx = self._data.varmap[modname]
                    varidx = varidx[varidx > 0] - 1
                    if isinstance(annot, pd.DataFrame):
                        annotations_names[modname] = annot.columns.to_list()
                        annotations[modname] = annot.iloc[varidx, :].to_numpy().T
                    else:
                        annotations[modname] = annot[varidx].T
        return annotations, annotations_names

    def _apply(
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

        ret = {}
        if by_group:
            for group_name, group_idx in self._groups.items():
                cgroup_kwargs = {
                    argname: kwargs[group_name] if group_name in kwargs else None
                    for argname, kwargs in group_kwargs.items()
                }
                cgroup_view_kwargs = {
                    argname: kwargs[group_name] if group_name in kwargs else None
                    for argname, kwargs in group_view_kwargs.items()
                }

                cret = {}
                for modname, mod in self._data[group_idx, :].mod.items():
                    cview_kwargs = {
                        argname: kwargs[modname] if modname in kwargs else None
                        for argname, kwargs in view_kwargs.items()
                    }
                    cview_kwargs.update(
                        {
                            argname: kwargs[modname] if kwargs is not None and modname in kwargs else None
                            for argname, kwargs in cgroup_view_kwargs.items()
                        }
                    )
                    cret[modname] = func(mod, group_name, modname, **kwargs, **cgroup_kwargs, **cview_kwargs)
                ret[group_name] = cret
        else:
            for modname, mod in self._data.mod.items():
                cview_kwargs = {
                    argname: kwargs[modname] if modname in kwargs else None for argname, kwargs in view_kwargs.items()
                }
                ret[modname] = func(mod, None, modname, **kwargs, **cview_kwargs)
        return ret
