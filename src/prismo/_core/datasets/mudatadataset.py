import logging
from typing import Any, Literal, TypeVar, Union

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
        cast_to: Union[np.ScalarType] | None = np.float32,  # noqa UP007
        sample_names: dict[str, NDArray[str]] | None = None,
        feature_names: dict[str, NDArray[str]] | None = None,
        **kwargs,
    ):
        super().__init__(mudata, preprocessor=preprocessor, cast_to=cast_to)
        self._groups = self._data.obs.groupby(group_by if group_by is not None else lambda x: "group_1").indices
        if feature_names is not None:
            need_update = False
            for view_name, view_feature_names in feature_names.items():
                if np.any(view_feature_names != self._data.mod[view_name].var_names):
                    self._data.mod[view_name] = self._data.mod[view_name][:, view_feature_names]
                    need_update = True
            if need_update:
                self._data.update_var()
        if sample_names is not None and any(
            np.any(sample_names[group_name] != self._data.obs_names[group_idx])
            for group_name, group_idx in self._groups.items()
        ):
            obs_idx = np.concatenate(
                [
                    self._data.obs_names[group_idx].get_indexer(sample_names[group_name])
                    for group_name, group_idx in self._groups.items()
                ]
            )
            self._data = self._data[obs_idx, :]
            if group_by is not None:
                self._groups = self._data.obs.groupby(group_by).indices

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

    def __getitems__(self, idx: dict[str, int | list[int]]) -> dict[str, dict]:
        data = {}
        nonmissing_obs = {}
        nonmissing_var = {}
        for group_name, group_idx in idx.items():
            group = {}
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
                group[modname] = arr
                gnonmissing_obs[modname] = (
                    np.nonzero(subdata.obsmap[modname] > 0)[0]
                    if self._needs_alignment[group_name][modname]
                    else slice(None)
                )
            data[group_name] = group
            idx[group_name] = np.asarray(group_idx)
            nonmissing_obs[group_name] = gnonmissing_obs
            nonmissing_var[group_name] = gnonmissing_var
        return {
            "data": data,
            "sample_idx": idx,
            "nonmissing_samples": nonmissing_obs,
            "nonmissing_features": nonmissing_var,
        }

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

        out = np.full(outshape, fill_value=fill_value, dtype=np.promote_types(type(fill_value), arr.dtype), order="C")
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
                modmissing = self._align_array_to_samples(modmissing, modname, subdata, fill_value=True)
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

    def _apply_by_group(
        self, func: ApplyCallable[T], gvkwargs: dict[str, dict[str, dict[str, Any]]], **kwargs
    ) -> dict[str, dict[str, T]]:
        ret = {}
        for group_name, group_idx in self._groups.items():
            cret = {}
            for modname, mod in self._data[group_idx, :].mod.items():
                cret[modname] = func(mod, group_name, modname, **kwargs, **gvkwargs[group_name][modname])
            ret[group_name] = cret
        return ret

    def _apply(self, func: ApplyCallable[T], vkwargs: dict[str, dict[str, Any]], **kwargs) -> dict[str, dict[str, T]]:
        ret = {}
        for modname, mod in self._data.mod.items():
            groups = np.empty((mod.n_obs,), dtype="O")
            for group, group_idx in self._groups.items():
                modidx = self._data.obsmap[modname][group_idx]
                modidx = modidx[modidx > 0] - 1
                groups[modidx] = group

            ret[modname] = func(mod, groups, modname, **kwargs, **vkwargs[modname])
        return ret
