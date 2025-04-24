import logging
from typing import Any, Literal, TypeVar, Union

import anndata as ad
import numpy as np
import pandas as pd
from mudata import MuData
from numpy.typing import NDArray
from scipy import sparse

from ..settings import settings
from .base import ApplyCallable, MofaFlexDataset, Preprocessor
from .utils import anndata_to_dask, apply_to_nested, array_to_dask, from_dask, have_dask

T = TypeVar("T")
_logger = logging.getLogger(__name__)


def _mudata_to_dask(mudata: MuData, with_extra=True):
    mods = {modname: anndata_to_dask(mod) for modname, mod in mudata.mod.items()}
    dask_mudata = MuData(mods, obs=mudata.obs, var=mudata.var, obsmap=mudata.obsmap, varmap=mudata.varmap)
    # MuData constructor runs update(), so we need to reset obs and var
    dask_mudata.obs = mudata.obs
    dask_mudata.var = mudata.var
    if with_extra:
        for attrname in ("obsm", "obsp", "varm", "varp"):
            attr = getattr(mudata, attrname)
            dask_attr = getattr(dask_mudata, attrname)
            for k, v in attr.items():
                dask_attr[k] = array_to_dask(v)
    return dask_mudata


class MuDataDataset(MofaFlexDataset):
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
        self._orig_data = self._data
        self._group_by = group_by
        self._sample_selection = self._feature_selection = slice(None)
        self._groups = None

        self.reindex_samples(sample_names)
        self.reindex_features(feature_names)

    def reindex_samples(self, sample_names: dict[str, NDArray[str]] | None = None):
        if sample_names is not None and (
            self._groups is None
            or any(
                sample_names[group_name].size != group_idx.size
                or np.any(sample_names[group_name] != self._data.obs_names[group_idx])
                for group_name, group_idx in self._groups.items()
                if group_name in sample_names
            )
        ):
            groups = self._get_groups(self._orig_data.obs)
            selection = pd.Index(())
            for group_name, group_idx in groups.items():
                group_sample_names = sample_names.get(group_name)
                if group_sample_names is not None:
                    group_sample_names = pd.Index(group_sample_names)
                    if np.any(~group_sample_names.isin(self._orig_data.obs_names[group_idx])):
                        _logger.warning(
                            f"Not all sample names given for group {group_name} are present in the data. Restricting alignment to group names present in th e data."
                        )
                        group_sample_names = group_sample_names.intersection(self._orig_data.obs_names[group_idx])
                else:
                    group_sample_names = self._orig_data.obs_names[group_idx]
                selection = selection.append(group_sample_names)
            self._data = self._orig_data[selection, self._feature_selection]
            self._sample_selection = selection
        else:
            self._data = self._orig_data[:, self._feature_selection]
            self._sample_selection = slice(None)

        self._groups = self._get_groups(self._data.obs)

        self._needs_alignment = {}
        for group_name, group_idx in self._groups.items():
            gneeds_align = set()
            for view_name, obsmap in self._data.obsmap.items():
                obsmap = obsmap[group_idx]
                if np.any(obsmap == 0) or np.any(np.diff(obsmap) != 1):
                    gneeds_align.add(view_name)
            self._needs_alignment[group_name] = gneeds_align

    def _get_groups(self, df):
        return df.groupby(
            pd.Categorical(df[self._group_by]).rename_categories(lambda x: str(x))
            if self._group_by is not None
            else lambda x: "group_1",
            observed=True,
        ).indices

    def reindex_features(self, feature_names: dict[str, NDArray[str]] | None = None):
        if feature_names is not None and any(
            feature_names[view_name].size != fnames.size or np.any(feature_names[view_name] != fnames)
            for view_name, fnames in self.feature_names.items()
            if view_name in feature_names
        ):
            selection = pd.Index(())
            for modname, mod in self._orig_data.mod.items():
                view_feature_names = feature_names.get(modname)
                if view_feature_names is not None:
                    view_feature_names = pd.Index(view_feature_names)
                    if np.any(~view_feature_names.isin(mod.var_names)):
                        _logger.warning(
                            f"Not all feature names given for view {modname} are present in the data. Restricting alignment to feature names present in the data."
                        )
                        view_feature_names = view_feature_names.intersection(mod.var_names)
                else:
                    view_feature_names = mod.var_names
                selection = selection.append(view_feature_names)
            self._data = self._orig_data[self._sample_selection, selection]
            self._feature_selection = selection
        else:
            self._data = self._orig_data[self._sample_selection, :]
            self._feature_selection = slice(None)

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
                cnonmissing_obs = (
                    np.nonzero(subdata.obsmap[modname] > 0)[0]
                    if modname in self._needs_alignment[group_name]
                    else slice(None)
                )
                arr, gnonmissing_obs[modname], gnonmissing_var[modname] = self.preprocessor(
                    mod.X, cnonmissing_obs, slice(None), group_name, modname
                )
                if self.cast_to is not None:
                    arr = arr.astype(self.cast_to, copy=False)
                if sparse.issparse(arr):
                    arr = arr.toarray()
                group[modname] = np.asarray(
                    arr
                )  # arr may be an anndata._core.views.ArrayView, which is not recognized by PyTorch
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
            if view_name not in self._needs_alignment[group_name]:
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

    def map_local_indices_to_global(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal["samples, features"]
    ) -> NDArray[int]:
        if align_to == "samples":
            subdata = self._data[self._groups[group_name], :]
            viewidx = subdata.obsmap[view_name]
            return np.argsort(viewidx)[(viewidx == 0).sum() :][idx]
        else:
            return idx

    def map_global_indices_to_local(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal["samples, features"]
    ) -> NDArray[int]:
        if align_to == "samples":
            subdata = self._data[self._groups[group_name], :]
            return subdata.obsmap[view_name][idx].astype(int) - 1
        else:
            return idx

    def get_obs(self) -> dict[str, pd.DataFrame]:
        # We don't want to duplicate MuData's push_obs logic, but at the same time
        # we don't want to modify the data object. So we create a temporary fake
        # MuData object with the same metadata, but no actual data
        fakeadatas = {
            modname: ad.AnnData(X=sparse.csr_array(mod.X.shape), obs=mod.obs, var=mod.var)
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
                    modmissing = ~(np.asarray(modmissing.sum(axis=1)).squeeze() == 0)
                else:
                    modmissing = np.isnan(mod.X).all(axis=1)
                modmissing = self._align_array_to_samples(modmissing, modname, subdata, fill_value=True)
                dfs.append(
                    pd.DataFrame(
                        {"view": modname, "group": group_name, "obs_name": subdata.obs_names, "missing": modmissing}
                    )
                )
        return pd.concat(dfs, axis=0, ignore_index=True)

    def get_covariates(
        self, obs_key: dict[str, str] | None = None, obsm_key: dict[str, str] | None = None
    ) -> tuple[dict[str, dict[str, NDArray]], dict[str, NDArray]]:
        covariates, covariates_names = {}, {}

        if obs_key is None:
            obs_key = {}
        if obsm_key is None:
            obsm_key = {}
        for group_name, group_idx in self._groups.items():
            obskey = obs_key.get(group_name, None)
            obsmkey = obsm_key.get(group_name, None)
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
                        ccov = self._align_array_to_samples(mod.obs[obskey].to_numpy(), modname, subdata)[:, None]
                    elif obskey in subdata.obs.columns:
                        ccov = subdata.obs[obskey].to_numpy()
                    if ccov is not None:
                        ccovs[modname] = ccov

                if len(ccovs):
                    covariates_names[group_name] = obskey
                else:
                    _logger.warn(f"No covariate data found in obs attribute for group {group_name}.")
            elif obsmkey is not None:
                covar_dim = set()
                for modname, mod in subdata.mod.items():
                    covar = None
                    needs_alignment = False
                    if obsmkey in mod.obsm:
                        covar = mod.obsm[obsmkey]
                        needs_alignment = True
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
                        covar_dim.add(covar.shape[1])

                        if needs_alignment:
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
                        annotations_names[modname] = annot.columns
                        annotations[modname] = annot.to_numpy().T
                    else:
                        annotations[modname] = annot.T
                elif key in self._data.varm:
                    annot = self._data.varm[key]
                    varidx = self._data.varmap[modname] > 0
                    if isinstance(annot, pd.DataFrame):
                        annotations_names[modname] = annot.columns
                        annotations[modname] = annot.iloc[varidx, :].to_numpy().T
                    else:
                        annotations[modname] = annot[varidx].T
        return annotations, annotations_names

    def _apply_by_group_view(
        self, func: ApplyCallable[T], gvkwargs: dict[str, dict[str, dict[str, Any]]], **kwargs
    ) -> dict[str, dict[str, T]]:
        if settings.use_dask:
            if have_dask():
                data = _mudata_to_dask(self._orig_data, with_extra=False)[
                    self._sample_selection, self._feature_selection
                ]
            else:
                _logger.warning("Could not import dask. Data arrays may be copied, resulting in high memory usage.")
                data = self._data
        ret = {}
        for group_name, group_idx in self._groups.items():
            cret = {}
            for modname, mod in data[group_idx, :].mod.items():
                ccret = func(mod, group_name, modname, **kwargs, **gvkwargs[group_name][modname])
                cret[modname] = apply_to_nested(ccret, from_dask)
            ret[group_name] = cret
        return ret

    def _apply_by_view(self, func: ApplyCallable[T], vkwargs: dict[str, dict[str, Any]], **kwargs) -> dict[str, T]:
        data = self._data
        if (self._sample_selection != slice(None) or self._feature_selection != slice(None)) and settings.use_dask:
            if have_dask():
                data = _mudata_to_dask(self._orig_data, with_extra=False)[
                    self._sample_selection, self._feature_selection
                ]
            else:
                _logger.warning("Could not import dask. Data arrays may be copied, resulting in high memory usage.")
        ret = {}
        for modname, mod in data.mod.items():
            groups = np.empty((mod.n_obs,), dtype="O")
            for group, group_idx in self._groups.items():
                modidx = self._data.obsmap[modname][group_idx]
                modidx = modidx[modidx > 0] - 1
                groups[modidx] = group

            cret = func(mod, groups, modname, **kwargs, **vkwargs[modname])
            ret[modname] = apply_to_nested(cret, from_dask)
        return ret

    def _apply_by_group(self, func: ApplyCallable[T], gkwargs: dict[str, dict[str, Any]], **kwargs) -> dict[str, T]:
        data = self._data
        if settings.use_dask:
            if have_dask():
                data = _mudata_to_dask(self._orig_data, with_extra=False)[
                    self._sample_selection, self._feature_selection
                ]
            else:
                _logger.warning("Could not import dask. Will copy all input arrays for stacking.")
        ret = {}
        for group_name, group_idx in self._groups.items():
            subdata = data[group_idx, :]
            gdata = {}
            convert = False
            for modname, mod in subdata.mod.items():
                if mod.n_obs != subdata.n_obs:
                    convert = True
                gdata[modname] = mod
            if convert:
                for modname, mod in gdata.items():
                    mod = mod.copy()
                    mod.X = mod.X.astype(np.promote_types(mod.X.dtype, type(np.nan)))
                    gdata[modname] = mod
            gdata = ad.concat(
                gdata, axis="var", join="outer", label="__view", merge="unique", uns_merge=None, fill_value=np.nan
            )
            if (gdata.obs_names != subdata.obs_names).any():
                gdata = gdata[subdata.obs_names, :]
            cret = func(gdata, group_name, gdata.var["__view"].to_numpy(), **kwargs, **gkwargs[group_name])
            ret[group_name] = apply_to_nested(cret, from_dask)
        return ret
