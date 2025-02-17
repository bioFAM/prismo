import logging
from functools import reduce
from typing import Any, Literal, TypeVar

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from .base import ApplyCallable, Preprocessor, PrismoDataset
from .utils import anndata_to_dask, apply_to_nested, from_dask, have_dask

T = TypeVar("T")
_logger = logging.getLogger(__name__)


class AnnDataDictDataset(PrismoDataset):
    def __init__(
        self,
        data: dict[str, dict[str, ad.AnnData]],
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
        self._use_obs = use_obs
        self._use_var = use_var

        obsfunc = (lambda x, y: x.intersection(y)) if use_obs == "intersection" else lambda x, y: x.union(y)
        varfunc = (lambda x, y: x.intersection(y)) if use_var == "intersection" else lambda x, y: x.union(y)

        for view_name in self.view_names:
            self._aligned_var[view_name] = reduce(
                varfunc, (group[view_name].var_names for group in self._data.values() if view_name in group)
            )

        for group_name, group in self._data.items():
            aligned_obs = reduce(obsfunc, (view.obs_names for view in group.values()))
            gobsmap = {}
            gvarmap = {}
            for view_name, view in group.items():
                obsmap = aligned_obs.get_indexer(view.obs_names)
                varmap = self._aligned_var[view_name].get_indexer(view.var_names)

                if np.sum(obsmap < 0) > 0 or np.any(np.diff(obsmap) != 1):
                    gobsmap[view_name] = obsmap
                if np.sum(varmap < 0) > 0 or np.any(np.diff(varmap) != 1):
                    gvarmap[view_name] = varmap
            self._obsmap[group_name] = gobsmap
            self._varmap[group_name] = varmap
            self._aligned_obs[group_name] = aligned_obs

    @staticmethod
    def _accepts_input(data):
        return isinstance(data, dict) and all(
            isinstance(group, dict) and all(isinstance(view, ad.AnnData) for view in group.values())
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

    def _get_minibatch(
        self, idx: dict[str, int], return_nonmissing: bool = True
    ) -> tuple[dict[str, dict[str, NDArray]], dict[str, int]]:
        data = {}
        if return_nonmissing:
            nonmissing_obs = {}
            nonmissing_var = {}
        for group_name, group_idx in idx.items():
            group = {}
            gobsmap = self._obsmap[group_name]
            gvarmap = self._varmap[group_name]
            if return_nonmissing:
                gnonmissing_obs = {}
                gnonmissing_var = {}
            for view_name, view in self._data[group_name].items():
                if view_name in gobsmap:
                    obsmap = gobsmap[view_name][group_idx]
                    obsidx = obsmap >= 0
                    arr = view.X[obsmap[obsidx], :]
                    if return_nonmissing:
                        gnonmissing_obs[view_name] = np.nonzero(obsidx)[0]
                else:
                    arr = view.X[group_idx, :]
                    if return_nonmissing:
                        gnonmissing_obs[view_name] = slice(None)
                    else:
                        obsmap = None

                if return_nonmissing:
                    if view_name in gvarmap:
                        gnonmissing_var[view_name] = np.nonzero(gvarmap[view_name] >= 0)[0]
                    else:
                        gnonmissing_var[view_name] = slice(None)

                arr = self.preprocessor(arr, group_name, view_name).astype(self.cast_to)
                if sparse.issparse(arr):
                    arr = arr.toarray()
                if not return_nonmissing:
                    arr = self._align_array_to_samples(arr, group_name, view_name, axis=(0, 1), obsmap=obsmap)
                group[view_name] = arr
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

    def __getitem__(self, idx: dict[str, int]) -> tuple[dict[str, dict[str, NDArray]], dict[str, int]]:
        return self._get_minibatch(idx, return_nonmissing=False)

    def __getitems__(self, idx: dict[str, int | list[int]]) -> dict[str, dict]:
        return self._get_minibatch(idx)

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
            axis = (axis,)
            align_to_both = False
        else:
            align_to_both = True

        if obsmap is None:
            need_obs_align = view_name in self._obsmap[group_name]
            if need_obs_align:
                obsmap = self._obsmap[group_name][view_name]
        else:
            need_obs_align = True

        if varmap is None:
            need_var_align = view_name in self._varmap[group_name]
            if need_var_align:
                varmap = self._varmap[group_name][view_name]
        else:
            need_var_align = True

        if (
            not align_to_both
            and (align_to == "obs" and not need_obs_align or align_to == "var" and not need_var_align)
            or align_to_both
            and not need_obs_align
            and not need_var_align
        ):
            return arr
        elif align_to_both:
            if not need_obs_align:
                align_to_both = False
                align_to = "var"
                axis = (axis[1],)
            elif not need_var_align:
                align_to_both = False
                align_to = "obs"
                axis = (axis[0],)

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
        return self._align_array_to_samples(
            arr, group_name, view_name, axis=axis, align_to="obs", fill_value=fill_value
        )

    def align_array_to_features(
        self, arr: NDArray[T], group_name: str, view_name: str, axis: int = 1, fill_value: np.ScalarType = np.nan
    ) -> NDArray[T]:
        return self._align_array_to_samples(
            arr, group_name, view_name, axis=axis, align_to="var", fill_value=fill_value
        )

    def align_array_to_data_samples(
        self, arr: NDArray[T], group_name: str, view_name: str, axis: int = 0
    ) -> NDArray[T]:
        if view_name not in self._obsmap[group_name]:
            return arr
        idx = self._obsmap[group_name][view_name]
        return np.take(arr, np.argsort(idx)[(idx < 0).sum() :], axis=axis)

    def align_array_to_data_features(
        self, arr: NDArray[T], group_name: str, view_name: str, axis: int = 1
    ) -> NDArray[T]:
        if view_name not in self._varmap[group_name]:
            return arr
        idx = self._varmap[group_name][view_name]
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

        havedask = have_dask()

        ret = {}
        if by_group:
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

                    cret[view_name] = func(view, group_name, view_name, **kwargs, **cgroup_kwargs, **cview_kwargs)
                ret[group_name] = cret
        else:
            for view_name in self.view_names:
                cview_kwargs = {
                    argname: kwargs[view_name] if view_name in kwargs else None
                    for argname, kwargs in view_kwargs.items()
                }

                if not havedask:
                    _logger.warning("Could not import dask. Will copy all input arrays for stacking.")

                data = {}
                for group_name, group in self._data.items():
                    data[group_name] = anndata_to_dask(group[view_name]) if havedask else group[view_name]
                data = ad.concat(
                    data,
                    join="inner" if self._use_var == "intersection" else "outer",
                    label="group",
                    merge="unique",
                    uns_merge=None,
                )
                if (data.var_names != self._aligned_var[view_name]).any():
                    data = data[:, self._aligned_var[view_name]]

                cret = func(data, group_name, view_name, **kwargs, **cview_kwargs)
                ret[view_name] = apply_to_nested(cret, from_dask)

        return ret
