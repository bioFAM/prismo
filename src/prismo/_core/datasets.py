import logging
from collections.abc import Callable
from typing import Any, Concatenate, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from numpy.typing import NDArray
from scipy import sparse
from torch.utils.data import Dataset, RandomSampler, Sampler

T = TypeVar("T")
ApplyCallable: TypeAlias = Callable[Concatenate[AnnData, str, str, ...], T]

_logger = logging.getLogger(__name__)


class Preprocessor:
    def __call__(self, arr: NDArray, group: str, view: str):
        return arr


class PrismoSampler(Sampler[dict[str, int]]):
    """A sampler for dicts.

    Given a dict with arbitrary keys and values indicating the number of data points in
    individual atasets, creates dicts of indices, such that the largest dataset is
    sampled without replacement, while for the smaller datasets multiple permutations
    are concatenated to yield the length of the largest dataset.
    """

    def __init__(self, n_samples: dict[str, int]):
        super().__init__()
        self._n_samples = n_samples
        self._largestgroup = max(n_samples.values())
        self._samplers = {
            k: RandomSampler(range(nsamples), num_samples=self._largestgroup) for k, nsamples in self._n_samples.items()
        }

    def __len__(self):
        return self._largestgroup

    def __iter__(self):
        iterators = {k: iter(smplr) for k, smplr in self._samplers.items()}
        for _ in range(len(self)):
            yield {k: next(smplr) for k, smplr in iterators.items()}


class MuDataDataset(Dataset):
    def __init__(
        self,
        mudata: MuData,
        group_by: str | list[str] | None = None,
        preprocessor: Preprocessor | None = None,
        cast_to: np.ScalarType = np.float32,
    ):
        super().__init__()
        self._data = mudata
        self._groups = mudata.obs.groupby(group_by if group_by is not None else lambda x: "group_1").groups
        self._viewstats = self._viewstats_per_group = self._viewstats_total = self._viewstats_total_per_group = (
            self._samplestats
        ) = None

        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = Preprocessor()

        self._cast_to = cast_to

    @property
    def preprocessor(self) -> Preprocessor:
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, preproc: Preprocessor):
        self._preprocessor = preproc

    @property
    def cast_to(self) -> np.ScalarType:
        return self._cast_to

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
    def view_names(self) -> list[str]:
        return list(self._data.mod.keys())

    @property
    def group_names(self) -> list[str]:
        return list(self._groups.keys())

    @property
    def sample_names(self) -> dict[str, list[str]]:
        return {groupname: self._data[groupidx, :].obs_names.to_list() for groupname, groupidx in self._groups.items()}

    @property
    def feature_names(self) -> dict[str, list[str]]:
        return {viewname: mod.var_names.to_list() for viewname, mod in self._data.mod.items()}

    def __len__(self):
        return max(self.n_samples.values())

    def __getitem__(self, idx: dict[str, int]) -> tuple[dict[str, dict[str, NDArray]], dict[str, int]]:
        ret = {}
        for group_name, group_idx in idx.items():
            group = {}
            glabel = self._groups[group_name][group_idx]
            for modname, mod in self._data[glabel, :].mod.items():
                arr = mod.X
                if arr.shape[0] == 0:
                    arr = np.full((1, arr.shape[1]), np.nan, dtype=arr.dtype)
                elif sparse.issparse(arr):
                    arr = arr.todense()
                group[modname] = self._preprocessor(arr, group_name, modname).astype(self._cast_to)
        return {"data": ret, "sample_idx": idx}

    def _align_array_to_samples(
        self,
        arr: NDArray,
        view_name: str,
        subdata: MuData | None = None,
        group_name: str | None = None,
        fill_value: np.ScalarType = np.nan,
    ):
        if subdata is None:
            if group_name is None:
                raise ValueError("Need either subdata or group_name, but both are None.")
            subdata = self._data[self._groups[group_name], :]

        outshape = list(arr.shape)
        outshape[0] = subdata.n_obs
        out = np.full(outshape, fill_value=fill_value, dtype=arr.dtype)

        viewidx = subdata.obsmap[view_name]
        nnz = viewidx > 0
        out[nnz] = arr[viewidx[nnz] - 1]
        return out

    def align_array_to_samples(self, arr: NDArray, view_name: str, group_name: str, fill_value: np.ScalarType = np.nan):
        return self._align_array_to_samples(arr, view_name, group_name=group_name, fill_value=fill_value)

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
                modname: mod.obs.reindex(self._data[group_idx, :].obs_names) for modname, mod in fakemudata.mod.items()
            }
            for group_name, group_idx in self._groups.items()
        }

    def get_missing_obs(self) -> pd.DataFrame:
        dfs = []
        for group_name, group_idx in self._groups.items():
            subdata = self._data[group_idx, :]
            for modname, mod in subdata.mod.items():
                if sparse.issparse(mod.X):
                    modmissing = (
                        sparse.csr_array((np.isnan(mod.X.data), mod.X.indices, mod.X.indptr), shape=mod.X.shape).sum(
                            axis=1
                        )
                        > 0
                    )
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
            if (
                group_name not in covariates_obs_key
                and group_name not in covariates_obsm_key
                or obskey is None
                and obsmkey is None
            ):
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
                    annotations[modname] = self._data[modname].varm[key]
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
            for group_name, group_idx in self._groups.items():
                cgroup_kwargs = {
                    argname: kwargs[group_name] for argname, kwargs in group_kwargs.items() if group_name in kwargs
                }
                cgroup_view_kwargs = {
                    argname: kwargs[group_name] for argname, kwargs in group_view_kwargs.items() if group_name in kwargs
                }

                cret = {}
                for modname, mod in self._data[group_idx, :].mod.items():
                    cview_kwargs = {
                        argname: kwargs[modname] for argname, kwargs in view_kwargs.items() if modname in kwargs
                    }
                    cview_kwargs.update(
                        {
                            argname: kwargs[modname]
                            for argname, kwargs in cgroup_view_kwargs.items()
                            if modname in kwargs
                        }
                    )
                    cret[modname] = func(mod, group_name, modname, **kwargs, **cgroup_kwargs, **cview_kwargs)
                ret[group_name] = cret
        else:
            for modname, mod in self._data.mod.items():
                cview_kwargs = {
                    argname: kwargs[modname] for argname, kwargs in view_kwargs.items() if modname in kwargs
                }
                ret[modname] = func(mod, None, modname, **kwargs, **cview_kwargs)
        return ret


class CovariatesDataset(Dataset):
    def __init__(
        self,
        data: MuDataDataset,
        covariates_obs_key: dict[str, str] | None = None,
        covariates_obsm_key: dict[str, str] | None = None,
    ):
        super().__init__()

        self.covariates, self.covariates_names = data.get_covariates(covariates_obs_key, covariates_obsm_key)
        self.covariates = {
            group_name: np.nanmean(np.stack(tuple(group_covars.values()), axis=0), axis=0)
            for group_name, group_covars in self.covariates.items()
        }
        self._n_samples = max(data.n_samples.values())
        self._cast_to = data.cast_to

    def __len__(self):
        return self._n_samples

    def __getitem__(self, idx: dict[str, int]) -> dict[str, NDArray]:
        return {
            group_name: self.covariates[group_name][group_idx, :].astype(self._cast_to)
            for group_name, group_idx in idx.items()
            if group_name in self.covariates
        }
