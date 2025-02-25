import logging
from collections import namedtuple

import numpy as np
from anndata import AnnData
from array_api_compat import array_namespace
from numpy.typing import NDArray
from scipy.sparse import issparse, sparray, spmatrix

from . import utils
from .datasets import Preprocessor, PrismoDataset

_logger = logging.getLogger(__name__)


ViewStatistics = namedtuple("ViewStatistics", ["mean", "min"])


class PrismoPreprocessor(Preprocessor):
    def __init__(
        self,
        dataset: PrismoDataset,
        likelihoods: dict[str, utils.Likelihood],
        nonnegative_weights: dict[str, bool],
        nonnegative_factors: dict[str, bool],
        scale_per_group: bool = True,
        constant_feature_var_threshold: float = 1e-16,
    ):
        super().__init__()

        self._scale_per_group = scale_per_group
        self._views_to_scale = {view_name for view_name, likelihood in likelihoods.items() if likelihood == "Normal"}
        self._nonnegative_weights = {k for k, v in nonnegative_weights.items() if v}
        self._nonnegative_factors = {k for k, v in nonnegative_factors.items() if v}

        view_vars = dataset.apply(lambda adata, group_name, view_name: utils.var(adata.X, axis=0), by_group=False)
        nonconstantfeatures = {}
        for view_name, viewvar in view_vars.items():
            # Storing a boolean mask is probably more memory-efficient than storing indices: indices are int64 (4 bytes), while
            # booleans are 1 byte. As long as we keep more than 1/ of the features this uses less memory.
            nonconst = viewvar > constant_feature_var_threshold
            _logger.debug(f"Removing {nonconst.size - nonconst.sum()} features from view {view_name}.")
            if issparse(nonconst):
                nonconst = nonconst.toarray()
            nonconstantfeatures[view_name] = dataset.feature_names[view_name][nonconst]

        dataset.reindex_features(nonconstantfeatures)

        meanfunc = lambda mod, group_name, view_name, axis: utils.mean(mod.X, axis=axis)
        self._feature_means = dataset.apply(meanfunc, axis=0)
        self._sample_means = dataset.apply(meanfunc, axis=1)

        self._viewstats = dataset.apply(
            lambda adata, group_name, view_name: ViewStatistics(
                utils.mean(adata.X, axis=0, keepdims=True), utils.min(adata.X, axis=0, keepdims=True)
            )
        )

        if self._scale_per_group:
            self._scale = dataset.apply(self._calc_scale_grouped, by_group=True)
        else:
            self._scale = dataset.apply(self._calc_scale_ungrouped, groups=dataset.group_names, by_group=False)

        for gstats in self._viewstats.values():
            for view_name, vstats in gstats.items():
                gstats[view_name] = ViewStatistics(*(x.toarray() if issparse(x) else x for x in vstats))

    def _calc_scale_ungrouped(self, adata: AnnData, group: NDArray[object], view_name: str, groups: list[str]):
        if view_name not in self._views_to_scale:
            return None

        arr = adata.X.copy()
        for group_name in groups:
            if view_name in self._nonnegative_weights and group_name in self._nonnegative_factors:
                attr = "min"
            else:
                attr = "mean"
            arr[group == group_name] -= align_local_array_to_global(  # noqa F821
                getattr(self._viewstats[group_name][view_name], attr),
                group_name,
                view_name,
                align_to="features",
                axis=0,
            )
        return np.sqrt(utils.var(arr, axis=None))

    def _calc_scale_grouped(self, adata: AnnData, group_name: str, view_name: str):
        if view_name not in self._views_to_scale:
            return None

        arr = self._center(adata.X, group_name, view_name)
        arr = utils.var(arr, axis=None)
        xp = array_namespace(arr)
        return xp.sqrt(arr)

    def _center(self, arr: NDArray, group_name: str, view_name: str):
        viewstats = self._viewstats[group_name][view_name]
        if view_name in self._nonnegative_weights and group_name in self._nonnegative_factors:
            arr = arr - viewstats.min  # can't use -= due to dask
        else:
            arr = arr - viewstats.mean
        return arr

    @property
    def feature_means(self) -> dict[str, dict[str, float]]:
        return self._feature_means

    @property
    def sample_means(self) -> dict[str, dict[str, float]]:
        return self._sample_means

    def __call__(
        self,
        arr: NDArray | sparray | spmatrix,
        nonmissing_samples: NDArray[int] | slice,
        nonmissing_features: NDArray[int] | slice,
        group: str,
        view: str,
    ) -> NDArray | sparray | spmatrix:
        if view in self._views_to_scale:
            arr = self._center(arr, group, view)
            arr /= self._scale[group][view] if self._scale_per_group else self._scale[view]

        return arr, nonmissing_samples, nonmissing_features


def infer_likelihood(view: AnnData, *args) -> utils.Likelihood:
    """Infer the likelihood for a view based on the data distribution."""
    data = view.X.data if issparse(view.X) else view.X
    xp = array_namespace(data)
    if xp.all(xp.isclose(data, 0) | xp.isclose(data, 1)):  # TODO: set correct atol value
        return "Bernoulli"
    elif xp.allclose(data, xp.round(data)) and data.min() >= 0:
        return "GammaPoisson"
    else:
        return "Normal"


def validate_likelihood(view: AnnData, group_name: str, view_name: str, likelihood: utils.Likelihood):
    """Validate the likelihood for a view based on the data distribution."""
    data = view.X.data if issparse(view.X) else view.X
    xp = array_namespace(data)
    if likelihood == "Bernoulli" and not xp.all(
        xp.isclose(data, 0) | xp.isclose(data, 1)
    ):  # TODO: set correct atol value
        raise ValueError(f"Bernoulli likelihood in view {view_name} must be used with binary data.")
    elif likelihood == "GammaPoisson" and not xp.allclose(data, xp.round(data)) and data.min() >= 0:
        raise ValueError(
            f"GammaPoisson likelihood in view {view_name} must be used with count (non-negative integer) data."
        )
