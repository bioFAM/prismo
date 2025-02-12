import logging
from collections import namedtuple

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import issparse

from . import utils
from .datasets import Preprocessor, PrismoDataset

_logger = logging.getLogger(__name__)


ViewStatistics = namedtuple("ViewStatistics", ["mean", "var", "min"])


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
        self._constant_feature_var_threshold = constant_feature_var_threshold
        self._views_to_scale = {view_name for view_name, likelihood in likelihoods.items() if likelihood == "Normal"}
        self._nonnegative_weights = {k for k, v in nonnegative_weights.items() if v}
        self._nonnegative_factors = {k for k, v in nonnegative_factors.items() if v}

        getstats = lambda mod, group_name, view_name, axis=0: ViewStatistics(
            utils.mean(mod.X, axis=axis), utils.var(mod.X, axis=axis), utils.min(mod.X, axis=axis)
        )
        self._feature_means = {
            group_name: {view_name: stats.mean for view_name, stats in group.items()}
            for group_name, group in dataset.apply(getstats).items()
        }
        self._sample_means = dataset.apply(
            lambda mod, group_name, view_name: dataset.align_array_to_samples(
                utils.mean(mod.X, axis=1), group_name=group_name, view_name=view_name
            )
        )

        self._viewstats = dataset.apply(getstats, by_group=False)
        self._nonconstantfeatures = {}
        for view_name, viewstats in self._viewstats.items():
            # Storing a boolean mask is probably more memory-efficient than storing indices: indices are int64 (4 bytes), while
            # booleans are 1 byte. As long as we keep more than 1/ of the features this uses less memory.
            nonconst = viewstats.var > self._constant_feature_var_threshold
            _logger.debug(f"Removing {viewstats.var.size - nonconst.sum()} features from view {view_name}.")
            self._nonconstantfeatures[view_name] = nonconst

            self._viewstats[view_name] = ViewStatistics(*(stat[nonconst] for stat in viewstats))

        self._scale = dataset.apply(
            lambda mod, *args, **kwargs: np.sqrt(utils.var(mod.X, axis=None)), by_group=self._scale_per_group
        )

    @property
    def feature_means(self) -> dict[str, dict[str, float]]:
        return self._feature_means

    @property
    def sample_means(self) -> dict[str, dict[str, float]]:
        return self._sample_means

    @property
    def used_features(self) -> dict[str, NDArray[np.bool]]:
        return self._nonconstantfeatures

    def __call__(self, arr: NDArray, group: str, view: str) -> NDArray:
        # remove constant features
        arr = arr[..., self._nonconstantfeatures[view]]

        if view in self._views_to_scale:
            viewstats = self._viewstats[view]

            # scale to unit variance
            arr /= self._scale[group][view] if self._scale_per_group else self._scale[view]

            # center
            if view in self._nonnegative_weights and group in self._nonnegative_factors:
                arr -= viewstats.min
            else:
                arr -= viewstats.mean
        return arr


def infer_likelihood(view: AnnData, *args) -> utils.Likelihood:
    """Infer the likelihood for a view based on the data distribution."""
    data = view.X.data if issparse(view.X) else view.X
    if np.all(np.isclose(data, 0) | np.isclose(data, 1)):  # TODO: set correct atol value
        return "Bernoulli"
    elif np.allclose(data, np.round(data)) and data.min() >= 0:
        return "GammaPoisson"
    else:
        return "Normal"


def validate_likelihood(view: AnnData, group_name: str, view_name: str, likelihood: utils.Likelihood):
    """Validate the likelihood for a view based on the data distribution."""
    data = view.X.data if issparse(view.X) else view.X
    if likelihood == "Bernoulli" and not np.all(
        np.isclose(data, 0) | np.isclose(data, 1)
    ):  # TODO: set correct atol value
        raise ValueError(f"Bernoulli likelihood in view {view_name} must be used with binary data.")
    elif likelihood == "GammaPoisson" and not np.allclose(data, np.round(data)) and data.min() >= 0:
        raise ValueError(
            f"GammaPoisson likelihood in view {view_name} must be used with count (non-negative integer) data."
        )
