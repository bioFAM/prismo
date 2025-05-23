import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from pyro import distributions as dist
from pyro.nn import pyro_method

from .base import EPS, R2, Likelihood, PyroLikelihood, PyroLikelihoodWithDispersion


class Normal(Likelihood):
    _priority = 0
    scale_data = True

    class _PyroNormal(PyroLikelihoodWithDispersion):
        def __init__(
            self,
            view_name: str,
            sample_dim: int,
            feature_dim: int,
            sample_means: dict[str, dict[str, NDArray[np.floating]]],
            feature_means: dict[str, dict[str, NDArray[np.floating]]],
            *,
            init_loc: float = 0.0,
            init_scale: float = 0.1,
        ):
            super().__init__(
                view_name,
                sample_dim,
                feature_dim,
                sample_means,
                feature_means,
                init_loc=init_loc,
                init_scale=init_scale,
            )

        @pyro_method
        def _model(
            self,
            estimate: torch.Tensor,
            group_name: str,
            sample_plate: pyro.plate,
            feature_plate: pyro.plate,
            nonmissing_samples: torch.Tensor | slice,
            nonmissing_features: torch.Tensor | slice,
        ) -> pyro.distributions.Distribution:
            dispersion = self._model_dispersion(
                estimate, group_name, sample_plate, feature_plate, nonmissing_samples, nonmissing_features
            )
            return dist.Normal(estimate, torch.reciprocal(dispersion + EPS))

    @classmethod
    def pyro_likelihood(
        cls,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        sample_means: dict[str, dict[str, NDArray[np.floating]]],
        feature_means: dict[str, dict[str, NDArray[np.floating]]],
        *,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        **kwargs,
    ) -> PyroLikelihood:
        return cls._PyroNormal(
            view_name, sample_dim, feature_dim, sample_means, feature_means, init_loc=init_loc, init_scale=init_scale
        )

    @classmethod
    def _validate(cls, data: NDArray, xp) -> bool:
        return True

    @classmethod
    def _r2(
        cls,
        r2_full: float,
        y_true: NDArray,
        factors: NDArray[np.floating],
        weights: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        # this is the same as MOFA2
        r2s = np.empty(factors.shape[1], dtype=np.float32)
        for k in range(factors.shape[1]):
            r2s[k] = cls._r2_impl_wrapper(y_true, factors[:, k, None], weights[None, k, :], dispersions, sample_means)
        return r2s

    @classmethod
    def _r2_impl(
        cls,
        y_true: NDArray,
        y_pred: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ) -> R2:
        ss_res = np.nansum(np.square(y_true - y_pred))
        ss_tot = np.nansum(np.square(y_true))  # data is centered
        return R2(ss_res, ss_tot)

    @classmethod
    def transform_prediction(cls, prediction: NDArray[np.floating], sample_means: NDArray[np.floating]):
        return prediction
