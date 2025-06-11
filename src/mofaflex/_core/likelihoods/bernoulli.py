import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from ..pyro.likelihoods import PyroLikelihood, _PyroBernoulli
from .base import R2, Likelihood


class Bernoulli(Likelihood):
    _priority = 10
    scale_data = False

    @classmethod
    def pyro_likelihood(
        cls,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        sample_means: dict[str, dict[str, NDArray[np.floating]]],
        feature_means: dict[str, dict[str, NDArray[np.floating]]],
        **kwargs,
    ) -> PyroLikelihood:
        return _PyroBernoulli(view_name, sample_dim, feature_dim, sample_means, feature_means)

    @classmethod
    def _validate(cls, data: NDArray, xp) -> bool:
        return xp.all(xp.isclose(data, 0) | xp.isclose(data, 1))  # TODO: set correct atol value

    @classmethod
    def _format_validate_exception(cls, view_name: str) -> str:
        return f"Bernoulli likelihood in view {view_name} must be used with binary data."

    @classmethod
    def _r2_impl(
        cls,
        y_true: NDArray,
        y_pred: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ) -> R2:
        ss_res = np.nansum(cls._dV_square(y_true, y_pred, -1, 1))
        ss_tot = np.nansum(cls._dV_square(y_true, np.nanmean(y_true), -1, 1))
        return R2(ss_res, ss_tot)

    @classmethod
    def transform_prediction(cls, prediction: NDArray[np.floating], sample_means: NDArray[np.floating]):
        return expit(prediction)
