import numpy as np
from numpy.typing import NDArray

from ..pyro.likelihoods import PyroLikelihood, PyroNegativeBinomial
from .base import R2, Likelihood


class NegativeBinomial(Likelihood):
    _priority = 5
    scale_data = False

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
        return PyroNegativeBinomial(
            view_name, sample_dim, feature_dim, sample_means, feature_means, init_loc=init_loc, init_scale=init_scale
        )

    @classmethod
    def _validate(cls, data: NDArray, xp) -> bool:
        return xp.allclose(data, xp.round(data)) and data.min() >= 0

    @classmethod
    def _format_validate_exception(cls, view_name: str) -> str:
        return f"NegativeBinomial likelihood in view {view_name} must be used with count (non-negative integer) data."

    @classmethod
    def _r2_impl(
        cls,
        y_true: NDArray,
        y_pred: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ):
        ss_res = np.nansum(cls._dV_square(y_true, y_pred, dispersions, 1))

        truemean = np.nanmean(y_true)
        nu2 = (np.nanvar(y_true) - truemean) / truemean**2  # method of moments estimator
        ss_tot = np.nansum(cls._dV_square(y_true, truemean, nu2, 1))

        return R2(ss_res, ss_tot)

    @classmethod
    def transform_prediction(cls, prediction: NDArray[np.floating], sample_means: NDArray[np.floating]):
        prediction = np.maximum(0, prediction)  # ReLU
        prediction *= sample_means[..., None]
        return prediction
