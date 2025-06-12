import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from pyro import distributions as dist

from .base import PyroLikelihood


class PyroBernoulli(PyroLikelihood):
    def __init__(
        self,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        sample_means: dict[str, dict[str, NDArray[np.floating]]],
        feature_means: dict[str, dict[str, NDArray[np.floating]]],
    ):
        super().__init__(view_name, sample_dim, feature_dim, sample_means, feature_means)

    def _model(
        self,
        estimate: torch.Tensor,
        group_name: str,
        sample_plate: pyro.plate,
        feature_plate: pyro.plate,
        nonmissing_samples: torch.Tensor | slice,
        nonmissing_features: torch.Tensor | slice,
    ) -> pyro.distributions.Distribution:
        return dist.Bernoulli(logits=estimate)

    def _guide(self, group_name: str, sample_plate: pyro.plate, feature_plate: pyro.plate):
        pass
