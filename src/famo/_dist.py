from typing import ClassVar

import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property


class GammaPoisson(dist.TorchDistribution):
    arg_constraints: ClassVar[dict[str, constraints.Constraint]] = {
        "mu": constraints.positive,
        "theta": constraints.positive,
    }
    support: ClassVar[constraints.Constraint] = constraints.nonnegative_integer

    def __init__(self, mu, theta, validate_args=None):
        self.mu, self.theta = broadcast_all(mu, theta)
        super().__init__(self.mu.shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.mean + (self.mean**2) / self.theta

    @lazy_property
    def _gamma(self):
        return torch.distributions.Gamma(concentration=self.theta, rate=self.theta / self.mu, validate_args=False)

    def sample(self, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size()
        with torch.no_grad():
            rate = self._gamma.sample(sample_shape=sample_shape)
            return torch.poisson(rate)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)

        eps = 1e-8
        log_theta_mu = torch.log(self.theta + self.mu + eps)
        res = (
            self.theta * (torch.log(self.theta + eps) - log_theta_mu)
            + value * (torch.log(self.mu + eps) - log_theta_mu)
            + torch.lgamma(value + self.theta)
            - torch.lgamma(self.theta)
            - torch.lgamma(value + 1)
        )

        return res

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GammaPoisson, _instance)
        batch_shape = torch.Size(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.theta = self.theta.expand(batch_shape)
        super(GammaPoisson, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
