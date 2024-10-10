from collections.abc import Iterable

import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import IndexKernel, Kernel, RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.priors import Prior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class MefistoKernel(Kernel):
    def __init__(
        self,
        base_kernel: Kernel | None,
        n_groups: int,
        rank: int = 1,
        lowrank_covar_prior: Prior | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.group_kernel = IndexKernel(
            num_tasks=n_groups, batch_shape=base_kernel.batch_shape, rank=rank, prior=lowrank_covar_prior
        )
        self.base_kernel = base_kernel

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, last_dim_is_batch=False, **params):
        group_idx1, group_idx2 = x1[..., 0, None], x2[..., 0, None]
        x1_, x2_ = x1[..., 1:], x2[..., 1:]
        base_mat = self.base_kernel(x1_, x2_, diag, last_dim_is_batch, **params)

        if not diag:
            group_cov = self.group_kernel(group_idx1, group_idx2)
            if x1 is x2 or x1.shape == x2.shape and x1.data_ptr() == x2.data_ptr():
                group_cov_diag1 = group_cov_diag2 = group_cov.diagonal().sqrt()
            else:
                group_cov_diag1 = self.group_kernel(group_idx1, diag=True).sqrt()
                group_cov_diag2 = self.group_kernel(group_idx2, diag=True).sqrt()
            group_cor = group_cov.div(group_cov_diag1[..., None]).div(group_cov_diag2[..., None, :])
            return base_mat.mul(group_cor)
        else:
            return base_mat

    @property
    def outputscale(self):
        return self.base_kernel.outputscale


class GP(ApproximateGP):
    """Gaussian Process model with RBF kernel."""

    def __init__(
        self, n_inducing: int, covariates: Iterable[torch.Tensor], n_factors: int, n_groups: int, rank: int = 1
    ):
        """Initialize the GP model.

        Parameters
        ----------
        n_inducing
            Number of inducing points.
        covariates
            Covariates to choose the inducing points from.
        n_factors
            Number of factors.
        n_groups
            Number of groups.
        rank
            Rank of the group correlation kernel.
        """
        inducing_points = setup_inducing_points(covariates, n_inducing, n_factors)
        if inducing_points.shape[-3] != n_factors:
            raise ValueError("The first dimension of inducing_points must be n_factors.")

        num_inducing_points = inducing_points.shape[-2]
        n_dims = inducing_points.shape[-1]
        batch_shape = [n_factors]
        device = inducing_points.device

        variational_distribution = CholeskyVariationalDistribution(num_inducing_points, batch_shape)

        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )

        super().__init__(variational_strategy)

        self.mean_module = ZeroMean(batch_shape)

        max_dist = torch.pdist(inducing_points.flatten(0, 1), p=n_dims).max()

        base_kernel = RBFKernel(batch_shape=batch_shape, lengthscale_constraint=Interval(max_dist / 20, max_dist))
        base_kernel = ScaleKernel(base_kernel, outputscale_constraint=Interval(1e-3, 1 - 1e-3), batch_shape=batch_shape)
        base_kernel.outputscale = torch.sigmoid(torch.randn(batch_shape, device=device)).clamp(1e-3, 1 - 1e-3)
        base_kernel.base_kernel.lengthscale = max_dist * torch.rand(batch_shape).to(device=device).clamp(0.1)

        self.covar_module = MefistoKernel(base_kernel, n_groups, rank)

    def forward(self, x):
        """Forward pass of the GP model."""
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


def setup_inducing_points(covariates: Iterable[torch.Tensor], n_inducing: int, n_factors: int):
    """Randomly initialize inducing points from the covariates.

    Parameters
    ----------
    covariates
        tensors of shape (n_samples, n_dims).
    n_inducing
        Number of inducing points.
    n_factors
        Number of factors.
    """
    if covariates is None:
        return None

    covariates = tuple(covariates)
    group_idx = torch.cat(tuple(torch.as_tensor(i).expand(c.shape[0]) for i, c in enumerate(covariates)), dim=0)
    covariates = torch.cat(covariates, dim=0)

    inducing_points = torch.zeros((n_factors, n_inducing, 1 + covariates.shape[-1]))
    for factor in range(n_factors):
        idx = torch.randint(0, covariates.shape[-2], (n_inducing,))
        inducing_points[factor, :, 1:] = covariates[idx]
        inducing_points[factor, :, 0] = group_idx[idx]
    return inducing_points
