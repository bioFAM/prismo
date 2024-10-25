from collections.abc import Iterable

import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import IndexKernel, Kernel, MaternKernel, RBFKernel, ScaleKernel
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
        self,
        n_inducing: int,
        covariates: Iterable[torch.Tensor],
        n_factors: int,
        n_groups: int,
        kernel: str = "RBF",
        rank: int = 1,
        **kwargs,
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
        kernel
            Can be "RBF" or "Matern".
        rank
            Rank of the group correlation kernel.
        """
        covariates = tuple(covariates)
        self._inducing_points_idx = get_inducing_points_idx(covariates, n_inducing, n_factors)
        self._n_inducing = n_inducing

        inducing_points = setup_inducing_points(covariates, self._inducing_points_idx, n_inducing)
        if inducing_points.shape[-3] != n_factors:
            raise ValueError("The first dimension of inducing_points must be n_factors.")

        n_dims = inducing_points.shape[-1]
        batch_shape = [n_factors]
        device = inducing_points.device

        variational_distribution = CholeskyVariationalDistribution(n_inducing, batch_shape)

        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )

        super().__init__(variational_strategy)

        self.mean_module = ZeroMean(batch_shape)

        max_dist = torch.pdist(inducing_points.flatten(0, 1), p=n_dims).max()

        if kernel == "RBF":
            base_kernel = RBFKernel(batch_shape=batch_shape, lengthscale_constraint=Interval(max_dist / 20, max_dist))
        if kernel == "Matern":
            base_kernel = MaternKernel(
                batch_shape=batch_shape,
                lengthscale_constraint=Interval(max_dist / 20, max_dist),
                nu=kwargs.get("nu", 2.5),
            )
        base_kernel = ScaleKernel(base_kernel, outputscale_constraint=Interval(1e-3, 1 - 1e-3), batch_shape=batch_shape)
        base_kernel.outputscale = torch.sigmoid(torch.randn(batch_shape, device=device)).clamp(1e-3, 1 - 1e-3)
        base_kernel.base_kernel.lengthscale = max_dist * torch.rand(batch_shape).to(device=device).clamp(0.1)

        self.covar_module = MefistoKernel(base_kernel, n_groups, rank)

    def __call__(self, group_idx: torch.Tensor | None, inputs: torch.Tensor | None, prior: bool = False, **kwargs):
        if group_idx is not None and inputs is not None:
            inputs = torch.cat((group_idx, inputs), dim=-1)
        return super().__call__(inputs, prior, **kwargs)

    def forward(self, x):
        """Forward pass of the GP model."""
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)

    def update_inducing_points(self, covariates):
        setup_inducing_points(
            covariates, self._inducing_points_idx, self._n_inducing, out=self.variational_strategy.inducing_points
        )


def get_inducing_points_idx(covariates: Iterable[torch.Tensor], n_inducing: int, n_factors: int):
    n = [0] + [c.shape[0] for c in covariates]
    totaln = sum(n)
    offsets = torch.cumsum(torch.as_tensor(n), 0)
    idx = tuple(torch.randint(0, totaln, (n_inducing,)).sort().values for _ in range(n_factors))
    return tuple(
        tuple(cidx[(s <= cidx) & (cidx < e)] - s for s, e in zip(offsets[:-1], offsets[1:], strict=False))
        for cidx in idx
    )


def setup_inducing_points(covariates: Iterable[torch.Tensor], idx, n_inducing, *, out=None):
    """Randomly initialize inducing points from the covariates.

    Parameters
    ----------
    covariates
        tensors of shape (n_samples, n_dims).
    idx
        output of get_inducing_points_idx
    n_inducing
        Number of inducing points.
    """
    if covariates is None:
        return None

    covariates = tuple(covariates)
    group_idx = tuple(torch.as_tensor(i) for i in range(len(covariates)))

    if out is None:
        out = torch.empty((len(idx), n_inducing, 1 + covariates[0].shape[-1]))
    for factor, factoridx in enumerate(idx):
        offset = 0
        for cov, cidx, gidx in zip(covariates, factoridx, group_idx, strict=False):
            noffset = offset + cidx.shape[0]
            out[factor, offset:noffset, 1:] = cov[cidx]
            out[factor, offset:noffset, 0] = gidx
            offset = noffset
    return out
