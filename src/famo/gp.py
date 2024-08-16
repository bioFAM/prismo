import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class GP(ApproximateGP):
    """Gaussian Process model with RBF kernel."""

    def __init__(self, inducing_points: torch.Tensor, n_factors: int):
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution=CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.shape[-2], batch_shape=[n_factors, 1]
            ),
        )

        super().__init__(variational_strategy)

        self.mean_module = ZeroMean(batch_shape=[n_factors, 1])
        self.covar_module = ScaleKernel(RBFKernel(batch_shape=[n_factors, 1]))

        self.covar_module.base_kernel.lengthscale = 0.1
        self.covar_module.outputscale = 1.0

    def forward(self, x):
        """Forward pass of the GP model."""
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


def setup_inducing_points(factor_prior: dict, covariates: dict, n_inducing: dict, n_factors: int, device: str):
    """Randomly initialize inducing points from the covariates.

    Parameters
    ----------
    n_inducing : int
        Number of inducing points.
    n_factors : int
        Number of factors.
    covariates : dict
        dictionary with group names and tensors of shape (n_samples, n_dims).
    device: str
    """
    inducing_points = {}
    for gn, gv in factor_prior.items():
        if gv == "GP":
            inducing_points[gn] = torch.zeros([n_factors, 1, n_inducing[gn], covariates[gn].shape[-1]], device=device)
            for factor in range(n_factors):
                inducing_points[gn][factor, 0] = covariates[gn][
                    torch.randint(0, covariates[gn].shape[-2], (n_inducing[gn],))
                ].to(device)

    return inducing_points
