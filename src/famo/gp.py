import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class GP(ApproximateGP):
    """Gaussian Process model with RBF kernel."""

    def __init__(
        self, inducing_points: torch.Tensor, n_factors: int, kernel: str = "RBF", matern_kernel_nu: float = 1.5
    ):
        """Initialize the GP model.

        Parameters
        ----------
        inducing_points : torch.Tensor
            Tensor of inducing points with shape (n_factors, n_inducing, n_dims).
        n_factors : int
            Number of factors.
        kernel : str
            The kernel to use for the GP model. Can be "RBF" or "Matern".
        matern_kernel_nu : float
            The smoothness parameter for the Matern kernel.
        """
        if inducing_points.shape[-3] != n_factors:
            raise ValueError("The first dimension of inducing_points must be n_factors.")

        num_inducing_points = inducing_points.shape[-2]
        n_dims = inducing_points.shape[-1]
        batch_shape = [n_factors]

        variational_distribution = CholeskyVariationalDistribution(num_inducing_points, batch_shape)

        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution)

        super().__init__(variational_strategy)

        self.mean_module = ZeroMean(batch_shape)

        max_dist = torch.pdist(inducing_points.flatten(0, 1), p=n_dims).max()

        if kernel == "RBF":
            base_kernel = RBFKernel(batch_shape=batch_shape, lengthscale_constraint=Interval(max_dist / 1000, max_dist))

        if kernel == "Matern":
            base_kernel = MaternKernel(
                batch_shape=batch_shape, nu=matern_kernel_nu, lengthscale_constraint=Interval(max_dist / 1000, max_dist)
            )

        self.covar_module = ScaleKernel(
            base_kernel, outputscale_constraint=Interval(1e-3, 1 - 1e-3), batch_shape=batch_shape
        )

        self.covar_module.outputscale = 0.9
        self.covar_module.base_kernel.lengthscale = max_dist * 0.1

    def forward(self, x):
        """Forward pass of the GP model."""
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


def setup_inducing_points(factor_prior: dict, covariates: dict, n_inducing: dict, n_factors: int, device: str):
    """Randomly initialize inducing points from the covariates.

    Parameters
    ----------
    factor_prior : dict
        Dictionary with group names and factor priors.
    covariates : dict
        dictionary with group names and tensors of shape (n_samples, n_dims).
    n_inducing : int
        Number of inducing points.
    n_factors : int
        Number of factors.
    device: str
        Device to use for the inducing points.
    """
    if covariates is None:
        return None

    inducing_points = {}
    for group_name, group_factor_prior in factor_prior.items():
        if group_factor_prior == "GP":
            inducing_points[group_name] = torch.zeros(
                [n_factors, n_inducing[group_name], covariates[group_name].shape[-1]], device=device
            )
            for factor in range(n_factors):
                inducing_points[group_name][factor] = covariates[group_name][
                    torch.randint(0, covariates[group_name].shape[-2], (n_inducing[group_name],))
                ].to(device)

    return inducing_points
