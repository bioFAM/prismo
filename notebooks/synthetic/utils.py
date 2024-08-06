import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

import random
import string

import anndata as ad
import numpy as np
import pandas as pd
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel
from gpytorch.means import ZeroMean


def generate_data(
    n_samples: dict[str, int],
    n_features: dict[str, int],
    n_factors: int,
    likelihoods: dict[str, float],
    covariates: dict[str, float] = None,
) -> dict:
    """Simulate data for experiments with synthetic data for model validation.

    Parameters
    ----------
    n_samples : dict[str, int]
        Number of samples per group.
    n_features : dict[str, int]
        Number of features per view.
    n_factors : int
        Number of factors.
    likelihoods : dict[str, float]
        Likelihoods for each view.
    covariates : dict[str, float]
        Covariates for each group (if available).

    Returns
    -------
    dict
        A nested dictionary of AnnData objects with the simulated data (group level -> view level)
    """
    if covariates is None:
        covariates = {group: None for group in n_samples}

    # factors
    z = {}
    lengthscales = {}
    obs_names = {}
    for group in n_samples:
        if group not in covariates:
            covariates[group] = None
        if covariates[group] is not None:
            cov_module = RBFKernel(batch_shape=torch.Size([n_factors]))
            lengthscales[group] = torch.rand([n_factors]).clamp(0.1, 0.8)
            cov_module.lengthscale = lengthscales[group]
            mean_module = ZeroMean(batch_shape=torch.Size([n_factors]))
            z[group] = (
                MultivariateNormal(
                    mean_module(covariates[group]), cov_module(covariates[group])
                )
                .sample()
                .T
            )
        else:
            lengthscales[group] = None
            z[group] = torch.randn(n_samples[group], n_factors)
        obs_names[group] = np.array(
            [
                "".join(random.choices(string.ascii_letters + string.digits, k=5))
                for _ in range(n_samples[group])
            ]
        )

    # weights
    w = {}
    var_names = {}
    for view in n_features:
        w[view] = torch.randn(n_factors, n_features[view])
        var_names[view] = np.array(
            [
                "".join(random.choices(string.ascii_letters + string.digits, k=4))
                for _ in range(n_features[view])
            ]
        )

    # observations
    data = {}
    for group in n_samples:
        data[group] = {}
        for view in n_features:
            # randomly remove some observations and features
            keep_obs_inds = torch.rand(n_samples[group]) > 0.1
            keep_var_inds = torch.rand(n_features[view]) > 0.1

            if likelihoods[view] == "Normal":
                loc = z[group][keep_obs_inds] @ w[view][:, keep_var_inds]
                obs = torch.distributions.Normal(loc, 0.1).sample()

                adata = ad.AnnData(
                    X=obs.numpy(),
                    obs=pd.DataFrame(index=obs_names[group][keep_obs_inds]),
                    var=pd.DataFrame(index=var_names[view][keep_var_inds]),
                )
                adata.obsm["z"] = z[group][keep_obs_inds].numpy()
                adata.varm["w"] = w[view][:, keep_var_inds].numpy().T

                if lengthscales[group] is not None:
                    adata.uns["lengthscales"] = lengthscales[group].numpy()

                if covariates[group] is not None:
                    adata.obsm["x"] = covariates[group][keep_obs_inds].numpy()

                data[group][view] = adata

            if likelihoods[view] == "Poisson":
                rate = torch.exp(z[group][keep_obs_inds] @ w[view][:, keep_var_inds])
                rate_scale = torch.distributions.Exponential(5.0).sample(
                    [rate.shape[-1]]
                )
                obs = torch.distributions.Poisson(rate * rate_scale).sample()

                data[group][view] = ad.AnnData(
                    X=obs.numpy(),
                    obs=pd.DataFrame(index=obs_names[group][keep_obs_inds]),
                    var=pd.DataFrame(index=var_names[view][keep_var_inds]),
                    obsm={
                        "z": z[group][keep_obs_inds].numpy(),
                    },
                    varm={
                        "w": w[view][:, keep_var_inds].numpy().T,
                        "rate_scale": rate_scale.numpy(),
                    },
                    uns={
                        "lengthscales": (
                            lengthscales[group].numpy()
                            if lengthscales[group] is not None
                            else None
                        )
                    },
                )
                if covariates[group] is not None:
                    data[group][view].obsm["covariates"] = covariates[group][
                        keep_obs_inds
                    ].numpy()

            if likelihoods[view] == "Bernoulli":
                logits = z[group][keep_obs_inds] @ w[view][:, keep_var_inds]
                obs = torch.distributions.Bernoulli(logits=logits).sample()

                data[group][view] = ad.AnnData(
                    X=obs.numpy(),
                    obs=pd.DataFrame(index=obs_names[group][keep_obs_inds]),
                    var=pd.DataFrame(index=var_names[view][keep_var_inds]),
                    obsm={
                        "z": z[group][keep_obs_inds].numpy(),
                    },
                    varm={"w": w[view][:, keep_var_inds].numpy().T},
                    uns={
                        "lengthscales": (
                            lengthscales[group].numpy()
                            if lengthscales[group] is not None
                            else None
                        )
                    },
                )
                if covariates[group] is not None:
                    data[group][view].obsm["covariates"] = covariates[group][
                        keep_obs_inds
                    ].numpy()

            if likelihoods[view] == "Binomial":
                loc = 1 / (
                    1 + np.exp(-(z[group][keep_obs_inds] @ w[view][:, keep_var_inds]))
                )

                obs_total = torch.distributions.Poisson(100).sample(loc.shape)
                obs_allelic = torch.distributions.Binomial(obs_total, loc).sample()

                var_names_ref = [
                    f"{name}_ref" for name in var_names[view][keep_var_inds]
                ]
                var_names_alt = [
                    f"{name}_alt" for name in var_names[view][keep_var_inds]
                ]

                adata_ref = ad.AnnData(
                    X=obs_allelic.numpy(),
                    var=pd.DataFrame(index=var_names_ref),
                    obs=pd.DataFrame(index=obs_names[group][keep_obs_inds]),
                )
                adata_alt = ad.AnnData(
                    X=obs_total.numpy() - obs_allelic.numpy(),
                    var=pd.DataFrame(index=var_names_alt),
                    obs=pd.DataFrame(index=obs_names[group][keep_obs_inds]),
                )
                adata = ad.concat([adata_ref, adata_alt], axis=1, merge="same")
                adata.obsm["z"] = z[group][keep_obs_inds].numpy()
                if covariates[group] is not None:
                    adata.obsm["covariates"] = covariates[group][keep_obs_inds].numpy()
                adata.uns["w"] = w[view][:, keep_var_inds].numpy().T
                adata.uns["lengthscale"] = (
                    lengthscales[group] if lengthscales[group] is not None else None
                )
                adata.uns["var_names"] = var_names[view][keep_var_inds]
                data[group][view] = adata

    return data


def match(
    reference: torch.Tensor | np.ndarray,
    permutable: torch.Tensor | np.ndarray,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the permutation and sign of permutable along one dimension to maximize correlation with reference.

    This is useful for comparing ground truth factor scores / loadings with inferred values because the factor order
    and sign is arbitrary.

    Parameters
    ----------
    reference : torch.Tensor | np.ndarray
        The reference tensor.
    permutable : torch.Tensor | np.ndarray
        The permutable tensor.
    dim : int
        The dimension along which to permute the tensor.

    Returns
    -------
    permutation : torch.tensor
        The permutation of permutable that maximizes the correlation with reference.
    signs : torch.tensor
        The sign of permutable that maximizes the correlation with reference.
    """
    # convert all tensors to numpy arrays
    if isinstance(reference, torch.Tensor):
        reference = reference.numpy()
    if isinstance(permutable, torch.Tensor):
        permutable = permutable.numpy()

    # move the assignment dimension to the end
    reference = np.moveaxis(reference, dim, -1)
    permutable = np.moveaxis(permutable, dim, -1)

    # compute the correlation matrix between reference and permutable
    correlation = np.zeros([reference.shape[-1], permutable.shape[-1]])
    for i in range(reference.shape[-1]):
        for j in range(permutable.shape[-1]):
            correlation[i, j] = pearsonr(
                reference[..., i].flatten(), permutable[..., j].flatten()
            )[0]
    correlation = np.nan_to_num(correlation, 0)

    # find the permutation that maximizes the correlation
    row_ind, permutation = linear_sum_assignment(-1 * np.abs(correlation))
    signs = np.ones_like(permutation)

    # if correlation is negative, flip the sign of the corresponding column
    for k in range(signs.shape[0]):
        if correlation[row_ind, permutation][k] < 0:
            signs[k] *= -1

    return permutation, signs
