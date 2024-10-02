import altair as alt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr


def match(
    reference: torch.Tensor | np.ndarray, permutable: torch.Tensor | np.ndarray, dim: int
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
            correlation[i, j] = pearsonr(reference[..., i].flatten(), permutable[..., j].flatten())[0]
    correlation = np.nan_to_num(correlation, 0)

    # find the permutation that maximizes the correlation
    row_ind, permutation = linear_sum_assignment(-1 * np.abs(correlation))
    signs = np.ones_like(permutation)

    # if correlation is negative, flip the sign of the corresponding column
    for k in range(signs.shape[0]):
        if correlation[row_ind, permutation][k] < 0:
            signs[k] *= -1

    return permutation, signs


def plot_factors_covariate_2d(factors, covariates):
    factor_charts = []
    z = factors
    df = pd.DataFrame(z)
    for i in range(covariates.shape[-1]):
        df[f"covariate_{i}"] = covariates[:, i]
    df.columns = df.columns.astype(str)

    for factor in range(factors.shape[1]):
        scatter_plot = (
            alt.Chart(df)
            .mark_point(filled=True, size=50)
            .encode(
                x=alt.X("covariate_0:O", title="", axis=alt.Axis(labels=False, ticks=False, grid=False, domain=False)),
                y=alt.Y("covariate_1:O", title="", axis=alt.Axis(labels=False, ticks=False, grid=False, domain=False)),
                color=alt.Color(f"{factor}:Q", scale=alt.Scale(scheme="redblue", domainMid=0), title=None),
            )
            .properties(width=300, height=300, title=f"Factor {factor+1}")
            .interactive()
        )

        factor_charts.append(scatter_plot)

    # Concatenate all the charts vertically
    final_chart = alt.hconcat(*factor_charts)

    # Display the chart
    final_chart.display()
