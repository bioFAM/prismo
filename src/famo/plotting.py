from typing import Optional, Union

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch

HEATMAP = "heatmap"
MATRIXPLOT = "matrixplot"
DOTPLOT = "dotplot"
TRACKSPLOT = "tracksplot"
VIOLIN = "violin"
STACKED_VIOLIN = "stacked_violin"
PL_TYPES = [HEATMAP, MATRIXPLOT, DOTPLOT, TRACKSPLOT, VIOLIN, STACKED_VIOLIN]


STRIPPLOT = "stripplot"
BOXPLOT = "boxplot"
BOXENPLOT = "boxenplot"
VIOLINPLOT = "violinplot"
GROUP_PL_TYPES = [STRIPPLOT, BOXPLOT, BOXENPLOT, VIOLINPLOT]

alt.data_transformers.enable("vegafusion")


def plot_overview(data):
    for group_name, group_data in data.items():
        missings = pd.DataFrame()
        for view_name, view_data in group_data.items():
            # Concatenate all data into one matrix
            missings = pd.concat(
                [
                    missings,
                    pd.DataFrame(
                        {
                            "view": view_name,
                            "group": group_name,
                            "obs_name": view_data.obs_names,
                            "missing": np.isnan(view_data.X).any(axis=1),
                        }
                    ),
                ],
                axis=0,
            )

        alt.Chart(missings).mark_rect().encode(
            x=alt.X("obs_name", axis=alt.Axis(labels=False, title=None)),
            y=alt.Y("view", axis=alt.Axis(title=None)),
            color=alt.Color("missing:N", scale=alt.Scale(range=["#214D83", "#8AB6D4"])),
            facet=alt.Facet("group:N", columns=3, title=None),
        ).properties(width=800, title="Missing Data Overview").configure_view(
            strokeWidth=2, strokeOpacity=1, stroke="black"
        ).display()


def _lines(ax, positions, ymin, ymax, horizontal=False, **kwargs):
    if horizontal:
        ax.hlines(positions, ymin, ymax, **kwargs)
    else:
        ax.vlines(positions, ymin, ymax, **kwargs)
    return ax


def lined_heatmap(data, figsize=None, hlines=None, vlines=None, **kwargs):
    """Plot heatmap with horizontal or vertical lines."""
    if figsize is None:
        figsize = (20, 2)
    fig, g = plt.subplots(figsize=figsize)
    g = sns.heatmap(data, ax=g, **kwargs)
    if hlines is not None:
        _lines(g, hlines, *sorted(g.get_xlim()), horizontal=True, lw=1.0, linestyles="dashed")
    if vlines is not None:
        _lines(g, vlines, *sorted(g.get_ylim()), horizontal=False, lw=1.0, linestyles="dashed")
    return g


def plot_training_curve(model, figsize=(600, 400)):
    """Plot the training curve, i.e. -ELBO vs epoch."""
    model._check_if_trained()

    if figsize is None:
        figsize = (600, 400)

    train_loss_elbo = model._cache["train_loss_elbo"]
    df = pd.DataFrame({"Epoch": range(len(train_loss_elbo)), "-ELBO": train_loss_elbo})
    alt.Chart(df).mark_line(color="#214D83").encode(alt.Y("-ELBO").scale(zero=False), x="Epoch").properties(
        title="Training Curve", width=figsize[0], height=figsize[1]
    ).display()


def plot_all_weights(model, clip=(-1, 1)):
    """Plot the weight matrices using Altair."""
    # Ensure the model has been trained
    model._check_if_trained()

    # Extract weights data
    weights = model._cache["weights"]

    charts = []
    for k, v in weights.items():
        # Convert the weights matrix to a dataframe
        df = pd.DataFrame(v.X.squeeze().T)
        df["Feature"] = df.index + 1
        df = df.melt("Feature", var_name="Factor", value_name="Weight")
        df["Factor"] += 1

        if clip is None:
            clip = (df["Weight"].min(), df["Weight"].max())

        chart = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x="Factor:O",
                y=alt.Y("Feature:O", axis=alt.Axis(ticks=False, labels=False, title="Features")),
                color=alt.Color("Weight:Q", scale=alt.Scale(scheme="redblue", domain=clip)),
                tooltip=["Feature", "Factor", "Weight"],
            )
            .properties(title=f"{k}", width=250, height=400)
        )

        charts.append(chart)

    # Combine charts
    combined_chart = (
        alt.hconcat(*charts).configure_view(strokeWidth=0).configure_concat(spacing=5).configure_title(fontSize=14)
    )
    combined_chart.display()


def plot_factor_correlation(model):
    """Plot the correlation between factors."""
    # Check if the model has been trained
    model._check_if_trained()

    # Get the factors from the model's cache
    factors = model._cache["factors"]

    # Create an empty list to hold all the charts
    charts = []

    for k, v in factors.items():
        # Calculate the correlation matrix
        # and melt the DataFrame to long format
        corr_df = pd.DataFrame(np.corrcoef(v.X.T), index=model.factor_names, columns=model.factor_names)
        # Increase index by 1 to match the factor number and then melt the dataframe

        corr_df["index"] = model.factor_names
        corr_df = corr_df.melt("index")
        corr_df.columns = ["Factor1", "Factor2", "Correlation"]

        # Sort by Factor 1, and make sure "Factor 10" is behind "Factor 2"
        # Extract factor id
        corr_df["Factor1_int"] = corr_df["Factor1"].str.extract(r"(\d+)").astype(int)
        corr_df["Factor2_int"] = corr_df["Factor2"].str.extract(r"(\d+)").astype(int)
        corr_df = corr_df.sort_values(["Factor1_int", "Factor2_int"])

        # Create the heatmap chart
        heatmap = (
            alt.Chart(corr_df)
            .mark_rect()
            .encode(
                x=alt.X("Factor1:O", title="Factor", sort=None),
                y=alt.Y("Factor2:O", title="Factor", sort=None),
                color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=(-1, 1))),
                tooltip=["Factor1", "Factor2", "Correlation"],
            )
            .properties(title=k, width=400, height=400)
        )

        # Add the chart to the list of charts
        charts.append(heatmap)

    # Concatenate all the charts horizontally
    final_chart = alt.hconcat(*charts).resolve_scale(color="independent")

    # Display the chart
    final_chart.display()


def plot_variance_explained(model, groupby="group"):
    """Plot the variance explained by each factor in each view."""
    # Check if the model has been trained
    model._check_if_trained()

    # Create an empty list to hold all the charts
    charts = []

    if groupby == "group":
        # Get the variance explained DataFrame from the model's cache
        df_r2 = model._cache["df_r2"]

        # Loop over all groups
        for group_name in model.group_names:
            # Get the variance explained DataFrame for the current group
            r2_df = df_r2[group_name]

            # Convert the DataFrame to long format
            r2_df = r2_df.reset_index().melt("index")
            r2_df.columns = ["Factor", "View", "Variance Explained"]
            # Increase Factor index by 1
            r2_df["Factor"] = r2_df["Factor"] + 1

            # Create the heatmap chart
            heatmap = (
                alt.Chart(r2_df)
                .mark_rect()
                .encode(
                    x=alt.X("View:O", title="View"),
                    y=alt.Y("Factor:O", title="Factor", sort="descending"),
                    color=alt.Color(
                        "Variance Explained:Q",
                        scale=alt.Scale(scheme="blues", domain=(0, 1.5 * max(r2_df["Variance Explained"]))),
                        title=None,
                    ),
                    tooltip=["Factor", "View", "Variance Explained"],
                )
                .properties(title=group_name, width=150, height=200)
            )

            # Add the chart to the list of charts
            charts.append(heatmap)

    elif groupby == "view":
        # exchange dict keys (groups) and column names (views) in model cache df_r2
        columns = list(next(iter(model._cache["df_r2"].values())).columns)
        r2_df = {col: pd.concat({k: df[col] for k, df in model._cache["df_r2"].items()}, axis=1) for col in columns}

        # Loop over all views
        for view_name in model.view_names:
            # Get the variance explained DataFrame for the current view
            r2 = r2_df[view_name]

            # Convert the DataFrame to long format
            r2 = r2.reset_index().melt("index")
            r2.columns = ["Factor", "Group", "Variance Explained"]
            # Increase Factor index by 1
            r2["Factor"] = r2["Factor"] + 1

            # Create the heatmap chart
            heatmap = (
                alt.Chart(r2)
                .mark_rect()
                .encode(
                    x=alt.X("Group:O", title="Group"),
                    y=alt.Y("Factor:O", title="Factor", sort="descending"),
                    color=alt.Color(
                        "Variance Explained:Q",
                        scale=alt.Scale(scheme="blues", domain=(0, 1.5 * max(r2["Variance Explained"]))),
                        title=None,
                    ),
                    tooltip=["Factor", "Group", "Variance Explained"],
                )
                .properties(title=view_name, width=150, height=200)
            )

            # Add the chart to the list of charts
            charts.append(heatmap)

    # Concatenate all the charts horizontally
    final_chart = alt.hconcat(*charts).resolve_scale(color="shared")

    # Display the chart
    final_chart.display()


def plot_factor(model, factor=1):
    """Plot factor values (y-axis) for each sample (x-axis)."""
    model._check_if_trained()

    # Create an empty list to hold all the charts
    charts = []

    factors = model._cache["factors"]

    for group_name in model.group_names:
        df = factors[group_name].to_df()
        df["id"] = df.index
        # Convert column names to strings
        df.columns = [f"Factor {i}" for i in range(1, model.n_factors + 1)] + ["id"]
        factor_name = f"Factor {factor}"

        # Create the scatter plot chart
        scatter_plot = (
            alt.Chart(df)
            .mark_point(filled=True)
            .encode(
                x=alt.X("id:O", title="", axis=alt.Axis(labels=False)),
                y=alt.Y(f"{factor_name}:Q", title=f"{factor_name}"),
                color=alt.Color(f"{factor_name}:Q", scale=alt.Scale(scheme="redblue", domainMid=0)),
                tooltip=["id", f"{factor_name}"],
            )
            .properties(width=600, height=300)
            .interactive()
        )

        # Add a horizontal rule at y=0
        rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black", strokeDash=[5, 5]).encode(y="y")

        # Combine the scatter plot and rule
        final_plot = scatter_plot + rule

        # Add the chart to the list of charts
        charts.append(final_plot)

    # Concatenate all the charts vertically
    final_chart = alt.vconcat(*charts)

    # Display the chart
    final_chart.display()


def plot_factors_covariate_2d(model, covariate: str):
    """Plot every factor against a 2D covariate."""
    model._check_if_trained()

    group_charts = []

    factors = model._cache["factors"]
    covariates = model.covariates

    for group_name in model.group_names:
        factor_charts = []
        z = factors[group_name].X.squeeze()
        df = pd.DataFrame(z)
        for i in range(covariates[group_name].shape[-1]):
            df[f"covariate_{i}"] = covariates[group_name][:, i]
        df.columns = df.columns.astype(str)

        for factor in range(model.n_factors):
            scatter_plot = (
                alt.Chart(df)
                .mark_point(filled=True)
                .encode(
                    x=alt.X("covariate_0:O", title="Covariate dim 1", axis=alt.Axis(labels=False)),
                    y=alt.Y("covariate_1:O", title="Covariate dim 2", axis=alt.Axis(labels=False)),
                    color=alt.Color(f"{factor}:Q", scale=alt.Scale(scheme="redblue", domainMid=0)),
                )
                .properties(width=300, height=300, title=f"Factor {factor+1} with covariates")
                .interactive()
            )

            factor_charts.append(scatter_plot)

        group_charts.append(alt.hconcat(*factor_charts))

    # Concatenate all the charts vertically
    final_chart = alt.vconcat(*group_charts)

    # Display the chart
    final_chart.display()


def plot_gps_covariate_2d(model, covariate: str):
    """Plot every gp against a 2D covariate."""
    model._check_if_trained()

    group_charts = []

    gps = model._cache["gps"]
    covariates = model.covariates

    for group_name in model.group_names:
        factor_charts = []
        f = gps[group_name].X.squeeze()
        df = pd.DataFrame(f)
        for i in range(covariates[group_name].shape[-1]):
            df[f"covariate_{i}"] = covariates[group_name][:, i]
        df.columns = df.columns.astype(str)

        for factor in range(model.n_factors):
            scatter_plot = (
                alt.Chart(df)
                .mark_point(filled=True)
                .encode(
                    x=alt.X("covariate_0:O", title="Covariate dim 1", axis=alt.Axis(labels=False)),
                    y=alt.Y("covariate_1:O", title="Covariate dim 2", axis=alt.Axis(labels=False)),
                    color=alt.Color(f"{factor}:Q", scale=alt.Scale(scheme="redblue", domainMid=0)),
                )
                .properties(width=300, height=300, title=f"GP {factor+1} with covariates")
                .interactive()
            )

            factor_charts.append(scatter_plot)

        group_charts.append(alt.hconcat(*factor_charts))

    # Concatenate all the charts vertically
    final_chart = alt.vconcat(*group_charts)

    # Display the chart
    final_chart.display()


def plot_factors_covariate_1d(model, covariate: str, color: str = None) -> None:
    """Plot every factor against a 1D covariate.

    Parameters
    ----------
    model
        The FAMO model.
    covariate: str
        The name of the covariate to plot against (needs to be in .obs DataFrame of data).
    color: list[str]
        The obs column to color by.

    Returns
    -------
    None
    """
    model._check_if_trained()

    df_factors = []
    for group in model.group_names:
        df_factors.append(
            pd.concat([model._cache["factors"][group].to_df(), model._cache["factors"][group].obs], axis=1)
        )
    df_factors = pd.concat(df_factors).T.drop_duplicates().T
    if "view" in df_factors.columns:
        df_factors.drop(columns=["view"], inplace=True)
    df_factors["identity"] = 1

    if color is None:
        color = alt.Color("identity", title=None)
    else:
        color = alt.Color(color, title=color)

    charts = []

    for factor in range(model.n_factors):
        scatter_plot = (
            alt.Chart(df_factors)
            .mark_point(filled=True)
            .encode(
                x=alt.X(covariate, title=covariate),
                y=alt.Y("Factor " + str(factor + 1), title="Factor " + str(factor + 1)),
                color=color,
            )
            .properties(width=200, height=200)
            .interactive()
        )

        rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black", strokeDash=[5, 5]).encode(y="y")
        final_plot = scatter_plot + rule
        charts.append(final_plot)

    final_chart = alt.hconcat(*charts)
    final_chart.display()


def plot_factors_scatter(model, x: int, y: int, color: list[str] = None) -> None:
    """Plot two factors against each other and color by covariates.

    Parameters
    ----------
    model
        The FAMO model.
    x : int
        The factor to plot on the x-axis.
    y : int
        The factor to plot on the y-axis.
    color : list[int]
        The covariate name(s) to color by.

    Returns
    -------
    None
    """
    model._check_if_trained()

    df_factors = []
    for group in model.group_names:
        df_factors.append(
            pd.concat([model._cache["factors"][group].to_df(), model._cache["factors"][group].obs], axis=1)
        )
    df_factors = pd.concat(df_factors).T.drop_duplicates().T
    if "view" in df_factors.columns:
        df_factors.drop(columns=["view"], inplace=True)
    df_factors["identity"] = 1

    charts = []

    if color is None:
        color = ["identity"]

    for color_name in color:
        scatter_plot = (
            alt.Chart(df_factors)
            .mark_point(filled=True)
            .encode(
                x=alt.X("Factor " + str(x), title="Factor " + str(x)),
                y=alt.Y("Factor " + str(y), title="Factor " + str(y)),
                color=alt.Color(color_name, title=color_name),
            )
            .properties(width=200, height=200)
            .interactive()
        )

        rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black", strokeDash=[5, 5]).encode(y="y")
        final_plot = scatter_plot + rule
        charts.append(final_plot)

    final_chart = alt.hconcat(*charts)
    final_chart.display()


def plot_gps_1d(model, x: torch.Tensor, n_samples: int = 100, color: str = None) -> None:
    """Plot the GP posterior mean and 95% confidence interval for each factor in each group.

    Parameters
    ----------
    model
        The FAMO model.
    x : torch.Tensor
        The input tensor to evaluate the GP at.
    n_samples : int
        The number of samples to draw from the GP.
    color : str
        The variable to color by.

    Returns
    -------
    None
    """
    model._check_if_trained()

    df_gps = []
    for group in model.group_names:
        f_dist = model.gps[group](x.to(model.device))
        f_samples = f_dist.sample(torch.Size([n_samples]))
        f_mean = f_samples.mean(dim=0)
        f_std = f_samples.std(dim=0)

        df = pd.DataFrame({"x": x.cpu().numpy(), "group": group})

        for factor in range(model.n_factors):
            df[f"f_mean_factor_{factor+1}"] = f_mean[factor].cpu().numpy()
            df[f"f_lower_factor_{factor+1}"] = f_mean[factor].cpu().numpy() - 2 * f_std[factor].cpu().numpy()
            df[f"f_upper_factor_{factor+1}"] = f_mean[factor].cpu().numpy() + 2 * f_std[factor].cpu().numpy()

        df_gps.append(df)

    df_gps = pd.concat(df_gps)
    df_gps["identity"] = 1

    if color is None:
        color = alt.Color("identity", title=None)
    else:
        color = alt.Color(color, title=color)

    charts = []

    for factor in range(model.n_factors):
        line = (
            alt.Chart(df_gps)
            .mark_line()
            .encode(
                x=alt.X("x:Q", title="x"),
                y=alt.Y(f"f_mean_factor_{factor+1}:Q", title=f"Factor {factor+1}"),
                color=color,
            )
        )

        area = (
            alt.Chart(df_gps)
            .mark_area(opacity=0.4)
            .encode(
                x="x",
                y="f_lower_factor_" + str(factor + 1),
                y2="f_upper_factor_" + str(factor + 1),
                color=alt.Color("group", title="Group"),
            )
        )

        plot = alt.layer(line, area).properties(width=200, height=200).interactive()

        rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black", strokeDash=[5, 5]).encode(y="y")
        final_plot = plot + rule
        charts.append(final_plot)

    final_chart = alt.hconcat(*charts)
    final_chart.display()


def plot_smoothness(model) -> None:
    """Plot the smoothness of the GP for each factor in each group.

    Parameters
    ----------
    model
        The FAMO model.

    Returns
    -------
    None
    """
    model._check_if_trained()

    charts = []

    for group in model.group_names:
        df = pd.DataFrame(index=np.arange(model.n_factors))
        df["smoothness"] = model.gps[group].covar_module.outputscale.cpu().detach().numpy()
        df["factor"] = [str(i + 1) for i in range(model.n_factors)]
        bar_plot = (
            alt.Chart(df)
            .mark_bar()
            .encode(y=alt.Y("factor", title="Factor"), x=alt.X("smoothness", title="Smoothness"))
            .properties(width=200, height=200, title=group)
        )
        charts.append(bar_plot)

    final_chart = alt.hconcat(*charts)
    final_chart.display()


def plot_top_weights(model, views: list[str] = None, n_features: int = 10, factors: list[int] = None):
    model._check_if_trained()

    if views is None:
        views = model.view_names

    if factors is None:
        factors = list(range(model.n_factors))

    charts = []

    for view in views:
        weights = model._cache["weights"][view].to_df()
        weights = weights.iloc[factors]
        weights = weights.iloc[:, weights.abs().max(axis=0).argsort()[-n_features:]]
        weights_melted = weights.reset_index().melt("index")
        weights_melted["value_abs"] = weights_melted["value"].abs()

        heatmap = (
            alt.Chart(weights_melted)
            .mark_circle()
            .encode(
                x=alt.X("index", title="Factor"),
                y=alt.Y("variable:O", title="Feature"),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", reverse=True), title="Weight"),
                size=alt.Size("value_abs:Q", title="Abs. Weight"),
            )
            .properties(width=800, height=n_features * 15, title=view)
        )

        charts.append(heatmap)

    final_chart = alt.hconcat(*charts)
    final_chart.display()


def plot_weights(model, view, factor=1, top_n_features=10):
    if isinstance(factor, int):
        factor = model.factor_names[factor - 1]

    weights = model.get_weights(return_type="pandas")[view].loc[factor, :]
    df_plot = pd.DataFrame({"weight": weights, "feature": weights.index})
    df_plot = df_plot.sort_values("weight", ascending=False)
    df_plot["rank"] = len(df_plot) - np.arange(len(df_plot))

    # Calculate x-axis limits with additional margin
    x_min = df_plot["weight"].min()
    x_max = df_plot["weight"].max()
    x_margin = (x_max - x_min) * 0.1  # 10% margin on each side

    # Select top n values with highest absolute weights
    # and split by pos/neg because we want to label them on different sides
    df_plot["abs_weight"] = np.abs(df_plot["weight"])
    top_n = df_plot.nlargest(top_n_features, "abs_weight").copy()
    top_n["sign"] = np.sign(top_n["weight"])
    top_n_pos = top_n.query("sign > 0").copy()
    top_n_neg = top_n.query("sign < 0").copy()

    # Adjust positions for labels
    # Create a unique y position for each label, invovles some manual tweaking
    top_n_pos["label_x"] = x_max + x_margin
    top_n_neg["label_x"] = x_min - x_margin
    top_n_pos["label_y"] = top_n_pos["rank"]
    top_n_neg["label_y"] = top_n_neg["rank"]
    max_rank = df_plot["rank"].max()
    top_n_pos["label_y"] = np.linspace(max_rank, 0.2 * max_rank, len(top_n_pos))
    top_n_neg["label_y"] = np.linspace(0.2 * max_rank, 0.8 * max_rank, len(top_n_neg))

    # Make an altair plot showing weight on x axis, rank on y axis
    points = (
        alt.Chart(df_plot)
        .mark_point(size=10)
        .encode(
            x=alt.X(
                "weight:Q", title="Weight", scale=alt.Scale(domain=[x_min - x_margin * 2.5, x_max + x_margin * 2.5])
            ),
            y=alt.Y("rank:Q", title="Rank", axis=alt.Axis(labels=False, ticks=False)),
        )
    )

    # Add feature names as text labels
    text_pos = (
        alt.Chart(top_n_pos)
        .mark_text(align="left", baseline="middle", dx=5)
        .encode(x="label_x:Q", y="label_y:Q", text="feature:N")
    )
    text_neg = (
        alt.Chart(top_n_neg)
        .mark_text(align="right", baseline="middle", dx=-5)
        .encode(x="label_x:Q", y="label_y:Q", text="feature:N")
    )

    # Add lines connecting points and labels
    lines_pos = (
        alt.Chart(top_n_pos)
        .mark_rule(color="gray", strokeDash=[1, 1])
        .encode(x="label_x:Q", y="label_y:Q", x2="weight:Q", y2="rank:Q")
    )
    lines_neg = (
        alt.Chart(top_n_neg)
        .mark_rule(color="gray", strokeDash=[1, 1])
        .encode(x="label_x:Q", y="label_y:Q", x2="weight:Q", y2="rank:Q")
    )

    chart = (points + text_pos + text_neg + lines_pos + lines_neg).properties(
        title=f"Top {view} weights for {factor}", width=600, height=400
    )

    chart.display()


def plot_top_weights_muvi(model, factor_idx, view_idx="all", top=25, ranked=True, figsize=None, **kwargs):
    """Scatterplot of factor loadings for specific factors."""
    if isinstance(factor_idx, int):
        factor_idx = model.factor_names[factor_idx - 1]
    if isinstance(factor_idx, str):
        factor_idx = [factor_idx]

    if view_idx == "all":
        view_idx = model.view_names

    if isinstance(view_idx, str):
        view_idx = [view_idx]

    n_views = len(view_idx)
    n_factors = len(factor_idx)

    if figsize is None:
        figsize = (8 * n_views, 8 * n_factors)

    fig, axs = plt.subplots(n_factors, n_views, squeeze=False, figsize=figsize)

    for m in range(n_views):
        view_name = view_idx[m]
        for k in range(n_factors):
            factor_name = factor_idx[k]
            i = k * n_views + m
            g = axs[k, m]
            # only last
            show_legend = i == n_views * n_factors - 1

            factor_loadings = model.get_weights("pandas")[view_name].loc[factor_name, :]
            factor_mask = pd.Series(False, index=factor_loadings.index)
            annotations = model.get_annotations("pandas")[view_name]
            if annotations is not None:
                factor_mask = annotations.loc[factor_name, :]
            factor_loadings_abs = np.abs(factor_loadings)
            factor_loadings_rank = factor_loadings.rank(ascending=False)

            name_col = "Feature"
            loading_col = "Loading"
            rank_col = "Rank"
            abs_loading_col = "Loading (abs)"

            data = pd.DataFrame(
                {
                    name_col: model.feature_names[view_name],
                    "Mask": factor_mask,
                    loading_col: factor_loadings,
                    abs_loading_col: factor_loadings_abs,
                    rank_col: factor_loadings_rank,
                    "FP": ~factor_mask & (factor_loadings_abs > 0.0),
                }
            )

            type_col = "Type"
            data[type_col] = data["FP"].map({False: "Annotated", True: "Inferred"})

            if top > 0:
                data = data.sort_values(abs_loading_col, ascending=True)
            x = loading_col
            y = name_col
            kwargs["hue"] = type_col
            kwargs["hue_order"] = ["Annotated", "Inferred"]
            kwargs["palette"] = {"Annotated": "black", "Inferred": "red"}
            if ranked:
                x = rank_col
                y = loading_col
            g = sns.scatterplot(
                ax=g,
                data=data.iloc[-top:],
                x=x,
                y=y,
                s=kwargs.pop("s", (64)),
                legend=show_legend and not ranked,
                **kwargs,
            )
            if ranked:
                g = sns.scatterplot(
                    ax=g, data=data.iloc[:-top], x=rank_col, y=loading_col, s=10, legend=show_legend, **kwargs
                )

                y_max = factor_loadings.max()
                y_min = factor_loadings.min()
                x_range = factor_loadings_rank.max()

                labeled_data = data.iloc[-top:].sort_values(loading_col, ascending=False).copy()

                labeled_data["is_positive"] = labeled_data[loading_col] > 0

                n_positive = labeled_data["is_positive"].sum()
                n_negative = top - n_positive
                num = max(n_positive, n_negative)

                labeled_data["x_arrow_pos"] = labeled_data[rank_col] + 0.02 * x_range
                labeled_data["x_text_pos"] = labeled_data["x_arrow_pos"] + 0.15 * x_range
                labeled_data["y_arrow_pos"] = labeled_data[loading_col]
                labeled_data["y_text_pos"] = (
                    np.linspace(y_max, 0.1 * y_max, num=num)[:n_positive].tolist()
                    + np.linspace(y_min, -0.1 * y_min, num=num)[:n_negative][::-1].tolist()
                )

                for _, row in labeled_data.iterrows():
                    g.text(
                        row["x_text_pos"],
                        row["y_text_pos"],
                        row[name_col],
                        color=kwargs["palette"][row[type_col]],
                        fontsize="medium",
                    )
                    g.annotate(
                        "",
                        (row["x_arrow_pos"], row["y_arrow_pos"]),
                        xytext=(row["x_text_pos"], row["y_text_pos"]),
                        # bbox=dict(boxstyle="round", alpha=0.1),
                        arrowprops={"arrowstyle": "simple,tail_width=0.01,head_width=0.15", "color": "black"},
                    )

            g.set_title(f"{factor_name} ({view_name})")

    fig.tight_layout()
    return fig, axs


def factor_activity(true_w, approx_w, true_mask, noisy_mask, factor_idx=0, ylim=None, top=None, **kwargs):
    true_w_col = true_w[factor_idx, :]
    w_col = approx_w[factor_idx, :]
    true_mask_col = true_mask[factor_idx, :]
    noisy_mask_col = noisy_mask[factor_idx, :]
    if top is not None:
        # descending order
        argsort_indices = np.argsort(-np.abs(w_col))[:top]
        w_col = w_col[argsort_indices]
        # remove zeros
        non_zero_indices = np.abs(w_col) > 0
        top = min(top, sum(non_zero_indices))
        # subset again
        argsort_indices = argsort_indices[:top]
        w_col = w_col[:top]
        true_w_col = true_w_col[argsort_indices]
        true_mask_col = true_mask_col[argsort_indices]
        noisy_mask_col = noisy_mask_col[argsort_indices]

    activity_df = pd.DataFrame(
        {
            "true_weight": true_w_col,
            "weight": w_col,
            "true_mask": true_mask_col,
            "noisy_mask": noisy_mask_col,
            "TP": true_mask_col & noisy_mask_col,
            "FP": ~true_mask_col & noisy_mask_col,
            "TN": ~true_mask_col & ~noisy_mask_col,
            "FN": true_mask_col & ~noisy_mask_col,
        }
    )
    activity_df.sort_values(["true_weight"], inplace=True)

    score_cols = ["TP", "FP", "TN", "FN"]

    assert (activity_df.loc[:, score_cols].values.sum(1) == 1).all()
    activity_df["state"] = (
        activity_df.loc[:, score_cols].astype(np.int32).dot(activity_df.loc[:, score_cols].columns + "+").str[:-1]
    )
    activity_df["true state"] = ["on" if f > 0.5 else "off" for f in activity_df["true_mask"]]
    activity_df["idx"] = list(range(len(w_col)))

    g = sns.scatterplot(
        data=activity_df,
        x="idx",
        y="weight",
        hue="state",
        hue_order=["TP", "FN", "TN", "FP"],
        style="true state",
        style_order=["on", "off"],
        size="state",
        sizes={"TP": 64, "FN": 64, "TN": 32, "FP": 32},
        linewidth=0.01,
        **kwargs,
    )
    g.set_xlabel("")
    joint_handles, joint_labels = g.get_legend_handles_labels()
    g.legend(
        handles=[h for i, h in enumerate(joint_handles) if i not in [0, 5]],
        labels=[h for i, h in enumerate(joint_labels) if i not in [0, 5]],
    )
    if ylim is not None:
        g.set(ylim=ylim)

    return g, activity_df


def savefig_or_show(
    writekey: str,
    show: Optional[bool] = None,
    dpi: Optional[int] = None,
    ext: Optional[str] = None,
    save: Union[bool, str, None] = None,
):
    return sc.pl._utils.savefig_or_show(writekey, show, dpi, ext, save)


def _subset_df(data, groupby, groups, include_rest=True):
    if groups is None:
        return data

    _groups = groups.copy()

    if include_rest:
        data[groupby] = data[groupby].cat.add_categories(include_rest)
        data.loc[~data[groupby].isin(_groups), groupby] = include_rest
        _groups.append(include_rest)
    data = data.loc[data[groupby].isin(_groups), :].copy()
    data[groupby] = data[groupby].cat.remove_unused_categories()

    if data.empty:
        raise ValueError("Empty data, check whether the provided `groups` are correct.")

    return data


def _setup_legend(g, bbox_to_anchor=(1, 0.5), loc="center left", frameon=False, remove_last=False, fontsize=None):
    kwargs = {"bbox_to_anchor": bbox_to_anchor, "loc": loc, "frameon": frameon}

    if remove_last:
        handles, labels = g.get_legend_handles_labels()
        kwargs["handles"] = handles[:-1]
        kwargs["labels"] = labels[:-1]

    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    g.legend(**kwargs)

    return g


def _get_color_dict(factor_adata, groupby, include_rest=True):
    uns_colors_key = f"{groupby}_colors"
    if uns_colors_key not in factor_adata.uns:
        return None
    color_dict = dict(
        zip(factor_adata.obs[groupby].astype("category").cat.categories, factor_adata.uns[uns_colors_key], strict=False)
    )
    if include_rest:
        color_dict[include_rest] = "#D3D3D3"
    return color_dict


# plot groups of observations against (subset of) factors
def group(model, factor_idx, groupby, groups=None, pl_type=HEATMAP, **kwargs):
    if isinstance(factor_idx, int):
        factor_idx = model.factor_names[factor_idx - 1]
    pl_type = pl_type.lower().strip()

    if (pl_type in MATRIXPLOT or pl_type in DOTPLOT) and "colorbar_title" not in kwargs:
        kwargs["colorbar_title"] = "Average scores\nin group"

    if pl_type in DOTPLOT and "size_title" not in kwargs:
        kwargs["size_title"] = "Fraction of samples\nin group (%)"

    type_to_fn = {
        HEATMAP: sc.pl.heatmap,
        MATRIXPLOT: sc.pl.matrixplot,
        DOTPLOT: sc.pl.dotplot,
        TRACKSPLOT: sc.pl.tracksplot,
        VIOLIN: sc.pl.violin,
        STACKED_VIOLIN: sc.pl.stacked_violin,
    }

    try:
        pl_fn = type_to_fn[pl_type]
    except KeyError as e:
        raise ValueError(f"`{pl_type}` is not valid. Select one of {','.join(PL_TYPES)}.") from e

    factor_adata = model._cache["factors"]
    factor_adata = factor_adata[list(factor_adata.keys())[0]]

    return pl_fn(
        factor_adata[_subset_df(factor_adata.obs.copy(), groupby, groups, include_rest=False).index, :],
        factor_idx,
        groupby,
        **kwargs,
    )


# plot ranked factors against groups of observations
def rank(model, n_factors=10, pl_type=None, sep_groups=True, **kwargs):
    factor_adata = model._cache["factors"]
    factor_adata = factor_adata[list(factor_adata.keys())[0]]
    if "rank_genes_groups" not in factor_adata.uns:
        raise ValueError("No group-wise ranking found, run `muvi.tl.rank first.`")

    if pl_type is None:
        print("`pl_type` is None, defaulting to `sc.pl.rank_genes_groups`.")
        pl_type = ""
    pl_type = pl_type.lower().strip()
    n_factors = kwargs.pop("n_genes", n_factors)

    type_to_fn = {
        "": sc.pl.rank_genes_groups,
        HEATMAP: sc.pl.rank_genes_groups_heatmap,
        MATRIXPLOT: sc.pl.rank_genes_groups_matrixplot,
        DOTPLOT: sc.pl.rank_genes_groups_dotplot,
        TRACKSPLOT: sc.pl.rank_genes_groups_tracksplot,
        VIOLIN: sc.pl.rank_genes_groups_violin,
        STACKED_VIOLIN: sc.pl.rank_genes_groups_stacked_violin,
    }

    n_groups = len(factor_adata.obs[factor_adata.uns["rank_genes_groups"]["params"]["groupby"]].unique())
    if "groups" in kwargs and kwargs["groups"] is not None:
        n_groups = len(kwargs["groups"])

    positions = np.linspace(n_factors, n_factors * n_groups, num=n_groups - 1, endpoint=False)

    try:
        pl_fn = type_to_fn[pl_type]
    except KeyError as e:
        raise ValueError(f"`{pl_type}` is not valid. Select one of {', '.join(PL_TYPES)}.") from e

    if not sep_groups:
        return pl_fn(factor_adata, n_genes=n_factors, **kwargs)

    show = kwargs.pop("show", None)
    save = kwargs.pop("save", None)
    _pl = pl_fn(factor_adata, n_genes=n_factors, show=False, save=None, **kwargs)

    # add line separation
    g = None
    if pl_type == HEATMAP:
        g = _pl["heatmap_ax"]
        ymin = -0.5
        ymax = factor_adata.n_obs
        positions -= 0.5
    if pl_type == MATRIXPLOT:
        g = _pl["mainplot_ax"]
        ymin = 0.0
        ymax = n_groups
    if pl_type in (DOTPLOT, STACKED_VIOLIN):
        g = _pl["mainplot_ax"]
        ymin = -0.5
        ymax = n_groups + 0.5

    if g is not None:
        g = _lines(
            g,
            positions,
            ymin=ymin,
            ymax=ymax,
            horizontal=kwargs.pop("swap_axes", False),
            lw=0.5,
            color="black",
            linestyles="dashed",
            zorder=10,
            clip_on=False,
        )
    writekey = "rank"
    if len(pl_type) > 0:
        writekey += f"_{pl_type}"
    savefig_or_show(writekey, show=show, save=save)
    if not show:
        return g


def _groupplot(
    model,
    factor_idx,
    groupby,
    pl_type=STRIPPLOT,
    groups=None,
    include_rest=True,
    rot: int = 45,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    if isinstance(factor_idx, int):
        factor_idx = model.factor_names[factor_idx - 1]
    factor_adata = model._cache["factors"]
    factor_adata = factor_adata[list(factor_adata.keys())[0]]
    if groupby not in factor_adata.obs.columns:
        raise ValueError(
            f"`{groupby}` not found in the metadata, " " add a new column onto `model._cache.factor_adata.obs`."
        )
    data = pd.concat([factor_adata.to_df().loc[:, factor_idx], factor_adata.obs[groupby]], axis=1)

    data = pd.melt(data, id_vars=[groupby], var_name="Factor", value_name="Score")

    data = _subset_df(data, groupby, groups, include_rest=include_rest)

    if pl_type is None:
        pl_type = STRIPPLOT
    pl_type = pl_type.lower().strip()

    type_to_fn = {STRIPPLOT: sns.stripplot, BOXPLOT: sns.boxplot, BOXENPLOT: sns.boxenplot, VIOLINPLOT: sns.violinplot}

    try:
        pl_fn = type_to_fn[pl_type]
    except KeyError as e:
        raise ValueError(f"`{pl_type}` is not valid. Select one of {', '.join(GROUP_PL_TYPES)}.") from e

    legend_fontsize = kwargs.pop("legend_fontsize", None)

    g = pl_fn(
        data=data,
        x="Factor",
        y="Score",
        hue=kwargs.pop("hue", groupby),
        palette=kwargs.pop("palette", _get_color_dict(factor_adata, groupby, include_rest=include_rest)),
        **kwargs,
    )
    if rot is not None:
        for label in g.get_xticklabels():
            label.set_rotation(rot)
    g = _setup_legend(g, remove_last=groups is not None and include_rest, fontsize=legend_fontsize)
    savefig_or_show(pl_type, show=show, save=save)
    if not show:
        return g


def stripplot(
    model,
    factor_idx,
    groupby,
    groups=None,
    include_rest=True,
    rot: int = 45,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    return _groupplot(
        model,
        factor_idx,
        groupby,
        pl_type=STRIPPLOT,
        groups=groups,
        include_rest=include_rest,
        rot=rot,
        show=show,
        save=save,
        **kwargs,
    )


def boxplot(
    model,
    factor_idx,
    groupby,
    groups=None,
    include_rest=True,
    rot: int = 45,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    return _groupplot(
        model,
        factor_idx,
        groupby,
        pl_type=BOXPLOT,
        groups=groups,
        include_rest=include_rest,
        rot=rot,
        show=show,
        save=save,
        **kwargs,
    )


def boxenplot(
    model,
    factor_idx,
    groupby,
    groups=None,
    include_rest=True,
    rot: int = 45,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    return _groupplot(
        model,
        factor_idx,
        groupby,
        pl_type=BOXENPLOT,
        groups=groups,
        include_rest=include_rest,
        rot=rot,
        show=show,
        save=save,
        **kwargs,
    )


def violinplot(
    model,
    factor_idx,
    groupby,
    groups=None,
    include_rest=True,
    rot: int = 45,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    return _groupplot(
        model,
        factor_idx,
        groupby,
        pl_type=VIOLINPLOT,
        groups=groups,
        include_rest=include_rest,
        rot=rot,
        show=show,
        save=save,
        **kwargs,
    )


def scatter(model, x, y, groupby=None, groups=None, include_rest=True, style=None, markers=True, **kwargs):
    if isinstance(x, int):
        x = model.factor_names[x - 1]

    if isinstance(y, int):
        y = model.factor_names[y - 1]

    kwargs["color"] = groupby
    factor_adata = model._cache["factors"]
    factor_adata = factor_adata[list(factor_adata.keys())[0]]

    data = pd.concat([factor_adata.to_df(), factor_adata.obs.copy()], axis=1)

    data = _subset_df(data, groupby, groups, include_rest=include_rest)
    palette = kwargs.pop("palette", _get_color_dict(factor_adata, groupby, include_rest=include_rest))

    if style is None:
        factor_adata = factor_adata.copy()
        if not include_rest:
            factor_adata = factor_adata[data.index, :]
        return sc.pl.scatter(factor_adata, x, y, groups=groups, **kwargs)

    size = kwargs.pop("size", None)
    show = kwargs.pop("show", None)
    save = kwargs.pop("save", None)
    legend_fontsize = kwargs.pop("legend_fontsize", None)

    kwargs = {}

    if size is None:
        size = 120000 / data.shape[0]
    g = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=groupby,
        style=style,
        markers=markers,
        s=size,
        palette=palette,
        linewidth=kwargs.pop("linewidth", 0),
        ax=kwargs.pop("ax", None),
        **kwargs,
    )

    # getting as close as possible to scanpy plotting style
    g = _setup_legend(g, remove_last=groups is not None and include_rest, fontsize=legend_fontsize)

    g.set_title(groupby)
    savefig_or_show("scatter", show=show, save=save)
    if not show:
        return g


def scatter_rank(model, groups=None, **kwargs):
    factor_adata = model._cache["factors"]
    factor_adata = factor_adata[list(factor_adata.keys())[0]]
    try:
        groupby = factor_adata.uns["rank_genes_groups"]["params"]["groupby"]
    except KeyError as e:
        raise ValueError("No group-wise ranking found, run `muvi.tl.rank first.`") from e
    group_df = sc.get.rank_genes_groups_df(factor_adata, group=groups)
    group_df["scores_abs"] = group_df["scores"].abs()

    relevant_factors_dict = {}
    for group in group_df["group"].unique():
        relevant_factors_dict[group] = (
            group_df[group_df["group"] == group].sort_values("scores_abs", ascending=False).iloc[:2]["names"].tolist()
        )

    show = kwargs.pop("show", None)
    save = kwargs.pop("save", None)
    gs = {}

    for group, relevant_factors in relevant_factors_dict.items():
        g = scatter(model, *relevant_factors[:2], groupby=groupby, groups=groups, show=False, save=False, **kwargs)

        g.set_title(f"{groupby} ({group})")
        savefig_or_show(f"scatter_rank_{group}", show=show, save=save)
        gs[group] = g
    if not show:
        return gs


def groupplot_rank(model, groups=None, pl_type=STRIPPLOT, top=1, **kwargs):
    factor_adata = model._cache["factors"]
    factor_adata = factor_adata[list(factor_adata.keys())[0]]
    try:
        groupby = factor_adata.uns["rank_genes_groups"]["params"]["groupby"]
    except KeyError as e:
        raise ValueError("No group-wise ranking found, run `muvi.tl.rank first.`") from e
    group_df = sc.get.rank_genes_groups_df(factor_adata, group=groups)
    group_df["scores_abs"] = group_df["scores"].abs()

    relevant_factors = []
    for group in group_df["group"].unique():
        rfs = group_df[group_df["group"] == group].sort_values("scores_abs", ascending=False).iloc[:top]["names"]
        for rf in rfs:
            if rf not in relevant_factors:
                relevant_factors.append(rf)

    show = kwargs.pop("show", None)
    save = kwargs.pop("save", None)

    return relevant_factors, _groupplot(
        model, relevant_factors, groupby, pl_type=pl_type, groups=groups, show=show, save=save, **kwargs
    )
