import altair as alt
import numpy as np
import pandas as pd

alt.data_transformers.enable("vegafusion")


def plot_overview(data):
    missings = pd.DataFrame()
    for group_name, group_data in data.items():
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
        color=alt.Color("missing:N", scale=alt.Scale(range=["#049DBF", "#023373"])),
        facet=alt.Facet("group:N", columns=3, title=None),
    ).properties(width=800, title="Missing Data Overview").display()


def plot_training_curve(model):
    """Plot the training curve, i.e. -ELBO vs epoch."""
    model._check_if_trained()

    train_loss_elbo = model._cache["train_loss_elbo"]
    df = pd.DataFrame({"Epoch": range(len(train_loss_elbo)), "-ELBO": train_loss_elbo})
    alt.Chart(df).mark_line().encode(x="Epoch", y="-ELBO").properties(title="Training Curve").display()


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
                y=alt.Y(
                    "Feature:O",
                    axis=alt.Axis(ticks=False, labels=False, title="Features"),
                ),
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
        corr_df = pd.DataFrame(np.corrcoef(v.X.T)).reset_index()
        # Increase index by 1 to match the factor number and then melt the dataframe

        corr_df["index"] += 1
        corr_df = corr_df.melt("index")
        corr_df.columns = ["Factor1", "Factor2", "Correlation"]

        # Create the heatmap chart
        heatmap = (
            alt.Chart(corr_df)
            .mark_rect()
            .encode(
                x=alt.X("Factor1:O", title="Factor"),
                y=alt.Y("Factor2:O", title="Factor"),
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


def plot_variance_explained(model):
    """Plot the variance explained by each factor in each view."""
    # Check if the model has been trained
    model._check_if_trained()

    # Get the variance explained DataFrame from the model's cache
    df_r2 = model._cache["df_r2"]

    # Create an empty list to hold all the charts
    charts = []

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
                    scale=alt.Scale(
                        scheme="blues",
                        domain=(0, 1.5 * max(r2_df["Variance Explained"])),
                    ),
                ),
                tooltip=["Factor", "View", "Variance Explained"],
            )
            .properties(title=group_name, width=750, height=350)
        )

        # Add the chart to the list of charts
        charts.append(heatmap)

    # Concatenate all the charts horizontally
    final_chart = alt.hconcat(*charts).resolve_scale(color="independent")

    # Display the chart
    final_chart.display()


def plot_factor(model, factor=1):
    """Plot factor values (y-axis) for each sample (x-axis)."""
    model._check_if_trained()

    # We reduce the factor value by one, because we internally start counting at 0
    factor -= 1

    # Create an empty list to hold all the charts
    charts = []

    factors = model._cache["factors"]

    for group_name in model.group_names:
        z = factors[group_name].X.squeeze()
        df = pd.DataFrame(z)
        df["id"] = range(len(df))
        # Convert column names to strings
        df.columns = df.columns.astype(str)

        # Create the scatter plot chart
        scatter_plot = (
            alt.Chart(df)
            .mark_point(filled=True)
            .encode(
                x=alt.X("id:O", title="", axis=alt.Axis(labels=False)),
                y=alt.Y(f"{factor}:Q", title=f"Factor {factor+1}"),
                color=alt.Color(f"{factor}:Q", scale=alt.Scale(scheme="redblue", domainMid=0)),
                tooltip=["id", f"{factor}"],
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


def plot_top_weights(model, view, factor=1, nfeatures=10, orientation="horizontal"):
    """Plot the top nfeatures weights for a given factor and view."""
    model._check_if_trained()

    factor = [factor] if not isinstance(factor, list) else factor

    # We reduce the factor value by one, because we internally start counting at 0
    factor = [x - 1 for x in factor]

    weights = model._cache["weights"]
    feature_names = model._cache["feature_names"][view]

    # Create an empty list to hold all the charts
    charts = []

    for f in factor:
        w = weights[view].X[f, :]
        signs = np.sign(w)
        w = np.abs(w)
        df = pd.DataFrame({"Feature": feature_names, "Weight": w, "Sign": signs})
        df = df.sort_values(by="Weight", ascending=False).head(nfeatures)
        df["Color"] = df["Sign"].apply(lambda x: "red" if x < 0 else "blue")

        if orientation == "horizontal":
            bar_chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Weight:Q", title="Abs. Weight"),
                    y=alt.Y("Feature:O", title="Feature", sort="-x"),
                    color=alt.Color("Color:N", scale=None),
                    tooltip=["Feature", "Weight"],
                )
                .properties(title=f"Factor {f+1} | {view}", width=300, height=300)
            )
        else:
            raise NotImplementedError("Vertical orientation not yet implemented")

        # Add the chart to the list of charts
        charts.append(bar_chart)

    # Concatenate all the charts horizontally or vertically based on the orientation
    if orientation == "horizontal":
        final_chart = alt.hconcat(*charts)
    else:
        final_chart = alt.vconcat(*charts)

    # Create a legend
    legend_data = pd.DataFrame({"Label": ["Negative", "Positive"], "Color": ["red", "blue"]})

    legend_boxes = (
        alt.Chart(legend_data)
        .mark_square(size=100)
        .encode(
            y=alt.Y("Label:N", axis=alt.Axis(title=None, labels=False, ticks=False)),
            color=alt.Color("Color:N", scale=None),
        )
        .properties(width=30, height=50)
    )

    legend_text = (
        alt.Chart(legend_data)
        .mark_text(align="left", dx=-100)
        .encode(y=alt.Y("Label:N", axis=None), text="Label:N")
        .properties(width=30, height=50)
    )

    legend = alt.hconcat(legend_boxes, legend_text).properties(title="Legend")

    # Add legend to the final chart
    final_chart = alt.hconcat(final_chart, legend).configure_view(stroke=None)  # Removes the border from the text box

    # Display the chart
    final_chart.display()


def plot_weights(model, view, factor=1, top_n_features=10):
    factor = factor - 1

    weights = model.get_weights()[view][factor]
    feature_names = model.feature_names[view]
    df_plot = pd.DataFrame({"weight": weights, "feature": feature_names})
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
                "weight:Q",
                title="Weight",
                scale=alt.Scale(domain=[x_min - x_margin * 2.5, x_max + x_margin * 2.5]),
            ),
            y=alt.Y("rank:Q", title="Rank", axis=alt.Axis(labels=False, ticks=False)),
        )
    )

    # Add feature names as text labels
    text_pos = (
        alt.Chart(top_n_pos)
        .mark_text(align="left", baseline="middle", dx=5)
        .encode(
            x="label_x:Q",
            y="label_y:Q",
            text="feature:N",
        )
    )
    text_neg = (
        alt.Chart(top_n_neg)
        .mark_text(align="right", baseline="middle", dx=-5)
        .encode(
            x="label_x:Q",
            y="label_y:Q",
            text="feature:N",
        )
    )

    # Add lines connecting points and labels
    lines_pos = (
        alt.Chart(top_n_pos)
        .mark_rule(color="gray", strokeDash=[1, 1])
        .encode(
            x="label_x:Q",
            y="label_y:Q",
            x2="weight:Q",
            y2="rank:Q",
        )
    )
    lines_neg = (
        alt.Chart(top_n_neg)
        .mark_rule(color="gray", strokeDash=[1, 1])
        .encode(
            x="label_x:Q",
            y="label_y:Q",
            x2="weight:Q",
            y2="rank:Q",
        )
    )

    chart = (points + text_pos + text_neg + lines_pos + lines_neg).properties(
        title=f"Top {view} weights for factor {factor}",
        width=600,
        height=400,
    )

    chart.display()
