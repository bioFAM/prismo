import pandas as pd
from anndata import AnnData


def load_mefisto_microbiome() -> dict[dict[str, AnnData]]:
    """Load microbiome dataset used in MEFISTO paper."""
    data = pd.read_csv("data/microbiome_data.csv")
    metadata = pd.read_csv("data/microbiome_features_metadata.csv")

    groups = data["group"].unique()

    data_dict = {}

    for group in groups:
        data_group = data[data["group"] == group]
        data_pivot = pd.pivot(data_group, index="sample", columns="feature", values="value")

        adata = AnnData(
            X=data_pivot.values,
            obs=data_group[~data_group.duplicated("sample")]
            .sort_values("sample")[["sample", "group", "month", "delivery", "diet", "sex"]]
            .set_index("sample"),
            var=metadata[metadata["SampleID"].isin(data_pivot.columns)]
            .sort_values("SampleID")[["SampleID", "Taxon", "Confidence"]]
            .rename(columns={"SampleID": "feature"})
            .set_index("feature"),
        )

        data_dict[group] = {"microbiome": adata}

    return data_dict
