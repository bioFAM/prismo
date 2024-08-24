import pandas as pd
from anndata import AnnData


def load_mefisto_microbiome() -> dict[dict[str, AnnData]]:
    """Load microbiome dataset used in MEFISTO paper."""
    data = pd.read_csv("data/microbiome_data.csv")
    metadata = pd.read_csv("data/microbiome_features_metadata.csv")

    data_pivot = pd.pivot(data, index="sample", columns="feature", values="value")

    adata = AnnData(
        X=data_pivot.values,
        obs=data[~data.duplicated("sample")]
        .sort_values("sample")[["sample", "group", "month", "delivery", "diet", "sex"]]
        .set_index("sample"),
        var=metadata[metadata["SampleID"].isin(data_pivot.columns)]
        .sort_values("SampleID")[["SampleID", "Taxon", "Confidence"]]
        .rename(columns={"SampleID": "feature"})
        .set_index("feature"),
    )

    return {"group_1": {"microbiome": adata}}
