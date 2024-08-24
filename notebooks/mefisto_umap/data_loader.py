import pandas as pd
from anndata import AnnData


def load_mefisto_umap() -> dict[dict[str, AnnData]]:
    """Load mouse development dataset with 2D UMAP coordinates used in MEFISTO paper."""
    data = pd.read_csv("./data/scnmt_data.txt", sep="\t")
    metadata = pd.read_csv("./data/scnmt_sample_metadata.txt", sep="\t").set_index("sample")

    adata_dict = {}
    adata_dict["group_1"] = {}

    for view in data["view"].unique():
        data_view = data[data.view == view]
        data_view_pivot = pd.pivot(data_view, index="sample", columns="feature", values="value")

        adata_dict["group_1"][view] = AnnData(
            X=data_view_pivot,
            obs=metadata[metadata.index.isin(data_view_pivot.index)].drop(columns=["UMAP1", "UMAP2"]),
            obsm={"umap": metadata[metadata.index.isin(data_view_pivot.index)][["UMAP1", "UMAP2"]].values},
        )

    return adata_dict
