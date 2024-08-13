import pandas as pd
from anndata import AnnData
from mudata import MuData


def load_mouse_development() -> MuData:
    """Load and return mouse development dataset used in MEFISTO paper."""
    # split up data into one dataframe per unique value in view column

    data = pd.read_csv("./data/scnmt_data.txt", sep="\t")
    metadata = pd.read_csv("./data/scnmt_sample_metadata.txt", sep="\t").set_index("sample")

    data_dict = {}

    for view in data["view"].unique():
        data_view = data[data["view"] == view]
        data_view_pivot = pd.pivot(data_view, index="sample", columns="feature", values="value")

        data_dict[view] = AnnData(data_view_pivot, obs=metadata[metadata.index.isin(data_view_pivot.index)])

    return MuData(data_dict)
