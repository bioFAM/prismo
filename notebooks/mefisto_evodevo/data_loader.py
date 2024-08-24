import pandas as pd
from anndata import AnnData


def load_mefisto_evodevo() -> dict[dict[str, AnnData]]:
    """Load evodevo dataset used in MEFISTO paper."""
    data = pd.read_csv("data/evodevo.csv")

    adata_dict = {}
    for group in data.group.unique():
        data_group = data[data.group == group]
        adata_dict[group] = {}
        for view in data_group.view.unique():
            data_group_view = data_group[data_group.view == view]
            data_pivot = pd.pivot(data_group_view, index="sample", columns="feature", values="value")

            adata_dict[group][view] = AnnData(
                X=data_pivot.values,
                obs=data_group_view[~data_group_view.duplicated("sample")]
                .sort_values("sample")[["sample", "group", "view", "time"]]
                .set_index("sample"),
                var=pd.DataFrame(
                    index=data_group_view[~data_group_view.duplicated("feature")]
                    .sort_values("feature")["feature"]
                    .values
                ),
            )

    return adata_dict
