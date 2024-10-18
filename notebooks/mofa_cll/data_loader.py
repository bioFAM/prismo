import pandas as pd
from anndata import AnnData
from mudata import MuData


def load_CLL() -> MuData:
    """Load and return CLL dataset."""
    modalities = {}

    obs = pd.read_csv(filepath_or_buffer="./data/metadata.txt", sep="\t", index_col="sample", encoding="utf_8")

    for ome in ["drugs", "methylation", "mrna", "mutations"]:
        modality = pd.read_csv(filepath_or_buffer=f"./data/cll_{ome}.csv", sep=",", index_col=0, encoding="utf_8").T
        modalities[ome] = AnnData(X=modality)

    # Replace with gene ID with gene name
    gene_ids = pd.read_csv("./data/cll_gene_ids.csv", index_col=0)
    cols = list(modalities["mrna"].var_names)

    # Replace each value in cols with the corrsponding value in the gene_ids dataframe
    cols = [gene_ids.loc[gene_ids["GENEID"] == gene, "SYMBOL"].item() for gene in cols]
    modalities["mrna"].var_names = cols

    # avoid duplicated names with the Mutations view
    modalities["mutations"].var_names = [f"m_{x}" for x in modalities["mutations"].var_names]

    # Replace drug names
    # Create mapping from drug_id to name
    drug_names = pd.read_csv("./data/drugs.txt", sep=",", index_col=0)
    mapping = drug_names["name"].to_dict()

    # Replace all substrings in drugs.columns as keys with the corresponding values in the mapping
    cols = []
    for k in modalities["drugs"].var_names:
        for v in mapping.keys():
            if v in k:
                cols.append(k.replace(v, mapping[v]))
                break

    modalities["drugs"].var_names = cols

    mdata = MuData(modalities)
    mdata.obs = mdata.obs.join(obs)

    return mdata
