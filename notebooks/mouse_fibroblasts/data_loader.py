import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData


def load_mouse_fibroblasts() -> MuData:
    """Load and return mouse fibroblasts datasset."""
    gene_metadata = pd.read_csv("./data/mouse_gene_annotation.csv", index_col=0)
    cell_cycle = pd.read_csv("./data/cell_cycle_annotation.csv", index_col=0)
    c57 = pd.read_csv("./data/SS3_c57_UMIs_concat.csv", index_col=0).T
    cast = pd.read_csv("./data/SS3_cast_UMIs_concat.csv", index_col=0).T

    obs = pd.DataFrame({"cell_cycle": cell_cycle["x"]}, index=c57.index)
    var = pd.merge(
        pd.DataFrame(index=c57.columns),
        gene_metadata,
        left_index=True,
        right_index=True,
        how="left",
    )

    # X chromosome genes
    chrX_gene_inds = var.chrom == "chrX"

    # total number of X chromosome counts for each allele
    c57_total_X = np.nansum(c57.values[:, chrX_gene_inds.values], axis=1)
    cast_total_X = np.nansum(cast.values[:, chrX_gene_inds.values], axis=1)

    # ratio of X chromosomes expression for every cell
    X_ratio = c57_total_X / (c57_total_X + cast_total_X)

    # active X chromosome for every cell
    obs["active_X"] = np.select(
        condlist=[X_ratio < 0.3, (X_ratio >= 0.3) & (X_ratio <= 0.7), X_ratio > 0.7],
        choicelist=["c57", np.nan, "cast"],
    )

    var["gene"] = var.index
    var_c57 = var.copy()
    var_c57.index = var.index.astype(str) + "_c57"
    var_cast = var.copy()
    var_cast.index = var.index.astype(str) + "_cast"

    adata_total = AnnData((c57 + cast).values, obs=obs, var=var)
    adata_c57 = AnnData(c57.values, obs=obs, var=var_c57)
    adata_cast = AnnData(cast.values, obs=obs, var=var_cast)
    adata_allelic = ad.concat(
        [adata_c57, adata_cast], axis=1, join="outer", merge="same"
    )

    return MuData({"mrna_allelic": adata_allelic, "mrna_total": adata_total})
