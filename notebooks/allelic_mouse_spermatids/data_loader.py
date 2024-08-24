import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from scipy.io import mmread


def load_mouse_spermatids() -> MuData:
    """Load and return mouse spermatids dataset."""
    genes = pd.read_csv("./data/genes.txt", sep="\t")
    metadata = pd.read_csv("./data/metadata.txt", sep="\t")

    genes_alt = genes.copy()
    genes_alt.index = genes_alt.index.astype(str) + "_alt"
    genes_ref = genes.copy()
    genes_ref.index = genes_ref.index.astype(str) + "_ref"

    alt = np.array(mmread("./data/allelic_counts_alternative.mtx").todense()).T
    ref = np.array(mmread("./data/allelic_counts_reference.mtx").todense()).T
    total = np.array(mmread("./data/total_counts.mtx").todense()).T

    allelic_sum = alt + ref
    gene_inds = np.array(allelic_sum.mean(axis=0) > 0.05).flatten()
    cell_inds = metadata.Species == "B6xCAST"

    genes = genes[gene_inds]
    genes_alt = genes_alt[gene_inds]
    genes_ref = genes_ref[gene_inds]

    metadata = metadata[cell_inds]

    alt = alt[cell_inds][:, gene_inds]
    ref = ref[cell_inds][:, gene_inds]
    total = total[cell_inds][:, gene_inds]

    adata_alt = AnnData(X=alt, obs=metadata, var=genes_alt, dtype=np.float32)
    adata_ref = AnnData(X=ref, obs=metadata, var=genes_ref, dtype=np.float32)
    adata_allelic = ad.concat([adata_alt, adata_ref], axis=1, join="outer", merge="same")
    adata_total = AnnData(X=total, obs=metadata, var=genes, dtype=np.float32)

    mdata = MuData({"mrna_allelic": adata_allelic, "mrna_total": adata_total})

    return mdata
