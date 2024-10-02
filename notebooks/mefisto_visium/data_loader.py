import os

import anndata as ad
import scanpy as sc


def load_mefisto_visium():
    if os.path.isfile("data/mefisto_visium.h5ad"):
        return ad.read_h5ad("data/mefisto_visium.h5ad")

    adata = sc.read_visium("data", count_file="V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()

    # filter low quality cells
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 20]

    # filter low count genes and cells
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_counts=100)

    # normalize data
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]

    adata.write_h5ad("data/mefisto_visium.h5ad")

    return adata
