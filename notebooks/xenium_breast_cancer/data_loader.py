import anndata as ad
import pandas as pd
import scanpy as sc


def load_xenium_breast_cancer():
    df = pd.read_csv(
        "data/gene_list",
        sep="\t",
        comment="#",
        names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
    )
    df_split = df["attribute"].str.split(";", expand=True)

    genes_df = pd.DataFrame(
        {
            "gene_id": df_split[0].str.split("=", expand=True).iloc[:, 1].values,
            "gene_symbol": df_split[3].str.split("=", expand=True).iloc[:, 1].values,
        }
    )
    genes_df["gene_id"] = genes_df["gene_id"].str.split(".", expand=True).iloc[:, 0]

    celltypes = pd.read_excel("data/Cell_Barcode_Type_Matrices.xlsx", sheet_name="scFFPE-Seq")
    celltypes.columns = ["barcode", "celltype"]
    celltypes.set_index("barcode", inplace=True)

    adata_scrna = sc.read_10x_h5(
        "data/Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5"
    )
    adata_scrna.var["gene_symbol"] = adata_scrna.var.index.copy()
    adata_scrna.var_names = adata_scrna.var["gene_ids"]
    adata_scrna.var.drop(columns=["feature_types", "genome", "gene_ids"], inplace=True)
    adata_scrna.var.index.name = "gene_id"
    adata_scrna.obs.index.name = "barcode"
    adata_scrna.obs = pd.merge(adata_scrna.obs, celltypes, left_index=True, right_index=True, how="left")
    adata_scrna.layers["counts"] = adata_scrna.X.copy()
    sc.pp.normalize_total(adata_scrna)
    sc.pp.log1p(adata_scrna)
    sc.pp.highly_variable_genes(adata_scrna, n_top_genes=5000, subset=True)
    sc.pp.normalize_total(adata_scrna)
    sc.pp.scale(adata_scrna)

    adata_xenium = ad.read_h5ad("data/adata_10X.h5ad")
    adata_xenium.var["gene_symbol"] = adata_xenium.var.index.copy()
    adata_xenium.var["gene_id"] = (
        pd.merge(adata_xenium.var, genes_df, on="gene_symbol", how="left")
        .drop_duplicates(subset="gene_symbol")
        .gene_id.values
    )
    adata_xenium.var_names = adata_xenium.var["gene_id"].copy()
    adata_xenium.var.drop(columns=["feature_types", "genome", "gene_id"], inplace=True)
    adata_xenium.obsm["spatial"] = adata_xenium.obs[["nucleus_centroid_x", "nucleus_centroid_y"]].values
    adata_xenium.obs.rename(columns={"celltype_major": "celltype"}, inplace=True)

    return {"group_spatial": {"rna": adata_xenium}, "group_scrna": {"rna": adata_scrna}}
