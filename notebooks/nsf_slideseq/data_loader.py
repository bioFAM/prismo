# to download the data, run data_download.R first

from os import path

import numpy as np
import scanpy as sc


def load_nsf_slideseq():
    if path.exists("data/sshippo_J2000.h5ad"):
        return sc.read_h5ad("data/sshippo_J2000.h5ad")

    np.random.seed(101)

    adata = sc.read_h5ad("data/sshippo.h5ad")
    adata.X = adata.raw.X
    adata.raw = None

    # add spatial coordinates to data and normalize them
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()
    adata.obs.drop(columns=["x", "y"], inplace=True)
    x_min = adata.obsm["spatial"].min(axis=0)
    adata.obsm["spatial"] -= x_min
    x_mean = np.exp(np.mean(np.log(adata.obsm["spatial"].max(axis=0))))
    adata.obsm["spatial"] *= 4.0 / x_mean
    adata.obsm["spatial"] -= adata.obsm["spatial"].mean(axis=0)

    # filter out cells with high mitochondrial counts
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 20]

    # filter out cells with low total counts
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=1)
    adata.layers = {"counts": adata.X.copy()}
    sc.pp.normalize_total(adata, inplace=True, layers=None, key_added="sizefactor")
    sc.pp.log1p(adata)

    # select top 2000 genes with highest deviance
    o = np.argsort(-adata.var["deviance_poisson"])
    idx = list(range(adata.shape[0]))
    np.random.shuffle(idx)
    adata = adata[idx, o]
    adata = adata[:, :2000]
    adata.X = adata.layers["counts"].copy()

    # sort cells and genes
    adata = adata[adata.obs.index.argsort()]
    adata = adata[:, adata.var.index.argsort()]

    adata.write_h5ad("data/sshippo_J2000.h5ad", compression="gzip")

    return adata
