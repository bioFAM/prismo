import anndata as ad


def load_nsf_slideseq():
    return ad.read_h5ad("data/sshippo_processed.h5ad")


def load_nsf_slideseq_subset_genes():
    return ad.read_h5ad("data/sshippo_J2000.h5ad")
