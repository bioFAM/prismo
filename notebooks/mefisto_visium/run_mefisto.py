if __name__ == "__main__":
    from data_loader import load_mefisto_visium
    from mofapy2.run.entry_point import entry_point

    adata = load_mefisto_visium()

    ent = entry_point()
    ent.set_data_options(use_float32=True)
    ent.set_data_from_anndata(adata)
    ent.set_model_options(factors=4)
    ent.set_train_options()
    ent.set_train_options(seed=54321)
    n_inducing = 1000

    ent.set_covariates([adata.obsm["spatial"]], covariates_names=["imagerow", "imagecol"])
    ent.set_smooth_options(sparseGP=True, frac_inducing=n_inducing / adata.n_obs, start_opt=10, opt_freq=10)
    ent.build()
    ent.run()

    expectations = ent.model.getExpectations()
