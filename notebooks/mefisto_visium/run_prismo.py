if __name__ == "__main__":
    import warnings

    from data_loader import load_mefisto_visium

    from famo.core import CORE

    warnings.simplefilter(action="ignore", category=FutureWarning)

    adata = load_mefisto_visium()

    model = CORE(device="cuda:0")

    model.fit(
        n_factors=4,
        data=adata,
        factor_prior="GP",
        weight_prior="SnS",
        covariates_obsm_key="spatial",
        lr=0.005,
        early_stopper_patience=100,
        print_every=100,
        center_groups=True,
        scale_views=False,
        scale_groups=True,
        max_epochs=10000,
        save=True,
        save_path="trained_model",
        init_factors="pca",
        init_scale=0.1,
        gp_n_inducing=1000,
        seed=54321,
    )
