from data_loader import load_xenium_breast_cancer

from famo.core import CORE

data = load_xenium_breast_cancer()

data["group_scrna"]["rna"].X = data["group_scrna"]["rna"].layers["counts"]
data["group_spatial"]["rna"].X = data["group_spatial"]["rna"].layers["raw"]

prismo_model = CORE(device="cuda:0")
prismo_model.fit(
    data=data,
    n_factors=10,
    weight_prior="Normal",
    factor_prior={"group_spatial": "GP", "group_scrna": "Normal"},
    likelihoods="GammaPoisson",
    covariates_obsm_key={"group_spatial": "spatial", "group_scrna": None},
    max_epochs=3000,
    early_stopper_patience=20,
    lr=5e-2,
    plot_data_overview=False,
    save=True,
    save_path="prismo_gammapoisson_nonnegative",
    init_scale=0.1,
    gp_n_inducing=2000,
    seed=5432,
    batch_size=10000,
    print_every=1,
)
