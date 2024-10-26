import logging

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import scarches as sca
import Spectra
from sklearn.decomposition import NMF
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    precision_recall_fscore_support,
    root_mean_squared_error,
)

# import muvi
from prismo.prismo import CORE

logger = logging.getLogger(__name__)


def generate_spectra_5d(
    rng,
    n_samples=20,
    n_features=500,
    n_factors=3,
    n_informative_factors=None,
    corr_coef=0.95,
    sparsity_interval=None,
    sigma=2,
):
    if sparsity_interval is None:
        sparsity_interval = (0.85, 0.95)
    if n_informative_factors is None:
        n_informative_factors = n_factors
    print(corr_coef)
    # factor scores
    z_mu = np.zeros(n_factors)
    z_cov = np.eye(n_factors)
    z_cov[0, 1] = corr_coef
    z_cov[1, 0] = corr_coef
    z = np.exp(rng.multivariate_normal(z_mu, z_cov, size=n_samples))

    w_shape = (n_factors, n_features)
    # half-cauchy
    w = np.abs(rng.standard_cauchy(w_shape).reshape(w_shape))
    true_mask = np.ones_like(w).astype(bool)

    for k in range(n_factors):
        sparsity_thresh = rng.uniform(*sparsity_interval)
        for true_thresh in sorted(set(w[k, :])):
            if (w[k, :] < true_thresh).mean() > sparsity_thresh:
                break
        true_mask[k, w[k, :] < true_thresh] = False

    # add some noise to avoid exactly zero values
    w = np.where(true_mask, w, np.abs(rng.standard_normal(w_shape) / 100))
    # assert ((w >= true_thresh) == true_mask).all()
    # print(f"true w threshold: {true_thresh}")

    z_informative = z[:, :n_informative_factors]
    w_informative = w[:n_informative_factors, :]

    rate = z_informative @ w_informative + np.abs(rng.normal(0, sigma))
    x = np.array(rng.poisson(rate), dtype=np.int32)

    return w, true_mask, z, x


def generate_spectra_5e(rng, n_samples=1000, n_features=500, n_factors=10, factor_size=20, overlap_coef=0.95, sigma=2):
    # factor scores
    z_shape = (n_samples, n_factors)
    z = rng.lognormal(0, 1, size=z_shape)
    z = np.where(z < 1, np.abs(rng.standard_normal(z_shape) / 100), z)

    w_shape = (n_factors, n_features)
    w = rng.exponential(16, size=w_shape)

    true_mask = np.zeros_like(w).astype(bool)

    idx_pairs = []
    indices = list(range(true_mask.shape[0]))
    while indices:
        idx_pairs.append((indices.pop(rng.choice(len(indices), 1)[0]), indices.pop(rng.choice(len(indices), 1)[0])))

    n_overlap = int(factor_size * overlap_coef)

    for k, l in idx_pairs:
        on_idx_k = rng.choice(true_mask.shape[1], factor_size, replace=False)
        on_idx_l = rng.choice(true_mask.shape[1], factor_size, replace=False)
        on_idx_l[:n_overlap] = on_idx_k[:n_overlap]
        true_mask[k, on_idx_k] = True
        true_mask[l, on_idx_l] = True

    w *= true_mask
    # add some noise to avoid exactly zero values
    w = np.where(true_mask, w, np.abs(rng.standard_normal(w_shape) / 100))

    rate = z @ w + np.abs(rng.normal(0, sigma))
    x = np.array(rng.poisson(rate), dtype=np.int32)

    return w, true_mask, z, x


def get_rand_noisy_mask(rng, true_mask, fpr=0.2, fnr=0.2):
    noisy_mask = np.array(true_mask, copy=True)

    for k in range(true_mask.shape[0]):
        active_idx = noisy_mask[k, :].nonzero()[0]
        inactive_idx = (~noisy_mask[k, :]).nonzero()[0]
        fp = int(fpr * len(active_idx))
        fn = int(fnr * len(active_idx))
        fp_idx = rng.choice(inactive_idx, fp, replace=False)
        fn_idx = rng.choice(active_idx, fn, replace=False)
        noisy_mask[k, fp_idx] = True
        noisy_mask[k, fn_idx] = False

    return noisy_mask


def train_nmf(data, mask, seed=0, **kwargs):
    n_components = kwargs.pop("n_components", mask.shape[0])
    init = kwargs.pop("init", "random")
    max_iter = kwargs.pop("max_iter", 1000)
    model = NMF(n_components=n_components, init=init, max_iter=max_iter, random_state=seed, **kwargs)
    model.fit(data)
    return model


def train_spectra(data, mask, seed=0, terms=None, **kwargs):
    adata = ad.AnnData(data)
    annot = pd.DataFrame(mask, columns=adata.var_names)
    if terms is not None:
        annot.index = pd.Index(terms, name="terms")

    annot = {f"all_{idx!s}": annot.columns[annot.loc[idx, :]].tolist() for idx in annot.index}

    lam = kwargs.pop("lam", 0.01)
    delta = kwargs.pop("delta", 0.001)
    kappa = kwargs.pop("kappa", None)
    rho = kwargs.pop("rho", 0.001)
    num_epochs = kwargs.pop("num_epochs", 10000)

    model = Spectra.est_spectra(
        adata=adata,
        gene_set_dictionary=annot,  # because we do not use the cell types
        # L=n_factors,
        # we will supply a regular dict
        # instead of the nested dict above
        use_highly_variable=False,
        cell_type_key=None,  # "cell_type_annotations"
        use_weights=True,
        lam=lam,
        delta=delta,
        kappa=kappa,
        rho=rho,
        use_cell_types=False,  # set to False to not use the cell type annotations
        n_top_vals=25,
        filter_sets=False,
        label_factors=False,
        clean_gs=False,
        min_gs_num=3,
        overlap_threshold=0.2,
        num_epochs=num_epochs,  # for demonstration purposes we will only run 2 epochs, we recommend 10,000 epochs
        **kwargs,
    )

    return model


def train_expimap(data, mask, seed=0, terms=None, **kwargs):
    adata = ad.AnnData(data)
    adata.obs["cond"] = "cond"
    adata.varm["I"] = mask.T
    if terms is not None:
        adata.uns["terms"] = terms
    else:
        adata.uns["terms"] = [f"factor_{k}" for k in range(mask.shape[0])]

    recon_loss = kwargs.pop("recon_loss", "mse")
    soft_mask = kwargs.pop("soft_mask", True)
    alpha = kwargs.pop("alpha", None)
    alpha_l1 = kwargs.pop("alpha_l1", None)
    n_epochs = kwargs.pop("n_epochs", 500)
    batch_size = kwargs.pop("batch_size", adata.shape[0])

    hidden_size_1 = kwargs.pop("hidden_size_1", 512)
    hidden_size_2 = kwargs.pop("hidden_size_2", 128)

    # create an instance of the model
    model = sca.models.EXPIMAP(
        adata=adata,
        condition_key="cond",
        hidden_layer_sizes=[hidden_size_1, hidden_size_2],
        recon_loss=recon_loss,
        soft_mask=soft_mask,
    )

    early_stopping_kwargs = {
        "early_stopping_metric": "val_unweighted_loss",  # val_unweighted_loss
        "threshold": 0,
        "patience": 50,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }
    model.train(
        n_epochs=n_epochs,
        # alpha_epoch_anneal=100,
        alpha=alpha,
        # # alpha_kl=0.5,
        alpha_l1=alpha_l1,
        # weight_decay=0.0,
        use_early_stopping=True,
        early_stopping_kwargs=early_stopping_kwargs,
        batch_size=batch_size,
        monitor_only_val=False,
        seed=seed,
        **kwargs,
    )

    return model


def train_muvi(data, mask, seed=0, terms=None, **kwargs):
    adata = ad.AnnData(data)
    if terms is None:
        terms = [f"factor_{k}" for k in range(mask.shape[0])]
    adata.varm["I"] = pd.DataFrame(mask, index=terms, columns=adata.var_names).T

    prior_confidence = kwargs.pop("prior_confidence", 0.99)
    n_dense = kwargs.pop("n_dense", 0)
    likelihood = kwargs.pop("likelihood", "normal")
    nmf = kwargs.pop("nmf", False)

    dense_scale = kwargs.pop("dense_scale", None)
    batch_size = kwargs.pop("batch_size", 0)
    n_epochs = kwargs.pop("n_epochs", 10000)
    n_particles = kwargs.pop("n_particles", 1)
    learning_rate = kwargs.pop("learning_rate", 0.003)

    tolerance = kwargs.pop("tolerance", 1e-5)
    patience = kwargs.pop("patience", 10)

    true_mask = kwargs.pop("true_mask", None)
    n_factors = mask.shape[0]

    model = muvi.tl.from_adata(
        adata,
        prior_mask_key="I",
        n_factors=n_factors + n_dense,
        prior_confidence=prior_confidence,
        likelihoods=[likelihood],
        nmf=[nmf],
        **kwargs,
    )

    if dense_scale is not None:
        for vn in model.view_names:
            model.prior_scales[vn][-n_dense:, :] = dense_scale

    callbacks = []
    if true_mask is not None:
        # this callback logs metrics of the training every X steps to better gauge the training progress
        log_callback = muvi.LogCallback(
            model,
            n_epochs,
            n_checkpoints=20,
            active_callbacks=["binary_scores", "avg_precision", "rmse"],
            # pass true masks
            masks={"view_0": true_mask},
            binary_scores_at=200,
            threshold=0.0,
            log=False,
            n_annotated=n_factors,
        )
        callbacks.append(log_callback)

    callbacks.append(
        muvi.EarlyStoppingCallback(n_epochs, min_epochs=n_epochs // 10, tolerance=tolerance, patience=patience)
    )

    model.fit(
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_particles=n_particles,
        learning_rate=learning_rate,
        optimizer="clipped",
        verbose=1,
        seed=seed,
        callbacks=callbacks,
    )

    return model


def train_famo(data, mask, seed=None, terms=None, **kwargs):
    adata = ad.AnnData(data)
    if terms is None:
        terms = [f"factor_{k}" for k in range(mask.shape[0])]
    adata.varm["I"] = pd.DataFrame(mask, index=terms, columns=adata.var_names).T

    device = kwargs.pop("device", "cpu")
    prior_penalty = kwargs.pop("prior_penalty", 0.005)
    n_factors = kwargs.pop("n_factors", 3)
    likelihood = kwargs.pop("likelihood", "Normal")
    nmf = kwargs.pop("nmf", False)

    batch_size = kwargs.pop("batch_size", 0)
    max_epochs = kwargs.pop("max_epochs", 10000)
    n_particles = kwargs.pop("n_particles", 1)
    lr = kwargs.pop("lr", 0.003)

    early_stopper_patience = kwargs.pop("early_stopper_patience", 100)

    model = CORE(device=device)
    model.fit(
        mu.MuData({"view_0": adata}),
        n_factors=n_factors,
        annotations={"view_0": adata.varm["I"].T},
        weight_prior="Horseshoe",
        factor_prior="Normal",
        likelihoods={"view_0": likelihood},
        nonnegative_weights=nmf,
        nonnegative_factors=nmf,
        prior_penalty=prior_penalty,
        batch_size=batch_size,
        max_epochs=max_epochs,
        n_particles=n_particles,
        lr=lr,
        early_stopper_patience=early_stopper_patience,
        print_every=500,
        plot_data_overview=False,
        scale_per_group=False,
        use_obs=None,
        use_var=None,
        seed=seed,
        **kwargs,
    )

    return model


def get_factor_loadings(model, with_dense=False):
    if type(model).__name__ == "NMF":
        return model.components_
    if type(model).__name__ == "CORE":
        w_hat = model.get_weights("numpy")["view_0"]
        if not with_dense and model.n_dense_factors > 0:
            return w_hat[model.n_dense_factors :, :]
        return w_hat
    if type(model).__name__ == "MuVI":
        w_hat = model.get_factor_loadings(as_df=False)["view_0"]
        if model.n_dense_factors > 0:
            return w_hat[: -model.n_dense_factors, :]
        return w_hat
    if type(model).__name__ == "SPECTRA_Model":
        return model.return_factors()[:-1, :]
    if type(model).__name__ == "EXPIMAP":
        return model.model.decoder.L0.expr_L.weight.cpu().detach().numpy().T

    raise ValueError(f"Unknown model type: {type(model)}")


def get_factor_scores(model, data, with_dense=False):
    if type(model).__name__ == "NMF":
        return model.transform(data)
    if type(model).__name__ == "CORE":
        z_hat = model.get_factors("numpy")["group_1"]
        if not with_dense and model.n_dense_factors > 0:
            return z_hat[:, model.n_dense_factors :]
        return z_hat
    if type(model).__name__ == "MuVI":
        z_hat = model.get_factor_scores(as_df=False)
        if model.n_dense_factors > 0:
            return z_hat[:, : -model.n_dense_factors]
        return z_hat
    if type(model).__name__ == "SPECTRA_Model":
        return model.return_cell_scores()[:, :-1]
    if type(model).__name__ == "EXPIMAP":
        return model.get_latent()

    raise ValueError(f"Unknown model type: {type(model)}")


def get_reconstructed(model, data):
    if type(model).__name__ == "NMF":
        return get_factor_scores(model, data) @ get_factor_loadings(model)
    if type(model).__name__ == "CORE":
        return get_factor_scores(model, data, with_dense=True) @ get_factor_loadings(model, with_dense=True)
    if type(model).__name__ == "MuVI":
        return model.get_reconstructed(as_df=False)["view_0"]
    if type(model).__name__ == "SPECTRA_Model":
        return model.return_cell_scores() @ (model.return_factors() * model.return_gene_scalings())
    if type(model).__name__ == "EXPIMAP":
        return model.get_y()

    raise ValueError(f"Unknown model type: {type(model)}")


def _r2(y_true, y_pred):
    ss_res = np.nansum(np.square(y_true - y_pred))
    ss_tot = np.nansum(np.square(y_true))
    return 1.0 - (ss_res / ss_tot)


def get_variance_explained(model, data, per_factor=False):
    z = get_factor_scores(model, data)
    w = get_factor_loadings(model)
    if not per_factor:
        return _r2(data, get_reconstructed(model, data))
    n_factors = z.shape[1]
    r2 = []
    for k in range(n_factors):
        y_pred_fac_k = np.outer(z[:, k], w[k, :])
        r2.append(_r2(data, y_pred_fac_k))
    return r2


def get_rmse(model, data, per_factor=False):
    z = get_factor_scores(model, data)
    w = get_factor_loadings(model)
    if not per_factor:
        return root_mean_squared_error(data, get_reconstructed(model, data))
    n_factors = z.shape[1]
    rmse = []
    for k in range(n_factors):
        y_pred_fac_k = np.outer(z[:, k], w[k, :])
        rmse.append(root_mean_squared_error(data, y_pred_fac_k))
    return rmse


def get_top_factors(model, data, r2_thresh=0.95):
    r2 = get_variance_explained(model, data, per_factor=True)
    r2_argsorted = np.argsort(r2)[::-1]
    r2_sorted = np.sort(r2)[::-1]

    if r2_thresh < 1.0:
        r2_thresh = (np.cumsum(r2_sorted) / np.sum(r2_sorted) < r2_thresh).sum() + 1

    return r2_argsorted[:r2_thresh], r2[:r2_thresh]


def sort_and_subset(w_hat, true_mask, top=None):
    # descending order
    argsort_indices = np.argsort(-np.abs(w_hat), axis=1)

    sorted_w_hat = np.array(list(map(lambda x, y: y[x], argsort_indices, w_hat)))
    sorted_true_mask = np.array(list(map(lambda x, y: y[x], argsort_indices, true_mask)))

    if top is not None:
        argsort_indices = argsort_indices[:, :top]
        sorted_w_hat = sorted_w_hat[:, :top]
        sorted_true_mask = sorted_true_mask[:, :top]

    return argsort_indices, sorted_w_hat, sorted_true_mask


def get_binary_scores(true_mask, model, threshold=0.0, per_factor=False, top=None):
    w_hat = get_factor_loadings(model)
    feature_idx, w_hat, true_mask = sort_and_subset(w_hat, true_mask, top)

    if threshold is not None:
        prec, rec, f1, supp = precision_recall_fscore_support(
            (true_mask).flatten(), (np.abs(w_hat) > threshold).flatten(), average="binary"
        )
    else:
        prec = None
        rec = None
        f1 = None
        supp = None
        sorted_w_hat = np.sort(np.abs(w_hat).flatten())
        for threshold_idx in np.linspace(0, len(sorted_w_hat), num=100, endpoint=False, dtype=int):
            threshold_ = sorted_w_hat[threshold_idx]
            prec_, rec_, f1_, supp_ = precision_recall_fscore_support(
                (true_mask).flatten(), (np.abs(w_hat) > threshold_).flatten(), average="binary"
            )

            if f1 is None or f1_ > f1:
                threshold = threshold_
                prec, rec, f1, supp = prec_, rec_, f1_, supp_
        print(f"best threshold: {threshold}")

    if not per_factor:
        return prec, rec, f1, threshold

    per_factor_prec = []
    per_factor_rec = []
    per_factor_f1 = []
    for k in range(w_hat.shape[0]):
        mask = true_mask[k, :]
        loadings_hat = np.abs(w_hat[k, :])
        order = np.argsort(loadings_hat)[::-1]
        prec, rec, f1, _ = precision_recall_fscore_support(
            mask[order], loadings_hat[order] > threshold, average="binary"
        )
        per_factor_prec.append(prec)
        per_factor_rec.append(rec)
        per_factor_f1.append(f1)

    return per_factor_prec, per_factor_rec, per_factor_f1, threshold


def get_reconstruction_fraction(true_mask, noisy_mask, model, top=None):
    fi_1, w_hat, true_mask = sort_and_subset(get_factor_loadings(model), true_mask, top)
    fi_2, w_hat, noisy_mask = sort_and_subset(get_factor_loadings(model), noisy_mask, top)
    assert (fi_1 == fi_2).all()
    feature_idx = fi_1
    # return true_mask & ~noisy_mask
    return feature_idx, w_hat, true_mask, noisy_mask
    # return (np.abs(w_hat) > 1e-3).mean()


def get_average_precision(true_mask, model, per_factor=False, top=None):
    w_hat = get_factor_loadings(model)
    feature_idx, w_hat, true_mask = sort_and_subset(w_hat, true_mask, top)
    if not per_factor:
        return average_precision_score((true_mask).flatten(), np.abs(w_hat).flatten())
    per_factor_aupr = []
    for k in range(w_hat.shape[0]):
        mask = true_mask[k, :]
        loadings_hat = np.abs(w_hat[k, :])
        order = np.argsort(loadings_hat)[::-1]
        per_factor_aupr.append(average_precision_score(mask[order], loadings_hat[order]))
    return per_factor_aupr


def plot_precision_recall(true_mask, model, top=None):
    w_hat = get_factor_loadings(model)
    feature_idx, w_hat, true_mask = sort_and_subset(w_hat, true_mask, top)
    return PrecisionRecallDisplay.from_predictions(
        (true_mask).flatten(), np.abs(w_hat).flatten(), plot_chance_level=True
    )
