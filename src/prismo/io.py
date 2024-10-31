import logging
from io import BytesIO
from pathlib import Path

import dill
import h5py
import numpy as np
import pandas as pd
import pyro
import torch

logger = logging.getLogger(__name__)


def save_model(model, path: str | Path, mofa_compat: bool = False):
    dset_kwargs = {"compression": "gzip", "compression_opts": 9}

    path = Path(path)
    if path.exists():
        logger.warning(f"{path} already exists, overwriting")
    with h5py.File(path, "w") as f:
        prismogrp = f.create_group("prismo")

        paramspkl, modelpkl = BytesIO(), BytesIO()
        torch.save(pyro.get_param_store().get_state(), paramspkl, pickle_module=dill)
        torch.save(model, modelpkl, pickle_module=dill)

        # TODO: a lot of things are stored twice: In the pickle and in the MOFA compat layer. Figure out how to reduce this
        prismogrp.create_dataset(
            "param_store", data=np.frombuffer(paramspkl.getbuffer(), dtype=np.uint8), **dset_kwargs
        )
        prismogrp.create_dataset("model", data=np.frombuffer(modelpkl.getbuffer(), dtype=np.uint8), **dset_kwargs)

        if mofa_compat:
            # save MOFA-compatible output
            f.create_dataset("groups/groups", data=model.group_names, **dset_kwargs)
            f.create_dataset("views/views", data=model.view_names, **dset_kwargs)

            samples_grp = f.create_group("samples")
            for group_name, group_samples in model.sample_names.items():
                samples_grp.create_dataset(group_name, data=group_samples, **dset_kwargs)

            features_grp = f.create_group("features")
            for view_name, view_features in model.feature_names.items():
                features_grp.create_dataset(view_name, data=view_features, **dset_kwargs)

            if model.covariates is not None:
                if len(model.covariates_names) == 1:
                    covar_names = next(model.covariates_names.values())
                else:
                    groups = list(model.covariates_names.keys())
                    lengths = [len(g) for g in model.covariates_names.values()]
                    refidx = np.argmax(lengths)
                    ref = set(model.covariates_names[groups[refidx]])
                    if all(set(gc) <= ref for gc in model.covariates_names.values()):
                        covar_names = model.covariates_names[groups[refidx]]
                    else:
                        covar_names = [None] * lengths[refidx]

                covar_names = [n if n is not None else f"covar_{i}" for i, n in enumerate(covar_names)]
                f.create_dataset("covariates/covariates", data=covar_names, **dset_kwargs)

                cov_grp = f.create_group("cov_samples")
                for g_name, covars in model.covariates.items():
                    cov_grp.create_dataset(g_name, data=covars.numpy(), **dset_kwargs)

                warped_covs = model.get_warped_covariates()
                if warped_covs is not None:
                    cov_grp = f.create_group("cov_samples_transformed")
                    for g_name, covars in warped_covs.items():
                        cov_grp.create_dataset(g_name, data=covars, **dset_kwargs)

            samples_meta_grp = f.create_group("samples_metadata")
            for group_name in model.group_names:
                cgrp = samples_meta_grp.create_group(group_name)
                df = pd.concat((v.obs for v in model.data[group_name].values()), axis=1).reset_index()
                for i in range(df.shape[1]):
                    col = df.iloc[:, i]
                    cgrp.create_dataset(col.name, data=col.to_numpy(), **dset_kwargs)

            intercept_grp = f.create_group("intercepts")
            for group_name, gintercepts in model.intercepts.items():
                for view_name, intercept in gintercepts.items():
                    cgrp = intercept_grp.require_group(view_name)
                    cgrp.create_dataset(group_name, data=intercept, **dset_kwargs)

            data_grp = f.create_group("data")
            for group_name, gdata in model.data.items():
                for view_name, data in gdata.items():
                    cgrp = data_grp.require_group(view_name)
                    cgrp.create_dataset(group_name, data=data.X, **dset_kwargs)

            imp_grp = f.create_group("imputed_data")
            imp_data = model.impute_data(missing_only=True)
            for group_name, gimp in imp_data.items():
                for view_name, imp in gimp.items():
                    vgrp = imp_grp.require_group(view_name)
                    ggrp = vgrp.require_group(group_name)
                    ggrp.create_dataset("mean", data=imp.X, **dset_kwargs)
                    # TODO: variance

            exp_grp = f.create_group("expectations")
            factor_grp = exp_grp.create_group("Z")
            for group_name, factors in model.get_factors(return_type="numpy").items():
                factor_grp.create_dataset(group_name, data=factors.T, **dset_kwargs)

            weight_grp = exp_grp.create_group("W")
            for view_name, weights in model.get_weights(return_type="numpy").items():
                weight_grp.create_dataset(view_name, data=weights, **dset_kwargs)

            # save Sigma?

            model_opts_grp = f.create_group("model_options")
            model_opts_grp.create_dataset(
                "likelihoods", data=[model.model_opts.likelihoods[v].lower() for v in model.view_names], **dset_kwargs
            )
            model_opts_grp.create_dataset(
                "spikeslab_factors", data=any(p == "SnS" for p in model.model_opts.factor_prior.values())
            )
            model_opts_grp.create_dataset(
                "spikeslab_weights", data=any(p == "SnS" for p in model.model_opts.weight_prior.values())
            )
            # ARD used unconditionally in SnS prior
            model_opts_grp.create_dataset(
                "ard_factors", data=any(p == "SnS" for p in model.model_opts.factor_prior.values())
            )
            model_opts_grp.create_dataset(
                "ard_weights", data=any(p == "SnS" for p in model.model_opts.weight_prior.values())
            )

            train_opts_grp = f.create_group("training_opts")
            train_opts_grp.create_dataset("maxiter", data=model.train_opts.max_epochs)
            train_opts_grp.create_dataset("freqELBO", data=1)
            train_opts_grp.create_dataset("start_elbo", data=0)
            train_opts_grp.create_dataset("gpu_mode", data=model.train_opts.device.type != "cpu")
            train_opts_grp.create_dataset("stochastic", data=True)

            if model.gp is not None:
                smooth_opts_grp = f.create_group("smooth_opts")
                smooth_opts_grp.create_dataset("scale_cov", data=b"False")
                smooth_opts_grp.create_dataset("start_opt", data=0)
                smooth_opts_grp.create_dataset("opt_freq", data=1)
                smooth_opts_grp.create_dataset("sparseGP", data=b"True")
                smooth_opts_grp.create_dataset("warping_freq", data=model.gp_opts.warp_interval)
                smooth_opts_grp.create_dataset("warping_ref", data=model.gp_opts.warp_reference_group)
                smooth_opts_grp.create_dataset(
                    "warping_open_begin", data=np.asarray(model.gp_opts.warp_open_begin).astype("S")
                )
                smooth_opts_grp.create_dataset(
                    "warping_open_end", data=np.asarray(model.gp_opts.warp_open_end).astype("S")
                )
                smooth_opts_grp.create_dataset("model_groups", data=b"True")

            varexp_grp = f.create_group("variance_explained")
            varexp_factor_grp = varexp_grp.create_group("r2_per_factor")
            for group_name, df in model.get_r2(total=False, ordered=True).items():
                varexp_factor_grp.create_dataset(group_name, data=df.to_numpy().T * 100, **dset_kwargs)

            varexp_total_grp = varexp_grp.create_group("r2_total")
            view_names = np.asarray(model.view_names)
            for group_name, df in model.get_r2(total=True).items():
                varexp_total_grp.create_dataset(
                    group_name, data=df[view_names[np.isin(view_names, df.index)]] * 100, **dset_kwargs
                )

            train_stats_grp = f.create_group("training_stats")
            train_stats_grp.create_dataset("elbo", data=model.get_training_loss(), **dset_kwargs)
            if model.gp is not None:
                train_stats_grp.create_dataset(
                    "length_scales",
                    data=model.gp.lengthscale.cpu().numpy().squeeze()[model.factor_order],
                    **dset_kwargs,
                )
                train_stats_grp.create_dataset(
                    "scales", data=model.gp.outputscale.cpu().numpy().squeeze()[model.factor_order], **dset_kwargs
                )
                train_stats_grp.create_dataset(
                    "Kg", data=model.gp.group_corr.cpu().numpy()[model.factor_order], **dset_kwargs
                )

    logger.info(f"Saved model to {path}")


def load_model(path: str | Path, with_params=True, map_location=None):
    path = Path(path)
    with h5py.File(path, "r") as f:
        prismogrp = f["prismo"]
        paramspkl = BytesIO(prismogrp["param_store"][()].tobytes())
        modelpkl = BytesIO(prismogrp["model"][()].tobytes())

        model = torch.load(modelpkl, map_location=map_location, pickle_module=dill)
        if with_params:
            pyro.get_param_store().set_state(torch.load(paramspkl, map_location=map_location, pickle_module=dill))

    logger.info(f"Loaded model from {path}")

    return model
