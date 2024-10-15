import time
from collections import defaultdict
from functools import reduce
import logging

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import scipy.stats as stats
import torch
from addict import Dict
from mudata import MuData
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam
from scipy.special import expit
from sklearn.decomposition import PCA
from tensordict import TensorDict
from torch.utils.data import DataLoader

from famo import gp, utils_data
from famo.model import Generative, Variational
from famo.plotting import plot_overview
from famo.utils_io import save_model
from famo.utils_training import EarlyStopper

logger = logging.getLogger(__name__)


class CORE(PyroModule):
    def __init__(self, device):
        super().__init__(name="CORE")
        self.device = self._setup_device(device)
        self.to(self.device)

        self._init()

    def _init(self):
        # data related attributes
        self.data = None
        self.metadata = None
        self.covariates = None
        self.intercepts = None
        self.annotations = None
        self.n_groups = 0
        self.n_views = 0
        self.n_samples = {}
        self.n_features = {}
        self.n_factors = 0

        self.group_names = pd.Index([])
        self.view_names = pd.Index([])
        self.sample_names = {}
        self.feature_names = {}
        self._factor_names = pd.Index([])
        self._factor_order = pd.Index([])

        self.likelihoods = {}
        self.nmf = {}
        self.prior_penalty = None
        self.prior_masks = None
        self.prior_scales = None

        # Training settings
        self.init_tensor = None

        # SVI related attributes
        self.generative = None
        self.variational = None
        self.gps = None
        self._optimizer = None
        self._svi = None

        # training related attributes
        self.train_loss_elbo = []
        self._is_trained = False
        self._cache = None

    @property
    def factor_order(self):
        return self._factor_order

    @factor_order.setter
    def factor_order(self, value):
        self._factor_order = self._factor_order[np.array(value)]

    @property
    def factor_names(self):
        return self._factor_names[np.array(self.factor_order)]

    def _to_device(self, data):
        tensor_dict = {}
        for k, v in data.items():
            if isinstance(v, dict):
                tensor_dict[k] = self._to_device(v)
            else:
                tensor_dict[k] = v.to(self.device)

        return tensor_dict

    def _setup_likelihoods(self, data, likelihoods):
        group_names = tuple(data.keys())
        view_names = tuple(reduce(np.union1d, [list(v.keys()) for v in data.values()]))

        # concatenate data across groups
        data_concatenated = defaultdict(list)
        for k_views in view_names:
            for k_groups in group_names:
                data_concatenated[k_views].append(data[k_groups][k_views])
            data_concatenated[k_views] = ad.concat(data_concatenated[k_views], axis=0)

        if likelihoods is None:
            logger.info("- No likelihoods provided. Inferring likelihoods from data.")
            likelihoods = utils_data.infer_likelihoods(data_concatenated)

        elif isinstance(likelihoods, dict):
            logger.info("- Checking compatibility of provided likelihoods with data.")
            utils_data.validate_likelihoods(data_concatenated, likelihoods)

        elif isinstance(likelihoods, str):
            logger.info("- Using provided likelihood for all views.")
            likelihoods = {k: likelihoods for k in view_names}
            # Still validate likelihoods
            utils_data.validate_likelihoods(data_concatenated, likelihoods)

        elif not (isinstance(likelihoods, dict) | isinstance(likelihoods, str)):
            raise ValueError("likelihoods must be a dictionary or string.")

        for k, v in likelihoods.items():
            logger.info(f"  - {k}: {v}")

        return likelihoods

    def _setup_annotations(self, n_factors, annotations, prior_penalty):
        informed = annotations is not None and len(annotations) > 0
        valid_n_factors = n_factors is not None and n_factors > 0

        if not informed and not valid_n_factors:
            raise ValueError(
                "Invalid latent configuration, "
                "please provide either a collection of prior masks, "
                "or set `n_factors` to a positive integer."
            )

        n_dense_factors = 0
        n_informed_factors = 0

        factor_names = []

        if n_factors is not None:
            n_dense_factors = n_factors
            factor_names += [f"Factor {k + 1}" for k in range(n_dense_factors)]

        prior_masks = None
        prior_scales = None

        if annotations is not None:
            # TODO: annotations need to be processed if not aligned or full
            n_informed_factors = annotations[self.view_names[0]].shape[0]
            if isinstance(annotations[self.view_names[0]], pd.DataFrame):
                for k, vm in annotations.items():
                    annotations[k] = vm.loc[:, self.feature_names[k]].copy()
                factor_names += annotations[self.view_names[0]].index.tolist()
            else:
                factor_names += [
                    f"Factor {k + 1}" for k in range(n_dense_factors, n_dense_factors + n_informed_factors)
                ]

            # keep only numpy arrays
            prior_masks = {
                vn: (vm.to_numpy().astype(bool) if isinstance(vm, pd.DataFrame) else vm.astype(bool))
                for vn, vm in annotations.items()
            }
            # add dense factors if necessary
            if n_dense_factors > 0:
                prior_masks = {
                    vn: np.concatenate([np.ones((n_dense_factors, self.n_features[vn])).astype(bool), vm], axis=0)
                    for vn, vm in annotations.items()
                }

            prior_scales = {
                vn: np.clip(vm.astype(np.float32) + prior_penalty, 1e-8, 1.0) for vn, vm in prior_masks.items()
            }

            if n_dense_factors > 0:
                dense_scale = 1.0
                for vn in self.view_names:
                    prior_scales[vn][:n_dense_factors, :] = dense_scale

        self.n_dense_factors = n_dense_factors
        self.n_informed_factors = n_informed_factors
        self.n_factors = n_dense_factors + n_informed_factors

        self._factor_names = pd.Index(factor_names)
        self._factor_order = np.arange(self.n_factors)

        # storing prior_masks as full annotations instead of partial annotations
        self.annotations = prior_masks
        self.prior_penalty = prior_penalty
        self.prior_masks = prior_masks
        self.prior_scales = prior_scales
        return self.annotations

    def _setup_svi(
        self,
        weight_prior,
        factor_prior,
        likelihoods,
        nonnegative_factors,
        nonnegative_weights,
        inducing_points,
        batch_size,
        max_epochs,
        n_particles,
        lr,
    ):
        self.gps = {}
        for group_name in self.group_names:
            if factor_prior[group_name] == "GP":
                self.gps[group_name] = gp.GP(inducing_points[group_name], self.n_factors).to(self.device)

        self.generative = Generative(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_factors=self.n_factors,
            prior_scales=self.prior_scales,
            factor_prior=factor_prior,
            weight_prior=weight_prior,
            likelihoods=likelihoods,
            nonnegative_factors=nonnegative_factors,
            nonnegative_weights=nonnegative_weights,
            gps=self.gps,
            device=self.device,
            feature_means=self.feature_means,
            sample_means=self.sample_means,
        )

        self.variational = Variational(self.generative, self.init_tensor)

        total_n_samples = sum(self.n_samples.values())

        n_iterations = int(max_epochs * (total_n_samples // batch_size))
        gamma = 0.1
        lrd = gamma ** (1 / n_iterations)
        logger.info(f"Decaying learning rate over {n_iterations} iterations.")
        self._optimizer = ClippedAdam({"lr": lr, "lrd": lrd})

        self._svi = SVI(
            model=pyro.poutine.scale(self.generative, scale=1.0 / total_n_samples),
            guide=pyro.poutine.scale(self.variational, scale=1.0 / total_n_samples),
            optim=self._optimizer,
            loss=TraceMeanField_ELBO(retain_graph=True, num_particles=n_particles, vectorize_particles=True),
        )

        return self._svi

    def _post_fit(self, save, save_path, covariates: dict[str, torch.Tensor] = None):
        # Sort factors by explained variance
        weights = self._get_weights_from_guide()
        factors = self._get_factors_from_guide()
        df_r2, self.factor_order = self._sort_factors(weights=weights, factors=factors)

        # Fill cache
        self._cache = {
            "weights": self.get_weights(return_type="anndata"),
            "factors": self.get_factors(return_type="anndata"),
            "train_loss_elbo": self.train_loss_elbo,
            # "intercepts": intercepts,
            "df_r2": df_r2,
        }
        self._cache["feature_names"] = self.feature_names

        if save:
            if save_path is None:
                save_path = f"model_{time.strftime('%Y%m%d_%H%M%S')}"
            logger.info("Saving results...")
            save_model(self, save_path)

    def _initialize_factors(self, init_factors="random", init_scale=1.0, impute_missings=True):
        init_tensor = Dict()
        logger.info(f"Initializing factors using `{init_factors}` method...")

        # Initialize factors
        if init_factors == "random":
            for group_name, n in self.n_samples.items():
                init_tensor[group_name]["loc"] = torch.rand(size=(n, self.n_factors), device=self.device)
        elif init_factors == "orthogonal":
            for group_name, n in self.n_samples.items():
                # Compute PCA of random vectors
                pca = PCA(n_components=self.n_factors, whiten=True)
                pca.fit(stats.norm.rvs(loc=0, scale=1, size=(n, self.n_factors)).T)
                init_tensor[group_name]["loc"] = torch.tensor(pca.components_.T, device=self.device)
        elif init_factors == "pca":
            for group_name in self.n_samples.keys():
                pca = PCA(n_components=self.n_factors, whiten=True)
                # Combine all views
                concat_data = torch.cat(
                    [
                        torch.tensor(self.data[group_name][view_name].X, dtype=torch.float)
                        for view_name in self.view_names
                    ],
                    dim=-1,
                )
                # Check if data has missings. If yes, and impute_missings is True, then impute, else raise an error
                if torch.isnan(concat_data).any():
                    if impute_missings:
                        from sklearn.impute import SimpleImputer

                        imp = SimpleImputer(missing_values=np.NaN, strategy="mean")
                        imp.fit(concat_data)
                        concat_data = torch.tensor(imp.transform(concat_data), dtype=torch.float)
                    else:
                        raise ValueError(
                            "Data has missing values. Please impute missings or set `impute_missings=True`."
                        )
                pca.fit(concat_data)
                init_tensor[group_name]["loc"] = torch.tensor(pca.transform(concat_data), device=self.device)

        else:
            raise ValueError(
                f"Initialization method `{init_factors}` not found. Please choose from `random`, `orthogonal`, or `PCA`."
            )

        for group_name, n in self.n_samples.items():
            # scale factor values from -1 to 1 (per factor)
            q = init_tensor[group_name]["loc"]
            q = 2.0 * (q - torch.min(q, dim=0)[0]) / (torch.max(q, dim=0)[0] - torch.min(q, dim=0)[0]) - 1

            # Add artifical dimension at dimension -2 for broadcasting
            init_tensor[group_name]["loc"] = q.T.unsqueeze(-2).float()
            init_tensor[group_name]["scale"] = (
                init_scale * torch.ones(size=(n, self.n_factors), device=self.device).T.unsqueeze(-2).float()
            )

        self.init_tensor = init_tensor

    def fit(
        self,
        data: MuData | dict[str, ad.AnnData] | dict[str, dict[str, ad.AnnData]],
        group_by: str | list[str] | dict[str] | dict[list[str]] | None = None,
        n_factors: int = None,
        annotations=None,
        weight_prior: dict[str, str] | str = None,
        factor_prior: dict[str, str] | str = None,
        likelihoods: dict[str, str] | str = None,
        covariates_obs_key: dict[str, str] | str = None,
        covariates_obsm_key: dict[str, str] | str = None,
        nonnegative_weights: dict[str, bool] | bool = False,
        nonnegative_factors: dict[str, bool] | bool = False,
        prior_penalty: float = 0.01,
        batch_size: int = 0,
        max_epochs: int = 10000,
        n_particles: int = 1,
        lr: float = 0.001,
        early_stopper_patience: int = 100,
        print_every: int = 100,
        plot_data_overview: bool = True,
        scale_per_group: bool = True,
        use_obs: str = "union",
        use_var: str = "union",
        save: bool = True,
        save_path: str = None,
        init_factors: str = "random",
        init_scale: float = 0.1,
        gp_n_inducing: dict[str, int] | int = 100,
        seed: int = None,
        **kwargs,
    ):
        """
        Fit the model using the provided data.

        Parameters
        ----------
        data : MuData | dict[str, ad.AnnData] | dict[str, MuData] | dict[str, dict[str, ad.AnnData]]
            can be any of:
            - MuData object
            - dict with view names as keys and AnnData objects as values
            - dict with view names as keys and torch.Tensor objects as values (single group only)
            - dict with group names as keys and MuData objects as values (incompatible with `group_by`)
            - Nested dict with group names as keys, view names as subkeys and AnnData objects as values (incompatible with `group_by`)
            - Nested dict with group names as keys, view names as subkeys and torch.Tensor objects as values (incompatible with `group_by`)
        group_by: Columns of `.obs` in MuData and AnnData objects to group data by. Can be any of:
            - String or list of strings. This will be applied to the MuData object or to all AnnData objects
            - Dict of strings or dict of lists of strings. This is only valid if a dict of AnnData objects
              is given as `data`, in which case each AnnData object will be grouped by the `.obs` columns
              in the corresponding `group_by` element.
        n_factors : int
            Number of latent factors.
        annotations : dict
            Dictionary with weight annotations for informed views.
        weight_prior : dict | str
            Weight priors for each view (if dict) or for all views (if str). Normal if None.
        factor_prior : dict | str
            Factor priors for each group (if dict) or for all groups (if str). Normal if None.
        likelihoods : dict | str
            Data likelihoods for each view (if dict) or for all views (if str). Inferred automatically if None.
        covariates_obs_key : dict | str
            Key of .obs attribute of each AnnData object that contains covariate values.
        covariates_obsm_key : dict | str
            Key of .obsm attribute of each AnnData object that contains covariate values.
        nonnegative_weights : dict | bool
            Non-negativity constraints for weights for each view (if dict) or for all views (if bool).
        nonnegative_factors : dict | bool
            Non-negativity constraints for factors for each group (if dict) or for all groups (if bool).
        prior_penalty : float
            Prior penalty for annotations. #TODO: add more detail
        batch_size : int
            Batch size.
        max_epochs : int
            Maximum number of training epochs.
        n_particles : int
            Number of particles for ELBO estimation.
        lr : float
            Learning rate.
        early_stopper_patience : int
            Number of steps without relevant improvement to stop training.
        print_every : int
            print loss every n steps.
        plot_data_overview: bool
            Plot data overview.
        scale_per_group : bool
            Scale Normal likelihood data per group, otherwise across all groups.
        use_obs : str
            How to align observations across views. One of 'union', 'intersection'.
        use_var : str
            How to align variables across groups. One of 'union', 'intersection'.
        save : bool
            Save model.
        save_path : str
            Path to save model.
        init_factors : str
            Initialization method for factors.
        init_scale: float
            Initialization scale of Normal distribution for factors.
        gp_n_inducing : dict | int
            Number of inducing points for each group (if dict) or for all groups (if int).
        seed : int
            Random seed.
        **kwargs
            Additional training arguments.
        """
        # convert input data to nested dictionary of AnnData objects (group level -> view level)
        self.data = utils_data.cast_data(data, group_by=None)
        self.data = utils_data.anndata_to_dense(self.data)

        # extract group and view names / numbers from data
        self.group_names = list(self.data.keys())
        self.n_groups = len(self.group_names)
        self.view_names = list(self.data[list(self.data.keys())[0]].keys())
        self.n_views = len(self.view_names)

        # convert input arguments to dictionaries if necessary
        weight_prior = {k: weight_prior for k in self.view_names} if isinstance(weight_prior, str) else weight_prior
        factor_prior = {k: factor_prior for k in self.group_names} if isinstance(factor_prior, str) else factor_prior
        covariates_obs_key = (
            {k: covariates_obs_key for k in self.group_names}
            if isinstance(covariates_obs_key, str)
            else covariates_obs_key
        )
        covariates_obsm_key = (
            {k: covariates_obsm_key for k in self.group_names}
            if isinstance(covariates_obsm_key, str)
            else covariates_obsm_key
        )
        gp_n_inducing = (
            {k: gp_n_inducing for k in self.group_names} if isinstance(gp_n_inducing, int) else gp_n_inducing
        )
        self.nonnegative_weights = (
            {k: nonnegative_weights for k in self.view_names}
            if isinstance(nonnegative_weights, bool)
            else nonnegative_weights
        )
        self.nonnegative_factors = (
            {k: nonnegative_factors for k in self.group_names}
            if isinstance(nonnegative_factors, bool)
            else nonnegative_factors
        )

        # validate or infer likelihoods
        self.likelihoods = self._setup_likelihoods(self.data, likelihoods)

        # process data
        self.data = utils_data.remove_constant_features(self.data, self.likelihoods)
        self.data = utils_data.scale_data(self.data, self.likelihoods, scale_per_group)
        self.data = utils_data.center_data(
            self.data, self.likelihoods, self.nonnegative_weights, self.nonnegative_factors
        )

        # align observations across views and variables across groups
        if use_obs is not None:
            self.data = utils_data.align_obs(self.data, use_obs)
        if use_var is not None:
            self.data = utils_data.align_var(self.data, self.likelihoods, use_var)

        # obtain observations DataFrame and covariates
        self.metadata = utils_data.extract_obs(self.data)
        self.covariates = utils_data.extract_covariate(self.data, covariates_obs_key, covariates_obsm_key)

        # extract feature and samples names / numbers from data
        self.feature_names = {k: self.data[list(self.data.keys())[0]][k].var_names.tolist() for k in self.view_names}
        self.n_features = {k: len(v) for k, v in self.feature_names.items()}
        self.sample_names = {k: self.data[k][list(self.data[k].keys())[0]].obs_names.tolist() for k in self.group_names}
        self.n_samples = {k: len(v) for k, v in self.sample_names.items()}

        for view_name in self.view_names:
            if self.likelihoods[view_name] == "BetaBinomial":
                feature_names_base = pd.Series(self.feature_names[view_name]).str.rsplit("_", n=1, expand=True)[0]
                if feature_names_base.nunique() == len(feature_names_base) // 2:
                    self.n_features[view_name] = self.n_features[view_name] // 2
                    self.feature_names[view_name] = feature_names_base.unique()

        # compute feature means for intercept terms
        self.feature_means = utils_data.get_data_mean(self.data, self.likelihoods, how="feature")
        self.sample_means = utils_data.get_data_mean(self.data, self.likelihoods, how="sample")

        if plot_data_overview:
            plot_overview(self.data)

        self._setup_annotations(n_factors, annotations, prior_penalty)
        # GP inducing point locations
        inducing_points = gp.setup_inducing_points(
            factor_prior, self.covariates, gp_n_inducing, self.n_factors, device=self.device
        )
        self._initialize_factors(init_factors, init_scale)
        n_samples_total = sum(self.n_samples.values())
        if batch_size is None or not (0 < batch_size <= n_samples_total):
            batch_size = n_samples_total

        self._setup_svi(
            weight_prior,
            factor_prior,
            self.likelihoods,
            self.nonnegative_factors,
            self.nonnegative_weights,
            inducing_points,
            batch_size,
            max_epochs,
            n_particles,
            lr,
        )

        # convert AnnData to torch.Tensor objects
        tensor_dict = {}
        for group_name, group_dict in self.data.items():
            tensor_dict[group_name] = {}
            if self.covariates is not None and group_name in self.covariates:
                if self.covariates[group_name] is not None:
                    tensor_dict[group_name]["covariates"] = self.covariates[group_name]

            for view_name, view_adata in group_dict.items():
                tensor_dict[group_name][view_name] = torch.from_numpy(view_adata.X)

        if batch_size < n_samples_total:
            batch_fraction = batch_size / n_samples_total

            # has to be a list of data loaders to zip over
            data_loaders = []

            for group_name in self.group_names:
                tensor_dict[group_name] = TensorDict(
                    {key: tensor_dict[group_name][key] for key in tensor_dict[group_name].keys()},
                    batch_size=[self.n_samples[group_name]],
                )
                tensor_dict[group_name]["sample_idx"] = torch.arange(self.n_samples[group_name])

                data_loaders.append(
                    DataLoader(
                        tensor_dict[group_name],
                        batch_size=max(1, int(batch_fraction * self.n_samples[group_name])),
                        shuffle=True,
                        num_workers=0,
                        collate_fn=lambda x: x,
                        pin_memory=str(self.device) != "cpu",
                        drop_last=False,
                    )
                )

            def step_fn():
                epoch_loss = 0

                for group_batch in zip(*data_loaders, strict=False):
                    epoch_loss += self._svi.step(
                        dict(zip(self.group_names, (batch.to(self.device) for batch in group_batch), strict=False))
                    )

                return epoch_loss

        else:
            # move all data to device once
            tensor_dict = self._to_device(tensor_dict)

            def step_fn():
                return self._svi.step(tensor_dict)

        if seed is not None:
            try:
                seed = int(seed)
            except ValueError:
                logger.info(f"Could not convert `{seed}` to integer.")
                seed = None

        if seed is None:
            seed = int(time.strftime("%y%m%d%H%M"))

        self.seed = seed

        logger.info(f"Setting training seed to `{seed}`.")
        pyro.set_rng_seed(seed)
        # clean start
        logger.info("Cleaning parameter store.")
        pyro.enable_validation(True)
        pyro.clear_param_store()

        # Train
        self.train_loss_elbo = []
        earlystopper = EarlyStopper(mode="min", min_delta=0.1, patience=early_stopper_patience, percentage=True)
        start_timer = time.time()

        try:
            for i in range(max_epochs):
                loss = step_fn()
                self.train_loss_elbo.append(loss)

                if i % print_every == 0:
                    logger.info(f"Epoch: {i:>7} | Time: {time.time() - start_timer:>10.2f}s | Loss: {loss:>10.2f}")

                if earlystopper.step(loss):
                    logger.info(f"Training finished after {i} steps.")
                    break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, stopping training and saving progress...")

        self._is_trained = True

        return self._post_fit(save, save_path, self.covariates)

    @staticmethod
    def _Vprime(mu, nu2, nu1):
        return 2 * nu2 * mu + nu1

    @staticmethod
    def _dV_square(a, b, nu2, nu1):
        dVb = __class__._Vprime(b, nu2, nu1)
        dVa = __class__._Vprime(a, nu2, nu1)
        sVb = np.sqrt(1 + dVb**2)
        sVa = np.sqrt(1 + dVa**2)
        return 1 / (16 * nu2**2) * (np.log((dVb + sVb) / (dVa + sVa)) + dVb * sVb - dVa * sVa) ** 2

    def _r2_impl(self, y_true, factor, weights, view_name):
        # this is based on Zhang: A Coefficient of Determination for Generalized Linear Models (2017)
        y_pred = factor.T @ weights
        likelihood = self.likelihoods[view_name]

        if likelihood == "Normal":
            ss_res = np.nansum(np.square(y_true - y_pred))
            ss_tot = np.nansum(np.square(y_true))  # data is centered
        elif likelihood == "GammaPoisson":
            y_pred = np.logaddexp(0, y_pred)  # softplus
            nu2 = self._get_dispersion_from_guide("mean")[view_name]
            ss_res = np.nansum(self._dV_square(y_true, y_pred, nu2, 1))

            truemean = np.nanmean(y_true)
            nu2 = (np.nanvar(y_true) - truemean) / truemean**2  # method of moments estimator
            ss_tot = np.nansum(self._dV_square(y_true, truemean, nu2, 1))
        elif likelihood == "Bernoulli":
            y_pred = expit(y_pred)
            ss_res = np.nansum(self._dV_square(y_true, y_pred, -1, 1))
            ss_tot = np.nansum(self._dV_square(y_true, np.nanmean(y_true), -1, 1))
        elif likelihood == "BetaBinomial":
            y_pred = expit(y_pred)
            obs_total = y_true[..., 1, :, :]
            y_true = y_true[..., 0, :, :]
            dispersion = self._get_dispersion_from_guide("mean")[view_name]
            nu2 = nu1 = obs_total * (1 + obs_total * dispersion) / (1 + dispersion)
            ss_res = np.nansum(self._dV_square(y_true, y_pred, nu2, nu1))

            pi = np.nansum(y_true) / np.nansum(obs_total)
            truemean = obs_total * pi
            truevar = obs_total * np.nanvar(y_true / obs_total)
            dispersion = (obs_total * pi * (1 - pi) - truevar) / (
                obs_total * (truevar - pi * (1 - pi))
            )  # method of moments estimator
            nu2 = nu1 = obs_total * (1 + obs_total * dispersion) / (1 + dispersion)
            ss_res = np.nansum(self._dV_square(y_true, truemean, nu2, nu1))
        else:
            raise NotImplementedError(likelihood)

        return max(0.0, 1.0 - ss_res / ss_tot)

    def _r2(self, y_true, factors, weights, view_name):
        r2_full = self._r2_impl(y_true, factors, weights, view_name)
        if r2_full < 1e-8:  # TODO: have some global definition/setting of EPS
            logger.info(f"R2 for view {view_name} is 0. Increase the number of factors and/or the number of training epochs.")
            return [0.0] * factors.shape[0]

        r2s = []
        if self.likelihoods[view_name] == "Normal":
            for k in range(factors.shape[0]):
                r2s.append(self._r2_impl(y_true, factors[None, k, :], weights[None, k, :], view_name))
        else:
            # For models with a link function that is not the identity, such as Bernoulli, calculating R2 of single
            # factors leads to erroneous results, in the case of Bernoulli it can lead to every factor having negative
            # R2 values. This is because an unimportant factor will not contribute much to the full model, but the zero
            # prediction of this single factor will be mapped by the link function to a non-zero value, which can result
            # in a worse prediction than the intercept-only null model. As a workaround, we therefore calculate R2 of
            # a model with all factors except for one and subtract it from the R2 value of the full model to arrive at
            # the R2 of the current factor.
            for k in range(factors.shape[0]):
                cfactors = np.delete(factors, k, 0)
                cweights = np.delete(weights, k, 0)
                cr2 = self._r2_impl(y_true, cfactors, cweights, view_name)
                r2s.append(max(0.0, r2_full - cr2))
        return r2s

    def _sort_factors(self, weights, factors, subsample=1000):
        # Loop over all groups
        dfs = {}

        for group_name, group_data in self.data.items():
            n_samples = self.n_samples[group_name]

            sample_idx = np.arange(n_samples)

            if subsample is not None and subsample > 0 and subsample < n_samples:
                sample_idx = np.random.choice(sample_idx, subsample, replace=False)

            group_r2 = {}
            for view_name, view_data in group_data.items():
                try:
                    group_r2[view_name] = self._r2(
                        view_data.X[sample_idx, :], factors[group_name][:, sample_idx], weights[view_name], view_name
                    )
                except NotImplementedError:
                    "R2 not yet implemented."

            dfs[group_name] = pd.DataFrame(group_r2)

        # sum the R2 values across all groups
        df_concat = pd.concat(dfs.values())
        df_sum = df_concat.groupby(df_concat.index).sum()

        try:
            # sort factors according to mean R2 across all views
            sorted_r2_means = df_sum.mean(axis=1).sort_values(ascending=False)
            factor_order = np.array(sorted_r2_means.index)
        except NameError:
            logger.info("Sorting factors failed. Using default order.")
            factor_order = np.array(list(range(self.n_factors)))

        for group_name in self.data.keys():
            dfs[group_name] = dfs[group_name].loc[factor_order].reset_index(drop=True)

        return dfs, factor_order

    def _check_if_trained(self):
        """Check if the model has been trained."""
        if not self._is_trained:
            raise ValueError("Model has not been trained yet. Please train first.")

    def _get_component(self, component, return_type="pandas"):
        if return_type == "numpy":
            component = {k: v.values for k, v in component.items()}
        if return_type == "torch":
            component = {k: torch.tensor(v.values, dtype=torch.float).clone().detach() for k, v in component.items()}
        if return_type == "anndata":
            component = {k: ad.AnnData(v) for k, v in component.items()}

        return component

    def get_factors(self, return_type="pandas", moment="mean"):
        """Get all factor matrices, z_x."""
        factors = {
            group_name: pd.DataFrame(
                group_factors[self.factor_order, :].T, index=self.sample_names[group_name], columns=self.factor_names
            )
            for group_name, group_factors in self._get_factors_from_guide(moment).items()
        }

        factors = self._get_component(factors, return_type)

        if return_type == "anndata":
            for group_name, group_adata in factors.items():
                group_adata.obs = pd.concat(self.metadata[group_name].values(), axis=1)

        return factors

    def get_weights(self, return_type="pandas", moment="mean"):
        """Get all weight matrices, w_x."""
        weights = {
            view_name: pd.DataFrame(
                view_weights[self.factor_order, :], index=self.factor_names, columns=self.feature_names[view_name]
            )
            for view_name, view_weights in self._get_weights_from_guide(moment).items()
        }

        return self._get_component(weights, return_type)

    def get_dispersion(self, return_type="pandas", moment="mean"):
        """Get all dispersion vectors, dispersion_x."""
        dispersion = {
            view_name: pd.Series(view_dispersion, index=self.feature_names[view_name])
            for view_name, view_dispersion in self._get_dispersion_from_guide(moment).items()
        }

        return self._get_component(dispersion, return_type)

    def get_gps(self, return_type="pandas", moment="mean", x: dict[str, torch.Tensor] = None, n_samples: int = 100):
        """Get all latent functions."""
        gps = {
            group_name: pd.DataFrame(group_f[self.factor_order, :].T, columns=self.factor_names)
            for group_name, group_f in self._get_gps_from_guide(moment, x, n_samples).items()
        }

        return self._get_component(gps, return_type)

    def get_annotations(self, return_type="pandas"):
        """Get all annotation matrices, a_x."""
        annotations = {
            k: pd.DataFrame(v[self.factor_order, :], index=self.factor_names, columns=self.feature_names[k])
            for k, v in self.annotations.items()
        }

        return self._get_component(annotations, return_type)

    @torch.no_grad()
    def _get_factors_from_guide(self, moment: str = "mean"):
        """Get all factor matrices, z_x."""
        self._check_if_trained()
        if moment not in ["mean", "std"]:
            raise ValueError("Invalid argument for `moment`. Must be one of ['mean', 'std'].")

        factors = {}
        for group_name in self.group_names:
            if moment == "mean":
                factors[group_name] = self.variational._get_loc_and_scale(f"z_{group_name}")[0].clone()
            else:
                factors[group_name] = self.variational._get_loc_and_scale(f"z_{group_name}")[1].clone()

            if self.generative.factor_prior[group_name] == "SnS":
                factors[group_name] *= self.variational._get_prob(f"s_z_{group_name}").clone()

            if self.generative.nonnegative_factors[group_name] and moment == "mean":
                factors[group_name] = self.generative.pos_transform(factors[group_name])

        return {group_name: group_factors.cpu().numpy().squeeze() for group_name, group_factors in factors.items()}

    @torch.no_grad()
    def _get_weights_from_guide(self, moment: str = "mean"):
        """Get all weight matrices, w_x."""
        self._check_if_trained()
        if moment not in ["mean", "std"]:
            raise ValueError("Invalid argument for `moment`. Must be one of ['mean', 'std'].")

        weights = {}
        for view_name in self.view_names:
            if moment == "mean":
                weights[view_name] = self.variational._get_loc_and_scale(f"w_{view_name}")[0].clone()
            else:
                weights[view_name] = self.variational._get_loc_and_scale(f"w_{view_name}")[1].clone()

            if self.generative.weight_prior == "SnS":
                weights[view_name] *= self.variational._get_prob(f"s_w_{view_name}").clone()

            if self.generative.nonnegative_weights[view_name] and moment == "mean":
                weights[view_name] = self.generative.pos_transform(weights[view_name])

        return {view_name: view_weights.cpu().numpy().squeeze() for view_name, view_weights in weights.items()}

    @torch.no_grad()
    def _get_dispersion_from_guide(self, moment: str = "mean"):
        """Get all dispersion vectors, dispersion_x."""
        self._check_if_trained()
        if moment not in ["mean", "std"]:
            raise ValueError("Invalid argument for `moment`. Must be one of ['mean', 'std'].")

        dispersion = {}
        for view_name in self.view_names:
            # TODO: use actual mean and std of LogNormal
            if moment == "mean":
                dispersion[view_name] = self.variational._get_loc_and_scale(f"dispersion_{view_name}")[0].clone()
            else:
                dispersion[view_name] = self.variational._get_loc_and_scale(f"dispersion_{view_name}")[1].clone()

        return {view_name: view_dispersion.cpu().numpy().squeeze() for view_name, view_dispersion in dispersion.items()}

    @torch.no_grad()
    def _get_gps_from_guide(self, moment: str = "mean", x: dict[str, torch.Tensor] = None, n_samples: int = 100):
        """Get all latent functions."""
        self._check_if_trained()
        if moment not in ["mean", "std"]:
            raise ValueError("Invalid argument for `moment`. Must be one of ['mean', 'std'].")

        if x is not None:
            group_names = x.keys()

        else:
            x = {group_name: self.covariates[group_name] for group_name in self.group_names}
            group_names = self.group_names

        f = {}
        for group_name in group_names:
            gp = self.gps[group_name]
            gp_dist = gp(x[group_name].to(self.device), prior=False)
            gp_samples = gp_dist.sample(torch.Size([n_samples]))
            if moment == "mean":
                f[group_name] = gp_samples.mean(axis=0).clone()
            else:
                f[group_name] = gp_samples.std(axis=0).clone()

        return {group_name: group_f.cpu().numpy().squeeze() for group_name, group_f in f.items()}

    def _setup_device(self, device):
        logger.info("Setting up device...")
        cuda_available = torch.cuda.is_available()

        device = str(device).lower()
        if "cuda" in device and not cuda_available:
            logger.info(f"- `{device}` not available...")
            device = "cpu"

        if "cuda" in device and cuda_available:
            # Check if device id is given, otherwise use default device
            if ":" in device:
                # Check if device_id is valid
                device_id = int(device.split(":")[1])
                if device_id >= torch.cuda.device_count():
                    logger.info(
                        f"- Device id `{device_id}` not available. Using default device: {torch.cuda.current_device()}"
                    )
                    device = f"cuda:{torch.cuda.current_device()}"
            else:
                logger.info(f"- No device id given. Using default device: {torch.cuda.current_device()}")
                device = f"cuda:{torch.cuda.current_device()}"

            # Set all cuda operations to specific cuda device
            torch.cuda.set_device(device)

        logger.info(f"- Running all computations on `{device}`")
        return torch.device(device)
