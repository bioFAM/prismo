import copy
import logging
import time
from collections import defaultdict
from dataclasses import MISSING, dataclass, field, fields
from functools import reduce
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import scipy.stats as stats
import torch
from dtw import dtw
from mudata import MuData
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import ClippedAdam
from scipy.special import expit
from sklearn.decomposition import NMF, PCA
from tensordict import TensorDict
from torch.utils.data import DataLoader

from prismo import gp, preprocessing
from prismo.io import save_model
from prismo.model import Generative, Variational
from prismo.plotting import plot_overview
from prismo.training import EarlyStopper

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class _Options:
    def __or__(self, other):
        if self.__class__ is not other.__class__:
            raise TypeError("Can only merge objects of the same type")

        kwargs = self.asdict()
        for f in fields(other):
            val = getattr(other, f.name)
            if (
                f.default is not MISSING
                and val != f.default
                or f.default_factory is not MISSING
                and val != f.default_factory()
            ):
                kwargs[f.name] = val
        return self.__class__(**kwargs)

    def __ior__(self, other):
        if self.__class__ is not other.__class__:
            raise TypeError("Can only merge objects of the same type")

        for f in fields(other):
            val = getattr(other, f.name)
            if (
                f.default is not MISSING
                and val != f.default
                or f.default_factory is not MISSING
                and val != f.default_factory()
            ):
                setattr(self, f.name, val)
        return self


@dataclass(kw_only=True)
class DataOptions(_Options):
    """
    Options for the data.

    Args:
        group_by: Columns of `.obs` in MuData and AnnData objects to group data by. Can be any of:
            - String or list of strings. This will be applied to the MuData object or to all AnnData objects
            - Dict of strings or dict of lists of strings. This is only valid if a dict of AnnData objects
              is given as `data`, in which case each AnnData object will be grouped by the `.obs` columns
              in the corresponding `group_by` element.
        scale_per_group: Scale Normal likelihood data per group, otherwise across all groups.
        covariates_obs_key: Key of .obs attribute of each AnnData object that contains covariate values.
        covariates_obsm_key: Key of .obsm attribute of each AnnData object that contains covariate values.
        use_obs: How to align observations across views. One of 'union', 'intersection'.
        use_var: How to align variables across groups. One of 'union', 'intersection'.
        plot_data_overview: Plot data overview.
    """

    group_by: str | list[str] | dict[str] | dict[list[str]] | None = None
    scale_per_group: bool = True
    covariates_obs_key: dict[str, str] | str | None = None
    covariates_obsm_key: dict[str, str] | str | None = None
    use_obs: Literal["union", "intersection"] | None = "union"
    use_var: Literal["union", "intersection"] | None = "union"
    plot_data_overview: bool = True


@dataclass(kw_only=True)
class ModelOptions(_Options):
    """
    Options for the model.

    Args:
        n_factors: Number of latent factors.
        weight_prior: Weight priors for each view (if dict) or for all views (if str).
        factor_prior: Factor priors for each group (if dict) or for all groups (if str).
        likelihoods: Data likelihoods for each view (if dict) or for all views (if str). Inferred automatically if None.
        nonnegative_weights: Non-negativity constraints for weights for each view (if dict) or for all views (if bool).
        nonnegative_factors: Non-negativity constraints for factors for each group (if dict) or for all groups (if bool).
        annotations: Dictionary with weight annotations for informed views.
        prior_penalty: Prior penalty for annotations. #TODO: add more detail
        init_factors: Initialization method for factors.
        init_scale: Initialization scale of Normal distribution for factors.
    """

    n_factors: int = 0
    weight_prior: dict[str, str] | str = "Normal"
    factor_prior: dict[str, str] | str = "Normal"
    likelihoods: dict[str, str] | str | None = None
    nonnegative_weights: dict[str, bool] | bool = False
    nonnegative_factors: dict[str, bool] | bool = False
    annotations: dict[str, pd.DataFrame] | dict[str, np.ndarray] | None = None
    prior_penalty: float = 0.01
    init_factors: Literal["random", "orthogonal", "pca"] = "random"
    init_scale: float = 0.1


@dataclass(kw_only=True)
class TrainingOptions(_Options):
    """
    Options for training.

    Args:
        device: Device to run training on.
        batch_size: Batch size.
        max_epochs: Maximum number of training epochs.
        n_particles: Number of particles for ELBO estimation.
        lr: Learning rate.
        early_stopper_patience: Number of steps without relevant improvement to stop training.
        print_every: Print loss every n steps.
        save: Save model.
        save_path: Path to save model.
        seed : Random seed.
    """

    device: str | torch.device = "cuda"
    batch_size: int = 0
    max_epochs: int = 10_000
    n_particles: int = 1
    lr: float = 0.001
    early_stopper_patience: int = 100
    print_every: int = 100
    save: bool = True
    save_path: str | None = None
    seed: int | None = None


@dataclass(kw_only=True)
class SmoothOptions(_Options):
    """
    Options for Gaussian processes.

    Args:
        n_inducing: Number of inducing points.
        kernel: Kernel function to use.
        warp_groups; List of groups to apply dynamic time warping to.
        warp_interval: Apply dynamic time warping every `warp_interval` epochs.
        warp_open_begin: Perform open-ended alignment.
        warp_open_end: Perform open-ended alignment.
        warp_reference_group: Reference group to align the others to. Defaults to the first group of `warp_groups`.
    """

    n_inducing: int = 100
    kernel: Literal["RBF", "Matern"] = "RBF"
    warp_groups: list[str] = field(default_factory=list)
    warp_interval: int = 20
    warp_open_begin: bool = True
    warp_open_end: bool = True
    warp_reference_group: str | None = None


class PRISMO:
    def __init__(self):
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

        self.group_names = []
        self.view_names = []
        self.sample_names = {}
        self.feature_names = {}
        self.gp_group_names = []
        self._factor_names = []
        self._factor_order = []

        self.nmf = {}
        self.prior_masks = None
        self.prior_scales = None

        # Training settings
        self.init_tensor = None

        # SVI related attributes
        self.generative = None
        self.variational = None
        self.gp = None
        self._orig_covariates = None
        self._gp_warp_groups_order = None
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

    def _to_device(self, data, device):
        tensor_dict = {}
        for k, v in data.items():
            if isinstance(v, dict):
                tensor_dict[k] = self._to_device(v, device)
            else:
                tensor_dict[k] = v.to(device)

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
            likelihoods = preprocessing.infer_likelihoods(data_concatenated)

        elif isinstance(likelihoods, dict):
            logger.info("- Checking compatibility of provided likelihoods with data.")
            preprocessing.validate_likelihoods(data_concatenated, likelihoods)

        elif isinstance(likelihoods, str):
            logger.info("- Using provided likelihood for all views.")
            likelihoods = {k: likelihoods for k in view_names}
            # Still validate likelihoods
            preprocessing.validate_likelihoods(data_concatenated, likelihoods)

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
        self.model_opts.n_factors = n_dense_factors + n_informed_factors

        self._factor_names = pd.Index(factor_names)
        self._factor_order = np.arange(self.model_opts.n_factors)

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
        gp_n_inducing,
        likelihoods,
        nonnegative_factors,
        nonnegative_weights,
        kernel,
        batch_size,
        max_epochs,
        n_particles,
        lr,
    ):
        self.gp_group_names = tuple(g for g in self.group_names if factor_prior[g] == "GP")
        if len(self.gp_group_names):
            if len(self.gp_opts.warp_groups) > 1:
                if not set(self.gp_opts.warp_groups) <= set(self.gp_group_names):
                    raise ValueError(
                        "The set of groups with dynamic time warping must be a subset of groups with a GP factor prior."
                    )
                self._gp_warp_groups_order = {}
                for g in self.gp_opts.warp_groups:
                    ccov = self.covariates[g].squeeze()
                    if ccov.ndim > 1:
                        raise ValueError(
                            f"Warping can only be performed with 1D covariates, but the covariate for group {g} has {ccov.ndim} dimensions."
                        )
                    self._gp_warp_groups_order[g] = ccov.argsort()
                self._orig_covariates = {g: c.clone() for g, c in self.covariates.items()}

                if self.gp_opts.warp_reference_group is None:
                    self.gp_opts.warp_reference_group = self.gp_opts.warp_groups[0]
            elif len(self.gp_opts.warp_groups) == 1:
                logger.warn("Need at least 2 groups for warping, but only one was given. Ignoring warping.")
                self.gp_opts.warp_groups = []

            self.gp = gp.GP(
                self.gp_opts.n_inducing,
                (self.covariates[g] for g in self.gp_group_names),
                self.model_opts.n_factors,
                len(self.gp_group_names),
                kernel,
            ).to(self.train_opts.device)

        self.generative = Generative(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_factors=self.model_opts.n_factors,
            prior_scales=self.prior_scales,
            factor_prior=factor_prior,
            weight_prior=weight_prior,
            likelihoods=likelihoods,
            nonnegative_factors=nonnegative_factors,
            nonnegative_weights=nonnegative_weights,
            gp=self.gp,
            gp_group_names=self.gp_group_names,
            feature_means=self.feature_means,
            sample_means=self.sample_means,
        ).to(self.train_opts.device)

        self.variational = Variational(self.generative, self.init_tensor).to(self.train_opts.device)

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
        weights = self.variational.get_weights()
        factors = self.variational.get_factors()
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
        init_tensor = defaultdict(dict)
        logger.info(f"Initializing factors using `{init_factors}` method...")

        # Initialize factors
        with self.train_opts.device:
            if init_factors == "random":
                for group_name, n in self.n_samples.items():
                    init_tensor[group_name]["loc"] = torch.rand(size=(n, self.model_opts.n_factors))
            elif init_factors == "orthogonal":
                for group_name, n in self.n_samples.items():
                    # Compute PCA of random vectors
                    pca = PCA(n_components=self.model_opts.n_factors, whiten=True)
                    pca.fit(stats.norm.rvs(loc=0, scale=1, size=(n, self.model_opts.n_factors)).T)
                    init_tensor[group_name]["loc"] = torch.tensor(pca.components_.T)
            elif init_factors in ["pca", "nmf"]:
                for group_name in self.n_samples.keys():
                    if init_factors == "pca":
                        pca = PCA(n_components=self.model_opts.n_factors, whiten=True)
                    elif init_factors == "nmf":
                        nmf = NMF(n_components=self.model_opts.n_factors, max_iter=1000)

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
                    if init_factors == "pca":
                        pca.fit(concat_data)
                        init_tensor[group_name]["loc"] = torch.tensor(pca.transform(concat_data))
                    elif init_factors == "nmf":
                        nmf.fit(concat_data)
                        init_tensor[group_name]["loc"] = torch.tensor(nmf.transform(concat_data))

            else:
                raise ValueError(
                    f"Initialization method `{init_factors}` not found. Please choose from `random`, `orthogonal`, `PCA`, or `NMF`."
                )

            for group_name, n in self.n_samples.items():
                # scale factor values from -1 to 1 (per factor)
                q = init_tensor[group_name]["loc"]
                q = 2.0 * (q - torch.min(q, dim=0)[0]) / (torch.max(q, dim=0)[0] - torch.min(q, dim=0)[0]) - 1

                # Add artifical dimension at dimension -2 for broadcasting
                init_tensor[group_name]["loc"] = q.T.unsqueeze(-2).float()
                init_tensor[group_name]["scale"] = (
                    init_scale * torch.ones(size=(n, self.model_opts.n_factors)).T.unsqueeze(-2).float()
                )

        self.init_tensor = init_tensor

    def fit(self, data: MuData | dict[str, ad.AnnData] | dict[str, dict[str, ad.AnnData]], *args: _Options):
        """
        Fit the model using the provided data.

        Parameters
        ----------
        data
            can be any of:
            - MuData object
            - dict with view names as keys and AnnData objects as values
            - dict with view names as keys and torch.Tensor objects as values (single group only)
            - dict with group names as keys and MuData objects as values (incompatible with `group_by`)
            - Nested dict with group names as keys, view names as subkeys and AnnData objects as values (incompatible with `group_by`)
            - Nested dict with group names as keys, view names as subkeys and torch.Tensor objects as values (incompatible with `group_by`)
        *args
            Options for training.
        """
        # convert input data to nested dictionary of AnnData objects (group level -> view level)
        self.data = preprocessing.cast_data(data, group_by=None)
        self.data = preprocessing.anndata_to_dense(self.data)

        # extract group and view names / numbers from data
        self.group_names = list(self.data.keys())
        self.n_groups = len(self.group_names)
        self.view_names = list(self.data[list(self.data.keys())[0]].keys())
        self.n_views = len(self.view_names)

        # process parameters
        self.data_opts = DataOptions()
        self.model_opts = ModelOptions()
        self.train_opts = TrainingOptions()
        self.gp_opts = SmoothOptions()

        for arg in args:
            match arg:
                case DataOptions():
                    self.data_opts |= arg
                case ModelOptions():
                    self.model_opts |= arg
                case TrainingOptions():
                    self.train_opts |= arg
                case SmoothOptions():
                    self.gp_opts |= arg

        # convert input arguments to dictionaries if necessary
        for opt_name, keys in zip(
            ("weight_prior", "factor_prior", "nonnegative_weights", "nonnegative_factors"),
            (self.view_names, self.group_names, self.view_names, self.group_names),
            strict=False,
        ):
            val = getattr(self.model_opts, opt_name)
            if not isinstance(val, dict):
                setattr(self.model_opts, opt_name, {k: val for k in keys})

        for opt_name in ("covariates_obs_key", "covariates_obsm_key"):
            val = getattr(self.data_opts, opt_name)
            if isinstance(val, str):
                setattr(self.data_opts, opt_name, {k: val for k in self.group_names})

        self.train_opts.device = self._setup_device(self.train_opts.device)
        # validate or infer likelihoods
        self.model_opts.likelihoods = self._setup_likelihoods(self.data, self.model_opts.likelihoods)

        # process data
        self.data = preprocessing.remove_constant_features(self.data, self.model_opts.likelihoods)
        self.data = preprocessing.scale_data(self.data, self.model_opts.likelihoods, self.data_opts.scale_per_group)
        self.data = preprocessing.center_data(
            self.data,
            self.model_opts.likelihoods,
            self.model_opts.nonnegative_weights,
            self.model_opts.nonnegative_factors,
        )

        # align observations across views and variables across groups
        if self.data_opts.use_obs is not None:
            self.data = preprocessing.align_obs(self.data, self.data_opts.use_obs)
        if self.data_opts.use_var is not None:
            self.data = preprocessing.align_var(self.data, self.model_opts.likelihoods, self.data_opts.use_var)

        # obtain observations DataFrame and covariates
        self.metadata = preprocessing.extract_obs(self.data)
        self.covariates = preprocessing.extract_covariate(
            self.data, self.data_opts.covariates_obs_key, self.data_opts.covariates_obsm_key
        )

        # extract feature and samples names / numbers from data
        self.feature_names = {k: self.data[list(self.data.keys())[0]][k].var_names.tolist() for k in self.view_names}
        self.n_features = {k: len(v) for k, v in self.feature_names.items()}
        self.sample_names = {k: self.data[k][list(self.data[k].keys())[0]].obs_names.tolist() for k in self.group_names}
        self.n_samples = {k: len(v) for k, v in self.sample_names.items()}

        for view_name in self.view_names:
            if self.model_opts.likelihoods[view_name] == "BetaBinomial":
                feature_names_base = pd.Series(self.feature_names[view_name]).str.rsplit("_", n=1, expand=True)[0]
                if feature_names_base.nunique() == len(feature_names_base) // 2:
                    self.n_features[view_name] = self.n_features[view_name] // 2
                    self.feature_names[view_name] = feature_names_base.unique()

        # compute feature means for intercept terms
        self.feature_means = preprocessing.get_data_mean(self.data, self.model_opts.likelihoods, how="feature")
        self.sample_means = preprocessing.get_data_mean(self.data, self.model_opts.likelihoods, how="sample")

        if self.data_opts.plot_data_overview:
            plot_overview(self.data)

        self._setup_annotations(self.model_opts.n_factors, self.model_opts.annotations, self.model_opts.prior_penalty)
        self._initialize_factors(self.model_opts.init_factors, self.model_opts.init_scale)
        n_samples_total = sum(self.n_samples.values())
        if self.train_opts.batch_size is None or not (0 < self.train_opts.batch_size <= n_samples_total):
            self.train_opts.batch_size = n_samples_total

        self._setup_svi(
            self.model_opts.weight_prior,
            self.model_opts.factor_prior,
            self.gp_opts.n_inducing,
            self.model_opts.likelihoods,
            self.model_opts.nonnegative_factors,
            self.model_opts.nonnegative_weights,
            self.gp_opts.kernel,
            self.train_opts.batch_size,
            self.train_opts.max_epochs,
            self.train_opts.n_particles,
            self.train_opts.lr,
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

        if self.train_opts.batch_size < n_samples_total:
            batch_fraction = self.train_opts.batch_size / n_samples_total

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
                        pin_memory=str(self.train_opts.device) != "cpu",
                        drop_last=False,
                        generator=torch.Generator(device=self.train_opts.device),
                    )
                )

            def step_fn():
                epoch_loss = 0

                for group_batch in zip(*data_loaders, strict=False):
                    epoch_loss += self._svi.step(
                        dict(
                            zip(
                                self.group_names,
                                (batch.to(self.train_opts.device) for batch in group_batch),
                                strict=False,
                            )
                        )
                    )

                return epoch_loss

        else:
            tensor_dict = self._to_device(tensor_dict, self.train_opts.device)

            def step_fn():
                return self._svi.step(tensor_dict)

        if self.train_opts.seed is not None:
            try:
                self.train_opts.seed = int(self.train_opts.seed)
            except ValueError:
                logger.warning(f"Could not convert `{self.train_opts.seed}` to integer.")
                self.train_opts.seed = None

        if self.train_opts.seed is None:
            self.train_opts.seed = int(time.strftime("%y%m%d%H%M"))

        logger.info(f"Setting training seed to `{self.train_opts.seed}`.")
        pyro.set_rng_seed(self.train_opts.seed)
        # clean start
        logger.info("Cleaning parameter store.")
        pyro.enable_validation(True)
        pyro.clear_param_store()

        with self.train_opts.device:
            # Train
            self.train_loss_elbo = []
            earlystopper = EarlyStopper(
                mode="min", min_delta=0.1, patience=self.train_opts.early_stopper_patience, percentage=True
            )
            start_timer = time.time()

            try:
                for i in range(self.train_opts.max_epochs):
                    loss = step_fn()
                    if len(self.gp_opts.warp_groups) and not i % self.gp_opts.warp_interval:
                        self._warp_covariates()
                    self.train_loss_elbo.append(loss)

                    if i % self.train_opts.print_every == 0:
                        logger.info(f"Epoch: {i:>7} | Time: {time.time() - start_timer:>10.2f}s | Loss: {loss:>10.2f}")

                    if earlystopper.step(loss):
                        logger.info(f"Training finished after {i} steps.")
                        break

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt, stopping training and saving progress...")

            self._is_trained = True

            return self._post_fit(self.train_opts.save, self.train_opts.save_path, self.covariates)

    def _warp_covariates(self):
        factormeans = self.variational.get_factors("mean")
        refgroup = self.gp_opts.warp_reference_group
        reffactormeans = factormeans[refgroup].mean(axis=0)
        refidx = self._gp_warp_groups_order[refgroup]
        for g in self.gp_opts.warp_groups[1:]:
            idx = self._gp_warp_groups_order[g]
            alignment = dtw(
                reffactormeans[refidx],
                factormeans[g][:, idx].mean(axis=0),
                open_begin=self.gp_opts.warp_open_begin,
                open_end=self.gp_opts.warp_open_end,
                step_pattern="asymmetric",
            )
            self.covariates[g] = self._orig_covariates[g].clone()
            self.covariates[g][idx[alignment.index2], 0] = self._orig_covariates[refgroup][refidx[alignment.index1], 0]
        self.gp.update_inducing_points(self.covariates.values())

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
        likelihood = self.model_opts.likelihoods[view_name]

        if likelihood == "Normal":
            ss_res = np.nansum(np.square(y_true - y_pred))
            ss_tot = np.nansum(np.square(y_true))  # data is centered
        elif likelihood == "GammaPoisson":
            y_pred = np.logaddexp(0, y_pred)  # softplus
            nu2 = self.variational.get_dispersion("mean")[view_name]
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
            dispersion = self.variational.get_dispersion("mean")[view_name]
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
            logger.info(
                f"R2 for view {view_name} is 0. Increase the number of factors and/or the number of training epochs."
            )
            return [0.0] * factors.shape[0]

        r2s = []
        if self.model_opts.likelihoods[view_name] == "Normal":
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
            factor_order = np.array(list(range(self.model_opts.n_factors)))

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
        self._check_if_trained()
        factors = {
            group_name: pd.DataFrame(
                group_factors[self.factor_order, :].T, index=self.sample_names[group_name], columns=self.factor_names
            )
            for group_name, group_factors in self.variational.get_factors(moment).items()
        }

        factors = self._get_component(factors, return_type)

        if return_type == "anndata":
            for group_name, group_adata in factors.items():
                group_adata.obs = pd.concat(self.metadata[group_name].values(), axis=1)

        return factors

    def get_weights(self, return_type="pandas", moment="mean"):
        """Get all weight matrices, w_x."""
        self._check_if_trained()
        weights = {
            view_name: pd.DataFrame(
                view_weights[self.factor_order, :], index=self.factor_names, columns=self.feature_names[view_name]
            )
            for view_name, view_weights in self.variational.get_weights(moment).items()
        }

        return self._get_component(weights, return_type)

    def get_dispersion(self, return_type="pandas", moment="mean"):
        """Get all dispersion vectors, dispersion_x."""
        self._check_if_trained()
        dispersion = {
            view_name: pd.Series(view_dispersion, index=self.feature_names[view_name])
            for view_name, view_dispersion in self.variational.get_dispersion(moment).items()
        }

        return self._get_component(dispersion, return_type)

    def get_gps(
        self, return_type="pandas", moment="mean", x: dict[str, torch.Tensor] | None = None, n_samples: int = 100
    ):
        """Get all latent functions."""
        if x is None:
            x = self.covariates
        gps = {
            group_name: pd.DataFrame(group_f[self.factor_order, :].T, columns=self.factor_names)
            for group_name, group_f in self.variational.get_gps(x, moment, n_samples).items()
        }

        return self._get_component(gps, return_type)

    def get_annotations(self, return_type="pandas"):
        """Get all annotation matrices, a_x."""
        annotations = {
            k: pd.DataFrame(v[self.factor_order, :], index=self.factor_names, columns=self.feature_names[k])
            for k, v in self.annotations.items()
        }

        return self._get_component(annotations, return_type)

    def _setup_device(self, device):
        logger.info("Setting up device...")

        device = torch.device(device)
        tens = torch.tensor(())
        try:
            tens.to(device)
        except (RuntimeError, AssertionError):
            default_device = tens.device
            logger.warning(f"Device {str(device)} is not available. Using default device: {default_device}")
            device = default_device

        return device

    def impute_missings(self):
        """Impute missing values in the training data using the trained model."""
        self._check_if_trained()

        imputed_data = copy.deepcopy(self.data)

        factors = self.get_factors(return_type="numpy")
        weights = self.get_weights(return_type="numpy")

        for k_groups in self.group_names:
            for k_views in self.view_names:
                if np.isnan(imputed_data[k_groups][k_views].X).any():
                    logger.debug(f"Imputing missing values for {k_groups} - {k_views}.")
                    imputed_data[k_groups][k_views].X = factors[k_groups] @ weights[k_views]

        return imputed_data
