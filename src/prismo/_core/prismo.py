import logging
import random
import time
from collections import defaultdict
from dataclasses import MISSING, asdict, dataclass, field, fields
from functools import reduce
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import numpy.typing as npt
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

from ..pl import plot_overview
from . import gp, preprocessing
from .io import load_model, save_model
from .model import Generative, Variational
from .training import EarlyStopper
from .utils import MeanStd

_logger = logging.getLogger(__name__)

_ResultsTypeDF = dict[str, pd.DataFrame | ad.AnnData | npt.NDArray[np.float32]]
_ResultsTypeSeries = dict[str, pd.Series | ad.AnnData | npt.NDArray[np.float32]]


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

    def __post_init__(self):
        # after an HDF5 roundtrip, these are numpy scalars, which PyTorch doesn't handle well'
        for f in fields(self):
            if f.type in (float, int, bool):
                setattr(self, f.name, f.type(getattr(self, f.name)))


@dataclass(kw_only=True)
class DataOptions(_Options):
    """Options for the data.

    Args:
        group_by: Columns of `.obs` in MuData and AnnData objects to group data by. Can be any of:

            - String or list of strings. This will be applied to the MuData object or to all AnnData objects
            - Dict of strings or dict of lists of strings. This is only valid if a dict of AnnData objects
              is given as `data`, in which case each AnnData object will be grouped by the `.obs` columns
              in the corresponding `group_by` element.

        scale_per_group: Scale Normal likelihood data per group, otherwise across all groups.
        covariates_obs_key: Key of .obs attribute of each AnnData object that contains covariate values.
        covariates_obsm_key: Key of .obsm attribute of each AnnData object that contains covariate values.
        use_obs: How to align observations across views.
        use_var: How to align variables across groups.
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
    """Options for the model.

    Args:
        n_factors: Number of latent factors.
        weight_prior: Weight priors for each view (if dict) or for all views (if str).
        factor_prior: Factor priors for each group (if dict) or for all groups (if str).
        likelihoods: Data likelihoods for each view (if dict) or for all views (if str). Inferred automatically if None.
        nonnegative_weights: Non-negativity constraints for weights for each view (if dict) or for all views (if bool).
        nonnegative_factors: Non-negativity constraints for factors for each group (if dict) or for all groups (if bool).
        annotations: Dictionary with weight annotations for informed views. Must have shape (n_factors, n_features).
        annotations_varm_key: Key of .varm attribute of each AnnData object that contains annotation values.
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
    annotations_varm_key: dict[str, str] | str | None = None
    prior_penalty: float = 0.01
    init_factors: float | Literal["random", "orthogonal", "pca"] = "random"
    init_scale: float = 0.1


@dataclass(kw_only=True)
class TrainingOptions(_Options):
    """Options for training.

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
        mofa_compat: Save model in MOFA2 compatible format.
        seed : Random seed.
    """

    device: str | torch.device = "cuda"
    batch_size: int = 0
    max_epochs: int = 10_000
    n_particles: int = 1
    lr: float = 0.001
    early_stopper_patience: int = 100
    print_every: int = 100
    save_path: str | None = None
    mofa_compat: bool = False
    seed: int | None = None

    def __post_init__(self):
        super().__post_init__()
        self.device = torch.device(self.device)


@dataclass(kw_only=True)
class SmoothOptions(_Options):
    """Options for Gaussian processes.

    Args:
        n_inducing: Number of inducing points.
        kernel: Kernel function to use.
        warp_groups: List of groups to apply dynamic time warping to.
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

    def __post_init__(self):
        super().__post_init__()
        self.warp_groups = list(self.warp_groups)  # in case the user passed a tuple here, we need a list for saving


def _to_device(data, device):
    tensor_dict = {}
    for k, v in data.items():
        if isinstance(v, dict):
            tensor_dict[k] = _to_device(v, device)
        else:
            tensor_dict[k] = v.to(device)

    return tensor_dict


class PRISMO:
    """Fit the model using the provided data.

    Args:
        data: can be any of:

            - MuData object
            - Nested dict with group names as keys, view names as subkeys and AnnData objects as values
              (incompatible with :class:`TrainingOptions` `.group_by`)

        *args: Options for training.
    """

    def __init__(self, data: MuData | dict[str, dict[str, ad.AnnData]], *args: _Options):
        self._process_options(*args)
        data = preprocessing.cast_data(data, group_by=self._data_opts.group_by)

        self._view_names = list(data[next(iter(data.keys()))].keys())
        self._group_names = list(data.keys())
        self._sample_names = {k: next(iter(adatas.values())).obs_names.tolist() for k, adatas in data.items()}

        self._adjust_options(data)
        data, feature_means, sample_means = self._preprocess_data(data)

        self._metadata = preprocessing.extract_obs(data)
        self._feature_names = {
            k: next(iter(data.values()))[k].var_names.tolist() for k in self.view_names
        }  # this must be after _preprocess_data

        for view_name in self.view_names:
            if self._model_opts.likelihoods[view_name] == "BetaBinomial":
                feature_names_base = pd.Series(self._feature_names[view_name]).str.rsplit("_", n=1, expand=True)[0]
                unique_feature_names = feature_names_base.unique()
                if len(unique_feature_names) == len(feature_names_base) // 2:
                    self._feature_names[view_name] = unique_feature_names

        self._setup_annotations(data)

        if self._data_opts.plot_data_overview:
            plot_overview(data).show()

        self._fit(data, feature_means, sample_means)

    @property
    def group_names(self) -> list[str]:
        """Group names."""
        return self._group_names

    @property
    def n_groups(self) -> int:
        """Number of groups."""
        return len(self.group_names)

    @property
    def view_names(self) -> list[str]:
        """View names."""
        return self._view_names

    @property
    def n_views(self) -> int:
        """Number of views."""
        return len(self.view_names)

    @property
    def feature_names(self) -> dict[str, list[str]]:
        """Feature names for each view."""
        return self._feature_names

    @property
    def n_features(self) -> dict[str, int]:
        """Number of features in each view."""
        return {k: len(v) for k, v in self.feature_names.items()}

    @property
    def n_features_total(self) -> int:
        """Total number of features."""
        return sum(self.n_features.values())

    @property
    def sample_names(self) -> dict[str, list[str]]:
        """Sample names for each group."""
        return self._sample_names

    @property
    def n_samples(self) -> dict[str, int]:
        """Number of samples in each group."""
        return {k: len(v) for k, v in self.sample_names.items()}

    @property
    def n_samples_total(self) -> int:
        """Total number of samples."""
        return sum(self.n_samples.values())

    @property
    def n_factors(self):
        """Total number of factors."""
        return self._model_opts.n_factors

    @property
    def n_dense_factors(self) -> int:
        """Number of dense (uninformed) factors."""
        return self._n_dense_factors

    @property
    def n_informed_factors(self) -> int:
        """Number of informed factors."""
        return self._n_informed_factors

    @property
    def factor_order(self) -> npt.NDArray[int]:
        """Ordering of factors by explained variance (highest to lowest)."""
        return self._factor_order

    @factor_order.setter
    def factor_order(self, value: npt.ArrayLike):
        order = np.asarray(value, dtype=int)
        if order.ndim != 1:
            raise ValueError(f"The ordering must have 1 dimension, but got {order.ndim}.")
        if order.size != self.n_factors:
            raise ValueError(f"The ordering must have {self.n_factors} items, but got {order.size}.")
        if order.min() != 0 or order.max() != self.n_factors - 1 or np.unique(order).size != order.size:
            raise ValueError(f"The ordering must contain all integers in [0, {self.n_factors}).")
        self._factor_order = order

    @property
    def factor_names(self) -> npt.NDArray[str | np.str_]:
        """Factor names."""
        return self._factor_names

    @property
    def warped_covariates(self) -> dict[str, npt.NDArray[np.float32]] | None:
        """Time-warped covariates for each group, if using a GP prior and dynamic time warping was enabled."""
        return self._covariates if hasattr(self, "_orig_covariates") else None

    @property
    def covariates(self) -> dict[str, npt.NDArray[np.float32]]:
        """Covariates for each group, if using a GP prior."""
        return self._orig_covariates if hasattr(self, "_orig_covariates") else self._covariates

    @property
    def covariates_names(self) -> dict[str, str | npt.NDArray[str | np.str_]]:
        """Covariate names for each group where they could be inferred from the input."""
        return self._covariates_names

    @property
    def gp_lengthscale(self) -> npt.NDArray[np.float32] | None:
        """Inferred lengthscales for each factor, if using a GP prior."""
        return self._gp.lengthscale.cpu().numpy().squeeze() if self._gp is not None else None

    @property
    def gp_scale(self) -> npt.NDArray[np.float32] | None:
        """Inferred variance scales (smoothness) for each factor, if using a GP prior."""
        return self._gp.outputscale.cpu().numpy().squeeze() if self._gp is not None else None

    @property
    def gp_group_correlation(self) -> npt.NDArray[np.float32]:
        """Between-group correlation for each factor, if using a GP prior."""
        return self._gp.group_corr.cpu().numpy() if self._gp is not None else None

    @property
    def training_loss(self) -> npt.NDArray[np.float32]:
        """Total loss (negative ELBO) for each training epoch."""
        return self._train_loss_elbo

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
            _logger.info("- No likelihoods provided. Inferring likelihoods from data.")
            likelihoods = preprocessing.infer_likelihoods(data_concatenated)

        elif isinstance(likelihoods, dict):
            _logger.info("- Checking compatibility of provided likelihoods with data.")
            preprocessing.validate_likelihoods(data_concatenated, likelihoods)

        elif isinstance(likelihoods, str):
            _logger.info("- Using provided likelihood for all views.")
            likelihoods = {k: likelihoods for k in view_names}
            # Still validate likelihoods
            preprocessing.validate_likelihoods(data_concatenated, likelihoods)

        elif not (isinstance(likelihoods, dict) | isinstance(likelihoods, str)):
            raise ValueError("likelihoods must be a dictionary or string.")

        for k, v in likelihoods.items():
            _logger.info(f"  - {k}: {v}")

        return likelihoods

    def _setup_annotations(self, data):
        annotations = self._model_opts.annotations
        if annotations is None and self._model_opts.annotations_varm_key is not None:
            annotations = {}
            for vn in self._model_opts.annotations_varm_key.keys():
                for gn in data.keys():
                    if self._model_opts.annotations_varm_key[vn] in data[gn][vn].varm:
                        view_annotations = data[gn][vn].varm[self._model_opts.annotations_varm_key[vn]]
                        if not isinstance(view_annotations, pd.DataFrame):
                            view_annotations = pd.DataFrame(view_annotations, index=data[gn][vn].var_names)
                        annotations[vn] = view_annotations.fillna(0).T
                        break

        informed = annotations is not None and len(annotations) > 0
        valid_n_factors = self._model_opts.n_factors is not None and self._model_opts.n_factors > 0

        if not informed and not valid_n_factors:
            raise ValueError(
                "Invalid latent configuration, "
                "please provide either a collection of prior masks, "
                "or set `n_factors` to a positive integer."
            )

        n_dense_factors = 0
        n_informed_factors = 0

        factor_names = []

        if self._model_opts.n_factors is not None:
            n_dense_factors = self._model_opts.n_factors
            factor_names += [f"Factor {k + 1}" for k in range(n_dense_factors)]

        prior_masks = None

        if annotations is not None:
            # TODO: annotations need to be processed if not aligned or full
            n_informed_factors = annotations[self.view_names[0]].shape[0]
            if isinstance(annotations[self.view_names[0]], pd.DataFrame):
                factor_names += annotations[self.view_names[0]].index.tolist()
                for k, vm in annotations.items():
                    annotations[k] = vm.loc[:, self.feature_names[k]].to_numpy()
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

            for vn in self.view_names:
                if vn not in prior_masks:
                    prior_masks[vn] = np.zeros((n_dense_factors + n_informed_factors, self.n_features[vn]), dtype=bool)

        self._n_dense_factors = n_dense_factors
        self._n_informed_factors = n_informed_factors
        self._model_opts.n_factors = n_dense_factors + n_informed_factors

        self._factor_names = np.asarray(factor_names)
        self._factor_order = np.arange(self._model_opts.n_factors)

        # storing prior_masks as full annotations instead of partial annotations
        self._annotations = prior_masks

    def _setup_gp(self, full_setup=True):
        gp_group_names = [g for g in self.group_names if self._model_opts.factor_prior[g] == "GP"]

        gp_warp_groups_order = None
        if len(gp_group_names):
            if full_setup:
                if len(self._gp_opts.warp_groups) > 1:
                    if not set(self._gp_opts.warp_groups) <= set(gp_group_names):
                        raise ValueError(
                            "The set of groups with dynamic time warping must be a subset of groups with a GP factor prior."
                        )
                    gp_warp_groups_order = {}
                    for g in self._gp_opts.warp_groups:
                        ccov = self._covariates[g].squeeze()
                        if ccov.ndim > 1:
                            raise ValueError(
                                f"Warping can only be performed with 1D covariates, but the covariate for group {g} has {ccov.ndim} dimensions."
                            )
                        gp_warp_groups_order[g] = ccov.argsort()
                    self._orig_covariates = {g: c.clone() for g, c in self._covariates.items()}

                    if self._gp_opts.warp_reference_group is None:
                        self._gp_opts.warp_reference_group = self._gp_opts.warp_groups[0]
                elif len(self._gp_opts.warp_groups) == 1:
                    _logger.warn("Need at least 2 groups for warping, but only one was given. Ignoring warping.")
                    self._gp_opts.warp_groups = []

            self._gp = gp.GP(
                self._gp_opts.n_inducing,
                (torch.as_tensor(self._covariates[g]) for g in gp_group_names),
                self._model_opts.n_factors,
                len(gp_group_names),
                self._gp_opts.kernel,
            ).to(self._train_opts.device)
            self._gp_group_names = gp_group_names
        else:
            self._gp = None
            self._gp_group_names = None
        return gp_warp_groups_order

    def _setup_svi(self, prior_scales, init_tensor, feature_means, sample_means):
        gp_warp_groups_order = self._setup_gp()
        generative = Generative(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_factors=self._model_opts.n_factors,
            prior_scales=prior_scales,
            factor_prior=self._model_opts.factor_prior,
            weight_prior=self._model_opts.weight_prior,
            likelihoods=self._model_opts.likelihoods,
            nonnegative_factors=self._model_opts.nonnegative_factors,
            nonnegative_weights=self._model_opts.nonnegative_weights,
            gp=self._gp,
            gp_group_names=self._gp_group_names,
            feature_means=feature_means,
            sample_means=sample_means,
        ).to(self._train_opts.device)

        variational = Variational(generative, init_tensor).to(self._train_opts.device)

        n_iterations = int(self._train_opts.max_epochs * (self.n_samples_total // self._train_opts.batch_size))
        gamma = 0.1
        lrd = gamma ** (1 / n_iterations)
        _logger.info(f"Decaying learning rate over {n_iterations} iterations.")
        optimizer = ClippedAdam({"lr": self._train_opts.lr, "lrd": lrd})

        svi = SVI(
            model=pyro.poutine.scale(generative, scale=1.0 / self.n_samples_total),
            guide=pyro.poutine.scale(variational, scale=1.0 / self.n_samples_total),
            optim=optimizer,
            loss=TraceMeanField_ELBO(
                retain_graph=True, num_particles=self._train_opts.n_particles, vectorize_particles=True
            ),
        )

        return svi, variational, gp_warp_groups_order

    def _post_fit(self, data, feature_means, variational, train_loss_elbo):
        self._weights = variational.get_weights()
        self._factors = variational.get_factors()
        self._df_r2_full, self._df_r2_factors, self._factor_order = self._sort_factors(
            data, weights=self._weights.mean, factors=self._factors.mean
        )
        self._sparse_factors_probabilities = variational.get_sparse_factor_probabilities()
        self._sparse_weights_probabilities = variational.get_sparse_weight_probabilities()
        self._sparse_factors_precisions = variational.get_sparse_factor_precisions()
        self._sparse_weights_precisions = variational.get_sparse_weight_precisions()
        self._gps = self._get_gps(self._covariates)
        self._dispersions = variational.get_dispersion()
        self._train_loss_elbo = np.asarray(train_loss_elbo)

        if self._covariates is not None:
            self._covariates = {g: cov.numpy() for g, cov in self._covariates.items()}
        if hasattr(self, "_orig_covariates"):
            self._orig_covariates = {g: cov.numpy() for g, cov in self._orig_covariates.items()}

        self._train_opts.save_path = self._train_opts.save_path or f"model_{time.strftime('%Y%m%d_%H%M%S')}.h5"
        _logger.info("Saving results...")
        self._save(self._train_opts.save_path, self._train_opts.mofa_compat, data, feature_means)

    def _initialize_factors(self, data, impute_missings=True):
        init_tensor = defaultdict(dict)
        _logger.info(f"Initializing factors using `{self._model_opts.init_factors}` method...")

        with self._train_opts.device:
            if not isinstance(self._model_opts.init_factors, str):
                for group_name, n in self.n_samples.items():
                    init_tensor[group_name]["loc"] = (
                        (torch.ones(size=(n, self._model_opts.n_factors)) * self._model_opts.init_factors)
                        .T.unsqueeze(-2)
                        .float()
                    )
                    init_tensor[group_name]["scale"] = (
                        (torch.ones(size=(n, self._model_opts.n_factors)) * self._model_opts.init_scale)
                        .T.unsqueeze(-2)
                        .float()
                    )
                return init_tensor
            match self._model_opts.init_factors:
                case "random":
                    for group_name, n in self.n_samples.items():
                        init_tensor[group_name]["loc"] = torch.rand(size=(n, self._model_opts.n_factors))
                case "orthogonal":
                    for group_name, n in self.n_samples.items():
                        # Compute PCA of random vectors
                        pca = PCA(n_components=self._model_opts.n_factors, whiten=True)
                        pca.fit(stats.norm.rvs(loc=0, scale=1, size=(n, self._model_opts.n_factors)).T)
                        init_tensor[group_name]["loc"] = torch.tensor(pca.components_.T)
                case "pca" | "nmf" as init:
                    for group_name in self.n_samples.keys():
                        if init == "pca":
                            pca = PCA(n_components=self._model_opts.n_factors, whiten=True)
                        elif init == "nmf":
                            nmf = NMF(n_components=self._model_opts.n_factors, max_iter=1000)

                        # Combine all views
                        concat_data = np.concatenate(
                            [data[group_name][view_name].X for view_name in self.view_names], axis=-1, dtype=np.float32
                        )

                        # Check if data has missings. If yes, and impute_missings is True, then impute, else raise an error
                        if np.isnan(concat_data).any():
                            if impute_missings:
                                from sklearn.impute import SimpleImputer

                                imp = SimpleImputer(missing_values=np.NaN, strategy="mean")
                                imp.fit(concat_data)
                            else:
                                raise ValueError(
                                    "Data has missing values. Please impute missings or set `impute_missings=True`."
                                )
                        if init == "pca":
                            pca.fit(concat_data)
                            init_tensor[group_name]["loc"] = torch.as_tensor(pca.transform(concat_data))
                        elif init == "nmf":
                            nmf.fit(concat_data)
                            init_tensor[group_name]["loc"] = torch.as_tensor(nmf.transform(concat_data))

                case _:
                    raise ValueError(
                        f"Initialization method `{self._model_opts.init_factors}` not found. Please choose from `random`, `orthogonal`, `PCA`, or `NMF`."
                    )

            for group_name, n in self.n_samples.items():
                # scale factor values from -1 to 1 (per factor)
                q = init_tensor[group_name]["loc"]
                q = 2.0 * (q - torch.min(q, dim=0)[0]) / (torch.max(q, dim=0)[0] - torch.min(q, dim=0)[0]) - 1

                # Add artifical dimension at dimension -2 for broadcasting
                init_tensor[group_name]["loc"] = q.T.unsqueeze(-2).float()
                init_tensor[group_name]["scale"] = (
                    self._model_opts.init_scale
                    * torch.ones(size=(n, self._model_opts.n_factors)).T.unsqueeze(-2).float()
                )

        return init_tensor

    def _process_options(self, *args: _Options):
        self._data_opts = DataOptions()
        self._model_opts = ModelOptions()
        self._train_opts = TrainingOptions()
        self._gp_opts = SmoothOptions()

        for arg in args:
            match arg:
                case DataOptions():
                    self._data_opts |= arg
                case ModelOptions():
                    self._model_opts |= arg
                case TrainingOptions():
                    self._train_opts |= arg
                case SmoothOptions():
                    self._gp_opts |= arg

        if self._train_opts.seed is not None:
            try:
                self._train_opts.seed = int(self._train_opts.seed)
            except ValueError:
                _logger.warning(f"Could not convert `{self._train_opts.seed}` to integer.")
                self._train_opts.seed = None

        if self._train_opts.seed is None:
            self._train_opts.seed = int(time.strftime("%y%m%d%H%M"))

    def _adjust_options(self, data: dict[dict[ad.AnnData]]):
        # convert input arguments to dictionaries if necessary
        for opt_name, keys in zip(
            ("weight_prior", "factor_prior", "nonnegative_weights", "nonnegative_factors"),
            (self.view_names, self.group_names, self.view_names, self.group_names),
            strict=False,
        ):
            val = getattr(self._model_opts, opt_name)
            if not isinstance(val, dict):
                setattr(self._model_opts, opt_name, {k: val for k in keys})

        for opt_name in ("covariates_obs_key", "covariates_obsm_key"):
            val = getattr(self._data_opts, opt_name)
            if isinstance(val, str):
                setattr(self._data_opts, opt_name, {k: val for k in self.group_names})

        self._train_opts.device = self._setup_device(self._train_opts.device)

        if self._train_opts.batch_size is None or not (0 < self._train_opts.batch_size <= self.n_samples_total):
            self._train_opts.batch_size = self.n_samples_total

    def _preprocess_data(self, data):
        data = preprocessing.anndata_to_dense(data)
        self._model_opts.likelihoods = self._setup_likelihoods(data, self._model_opts.likelihoods)
        data = preprocessing.remove_constant_features(data, self._model_opts.likelihoods)
        data = preprocessing.scale_data(data, self._model_opts.likelihoods, self._data_opts.scale_per_group)
        data = preprocessing.center_data(
            data,
            self._model_opts.likelihoods,
            self._model_opts.nonnegative_weights,
            self._model_opts.nonnegative_factors,
        )

        # align observations across views and variables across groups
        if self._data_opts.use_obs is not None:
            data = preprocessing.align_obs(data, self._data_opts.use_obs)
        if self._data_opts.use_var is not None:
            data = preprocessing.align_var(data, self._model_opts.likelihoods, self._data_opts.use_var)

        # obtain observations DataFrame and covariates
        self._covariates, self._covariates_names = preprocessing.extract_covariate(
            data, self._data_opts.covariates_obs_key, self._data_opts.covariates_obsm_key
        )  # names for MOFA output

        # compute feature means for intercept terms
        feature_means = preprocessing.get_data_mean(data, self._model_opts.likelihoods, how="feature")
        sample_means = preprocessing.get_data_mean(data, self._model_opts.likelihoods, how="sample")

        return data, feature_means, sample_means

    def _fit(self, data, feature_means, sample_means):
        init_tensor = self._initialize_factors(data)

        prior_scales = None
        if self._annotations is not None:
            prior_scales = {
                vn: np.clip(vm.astype(np.float32) + self._model_opts.prior_penalty, 1e-8, 1.0)
                for vn, vm in self._annotations.items()
            }
            if self._n_dense_factors > 0:
                dense_scale = 1.0
                for vn in self._annotations.keys():
                    prior_scales[vn][: self._n_dense_factors, :] = dense_scale

        svi, variational, gp_warp_groups_order = self._setup_svi(prior_scales, init_tensor, feature_means, sample_means)

        # convert AnnData to torch.Tensor objects
        tensor_dict = {}
        for group_name, group_dict in data.items():
            tensor_dict[group_name] = {}
            if self._covariates is not None and group_name in self._covariates:
                if self._covariates[group_name] is not None:
                    tensor_dict[group_name]["covariates"] = torch.as_tensor(self._covariates[group_name])

            for view_name, view_adata in group_dict.items():
                tensor_dict[group_name][view_name] = torch.from_numpy(view_adata.X)

        if self._train_opts.batch_size < self.n_samples_total:
            batch_fraction = self._train_opts.batch_size / self.n_samples_total

            # has to be a list of data loaders to zip over
            data_loaders = []

            for group_name in data.keys():
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
                        pin_memory=str(self._train_opts.device) != "cpu",
                        drop_last=False,
                    )
                )

            def step_fn():
                epoch_loss = 0

                for group_batch in zip(*data_loaders, strict=False):
                    with self._train_opts.device:
                        epoch_loss += svi.step(
                            dict(
                                zip(
                                    data.keys(),
                                    (batch.to(self._train_opts.device) for batch in group_batch),
                                    strict=False,
                                )
                            )
                        )

                return epoch_loss

        else:
            tensor_dict = _to_device(tensor_dict, self._train_opts.device)

            def step_fn():
                with self._train_opts.device:
                    return svi.step(tensor_dict)

        _logger.info(f"Setting training seed to `{self._train_opts.seed}`.")
        random.seed(self._train_opts.seed)
        np.random.seed(self._train_opts.seed)
        torch.manual_seed(self._train_opts.seed)
        torch.cuda.manual_seed_all(self._train_opts.seed)
        pyro.set_rng_seed(self._train_opts.seed)

        # clean start
        _logger.info("Cleaning parameter store.")
        pyro.enable_validation(True)
        pyro.clear_param_store()

        # Train
        train_loss_elbo = []
        earlystopper = EarlyStopper(
            mode="min", min_delta=0.1, patience=self._train_opts.early_stopper_patience, percentage=True
        )
        start_timer = time.time()

        for i in range(self._train_opts.max_epochs):
            loss = step_fn()
            if self._gp is not None and len(self._gp_opts.warp_groups) and not i % self._gp_opts.warp_interval:
                self._warp_covariates(variational, gp_warp_groups_order)
            train_loss_elbo.append(loss)

            if i % self._train_opts.print_every == 0:
                _logger.info(f"Epoch: {i:>7} | Time: {time.time() - start_timer:>10.2f}s | Loss: {loss:>10.2f}")

            if earlystopper.step(loss):
                _logger.info(f"Training finished after {i} steps.")
                break

        self._post_fit(data, feature_means, variational, train_loss_elbo)

    def _warp_covariates(self, variational, warp_groups_order):
        factormeans = variational.get_factors().mean
        refgroup = self._gp_opts.warp_reference_group
        reffactormeans = factormeans[refgroup].mean(axis=0)
        refidx = warp_groups_order[refgroup]
        for g in self._gp_opts.warp_groups[1:]:
            idx = warp_groups_order[g]
            alignment = dtw(
                reffactormeans[refidx],
                factormeans[g][:, idx].mean(axis=0),
                open_begin=self._gp_opts.warp_open_begin,
                open_end=self._gp_opts.warp_open_end,
                step_pattern="asymmetric",
            )
            self._covariates[g] = self._orig_covariates[g].clone()
            self._covariates[g][idx[alignment.index2], 0] = self._orig_covariates[refgroup][refidx[alignment.index1], 0]
        self._gp.update_inducing_points(self._covariates.values())

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
        likelihood = self._model_opts.likelihoods[view_name]

        if likelihood == "Normal":
            ss_res = np.nansum(np.square(y_true - y_pred))
            ss_tot = np.nansum(np.square(y_true))  # data is centered
        elif likelihood == "GammaPoisson":
            y_pred = np.logaddexp(0, y_pred)  # softplus
            nu2 = self._dispersions.mean[view_name]
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
            dispersion = self._dispersions.mean[view_name]
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
            _logger.info(
                f"R2 for view {view_name} is 0. Increase the number of factors and/or the number of training epochs."
            )
            return r2_full, [0.0] * factors.shape[0]

        r2s = []
        if self._model_opts.likelihoods[view_name] == "Normal":
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
        return r2_full, r2s

    def _sort_factors(self, data, weights, factors, subsample=1000):
        # Loop over all groups
        dfs_factors, dfs_full = {}, {}

        for group_name, group_data in data.items():
            n_samples = self.n_samples[group_name]

            sample_idx = np.arange(n_samples)

            if subsample is not None and subsample > 0 and subsample < n_samples:
                sample_idx = np.random.choice(sample_idx, subsample, replace=False)

            group_r2_factors, group_r2_full = {}, {}
            for view_name, view_data in group_data.items():
                try:
                    group_r2_full[view_name], group_r2_factors[view_name] = self._r2(
                        view_data.X[sample_idx, :], factors[group_name][:, sample_idx], weights[view_name], view_name
                    )
                except NotImplementedError:
                    _logger.warning(
                        f"R2 calculation for {self.model_opts.likelihoods[view_name]} likelihood has not yet been implemented. Skipping view {view_name} for group {group_name}."
                    )
            if len(group_r2_factors) == 0:
                logging.warning(f"No R2 values found for group {group_name}. Skipping...")
                continue

            dfs_factors[group_name] = pd.DataFrame(group_r2_factors)
            dfs_full[group_name] = pd.Series(group_r2_full)

        # sum the R2 values across all groups
        df_concat = pd.concat(dfs_factors.values())
        df_sum = df_concat.groupby(df_concat.index).sum()
        dfs_full = pd.DataFrame(dfs_full)

        try:
            # sort factors according to mean R2 across all views
            sorted_r2_means = df_sum.mean(axis=1).sort_values(ascending=False)
            factor_order = sorted_r2_means.index.to_numpy()
        except NameError:
            _logger.info("Sorting factors failed. Using default order.")
            factor_order = np.array(list(range(self.model_opts.n_factors)))

        return dfs_full, dfs_factors, factor_order

    def _get_component(self, component, return_type="pandas"):
        match return_type:
            case "numpy":
                return {k: v.to_numpy() for k, v in component.items()}
            case "pandas":
                return component
            case "torch":
                return {k: torch.tensor(v.values, dtype=torch.float).clone().detach() for k, v in component.items()}
            case "anndata":
                return {k: ad.AnnData(v) for k, v in component.items()}

    def _get_sparse(self, what, moment, sparse_type):
        ret = {}
        probs = getattr(self, f"_sparse_{what}_probabilities")
        vals = getattr(self, "_" + what)
        precs = getattr(self, f"_sparse_{what}_precisions")
        for name, cvals in getattr(vals, moment).items():
            if name in probs:
                if sparse_type == "mix":
                    if moment == "mean":
                        cvals = cvals * probs[name]
                    else:
                        p = probs[name]
                        a = precs.mean[name][:, None]
                        cvals = np.sqrt(vals.mean[name] ** 2 * p * (1 - p) + p * cvals**2 + (1 - p) / a**2)
                elif sparse_type == "thresh":
                    if moment == "mean":
                        cvals = cvals * (vals[name].mean >= 0.5)
                    else:
                        cvals = 1 / precs.mean[name]
            ret[name] = cvals
        return ret

    def get_factors(
        self,
        return_type: Literal["pandas", "anndata", "numpy"] = "pandas",
        moment: Literal["mean", "std"] = "mean",
        sparse_type: Literal["raw", "mix", "thresh"] = "mix",
        ordered: bool = False,
    ) -> _ResultsTypeDF:
        """Get the factor matrices Z for each group.

        Args:
             return_type: Format of the returned object.
             moment: Which moment of the posterior distribution to return.
             sparse_type: How to handle sparsity when using the spike and slab prior.

                 - raw: Do nothing, return inferred values for all entries.
                 - mix: Return the corresponding moment of a mixture distribution of two
                   Normal distributions: One centered at 0 and the other centered at the
                   inferred non-sparse value. The mixture is weighted by the inferred
                   sparsity probability. This is what MOFA does.
                 - thresh: Set all values with a sparsity probablity > 0.5 to 0.

             ordered: Whether to return the factors ordered by explained variance (highest to lowest).
        """
        factors = {
            group_name: pd.DataFrame(
                group_factors.T, index=self.sample_names[group_name], columns=self.factor_names
            ).iloc[:, self.factor_order if ordered else slice(None)]
            for group_name, group_factors in self._get_sparse("factors", moment, sparse_type).items()
        }
        factors = self._get_component(factors, return_type)

        if return_type == "anndata":
            for group_name, group_adata in factors.items():
                group_adata.obs = pd.concat(self._metadata[group_name].values(), axis=1)
                group_adata.obs = group_adata.obs.loc[:, ~group_adata.obs.columns.duplicated()]

        return factors

    def get_r2(self, total: bool = False, ordered: bool = False) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Get the fraction of explained variance for each view and group.

        Args:
             total: If `True`, returns a DataFrame with fraction of explained variance for the full
                 model for each group (columns) and view (rows). Otherwise returns a dict with group
                 names as keys containing DataFrames with the fraction of explained variance for each
                 view (columns) and factor(rows).
             ordered: Whether to return the factors ordered by explained variance (highest to lowest).
                 Has no effect if `total == True`.
        """
        if total:
            return self._df_r2_full
        else:
            return {
                group_name: df.set_index(self.factor_names).iloc[self.factor_order if ordered else slice(None), :]
                for group_name, df in self._df_r2_factors.items()
            }

    def get_weights(
        self,
        return_type: Literal["pandas", "anndata", "numpy"] = "pandas",
        moment: Literal["mean", "std"] = "mean",
        sparse_type: Literal["raw", "mix", "thresh"] = "mix",
        ordered: bool = False,
    ) -> _ResultsTypeDF:
        """Get the weight matrices W for each view.

        Args:
             return_type: Format of the returned object.
             moment: Which moment of the posterior distribution to return.
             sparse_type: How to handle sparsity when using the spike and slab prior.

                 - raw: Do nothing, return inferred values for all entries.
                 - mix: Return the corresponding moment of a mixture distribution of two
                   Normal distributions: One centered at 0 and the other centered at the
                   inferred non-sparse value. The mixture is weighted by the inferred
                   sparsity probability. This is what MOFA does.
                 - thresh: Set all values with a sparsity probablity > 0.5 to 0.

             ordered: Whether to return the factors ordered by explained variance (highest to lowest).
        """
        weights = {
            view_name: pd.DataFrame(view_weights, index=self.factor_names, columns=self.feature_names[view_name]).iloc[
                self.factor_order if ordered else slice(None), :
            ]
            for view_name, view_weights in self._get_sparse("weights", moment, sparse_type).items()
        }

        return self._get_component(weights, return_type)

    def get_sparse_factor_probabilities(
        self, return_type: Literal["pandas", "anndata", "numpy"] = "pandas", ordered: bool = False
    ) -> _ResultsTypeDF:
        """Get the probabilties that a factor value is non-sparse for each group with a spike and slab factor prior.

        Args:
             return_type: Format of the returned object.
             ordered: Whether to return the factors ordered by explained variance (highest to lowest).
        """
        probs = {
            group_name: pd.DataFrame(group_prob.T, index=self.sample_names[group_name], columns=self.factor_names).iloc[
                :, self.factor_order if ordered else slice(None)
            ]
            for group_name, group_prob in self._sparse_factors_probabilities.items()
        }
        return self._get_component(probs, return_type)

    def get_sparse_weight_probabilities(
        self, return_type: Literal["pandas", "anndata", "numpy"] = "pandas", ordered: bool = False
    ) -> _ResultsTypeDF:
        """Get the probabilties that a weight value is non-sparse for each view with a spike and slab view prior.

        Args:
             return_type: Format of the returned object.
             ordered: Whether to return the factors ordered by explained variance (highest to lowest).
        """
        probs = {
            view_name: pd.DataFrame(view_prob, index=self.factor_names, columns=self.feature_names[view_name]).iloc[
                self.factor_order if ordered else slice(None), :
            ]
            for view_name, view_prob in self._sparse_weights_probabilities.items()
        }
        return self._get_component(probs, return_type)

    def get_dispersion(
        self, return_type: Literal["pandas", "anndata", "numpy"] = "pandas", moment: Literal["mean", "std"] = "mean"
    ) -> _ResultsTypeSeries:
        """Get the dispersion vectors for each view.

        Args:
             return_type: Format of the returned object.
             moment: Which moment of the posterior distribution to return.
        """
        dispersion = {
            view_name: pd.Series(view_dispersion, index=self.feature_names[view_name])
            for view_name, view_dispersion in getattr(self._dispersions, moment).items()
        }

        return self._get_component(dispersion, return_type)

    def get_gps(
        self,
        return_type: Literal["pandas", "anndata", "numpy"] = "pandas",
        moment: Literal["mean", "std"] = "mean",
        x: dict[str, np.ndarray | torch.Tensor] | None = None,
        batch_size: int | None = None,
        ordered: bool = False,
    ) -> _ResultsTypeDF:
        """Get all latent functions.

        Args:
             return_type: Format of the returned object.
             moment: Which moment of the posterior distribution to return.
             x: Covariate values for each group. If `None`, will return latent function values at
                 covariate coordinates used for training.
             batch_size: Minibatch size. Only has an effect if `x` is not `None`. Defaults to the
                 minibatch size used for training.
             ordered: Whether to return the factors ordered by explained variance (highest to lowest).
        """
        gps = getattr(self._gps if x is None else self._get_gps(x, batch_size), moment)
        gps = {
            group_name: pd.DataFrame(
                group_f[self.factor_order if ordered else slice(None), :].T, columns=self.factor_names
            )
            for group_name, group_f in gps.items()
        }

        if x is None:
            for gname, df in gps.items():
                df.set_index(np.asarray(self.sample_names[gname]), inplace=True)

        return self._get_component(gps, return_type)

    def _get_gps(self, x: dict[str, np.ndarray | torch.Tensor], batch_size: int | None = None):
        if batch_size is None:
            batch_size = self._train_opts.batch_size
        gps = MeanStd({}, {})
        if self._gp is not None:
            with (
                torch.inference_mode(),
                self._train_opts.device,
            ):  # FIXME: allow user to run this in a `with device` context?
                for group_idx, group_name in enumerate(self._gp_group_names):
                    gidx = torch.as_tensor(group_idx)
                    gdata = x[group_name]
                    mean, std = [], []

                    for start_idx in range(0, gdata.shape[0], batch_size):
                        end_idx = min(start_idx + batch_size, gdata.shape[0])
                        minibatch = gdata[start_idx:end_idx]

                        gp_dist = self._gp(
                            gidx.expand(minibatch.shape[0], 1),
                            torch.as_tensor(minibatch, dtype=torch.float32),
                            prior=False,
                        )

                        mean.append(gp_dist.mean.cpu().numpy().squeeze())
                        std.append(gp_dist.stddev.cpu().numpy().squeeze())

                    gps.mean[group_name] = np.concatenate(mean, axis=1)
                    gps.std[group_name] = np.concatenate(std, axis=1)
        return gps

    def get_annotations(
        self, return_type: Literal["pandas", "anndata", "numpy"] = "pandas", ordered=False
    ) -> _ResultsTypeDF:
        """Get the annotation matrices for each view.

        Args:
            return_type: Format of the returned object.
            ordered: Whether to return the factors ordered by explained variance (highest to lowest).
        """
        annotations = {
            k: pd.DataFrame(v, index=self.factor_names, columns=self.feature_names[k])
            .astype(bool)
            .iloc[self.factor_order if ordered else slice(None), :]
            for k, v in self._annotations.items()
        }

        return self._get_component(annotations, return_type)

    def _setup_device(self, device):
        _logger.info("Setting up device...")

        device = torch.device(device)
        tens = torch.tensor(())
        try:
            tens.to(device)
        except (RuntimeError, AssertionError):
            default_device = tens.device
            _logger.warning(f"Device {str(device)} is not available. Using default device: {default_device}")
            device = default_device

        return device

    def impute_data(
        self, data: MuData | dict[str, dict[str, ad.AnnData]], missing_only=False
    ) -> dict[dict[str, ad.AnnData]]:
        """Impute values in the training data using the trained factorization.

        Args:
            data: can be any of:

                - MuData object
                - Nested dict with group names as keys, view names as subkeys and AnnData objects as values
                  (incompatible with :py:class:`TrainingOptions` `.group_by`)

            missing_only: Only impute missing values in the data.
        """
        imputed_data = preprocessing.cast_data(data, group_by=self._data_opts.group_by, copy=True)

        factors = self.get_factors(return_type="numpy")
        weights = self.get_weights(return_type="numpy")

        for k_groups in self.group_names:
            for k_views in self.view_names:
                if missing_only and not np.isnan(imputed_data[k_groups][k_views].X).any():
                    continue

                imputation = factors[k_groups] @ weights[k_views]

                if self._model_opts.likelihoods[k_views] != "Normal":
                    if self._model_opts.likelihoods[k_views] == "Bernoulli":
                        imputation = expit(imputation)
                    else:
                        raise NotImplementedError(
                            f"Imputation for {self._model_opts.likelihoods[k_views]} not implemented."
                        )

                if not missing_only:
                    imputed_data[k_groups][k_views].X = imputation
                else:
                    _logger.debug(f"Imputing missing values for {k_groups} - {k_views}.")
                    mask = np.isnan(imputed_data[k_groups][k_views].X)
                    imputed_data[k_groups][k_views].X[mask] = imputation[mask]

        return imputed_data

    def _save(
        self,
        path: str | Path,
        mofa_compat: bool = False,
        data: dict[str, dict[str, ad.AnnData]] | None = None,
        intercepts: dict[str, dict[str, np.ndarray]] | None = None,
    ):
        state = {
            "weights": self._weights._asdict(),
            "factors": self._factors._asdict(),
            "covariates": self._covariates,
            "covariates_names": self._covariates_names,
            "df_r2_full": self._df_r2_full,
            "df_r2_factors": self._df_r2_factors,
            "n_dense_factors": self._n_dense_factors,
            "n_informed_factors": self._n_informed_factors,
            "factor_names": self._factor_names,
            "factor_order": self._factor_order,
            "sparse_factors_probabilities": self._sparse_factors_probabilities,
            "sparse_weights_probabilities": self._sparse_weights_probabilities,
            "sparse_factors_precisions": self._sparse_factors_precisions._asdict(),
            "sparse_weights_precisions": self._sparse_weights_precisions._asdict(),
            "gps": self._gps._asdict(),
            "dispersions": self._dispersions._asdict(),
            "train_loss_elbo": self._train_loss_elbo,
            "group_names": self._group_names,
            "view_names": self._view_names,
            "feature_names": self._feature_names,
            "sample_names": self._sample_names,
            "annotations": self._annotations,
            "metadata": self._metadata,
            "data_opts": asdict(self._data_opts),
            "model_opts": asdict(self._model_opts),
            "train_opts": asdict(self._train_opts),
            "gp_opts": asdict(self._gp_opts),
        }
        state["train_opts"]["device"] = str(state["train_opts"]["device"])
        if hasattr(self, "_orig_covariates"):
            state["orig_covariates"] = self._orig_covariates

        pickle = None
        if self._gp is not None and self._gp_group_names is not None:
            pickle = self._gp.state_dict()
            state["gp_group_names"] = self._gp_group_names
        save_model(state, pickle, path, mofa_compat, self, data, intercepts)

    @classmethod
    def load(cls, path: str | Path, map_location=None) -> "PRISMO":
        """Load a saved PRISMO model.

        Args:
            path: Path to the saved model file.
            map_location: Specify how to remap storage locations for PyTorch tensors. See the `torch.load`
                documentation for details.
        """
        state, pickle = load_model(path)

        model = cls.__new__(cls)
        model._weights = MeanStd(**state["weights"])
        model._factors = MeanStd(**state["factors"])
        model._covariates = state.get("covariates")
        if "orig_covariates" in state:
            model._orig_covariates = state["orig_covariates"]
        model._covariates_names = state.get("covariates_names")
        model._df_r2_full = state["df_r2_full"]
        model._df_r2_factors = state["df_r2_factors"]
        model._n_dense_factors = state["n_dense_factors"]
        model._n_informed_factors = state["n_informed_factors"]
        model._factor_names = state["factor_names"]
        model._factor_order = state["factor_order"]
        model._sparse_factors_probabilities = state["sparse_factors_probabilities"]
        model._sparse_weights_probabilities = state["sparse_weights_probabilities"]
        model._sparse_factors_precisions = MeanStd(**state["sparse_factors_precisions"])
        model._sparse_weights_precisions = MeanStd(**state["sparse_weights_precisions"])
        model._gps = MeanStd(**state["gps"])
        model._dispersions = MeanStd(**state["dispersions"])
        model._train_loss_elbo = state["train_loss_elbo"]
        model._group_names = state["group_names"]
        if "gp_group_names" in state:
            model._gp_group_names = state["gp_group_names"]
        model._view_names = state["view_names"]
        model._feature_names = {v: n.tolist() for v, n in state["feature_names"].items()}
        model._sample_names = {v: n.tolist() for v, n in state["sample_names"].items()}
        model._annotations = state.get("annotations")
        model._metadata = state["metadata"]
        model._data_opts = DataOptions(**state["data_opts"])
        model._model_opts = ModelOptions(**state["model_opts"])
        model._train_opts = TrainingOptions(**state["train_opts"])
        model._gp_opts = SmoothOptions(**state["gp_opts"])

        model._setup_gp(False)
        if model._gp is not None and len(pickle):
            model._gp.load_state_dict(pickle)

        return model
