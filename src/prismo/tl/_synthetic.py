"""Data module."""

import itertools
import logging
import math
from collections.abc import Iterable

import anndata as ad
import mudata as mu
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generator class for creating synthetic multi-view data with latent factors.

    This class generates synthetic data with specified properties including shared and private
    latent factors, different likelihoods, and optional covariates and response variables.

    Attributes:
        n_features: List of feature counts for each view.
        n_samples: Number of samples to generate.
        n_views: Number of views in the dataset.
        n_fully_shared_factors: Number of factors shared across all views.
        n_partially_shared_factors: Number of factors shared between some views.
        n_private_factors: Number of factors unique to individual views.
        n_covariates: Number of observed covariates.
        n_response: Number of response variables.
        likelihoods: List of likelihood types for each view.
        factor_size_params: Parameters for factor size distribution.
        factor_size_dist: Type of distribution for factor sizes.
        n_active_factors: Number or fraction of active factors.
        nmf: List indicating which views should use non-negative matrix factorization.
    """

    def __init__(
        self,
        n_features: list[int],
        n_samples: int = 1000,
        likelihoods: list[str] | None = None,
        n_fully_shared_factors: int = 2,
        n_partially_shared_factors: int = 15,
        n_private_factors: int = 3,
        factor_size_params: tuple[float] | None = None,
        factor_size_dist: str = "uniform",
        n_active_factors: float = 1.0,
        n_covariates: int = 0,
        n_response: int = 0,
        nmf: list[bool] | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_views = len(self.n_features)
        self.n_fully_shared_factors = n_fully_shared_factors
        self.n_partially_shared_factors = n_partially_shared_factors
        self.n_private_factors = n_private_factors
        self.n_covariates = n_covariates
        self.n_response = n_response

        if factor_size_params is None:
            if factor_size_dist == "uniform":
                logger.warning(
                    "Using a uniform distribution with parameters 0.05 and 0.15 "
                    "for generating the number of active factor loadings."
                )
                factor_size_params = (0.05, 0.15)
            elif factor_size_dist == "gamma":
                logger.warning(
                    "Using a uniform distribution with shape of 1 and scale of 50 "
                    "for generating the number of active factor loadings."
                )
                factor_size_params = (1.0, 50.0)

        if isinstance(factor_size_params, tuple):
            factor_size_params = [factor_size_params for _ in range(self.n_views)]

        self.factor_size_params = factor_size_params
        self.factor_size_dist = factor_size_dist

        # custom assignment
        if likelihoods is None:
            likelihoods = ["normal" for _ in range(self.n_views)]
        self.likelihoods = likelihoods

        self.n_active_factors = n_active_factors

        if nmf is None:
            nmf = [False for _ in range(self.n_views)]
        self.nmf = nmf

        # set upon data generation
        # covariates
        self._x = None
        # covariate coefficients
        self._betas = None
        # latent factors
        self._z = None
        # factor loadings
        self._ws = None
        self._sigmas = None
        self._ys = None
        self._w_masks = None
        self._noisy_w_masks = None
        self._active_factor_indices = None
        self._view_factor_mask = None
        # set when introducing missingness
        self._presence_masks = None
        self._missing_ys = None

        self._response_w = None
        self._response_sigma = None
        self._response = None

    @property
    def n_factors(self) -> int:
        """Total number of factors."""
        return self.n_fully_shared_factors + self.n_partially_shared_factors + self.n_private_factors

    def _to_matrix(self, matrix_list):
        return np.concatenate(matrix_list, axis=1)

    def _attr_to_matrix(self, attr_name):
        attr = getattr(self, attr_name)
        if attr is not None:
            attr = self._to_matrix(attr)
        return attr

    def _mask_to_nan(self):
        nan_masks = []
        for mask in self._presence_masks:
            nan_mask = np.array(mask, dtype=np.float32, copy=True)
            nan_mask[nan_mask == 0] = np.nan
            nan_masks.append(nan_mask)
        return nan_masks

    def _mask_to_bool(self):
        bool_masks = []
        for mask in self._presence_masks:
            bool_mask = mask == 1.0
            bool_masks.append(bool_mask)
        return bool_masks

    @property
    def missing_ys(self) -> list[npt.NDArray[np.float32]]:
        """Generated data with non-missing values replaced with `np.nan`."""
        if self._ys is None:
            logger.warning("Generate data first by calling `generate`.")
            return []
        if self._presence_masks is None:
            logger.warning("Introduce missing data first by calling `generate_missingness`.")
            return self._ys

        nan_masks = self._mask_to_nan()

        return [self._ys[m] * nan_masks[m] for m in range(self.n_views)]

    @property
    def y(self) -> npt.NDArray[np.float32]:
        """Generated data."""
        return self._attr_to_matrix("_ys")

    @property
    def missing_y(self) -> npt.NDArray[np.float32]:
        """Generated data with non-missing values replaced with `np.nan`."""
        return self._attr_to_matrix("_missing_ys")

    @property
    def w(self) -> npt.NDArray[np.float32]:
        """Generated weights."""
        return self._attr_to_matrix("_ws")

    @property
    def z(self) -> npt.NDArray[np.float32]:
        """Generated latent factors."""
        return self._z

    @property
    def w_mask(self) -> npt.NDArray[np.bool]:
        """Gene set mask describing co-expressed genes."""
        return self._attr_to_matrix("_w_masks")

    @property
    def noisy_w_mask(self) -> npt.NDArray[np.bool]:
        """Gene set mask describing co-expressed genes, with added noise."""
        return self._attr_to_matrix("_noisy_w_masks")

    def _generate_view_factor_mask(self, rng=None, all_combs=False):
        if all_combs and self.n_views == 1:
            logger.warning("Single view dataset, cannot generate factor combinations for a single view.")
            all_combs = False
        if all_combs:
            logger.warning(f"Generating all possible binary combinations of {self.n_views} variables.")
            self.n_fully_shared_factors = 1
            self.n_private_factors = self.n_views
            self.n_partially_shared_factors = 2**self.n_views - 2 - self.n_private_factors
            logger.warning(
                "New factor configuration: "
                f"{self.n_fully_shared_factors} fully shared, "
                f"{self.n_partially_shared_factors} partially shared, "
                f"{self.n_private_factors} private factors."
            )

            return np.array([list(i) for i in itertools.product([1, 0], repeat=self.n_views)], dtype=bool)[:-1, :].T
        if rng is None:
            rng = np.random.default_rng()

        view_factor_mask = np.ones([self.n_views, self.n_factors], dtype=bool)

        for factor_idx in range(self.n_fully_shared_factors, self.n_factors):
            # exclude view subsets for partially shared factors
            if factor_idx < self.n_fully_shared_factors + self.n_partially_shared_factors:
                if self.n_views > 2:
                    exclude_view_subset_size = rng.integers(1, self.n_views - 1)
                else:
                    exclude_view_subset_size = 0

                exclude_view_subset = rng.choice(self.n_views, exclude_view_subset_size, replace=False)
            # exclude all but one view for private factors
            else:
                include_view_idx = rng.integers(self.n_views)
                exclude_view_subset = [i for i in range(self.n_views) if i != include_view_idx]

            for m in exclude_view_subset:
                view_factor_mask[m, factor_idx] = 0

        if self.n_private_factors >= self.n_views:
            view_factor_mask[-self.n_views :, -self.n_views :] = np.eye(self.n_views)

        return view_factor_mask

    def normalise(self, with_std: bool = False):
        """Normalize data with a Gaussian likelihood to zero mean and optionally unit variance.

        Args:
            with_std: If `True`, also normalize to unit variance. Otherwise, only shift to zero mean.
        """
        for m in range(self.n_views):
            if self.likelihoods[m] == "normal":
                y = np.array(self._ys[m], dtype=np.float32, copy=True)
                y -= y.mean(axis=0)
                if with_std:
                    y_std = y.std(axis=0)
                    y = np.divide(y, y_std, out=np.zeros_like(y), where=y_std != 0)
                self._ys[m] = y

    def _sigmoid(self, x: float):
        return 1.0 / (1 + np.exp(-x))

    def generate(self, seed: int | None = None, all_combs: bool = False, overwrite: bool = False):
        """Generate synthetic data.

        Args:
            seed: Seed for the pseudorandom number generator.
            all_combs: Wether to generate all combinations of active factors and views. If `True`, the model
                will have 1 shared factor, `n_views` private factors, and `2**n_views - n_views -2` partially
                shared factors.
            overwrite: Whether to overwrite already generated data
        """
        rng = np.random.default_rng(seed)

        if self._ys is not None and not overwrite:
            raise ValueError("Data has already been generated, to generate new data please set `overwrite` to True.")

        view_factor_mask = self._generate_view_factor_mask(rng, all_combs)

        n_active_factors = self.n_active_factors
        if n_active_factors <= 1.0:
            # if fraction of active factors convert to int
            n_active_factors = int(n_active_factors * self.n_factors)

        active_factor_indices = sorted(rng.choice(self.n_factors, size=math.ceil(n_active_factors), replace=False))

        # generate factor scores which lie in the latent space
        z = rng.standard_normal((self.n_samples, self.n_factors))

        if any(self.nmf):
            z = np.abs(z)

        if self.n_covariates > 0:
            x = rng.standard_normal((self.n_samples, self.n_covariates))
            if any(self.nmf):
                x = np.abs(x)

        betas = []
        ws = []
        sigmas = []
        ys = []
        w_masks = []

        for factor_idx in range(self.n_factors):
            if factor_idx not in active_factor_indices:
                view_factor_mask[:, factor_idx] = 0

        for m in range(self.n_views):
            n_features = self.n_features[m]
            w_shape = (self.n_factors, n_features)
            w = rng.standard_normal(w_shape)
            w_mask = np.zeros(w_shape, dtype=np.bool)

            fraction_active_features = {
                "gamma": (
                    lambda shape, scale, n_features=n_features: (rng.gamma(shape, scale, self.n_factors) + 20)
                    / n_features
                ),
                "uniform": lambda low, high, n_features=n_features: rng.uniform(low, high, self.n_factors),
            }[self.factor_size_dist](self.factor_size_params[m][0], self.factor_size_params[m][1])

            for factor_idx, faft in enumerate(fraction_active_features):
                if view_factor_mask[m, factor_idx]:
                    w_mask[factor_idx] = rng.choice(2, n_features, p=[1 - faft, faft])

            # set small values to zero
            tiny_w_threshold = 0.1
            w_mask[np.abs(w) < tiny_w_threshold] = False
            # add some noise to avoid exactly zero values
            w = np.where(w_mask, w, rng.standard_normal(w_shape) / 100)
            assert ((np.abs(w) > tiny_w_threshold) == w_mask).all()

            if self.nmf[m]:
                w = np.abs(w)

            y_loc = np.matmul(z, w)

            if self.n_covariates > 0:
                beta_shape = (self.n_covariates, n_features)
                # reduce effect of betas by scaling them down
                beta = rng.standard_normal(beta_shape) / 10
                if self.nmf[m]:
                    beta = np.abs(beta)
                y_loc = y_loc + np.matmul(x, beta)
                betas.append(beta)

            # generate feature sigmas
            sigma = 1.0 / np.sqrt(rng.gamma(10.0, 1.0, n_features))

            match self.likelihoods[m]:
                case "normal":
                    y = rng.normal(loc=y_loc, scale=sigma)
                    if self.nmf[m]:
                        y = np.abs(y)
                case "bernoulli":
                    y = rng.binomial(1, self._sigmoid(y_loc))
                case "poisson":
                    rate = np.exp(y_loc)
                    y = rng.poisson(rate)

            ws.append(w)
            sigmas.append(sigma)
            ys.append(y)
            w_masks.append(w_mask)

        if self.n_covariates > 0:
            self._x = x
            self._betas = betas

        self._z = z
        self._ws = ws
        self._w_masks = w_masks
        self._sigmas = sigmas
        self._ys = ys
        self._active_factor_indices = active_factor_indices
        self._view_factor_mask = view_factor_mask

        if self.n_response > 0:
            self._response_w = rng.standard_normal((self.n_factors, self.n_response))
            self._response_sigma = 1.0 / np.sqrt(rng.gamma(10.0, 1.0, self.n_response))
            self._response = rng.normal(loc=np.matmul(z, self._response_w), scale=self._response_sigma)

    def get_noisy_mask(
        self, seed: int | None = None, noise_fraction: float = 0.1, informed_view_indices: Iterable[int] | None = None
    ) -> list[npt.NDArray[bool]]:
        """Generate a noisy version of `w_mask`, the mask describing co-expressed genes.

        Noisy in this context means that some annotations are wrong, i.e. some genes active in a particular factor
        are marked as inactive, and some genes inactive in a factor are marked as active.

        Args:
            seed: Seed for the pseudorandom number generator.
            noise_fraction: Fraction of active genes per factor that will be marked as inactive. The same number of
                inactive genes will be marked as active.
            informed_view_indices: Indices of views that will be used to benchmark informed models. Noisy masks will
                be generated only for those views. For uninformed views, th enoisy masks will be filled with `False`.

        Returns:
            A list with a noisy mask for each view.
        """
        rng = np.random.default_rng(seed)

        if informed_view_indices is None:
            logger.warning("Parameter `informed_view_indices` set to None, adding noise to all views.")
            informed_view_indices = list(range(self.n_views))

        noisy_w_masks = [mask.copy() for mask in self._w_masks]

        if len(informed_view_indices) == 0:
            logger.warning(
                "Parameter `informed_view_indices` set to an empty list, removing information from all views."
            )
            self._noisy_w_masks = [np.ones_like(mask) for mask in noisy_w_masks]
            return self._noisy_w_masks

        for m in range(self.n_views):
            noisy_w_mask = noisy_w_masks[m]

            if m in informed_view_indices:
                fraction_active_cells = noisy_w_mask.mean(axis=1).sum() / self._view_factor_mask[0].sum()
                for factor_idx in range(self.n_factors):
                    active_cell_indices = noisy_w_mask[factor_idx, :].nonzero()[0]
                    # if all features turned off
                    # => simulate random noise in terms of false positives only
                    if len(active_cell_indices) == 0:
                        logger.warning(
                            f"Factor {factor_idx} is completely off, inserting "
                            f"{(100 * fraction_active_cells):.2f}%% false positives."
                        )
                        active_cell_indices = rng.choice(
                            self.n_features[m], int(self.n_features[m] * fraction_active_cells), replace=False
                        )

                    inactive_cell_indices = (noisy_w_mask[factor_idx, :] == 0).nonzero()[0]
                    n_noisy_cells = int(noise_fraction * len(active_cell_indices))
                    swapped_indices = zip(
                        rng.choice(len(active_cell_indices), n_noisy_cells, replace=False),
                        rng.choice(len(inactive_cell_indices), n_noisy_cells, replace=False),
                        strict=False,
                    )

                    for on_idx, off_idx in swapped_indices:
                        noisy_w_mask[factor_idx, active_cell_indices[on_idx]] = False
                        noisy_w_mask[factor_idx, inactive_cell_indices[off_idx]] = True

            else:
                noisy_w_mask.fill(False)

        self._noisy_w_masks = noisy_w_masks
        return self._noisy_w_masks

    def generate_missingness(
        self,
        seed: int | None = None,
        n_partial_samples: int = 0,
        n_partial_features: int = 0,
        missing_fraction_partial_features: float = 0.0,
        random_fraction: float = 0.0,
    ):
        """Mark observations as missing.

        Args:
            seed: Seed for the pseudorandom number generator.
            n_partial_samples: Number of samples marked as missing in one random view. If the model has only
                one view, this has no effect.
            n_partial_features: Number of features marked as missing in some samples.
            missing_fraction_partial_features: Fraction of samples marked as missing due to `n_partial_features`.
            random_fraction: Fraction of all observations marked as missing at random.
        """
        rng = np.random.default_rng(seed)

        sample_view_mask = np.ones((self.n_samples, self.n_views), dtype=np.bool)
        missing_sample_indices = rng.choice(self.n_samples, n_partial_samples, replace=False)

        # partially missing samples
        for ms_idx in missing_sample_indices:
            if self.n_views > 1:
                exclude_view_subset_size = rng.integers(1, self.n_views)
            else:
                exclude_view_subset_size = 0
            exclude_view_subset = rng.choice(self.n_views, exclude_view_subset_size, replace=False)
            sample_view_mask[ms_idx, exclude_view_subset] = 0

        mask = np.repeat(sample_view_mask, self.n_features, axis=1)

        # partially missing features
        missing_feature_indices = rng.choice(sum(self.n_features), n_partial_features, replace=False)

        for mf_idx in missing_feature_indices:
            random_sample_indices = rng.choice(
                self.n_samples, int(self.n_samples * missing_fraction_partial_features), replace=False
            )
            mask[random_sample_indices, mf_idx] = 0

        # remove random fraction
        mask *= rng.choice([0, 1], mask.shape, p=[random_fraction, 1 - random_fraction])

        view_feature_offsets = [0, *np.cumsum(self.n_features).tolist()]
        masks = []
        for offset_idx in range(len(view_feature_offsets) - 1):
            start_offset = view_feature_offsets[offset_idx]
            end_offset = view_feature_offsets[offset_idx + 1]
            masks.append(mask[:, start_offset:end_offset])

        self._presence_masks = masks

    def _permute_features(self, lst, new_order):
        return [np.array(lst[m][:, o], copy=True) for m, o in enumerate(new_order)]

    def _permute_factors(self, lst, new_order):
        return [np.array(lst[m][o, :], copy=True) for m, o in enumerate(new_order)]

    def permute_features(self, new_feature_order: Iterable[int]):
        """Permute features.

        Args:
            new_feature_order: New ordering of features.
        """
        n_features = sum(self.n_features)
        if len(new_feature_order) != n_features:
            raise ValueError("Length of new order list must equal the number of features.")
        new_feature_order = np.asarray(new_feature_order, dtype=np.int)
        if (
            new_feature_order.min() != 0
            or new_feature_order.max() != n_features - 1
            or np.unique(new_feature_order).size != new_feature_order.size
        ):
            raise ValueError(f"New order must contain all integers in [0, {n_features}).")

        if self._betas is not None:
            self._betas = self._permute_features(self._betas, new_feature_order)
        self._ws = self._permute_features(self._ws, new_feature_order)
        self._w_masks = self._permute_features(self._w_masks, new_feature_order)
        if self._noisy_w_masks is not None:
            self._noisy_w_masks = self._permute_features(self._noisy_w_masks, new_feature_order)
        self._sigmas = self._permute_features(self._sigmas, new_feature_order)
        self._ys = self._permute_features(self._ys, new_feature_order)
        if self._presence_masks is not None:
            self._missing_ys = self._permute_features(self._missing_ys, new_feature_order)
            self._presence_masks = self._permute_features(self._presence_masks, new_feature_order)

    def permute_factors(self, new_factor_order: Iterable[int]):
        """Permute factors.

        Args:
            new_factor_order: New ordering of factors.
        """
        if len(new_factor_order) != self.n_factors:
            raise ValueError("Length of new order list must equal the number of factors.")
        new_factor_order = np.asarray(new_factor_order)
        if (
            new_factor_order.min() != 0
            or new_factor_order.max() != self.n_factors - 1
            or np.unique(new_factor_order).size != new_factor_order.size
        ):
            raise ValueError(f"New order must contain all integers in [0, {self.n_factors}).")

        self._z = np.array(self._z[:, np.array(new_factor_order)], copy=True)
        self._ws = self._permute_factors(self._ws, new_factor_order)
        self._w_masks = self._permute_factors(self._w_masks, new_factor_order)
        if self._noisy_w_masks is not None:
            self._noisy_w_masks = self._permute_factors(self._noisy_w_masks, new_factor_order)
        self._view_factor_mask = [self._view_factor_mask[m, np.array(new_factor_order)] for m in range(self.n_views)]
        self._active_factor_indices = np.array(self._active_factor_indices[np.array(new_factor_order)], copy=True)

    def to_mdata(self, noisy=False) -> mu.MuData:
        """Export the generated data as a `MuData` object.

        The `AnnData` objects generated for each view will have their weights in `.varm["w"]` and the gene set mask
        in `.varm["w_mask"]. The latent factors will be in `.obsm["z"]` of the `MuData` object, the likelihoods in
        `.uns["likelihoods"]` and the number of active factors in `.uns["n_active_factors"]`. If covariates
        were simulated, they will be in `.obsm["x"]`.

        Args:
            noisy: Whether to export the noisy or noise-free gene set mask.
        """
        view_names = []
        ad_dict = {}
        for m in range(self.n_views):
            adata = ad.AnnData(self._ys[m], dtype=np.float32)
            adata.var_names = f"feature_group_{m}:" + adata.var_names
            adata.varm["w"] = self._ws[m].T
            w_mask = self._w_masks[m].T
            if noisy:
                w_mask = self._noisy_w_masks[m].T
            adata.varm["w_mask"] = w_mask
            view_name = f"feature_group_{m}"
            ad_dict[view_name] = adata
            view_names.append(view_name)

        mdata = mu.MuData(ad_dict)
        mdata.uns["likelihoods"] = dict(zip(view_names, self.likelihoods, strict=False))
        mdata.uns["n_active_factors"] = self.n_active_factors
        mdata.obsm["z"] = self._z
        if self._x is not None:
            mdata.obsm["x"] = self._x

        return mdata
