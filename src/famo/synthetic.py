"""Data module."""

import itertools
import logging
import math

from typing import Optional

import anndata as ad
import mudata as mu
import numpy as np


logger = logging.getLogger(__name__)


class DataGenerator:
    def __init__(
        self,
        n_samples: int = 1000,
        n_features: Optional[list[int]] = None,
        likelihoods: Optional[list[str]] = None,
        n_fully_shared_factors: int = 2,
        n_partially_shared_factors: int = 15,
        n_private_factors: int = 3,
        factor_size_params: Optional[tuple[float]] = None,
        factor_size_dist: str = "uniform",
        n_active_factors: float = 1.0,
        n_covariates: int = 0,
        n_response: int = 0,
        nmf: Optional[list[bool]] = None,
        **kwargs,
    ) -> None:
        """Generate synthetic data

        Parameters
        ----------
        n_samples : int, optional
            Number of samples, by default 1000
        n_features : list[int], optional
            Number of features for each view, by default None
        likelihoods : list[str], optional
            Likelihoods for each view, 'normal' or 'bernoulli', by default None
        n_fully_shared_factors : int, optional
            Number of fully shared latent factors, by default 2
        n_partially_shared_factors : int, optional
            Number of partially shared latent factors, by default 15
        n_private_factors : int, optional
            Number of private latent factors, by default 3
        factor_size_params : tuple[float], optional
            Parameters for the distribution of the number
            of active factor loadings for the latent factors,
            by default None
        factor_size_dist : str, optional
            Distribution of the number of active factor loadings,
            either "uniform" or "gamma",
            by default "uniform"
        n_active_factors : float, optional
            Number or fraction of active factors, by default 1.0 (all)
        n_covariates : int, optional
            Number of observed covariates, by default 0
        n_response : int, optional
            Number of response variables from the latent factors, by default 0
        nmf : list[bool], optional
            Whether to generate data from a non-negative matrix factorization,
            by default False
        """

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
        self.x = None
        # covariate coefficients
        self.betas = None
        # latent factors
        self.z = None
        # factor loadings
        self.ws = None
        self.sigmas = None
        self.ys = None
        self.w_masks = None
        self.noisy_w_masks = None
        self.active_factor_indices = None
        self.view_factor_mask = None
        # set when introducing missingness
        self.presence_masks = None

        self.response_w = None
        self.response_sigma = None
        self.response = None

    @property
    def n_factors(self):
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
        for mask in self.presence_masks:
            nan_mask = np.array(mask, dtype=np.float32, copy=True)
            nan_mask[nan_mask == 0] = np.nan
            nan_masks.append(nan_mask)
        return nan_masks

    def _mask_to_bool(self):
        bool_masks = []
        for mask in self.presence_masks:
            bool_mask = mask == 1.0
            bool_masks.append(bool_mask)
        return bool_masks

    @property
    def missing_ys(self):
        if self.ys is None:
            logger.warning("Generate data first by calling `generate`.")
            return []
        if self.presence_masks is None:
            logger.warning("Introduce missing data first by calling `generate_missingness`.")
            return self.ys

        nan_masks = self._mask_to_nan()

        return [self.ys[m] * nan_masks[m] for m in range(self.n_views)]

    @property
    def y(self):
        return self._attr_to_matrix("ys")

    @property
    def missing_y(self):
        return self._attr_to_matrix("missing_ys")

    @property
    def w(self):
        return self._attr_to_matrix("ws")

    @property
    def w_mask(self):
        return self._attr_to_matrix("w_masks")

    @property
    def noisy_w_mask(self):
        return self._attr_to_matrix("noisy_w_masks")

    def _generate_view_factor_mask(self, rng=None, all_combs=False):
        if all_combs and self.n_views == 1:
            logger.warning("Single view dataset, " "cannot generate factor combinations for a single view.")
            all_combs = False
        if all_combs:
            logger.warning("Generating all possible binary combinations of " f"{self.n_views} variables.")
            self.n_fully_shared_factors = 1
            self.n_private_factors = self.n_views
            self.n_partially_shared_factors = 2**self.n_views - 2 - self.n_private_factors
            logger.warning(
                "New factor configuration: "
                f"{self.n_fully_shared_factors} fully shared, "
                f"{self.n_partially_shared_factors} partially shared, "
                f"{self.n_private_factors} private factors."
            )

            return np.array([list(i) for i in itertools.product([1, 0], repeat=self.n_views)])[:-1, :].T
        if rng is None:
            rng = np.random.default_rng()

        view_factor_mask = np.ones([self.n_views, self.n_factors])

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

    def normalise(self, with_std=False):
        for m in range(self.n_views):
            if self.likelihoods[m] == "normal":
                y = np.array(self.ys[m], dtype=np.float32, copy=True)
                y -= y.mean(axis=0)
                if with_std:
                    y_std = y.std(axis=0)
                    y = np.divide(y, y_std, out=np.zeros_like(y), where=y_std != 0)
                self.ys[m] = y

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def generate(self, seed: Optional[int] = None, all_combs: bool = False, overwrite: bool = False) -> None:
        rng = np.random.default_rng()

        if seed is not None:
            rng = np.random.default_rng(seed)

        if self.ys is not None and not overwrite:
            logger.warning("Data has already been generated, " "to generate new data please set `overwrite` to True.")
            return rng

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
                view_factor_mask[:, factor_idx] = 0.0

        for m in range(self.n_views):
            n_features = self.n_features[m]
            w_shape = (self.n_factors, n_features)
            w = rng.standard_normal(w_shape)
            w_mask = np.zeros(w_shape)

            fraction_active_features = {
                "gamma": (
                    lambda shape, scale, n_features=n_features: (rng.gamma(shape, scale, self.n_factors) + 20)
                    / n_features
                ),
                "uniform": lambda low, high, n_features=n_features: rng.uniform(low, high, self.n_factors),
            }[self.factor_size_dist](self.factor_size_params[m][0], self.factor_size_params[m][1])

            for factor_idx, faft in enumerate(fraction_active_features):
                if view_factor_mask[m, factor_idx] > 0:
                    w_mask[factor_idx] = rng.choice(2, n_features, p=[1 - faft, faft])

            # set small values to zero
            tiny_w_threshold = 0.1
            w_mask[np.abs(w) < tiny_w_threshold] = 0.0
            w_mask = w_mask.astype(bool)
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

            if self.likelihoods[m] == "normal":
                y = rng.normal(loc=y_loc, scale=sigma)
                if self.nmf[m]:
                    y = np.abs(y)
            elif self.likelihoods[m] == "bernoulli":
                y = rng.binomial(1, self.sigmoid(y_loc))
            elif self.likelihoods[m] == "poisson":
                rate = np.exp(y_loc)
                y = rng.poisson(rate)

            ws.append(w)
            sigmas.append(sigma)
            ys.append(y)
            w_masks.append(w_mask)

        if self.n_covariates > 0:
            self.x = x
            self.betas = betas

        self.z = z
        self.ws = ws
        self.w_masks = w_masks
        self.sigmas = sigmas
        self.ys = ys
        self.active_factor_indices = active_factor_indices
        self.view_factor_mask = view_factor_mask

        if self.n_response > 0:
            self.response_w = rng.standard_normal((self.n_factors, self.n_response))
            self.response_sigma = 1.0 / np.sqrt(rng.gamma(10.0, 1.0, self.n_response))
            self.response = rng.normal(loc=np.matmul(z, self.response_w), scale=self.response_sigma)

        return rng

    def get_noisy_mask(self, rng=None, noise_fraction=0.1, informed_view_indices=None):
        if rng is None:
            rng = np.random.default_rng()

        if informed_view_indices is None:
            logger.warning("Parameter `informed_view_indices` set to None, " "adding noise to all views.")
            informed_view_indices = list(range(self.n_views))

        noisy_w_masks = [np.array(mask, copy=True) for mask in self.w_masks]

        if len(informed_view_indices) == 0:
            logger.warning(
                "Parameter `informed_view_indices` " "set to an empty list, removing information from all views."
            )
            self.noisy_w_masks = [np.ones_like(mask) for mask in noisy_w_masks]
            return self.noisy_w_masks

        for m in range(self.n_views):
            noisy_w_mask = noisy_w_masks[m]

            if m in informed_view_indices:
                fraction_active_cells = noisy_w_mask.mean(axis=1).sum() / self.view_factor_mask[0].sum()
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
                    )

                    for on_idx, off_idx in swapped_indices:
                        noisy_w_mask[factor_idx, active_cell_indices[on_idx]] = 0.0
                        noisy_w_mask[factor_idx, inactive_cell_indices[off_idx]] = 1.0

            else:
                noisy_w_mask.fill(0.0)

        self.noisy_w_masks = noisy_w_masks
        return self.noisy_w_masks

    def generate_missingness(
        self,
        random_fraction: float = 0.0,
        n_partial_samples: int = 0,
        n_partial_features: int = 0,
        missing_fraction_partial_features: float = 0.0,
        seed=None,
    ):
        rng = np.random.default_rng()

        if seed is not None:
            rng = np.random.default_rng(seed)

        n_partial_samples = int(n_partial_samples)
        n_partial_features = int(n_partial_features)

        sample_view_mask = np.ones((self.n_samples, self.n_views))
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

        self.presence_masks = masks

        return rng

    def _permute_features(self, lst, new_order):
        return [np.array(lst[m][:, o], copy=True) for m, o in enumerate(new_order)]

    def _permute_factors(self, lst, new_order):
        return [np.array(lst[m][o, :], copy=True) for m, o in enumerate(new_order)]

    def permute_features(self, new_feature_order):
        if len(new_feature_order) != self.n_features:
            raise ValueError("Length of new order list must equal the number of features.")

        if self.betas is not None:
            self.betas = self._permute_features(self.betas, new_feature_order)
        self.ws = self._permute_features(self.ws, new_feature_order)
        self.w_masks = self._permute_features(self.w_masks, new_feature_order)
        if self.noisy_w_masks is not None:
            self.noisy_w_masks = self._permute_features(self.noisy_w_masks, new_feature_order)
        self.sigmas = self._permute_features(self.sigmas, new_feature_order)
        self.ys = self._permute_features(self.ys, new_feature_order)
        if self.presence_masks is not None:
            self.missing_ys = self._permute_features(self.missing_ys, new_feature_order)
            self.presence_masks = self._permute_features(self.presence_masks, new_feature_order)

    def permute_factors(self, new_factor_order):
        if len(new_factor_order) != self.n_factors:
            raise ValueError("Length of new order list must equal the number of factors.")

        self.z = np.array(self.z[:, np.array(new_factor_order)], copy=True)
        self.ws = self._permute_factors(self.ws, new_factor_order)
        self.w_masks = self._permute_factors(self.w_masks, new_factor_order)
        if self.noisy_w_masks is not None:
            self.noisy_w_masks = self._permute_factors(self.noisy_w_masks, new_factor_order)
        self.view_factor_mask = [self.view_factor_mask[m, np.array(new_factor_order)] for m in range(self.n_views)]
        self.active_factor_indices = np.array(self.active_factor_indices[np.array(new_factor_order)], copy=True)

    def to_mdata(self, noisy=False) -> mu.MuData:
        view_names = []
        ad_dict = {}
        for m in range(self.n_views):
            adata = ad.AnnData(self.ys[m], dtype=np.float32)
            adata.var_names = f"feature_group_{m}:" + adata.var_names
            adata.varm["w"] = self.ws[m].T
            w_mask = self.w_masks[m].T
            if noisy:
                w_mask = self.noisy_w_masks[m].T
            adata.varm["w_mask"] = w_mask
            view_name = f"feature_group_{m}"
            ad_dict[view_name] = adata
            view_names.append(view_name)

        mdata = mu.MuData(ad_dict)
        mdata.uns["likelihoods"] = dict(zip(view_names, self.likelihoods))
        mdata.uns["n_active_factors"] = self.n_active_factors
        if self.x is not None:
            mdata.obsm["x"] = self.x
        if self.z is not None:
            mdata.obsm["z"] = self.z

        return mdata
