import time
from collections import defaultdict
from functools import reduce

import anndata as ad
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from addict import Dict
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam
from sklearn.decomposition import PCA
from tensordict import TensorDict
from torch.utils.data import DataLoader

from famo.model import Generative, Variational
from famo.plotting import plot_overview
from famo.utils import data as utils_data
from famo.utils.io import save_model
from famo.utils.training import EarlyStopper

# Set 16bit cuda float as default
# torch.set_default_dtype(torch.float32)


class CORE(PyroModule):
    def __init__(self, device):
        super().__init__(name="CORE")
        self.device = self._setup_device(device)
        self.to(self.device)

        self._init()

    def _init(self):
        # data related attributes
        self.data = None
        self.intercepts = None
        self.annotations = None
        self.n_groups = 0
        self.n_views = 0
        self.n_samples = {}
        self.n_features = {}
        self.n_factors = 0

        self.group_names = []
        self.view_names = []
        self.sample_names = {}
        self.feature_names = {}
        self.factor_names = []
        self.factor_index = []

        self.likelihoods = []

        # Training settings
        self.init_tensor = None

        # SVI related attributes
        self.generative = None
        self.variational = None
        self._optimizer = None
        self._svi = None

        # training related attributes
        self.train_loss_elbo = []
        self._is_trained = False
        self._cache = None

    def _to_device(self, data):
        tensor_dict = {}
        for k, v in data.items():
            if isinstance(v, dict):
                tensor_dict[k] = self._to_device(v)
            else:
                tensor_dict[k] = v.to(self.device)

        return tensor_dict

    def _setup_data(
        self,
        data,
        likelihoods,
        scale_per_group,
        use_obs,
        use_var,
    ):
        data = utils_data.cast_data(data)
        likelihoods = self._setup_likelihoods(data, likelihoods)
        data = utils_data.remove_constant_features(data, likelihoods)
        data = utils_data.center_data(data, likelihoods)
        data = utils_data.scale_data(data, likelihoods, scale_per_group)
        data = utils_data.align_obs(data, use_obs)
        data = utils_data.align_var(data, likelihoods, use_var)

        self.feature_means = utils_data.get_feature_mean(data, likelihoods)
        self.likelihoods = likelihoods
        self.data = data

        (
            self.group_names,
            self.n_groups,
            self.view_names,
            self.n_views,
            self.feature_names,
            self.n_features,
            self.sample_names,
            self.n_samples,
        ) = utils_data.get_sizes_and_names(data)

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
            print("- No likelihoods provided. Inferring likelihoods from data.")
            likelihoods = utils_data.infer_likelihoods(data_concatenated)

        elif isinstance(likelihoods, dict):
            print("- Checking compatibility of provided likelihoods with data.")
            utils_data.validate_likelihoods(data_concatenated, likelihoods)

        elif isinstance(likelihoods, str):
            print("- Using provided likelihood for all views.")
            likelihoods = {k: likelihoods for k in view_names}
            # Still validate likelihoods
            utils_data.validate_likelihoods(data_concatenated, likelihoods)

        elif not (isinstance(likelihoods, dict) | isinstance(likelihoods, str)):
            raise ValueError("likelihoods must be a dictionary or string.")

        for k, v in likelihoods.items():
            print(f"  - {k}: {v}")

        return likelihoods

    def _setup_annotations(self, n_factors, annotations):
        if n_factors is None and annotations is None:
            raise ValueError("`n_factors` or `annotations` must be provided.")

        if n_factors is not None:
            self.n_factors = n_factors
            self.factor_names = [f"Factor {k + 1}" for k in range(n_factors)]

        if annotations is not None:
            # TODO: annotations need to be processed if not aligned or full
            self.n_factors = annotations[self.view_names[0]].shape[0]
            if isinstance(annotations[self.view_names[0]], pd.DataFrame):
                self.factor_names = list(annotations[self.view_names[0]].index)

        self.annotations = annotations
        return self.annotations

    def _setup_svi(self, weight_prior, factor_prior, likelihoods, lr):
        self.generative = Generative(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_factors=self.n_factors,
            annotations=self.annotations,
            factor_prior=factor_prior,
            weight_prior=weight_prior,
            likelihoods=likelihoods,
            device=self.device,
        )

        self.variational = Variational(self.generative, self.init_tensor)

        self._optimizer = ClippedAdam({"lr": lr})
        self._svi = SVI(
            self.generative,
            self.variational,
            self._optimizer,
            loss=TraceMeanField_ELBO(),
        )

        return self._svi

    def _post_fit(self, save, save_path):
        # Sort factors by explained variance
        weights = self._get_weights_from_guide()
        factors = self._get_factors_from_guide()
        df_r2, self.factor_index = self._sort_factors(weights=weights, factors=factors)

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
            print("Saving results...")
            save_model(self, save_path)

    def _initialize_factors(self, init_factors="random", init_scale=1.0, impute_missings=True):
        init_tensor = Dict()
        print(f"Initializing factors using `{init_factors}` method...")

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
        data,
        n_factors=None,
        annotations=None,
        weight_prior=None,
        factor_prior=None,
        likelihoods=None,
        lr=0.001,
        batch_size=0,
        early_stopper_patience=500,
        print_every=500,
        plot_data_overview=True,
        scale_per_group=True,
        use_obs="union",
        use_var="union",
        max_epochs=100000,
        save=True,
        save_path=None,
        init_factors="random",
        init_scale=0.1,
    ):
        """
        Fit the model using the provided data.

        Parameters
        ----------
        data : MuData or dict
            MuData object or dictionary with AnnData objects for each view.
        n_factors : int
            Number of latent factors.
        annotations : dict
            Dictionary with weight annotations for informed views.
        weight_prior : dict or str
            Dictionary with weight priors for each view.
        factor_prior : dict or str
            Dictionary with factor priors for each group.
        likelihoods : dict or str
            Dictionary with likelihoods for each view.
        lr : float
            Learning rate.
        batch_size : int
            Batch size.
        early_stopper_patience : int
            Number of steps without relevant improvement to stop training.
        print_every : int
            Print loss every n steps.
        plot_data_overview: bool
            Plot data overview.
        init_factors : str
            Initialization method for factors.
        init_scale: float
            Initialization scale of Normal distribution for factors.
        **kwargs
            Additional training arguments.
        """
        print("Fitting model...")

        self._setup_data(
            data,
            likelihoods,
            scale_per_group,
            use_obs,
            use_var,
        )

        if plot_data_overview:
            plot_overview(self.data)

        self._setup_annotations(n_factors, annotations)
        self._initialize_factors(init_factors, init_scale)
        self._setup_svi(weight_prior, factor_prior, self.likelihoods, lr)

        # convert AnnData to torch.Tensor objects
        tensor_dict = {}
        for k_groups, v_groups in self.data.items():
            tensor_dict[k_groups] = {}
            for k_views, v_views in v_groups.items():
                tensor_dict[k_groups][k_views] = torch.from_numpy(v_views.X)

        n_samples_total = sum(self.n_samples.values())
        if batch_size is None or not (0 < batch_size <= n_samples_total):
            batch_size = n_samples_total

        if batch_size < n_samples_total:
            batch_fraction = batch_size / n_samples_total

            # has to be a list of data loaders to zip over
            data_loaders = []

            for group_name in self.group_names:
                tensor_dict[group_name] = TensorDict(
                    {view_name: tensor_dict[group_name][view_name] for view_name in self.view_names},
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
                        dict(
                            zip(
                                self.group_names,
                                (batch.to(self.device) for batch in group_batch),
                                strict=False,
                            )
                        )
                    )

                return epoch_loss

        else:
            # move all data to device once
            tensor_dict = self._to_device(tensor_dict)

            def step_fn():
                return self._svi.step(tensor_dict)

        # Train
        self.train_loss_elbo = []
        earlystopper = EarlyStopper(
            mode="min",
            min_delta=0.1,
            patience=early_stopper_patience,
            percentage=True,
        )
        start_timer = time.time()
        for i in range(max_epochs):
            # import ipdb; ipdb.set_trace()
            loss = step_fn()
            self.train_loss_elbo.append(loss)

            if i % print_every == 0:
                print(f"Epoch: {i:>7} | Time: {time.time() - start_timer:>10.2f}s | Loss: {loss:>10.2f}")

            if earlystopper.step(loss):
                print(f"Training finished after {i} steps.")
                break

        self._is_trained = True

        return self._post_fit(save, save_path)

    def _sort_factors(self, weights, factors):
        def _r2(y_true, y_pred):
            ss_res = np.nansum(np.square(y_true - y_pred))
            ss_tot = np.nansum(np.square(y_true))
            return 1.0 - (ss_res / ss_tot)

        # Loop over all groups
        dfs = {}

        for group_name, group_data in self.data.items():
            group_r2 = {}
            for view_name, view_data in group_data.items():
                if self.likelihoods[view_name] == "Normal":
                    group_r2[view_name] = []
                    for k in range(factors[group_name].shape[0]):
                        y_pred = np.outer(factors[group_name][k, :], weights[view_name][k, :])
                        group_r2[view_name].append(_r2(view_data.X, y_pred))

                # TODO: Implement R2 for other likelihoods
                else:
                    print(f"Skipping view {view_name} for group {group_name} as it does not have a Normal likelihood.")

            if len(group_r2) == 0:
                print(f"No R2 values found for group {group_name}. Skipping...")
                continue

            dfs[group_name] = pd.DataFrame(group_r2)
            # Sort by mean R2
            sorted_r2_means = dfs[group_name].mean(axis=1).sort_values(ascending=False)
            # Resort index according to sorted mean R2
            dfs[group_name] = dfs[group_name].loc[sorted_r2_means.index].reset_index(drop=True)

        # TODO: As of now, we only pick the sorting accoring to the last group...
        try:
            factor_index = list(sorted_r2_means.index)
        except NameError:
            print("Sorting factors failed. Using default order.")
            factor_index = list(range(self.n_factors))

        return dfs, factor_index

    def _check_if_trained(self):
        """Check if the model has been trained."""
        if not self._is_trained:
            raise ValueError("Model has not been trained yet. Please train first.")

    def get_factors(self, return_type="numpy"):
        """Get all factor matrices, z_x."""
        factors = {k: v[self.factor_index] for k, v in self._get_factors_from_guide().items()}

        if return_type == "torch":
            factors = {k: torch.tensor(v, dtype=torch.float).clone().detach() for k, v in factors.items()}
        if return_type == "anndata":
            factors = {
                k: ad.AnnData(X=v.T, obs=pd.DataFrame({}, index=self.sample_names[k])) for k, v in factors.items()
            }

        return factors

    def get_weights(self, return_type="numpy"):
        """Get all weight matrices, w_x."""
        weights = {k: v[self.factor_index] for k, v in self._get_weights_from_guide().items()}

        if return_type == "torch":
            weights = {k: torch.tensor(v, dtype=torch.float).clone().detach() for k, v in weights.items()}
        if return_type == "anndata":
            weights = {
                k: ad.AnnData(X=v, var=pd.DataFrame({}, index=self.feature_names[k])) for k, v in weights.items()
            }

        return weights

    def _get_factors_from_guide(self):
        """Get all factor matrices, z_x."""
        self._check_if_trained()

        factors = {k: self.variational.expectation(f"z_{k}").detach() for k in self.group_names}
        if self.generative.factor_prior == "ARD_Spike_and_Slab":
            for k in self.group_names:
                factors[k] *= self.variational.expectation(f"s_z_{k}").detach()
        return {k: f.cpu().numpy().squeeze() for k, f in factors.items()}

    def _get_weights_from_guide(self):
        """Get all weight matrices, w_x."""
        self._check_if_trained()

        weights = {k: self.variational.expectation(f"w_{k}").detach() for k in self.view_names}
        if self.generative.weight_prior == "ARD_Spike_and_Slab":
            for k in self.view_names:
                weights[k] *= self.variational.expectation(f"s_w_{k}").detach()
        return {k: w.cpu().numpy().squeeze() for k, w in weights.items()}

    def _setup_device(self, device):
        print("Setting up device...")
        cuda_available = torch.cuda.is_available()

        device = str(device).lower()
        if "cuda" in device and not cuda_available:
            print(f"- `{device}` not available...")
            device = "cpu"

        if "cuda" in device and cuda_available:
            # Check if device id is given, otherwise use default device
            if ":" in device:
                # Check if device_id is valid
                device_id = int(device.split(":")[1])
                if device_id >= torch.cuda.device_count():
                    print(
                        f"- Device id `{device_id}` not available. Using default device: {torch.cuda.current_device()}"
                    )
                    device = f"cuda:{torch.cuda.current_device()}"
            else:
                print(f"- No device id given. Using default device: {torch.cuda.current_device()}")
                device = f"cuda:{torch.cuda.current_device()}"

            # Set all cuda operations to specific cuda device
            torch.cuda.set_device(device)

        print(f"- Running all computations on `{device}`")
        return torch.device(device)
