import copy
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.infer.autoguide.guides import deep_getattr, deep_setattr
from pyro.nn import PyroModule, PyroParam

from famo.dist import ReinMaxBernoulli

EPS = 1e-8


class Generative(PyroModule):
    def __init__(
        self,
        n_samples: dict[str, int],
        n_features: dict[str, int],
        n_factors: int,
        prior_scales=None,
        factor_prior="Normal",
        weight_prior="Normal",
        likelihoods="Normal",
        nmf=False,
        device=None,
        **kwargs,
    ):
        super().__init__("Generative")
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_factors = n_factors
        self.prior_scales = prior_scales

        self.factor_prior = factor_prior
        self.weight_prior = weight_prior
        self.likelihoods = likelihoods

        self.nmf = nmf
        self.pos_transform = torch.nn.ReLU()

        self.scale_elbo = True
        n_views = len(n_features)
        self.view_scales = {vn: 1.0 for vn in n_features}
        if self.scale_elbo and n_views > 1:
            for vn, nf in n_features.items():
                self.view_scales[vn] = (n_views / (n_views - 1)) * (
                    1.0 - nf / sum(n_features.values())
                )
        print(self.view_scales)

        self.device = device
        self.to(self.device)

        self._setup_samplers()

        self.sample_dict: dict[str, torch.Tensor] = {}

    def get_plate(self, site_name: str, **kwargs):
        plate_kwargs = {
            "factors": {"name": "plate_factors", "size": self.n_factors, "dim": -3}
        }

        for group_name, group_n_samples in self.n_samples.items():
            plate_kwargs[f"samples_{group_name}"] = {
                "name": f"plate_samples_{group_name}",
                "size": group_n_samples,
                "dim": -1,
            }

        for view_name, view_n_features in self.n_features.items():
            plate_kwargs[f"features_{view_name}"] = {
                "name": f"plate_features_{view_name}",
                "size": view_n_features,
                "dim": -2,
            }

        return pyro.plate(device=self.device, **{**plate_kwargs[site_name], **kwargs})

    def _sample(self, site_name, dist_fn):
        return pyro.sample(site_name, dist_fn)

    def _sample_vector(self, site_name, plate, dist_fn):
        with plate:
            return self._sample(site_name, dist_fn)

    def _sample_component(self, site_name, outer_plate, inner_plate, dist_fn, **kwargs):
        with outer_plate, inner_plate:
            return pyro.sample(site_name, dist_fn)

    def _sample_vector_lognormal(self, site_name, plate, **kwargs):
        return self._sample_vector(
            site_name, plate, dist.LogNormal(self._zeros((1,)), self._ones((1,)))
        )

    def _sample_vector_gamma(self, site_name, plate, **kwargs):
        return self._sample_vector(
            site_name,
            plate,
            dist.Gamma(1e-10 * self._ones((1,)), 1e-10 * self._ones((1,))),
        )

    def _sample_component_normal(self, site_name, outer_plate, inner_plate, **kwargs):
        return self._sample_component(
            site_name,
            outer_plate,
            inner_plate,
            dist.Normal(self._zeros((1,)), self._ones((1,))),
        )

    def _sample_component_laplace(self, site_name, outer_plate, inner_plate, **kwargs):
        return self._sample_component(
            site_name,
            outer_plate,
            inner_plate,
            dist.Laplace(self._zeros((1,)), self._ones((1,))),
        )

    def _sample_component_horseshoe(
        self,
        site_name,
        outer_plate,
        inner_plate,
        regularized=True,
        prior_scales=None,
        **kwargs,
    ):
        regularized |= prior_scales is not None

        global_scale = self._sample(
            f"global_scale_{site_name}", dist.HalfCauchy(self._ones((1,)))
        )
        with outer_plate:
            inter_scale = self._sample(
                f"inter_scale_{site_name}", dist.HalfCauchy(self._ones((1,)))
            )
            with inner_plate:
                local_scale = self._sample(
                    f"local_scale_{site_name}", dist.HalfCauchy(self._ones((1,)))
                )
                local_scale = local_scale * inter_scale * global_scale

                if regularized:
                    caux = self._sample(
                        f"caux_{site_name}",
                        dist.InverseGamma(
                            self._ones((1,)) * 0.5, self._ones((1,)) * 0.5
                        ),
                    )
                    c = torch.sqrt(caux)
                    if prior_scales is not None:
                        c = c * prior_scales.unsqueeze(-1)
                    local_scale = (c * local_scale) / torch.sqrt(c**2 + local_scale**2)

                return self._sample(
                    site_name,
                    dist.Normal(self._zeros((1,)), self._ones((1,)) * local_scale),
                )

    def _sample_component_ard_spike_and_slab(
        self, site_name, outer_plate, inner_plate, **kwargs
    ):
        with outer_plate:
            alpha = self._sample(
                f"alpha_{site_name}",
                dist.Gamma(1e-10 * self._ones((1,)), 1e-10 * self._ones((1,))),
            )
            theta = self._sample(
                f"theta_{site_name}", dist.Beta(self._ones((1,)), self._ones((1,)))
            )
            with inner_plate:
                s = self._sample(f"s_{site_name}", dist.Bernoulli(theta))
                return (
                    self._sample(site_name, dist.Normal(0.0, 1.0 / (alpha + EPS))) * s
                )

    def _setup_samplers(self):
        # factor_prior
        if self.factor_prior is None:
            self.factor_prior = "Normal"

        if self.factor_prior == "Normal":
            self.sample_factors = self._sample_component_normal
        elif self.factor_prior == "Laplace":
            self.sample_factors = self._sample_component_laplace
        elif self.factor_prior == "Horseshoe":
            self.sample_factors = self._sample_component_horseshoe
        elif self.factor_prior == "ARD_Spike_and_Slab":
            self.sample_factors = self._sample_component_ard_spike_and_slab
        else:
            raise ValueError(f"Invalid factor_prior: {self.factor_prior}")

        # weight_prior
        if self.weight_prior is None:
            self.weight_prior = "Normal"

        if self.weight_prior == "Normal":
            self.sample_weights = self._sample_component_normal
        elif self.weight_prior == "Laplace":
            self.sample_weights = self._sample_component_laplace
        elif self.weight_prior == "Horseshoe":
            self.sample_weights = self._sample_component_horseshoe
        elif self.weight_prior == "ARD_Spike_and_Slab":
            self.sample_weights = self._sample_component_ard_spike_and_slab
        else:
            raise ValueError(f"Invalid weight_prior: {self.weight_prior}")

        # dispersion_prior
        self.sample_dispersion = self._sample_vector_gamma

    def _zeros(self, size):
        return torch.zeros(size, device=self.device)

    def _ones(self, size):
        return torch.ones(size, device=self.device)

    def forward(self, data):
        current_group_names = list(data.keys())
        subsample = {group_name: None for group_name in current_group_names}
        for group_name in current_group_names:
            if "sample_idx" in data[group_name].keys():
                subsample[group_name] = data[group_name]["sample_idx"]

        plates = {}
        for group_name in current_group_names:
            plates[f"samples_{group_name}"] = self.get_plate(
                f"samples_{group_name}", subsample=subsample[group_name]
            )

        for view_name in self.n_features:
            plates[f"features_{view_name}"] = self.get_plate(f"features_{view_name}")
        plates["factors"] = self.get_plate("factors")

        for group_name in current_group_names:
            self.sample_dict[f"z_{group_name}"] = self.sample_factors(
                f"z_{group_name}", plates["factors"], plates[f"samples_{group_name}"]
            )

            if any(self.nmf.values()):
                self.sample_dict[f"z_{group_name}"] = self.pos_transform(
                    self.sample_dict[f"z_{group_name}"]
                )

        for view_name in self.n_features:
            prior_scales = None
            if self.prior_scales is not None:
                prior_scales = self.prior_scales[view_name]
            self.sample_dict[f"w_{view_name}"] = self.sample_weights(
                f"w_{view_name}",
                plates["factors"],
                plates[f"features_{view_name}"],
                prior_scales=prior_scales,
            )
            if self.nmf[view_name]:
                self.sample_dict[f"w_{view_name}"] = self.pos_transform(
                    self.sample_dict[f"w_{view_name}"]
                )

            if self.likelihoods[view_name] in [
                "Normal",
                "GammaPoisson",
                "BetaBinomial",
            ]:
                self.sample_dict[f"dispersion_{view_name}"] = self.sample_dispersion(
                    f"dispersion_{view_name}", plates[f"features_{view_name}"]
                )

        for group_name in current_group_names:
            for view_name in self.n_features:
                view_obs = data[group_name][view_name]

                z = self.sample_dict[f"z_{group_name}"]
                w = self.sample_dict[f"w_{view_name}"]

                loc = torch.einsum("...kji,...kji->...ji", z, w)

                site_name = f"x_{group_name}_{view_name}"

                obs = view_obs.T
                obs_mask = torch.logical_not(torch.isnan(obs))
                obs = torch.nan_to_num(obs, nan=0)

                if self.likelihoods[view_name] == "Normal":
                    precision = self.sample_dict[f"dispersion_{view_name}"]
                    dist_parametrized = dist.Normal(
                        loc * self._ones(1), self._ones(1) / (precision + EPS)
                    )

                elif self.likelihoods[view_name] == "GammaPoisson":
                    dispersion = self.sample_dict[f"dispersion_{view_name}"]
                    # TODO: include intercept
                    rate = torch.nn.Softplus()(loc)
                    dist_parametrized = dist.GammaPoisson(
                        1 / dispersion, 1 / (rate * dispersion + EPS)
                    )

                elif self.likelihoods[view_name] == "Bernoulli":
                    # TODO: include intercept
                    dist_parametrized = dist.Bernoulli(logits=loc)

                elif self.likelihoods[view_name] == "BetaBinomial":
                    # pairs of features are sorted and thus can be reshaped to extra dimension
                    obs_reshaped = obs.reshape(obs.shape[0] // 2, 2, obs.shape[-1])
                    obs_mask_reshaped = obs_mask.reshape(
                        obs.shape[0] // 2, 2, obs_mask.shape[-1]
                    )
                    obs_total = obs_reshaped.sum(dim=1)
                    obs = obs_reshaped[:, 0, :]  # equals success counts

                    # add feature to mask if any of paired features is masked
                    obs_mask = obs_mask_reshaped.all(dim=1)

                    dispersion = self.sample_dict[f"dispersion_{view_name}"]
                    # TODO: include intercept
                    probs = torch.nn.Sigmoid()(loc)
                    dist_parametrized = dist.BetaBinomial(
                        concentration1=(dispersion * probs).clamp(EPS),
                        concentration0=(dispersion * (1 - probs)).clamp(EPS),
                        total_count=obs_total,
                    )

                with (
                    pyro.poutine.mask(mask=obs_mask),
                    pyro.poutine.scale(scale=self.view_scales[view_name]),
                    plates[f"samples_{group_name}"],
                    plates[f"features_{view_name}"],
                ):
                    self.sample_dict[site_name] = pyro.sample(
                        site_name,
                        dist_parametrized,
                        obs=obs,
                    )

        return self.sample_dict


class Variational(PyroModule):
    def __init__(
        self,
        generative,
        init_tensor: dict = None,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        init_prob: float = 0.5,
        init_alpha: float = 1.0,
        init_beta: float = 1.0,
        init_shape: float = 10.0,
        init_rate: float = 10.0,
        **kwargs,
    ):
        super().__init__("Variational")
        self.generative = generative
        self.locs = PyroModule()
        self.scales = PyroModule()
        self.probs = PyroModule()
        self.alphas = PyroModule()
        self.betas = PyroModule()
        self.shapes = PyroModule()
        self.rates = PyroModule()

        self.init_loc = init_loc
        self.init_scale = init_scale
        self.init_prob = init_prob
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.init_shape = init_shape
        self.init_rate = init_rate
        self.init_tensor = init_tensor

        self.site_to_dist = self.setup()

        self._setup_samplers()

        self.device = self.generative.device
        self.to(self.device)

        self.sample_dict: dict[str, torch.Tensor] = {}

    def _get_loc_and_scale(self, site_name: str):
        site_loc = deep_getattr(self.locs, site_name)
        site_scale = deep_getattr(self.scales, site_name)
        return site_loc, site_scale

    def _get_prob(self, site_name: str):
        site_prob = deep_getattr(self.probs, site_name)
        return site_prob

    def _get_alpha_and_beta(self, site_name: str):
        site_alpha = deep_getattr(self.alphas, site_name)
        site_beta = deep_getattr(self.betas, site_name)
        return site_alpha, site_beta

    def _get_shape_and_rate(self, site_name: str):
        site_shape = deep_getattr(self.shapes, site_name)
        site_rate = deep_getattr(self.rates, site_name)
        return site_shape, site_rate

    def _zeros(self, size):
        return torch.zeros(size, device=self.device)

    def _ones(self, size):
        return torch.ones(size, device=self.device)

    def setup(self):
        """Setup parameters and sampling sites."""
        n_samples = self.generative.n_samples
        n_features = self.generative.n_features
        n_factors = self.generative.n_factors

        name_to_shape = {}

        normal_sites = []
        lognormal_sites = []
        gamma_sites = []
        bernoulli_sites = []
        beta_sites = []

        for gn in n_samples.keys():
            if self.generative.factor_prior == "Horseshoe":
                name_to_shape[f"global_scale_z_{gn}"] = (1,)
                name_to_shape[f"inter_scale_z_{gn}"] = (n_factors, 1, 1)
                name_to_shape[f"local_scale_z_{gn}"] = (n_factors, 1, n_samples[gn])
                name_to_shape[f"caux_z_{gn}"] = (n_factors, 1, n_samples[gn])
            if self.generative.factor_prior == "ARD_Spike_and_Slab":
                name_to_shape[f"alpha_z_{gn}"] = (n_factors, 1, 1)
                name_to_shape[f"theta_z_{gn}"] = (n_factors, 1, 1)
                name_to_shape[f"s_z_{gn}"] = (n_factors, 1, n_samples[gn])

            name_to_shape[f"z_{gn}"] = (n_factors, 1, n_samples[gn])

            normal_sites.extend([f"z_{gn}"])
            lognormal_sites.extend(
                [
                    f"global_scale_z_{gn}",
                    f"inter_scale_z_{gn}",
                    f"local_scale_z_{gn}",
                    f"caux_z_{gn}",
                ]
            )
            gamma_sites.extend([f"alpha_z_{gn}"])
            bernoulli_sites.extend([f"s_z_{gn}"])
            beta_sites.extend([f"theta_z_{gn}"])

        for vn in n_features.keys():
            if self.generative.weight_prior == "Horseshoe":
                name_to_shape[f"global_scale_w_{vn}"] = (1,)
                name_to_shape[f"inter_scale_w_{vn}"] = (n_factors, 1, 1)
                name_to_shape[f"local_scale_w_{vn}"] = (n_factors, n_features[vn], 1)
                name_to_shape[f"caux_w_{vn}"] = (n_factors, n_features[vn], 1)
            if self.generative.weight_prior == "ARD_Spike_and_Slab":
                name_to_shape[f"alpha_w_{vn}"] = (n_factors, 1, 1)
                name_to_shape[f"theta_w_{vn}"] = (n_factors, 1, 1)
                name_to_shape[f"s_w_{vn}"] = (n_factors, n_features[vn], 1)

            name_to_shape[f"w_{vn}"] = (n_factors, n_features[vn], 1)
            name_to_shape[f"dispersion_{vn}"] = (n_features[vn], 1)

            normal_sites.extend([f"w_{vn}"])
            lognormal_sites.extend(
                [
                    f"global_scale_w_{vn}",
                    f"inter_scale_w_{vn}",
                    f"local_scale_w_{vn}",
                    f"caux_w_{vn}",
                    f"dispersion_{vn}",
                ]
            )
            gamma_sites.extend([f"alpha_w_{vn}"])
            bernoulli_sites.extend([f"s_w_{vn}"])
            beta_sites.extend([f"theta_w_{vn}"])

        site_to_dist = {}
        site_to_dist.update({k: "Normal" for k in normal_sites})
        site_to_dist.update({k: "LogNormal" for k in lognormal_sites})
        site_to_dist.update({k: "Gamma" for k in gamma_sites})
        site_to_dist.update({k: "Bernoulli" for k in bernoulli_sites})
        site_to_dist.update({k: "Beta" for k in beta_sites})

        for site_name, site_shape in name_to_shape.items():
            if site_name in normal_sites + lognormal_sites:
                if (self.init_tensor is not None) and site_name.startswith("z_"):
                    # remove leading "z_"
                    gn = site_name[2:]
                    loc_val = self.init_tensor[gn]["loc"].clone()
                    scale_val = self.init_tensor[gn]["scale"].clone()
                else:
                    loc_val = self.init_loc * torch.ones(site_shape)
                    scale_val = self.init_scale * torch.ones(site_shape)

                deep_setattr(self.locs, site_name, PyroParam(loc_val, constraints.real))
                deep_setattr(
                    self.scales,
                    site_name,
                    PyroParam(scale_val, constraints.softplus_positive),
                )
            elif site_name in bernoulli_sites:
                loc_prob_val = self.init_prob * torch.ones(site_shape)
                deep_setattr(
                    self.probs,
                    site_name,
                    PyroParam(loc_prob_val, constraints.unit_interval),
                )
            elif site_name in gamma_sites:
                shape_val = self.init_shape * torch.ones(site_shape)
                rate_val = self.init_rate * torch.ones(site_shape)
                deep_setattr(
                    self.shapes,
                    site_name,
                    PyroParam(shape_val, constraints.softplus_positive),
                )
                deep_setattr(
                    self.rates,
                    site_name,
                    PyroParam(rate_val, constraints.softplus_positive),
                )
            else:
                loc_alpha = self.init_alpha * torch.ones(site_shape)
                loc_beta = self.init_beta * torch.ones(site_shape)
                deep_setattr(
                    self.alphas,
                    site_name,
                    PyroParam(loc_alpha, constraints.softplus_positive),
                )
                deep_setattr(
                    self.betas,
                    site_name,
                    PyroParam(loc_beta, constraints.softplus_positive),
                )

        return site_to_dist

    @torch.no_grad()
    def expectation(self, site_name: str):
        if self.site_to_dist[site_name] in ["Normal", "LogNormal"]:
            loc, scale = self._get_loc_and_scale(site_name)
            expectation = loc
            if self.site_to_dist[site_name] == "LogNormal":
                expectation = (loc - scale.square()).exp()

            for gn in self.generative.n_samples:
                if site_name == f"z_{gn}" and any(self.generative.nmf.values()):
                    expectation = self.generative.pos_transform(expectation)

            for vn in self.generative.n_features:
                if site_name == f"w_{vn}" and self.generative.nmf[vn]:
                    expectation = self.generative.pos_transform(expectation)

            return expectation.clone()

        if self.site_to_dist[site_name] == "Gamma":
            shape, rate = self._get_shape_and_rate(site_name)
            return (shape / rate).clone()

        if self.site_to_dist[site_name] == "Bernoulli":
            prob = self._get_prob(site_name)
            expectation = prob
            return expectation.clone()

        if self.site_to_dist[site_name] == "Beta":
            alpha, beta = self._get_alpha_and_beta(site_name)
            expectation = (alpha - 1) / (alpha + beta - 2)
            return expectation.clone()

    def _sample(self, site_name, index=None, dim=0):
        if self.site_to_dist[site_name] in ["Normal", "LogNormal"]:
            loc, scale = self._get_loc_and_scale(site_name)
            if index is not None:
                loc = loc.index_select(dim, index)
                scale = scale.index_select(dim, index)
            if self.site_to_dist[site_name] == "LogNormal":
                return pyro.sample(site_name, dist.LogNormal(loc, scale))
            return pyro.sample(site_name, dist.Normal(loc, scale))

        if self.site_to_dist[site_name] == "Gamma":
            shape, rate = self._get_shape_and_rate(site_name)
            if index is not None:
                shape = shape.index_select(dim, index)
                rate = rate.index_select(dim, index)
            return pyro.sample(site_name, dist.Gamma(shape, rate))

        if self.site_to_dist[site_name] == "Bernoulli":
            prob = self._get_prob(site_name)
            if index is not None:
                prob = prob.index_select(dim, index)
            return pyro.sample(site_name, ReinMaxBernoulli(temperature=2.0, probs=prob))

        if self.site_to_dist[site_name] == "Beta":
            alpha, beta = self._get_alpha_and_beta(site_name)
            if index is not None:
                alpha = alpha.index_select(dim, index)
                beta = beta.index_select(dim, index)
            return pyro.sample(site_name, dist.Beta(alpha, beta))

    def _sample_vector(self, site_name, plate, **kwargs):
        with plate as index:
            return self._sample(site_name, index=index, dim=plate.dim)

    def _sample_component(self, site_name, outer_plate, inner_plate, **kwargs):
        with outer_plate:
            with inner_plate as index:
                return self._sample(site_name, index=index, dim=inner_plate.dim)

    def _sample_component_horseshoe(
        self,
        site_name,
        outer_plate,
        inner_plate,
        regularized=True,
        prior_scales=None,
        **kwargs,
    ):
        regularized |= prior_scales is not None

        self._sample(f"global_scale_{site_name}")
        with outer_plate:
            self._sample(f"inter_scale_{site_name}")
            with inner_plate as index:
                self._sample(
                    f"local_scale_{site_name}", index=index, dim=inner_plate.dim
                )

                if regularized:
                    self._sample(f"caux_{site_name}", index=index, dim=inner_plate.dim)

                return self._sample(site_name, index=index, dim=inner_plate.dim)

    def _sample_component_ard_spike_and_slab(
        self, site_name, outer_plate, inner_plate, **kwargs
    ):
        with outer_plate:
            self._sample(f"alpha_{site_name}")
            self._sample(f"theta_{site_name}")
            with inner_plate as index:
                self._sample(f"s_{site_name}", index=index, dim=inner_plate.dim)
                return self._sample(site_name, index=index, dim=inner_plate.dim)

    def _setup_samplers(self):
        # factor_prior
        self.sample_factors = self._sample_component
        if self.generative.factor_prior == "Horseshoe":
            self.sample_factors = self._sample_component_horseshoe
        if self.generative.factor_prior == "ARD_Spike_and_Slab":
            self.sample_factors = self._sample_component_ard_spike_and_slab

        # weight_prior
        self.sample_weights = self._sample_component
        if self.generative.weight_prior == "Horseshoe":
            self.sample_weights = self._sample_component_horseshoe
        if self.generative.weight_prior == "ARD_Spike_and_Slab":
            self.sample_weights = self._sample_component_ard_spike_and_slab

        # dispersion_prior
        self.sample_dispersion = self._sample_vector

    def forward(self, data):
        current_group_names = list(data.keys())
        subsample = {group_name: None for group_name in current_group_names}
        for group_name in current_group_names:
            if "sample_idx" in data[group_name].keys():
                subsample[group_name] = data[group_name]["sample_idx"]

        plates = {}
        for group_name in current_group_names:
            plates[f"samples_{group_name}"] = self.generative.get_plate(
                f"samples_{group_name}", subsample=subsample[group_name]
            )

        for view_name in self.generative.n_features:
            plates[f"features_{view_name}"] = self.generative.get_plate(
                f"features_{view_name}"
            )
        plates["factors"] = self.generative.get_plate("factors")

        for group_name in current_group_names:
            self.sample_dict[f"z_{group_name}"] = self.sample_factors(
                f"z_{group_name}", plates["factors"], plates[f"samples_{group_name}"]
            )

        for view_name in self.generative.n_features:
            self.sample_dict[f"w_{view_name}"] = self.sample_weights(
                f"w_{view_name}", plates["factors"], plates[f"features_{view_name}"]
            )

            if self.generative.likelihoods[view_name] in [
                "Normal",
                "GammaPoisson",
                "BetaBinomial",
            ]:
                self.sample_dict[f"dispersion_{view_name}"] = self.sample_dispersion(
                    f"dispersion_{view_name}", plates[f"features_{view_name}"]
                )

        return self.sample_dict
