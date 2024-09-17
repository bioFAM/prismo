import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.infer.autoguide.guides import deep_getattr, deep_setattr
from pyro.nn import PyroModule, PyroParam

from famo.dist import ReinMaxBernoulli
from famo.gp import GP

EPS = 1e-8


class Generative(PyroModule):
    def __init__(
        self,
        n_samples: dict[str, int],
        n_features: dict[str, int],
        n_factors: int,
        prior_scales=None,
        factor_prior: dict[str, str] | str = "Normal",
        weight_prior: dict[str, str] | str = "Normal",
        likelihoods: dict[str, str] | str = "Normal",
        nonnegative_weights: dict[str, bool] | bool = False,
        nonnegative_factors: dict[str, bool] | bool = False,
        gps: dict[str, GP] = None,
        device: str = None,
        **kwargs,
    ):
        super().__init__("Generative")

        self.group_names = list(n_samples.keys())
        self.view_names = list(n_features.keys())

        if isinstance(factor_prior, str):
            factor_prior = {group_name: factor_prior for group_name in self.group_names}

        if isinstance(weight_prior, str):
            weight_prior = {view_name: weight_prior for view_name in self.view_names}

        if isinstance(likelihoods, str):
            likelihoods = {view_name: likelihoods for view_name in self.view_names}

        if isinstance(nonnegative_weights, bool):
            nonnegative_weights = {view_name: nonnegative_weights for view_name in self.view_names}

        if isinstance(nonnegative_factors, bool):
            nonnegative_factors = {group_name: nonnegative_factors for group_name in self.group_names}

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_factors = n_factors
        self.prior_scales = prior_scales
        self.factor_prior = factor_prior
        self.weight_prior = weight_prior
        self.likelihoods = likelihoods
        self.nonnegative_weights = nonnegative_weights
        self.nonnegative_factors = nonnegative_factors
        self.gps = gps

        self.pos_transform = torch.nn.ReLU()

        self.scale_elbo = True
        n_views = len(self.view_names)
        self.view_scales = {view_name: 1.0 for view_name in self.view_names}
        if self.scale_elbo and n_views > 1:
            for view_name, view_n_features in n_features.items():
                self.view_scales[view_name] = (n_views / (n_views - 1)) * (
                    1.0 - view_n_features / sum(n_features.values())
                )

        self.device = device
        self.to(self.device)

        self._setup_distributions()

        self.sample_dict: dict[str, torch.Tensor] = {}

    def _zeros(self, size):
        return torch.zeros(size, device=self.device)

    def _ones(self, size):
        return torch.ones(size, device=self.device)

    def _get_plates(self, **kwargs):
        plates = {}
        for group_name in self.group_names:
            subsample = kwargs.get("subsample", None)
            if subsample is not None:
                subsample = subsample[group_name]

            plates[f"samples_{group_name}"] = pyro.plate(
                "plate_samples_" + group_name,
                self.n_samples[group_name],
                dim=-1,
                device=self.device,
                subsample=subsample,
            )

        for view_name in self.view_names:
            plates[f"features_{view_name}"] = pyro.plate(
                "plate_features_" + view_name, self.n_features[view_name], dim=-2, device=self.device
            )

        plates["factors"] = pyro.plate("plate_factors", self.n_factors, dim=-3, device=self.device)

        # needs to be at dim=-2 to work with GPyTorch
        plates["gp_batch"] = pyro.plate("gp_batch", self.n_factors, dim=-2, device=self.device)

        return plates

    def _setup_distributions(self):
        self.sample_factors = {}
        for group_name in self.group_names:
            if self.factor_prior[group_name] is None:
                self.factor_prior[group_name] = "Normal"

            if self.factor_prior[group_name] == "Normal":
                self.sample_factors[group_name] = self._sample_factors_normal
            elif self.factor_prior == "Laplace":
                self.sample_factors[group_name] = self._sample_factors_laplace
            elif self.factor_prior[group_name] == "Horseshoe":
                self.sample_factors[group_name] = self._sample_factors_horseshoe
            elif self.factor_prior[group_name] == "SnS":
                self.sample_factors[group_name] = self._sample_factors_sns
            elif self.factor_prior[group_name] == "GP":
                self.sample_factors[group_name] = self._sample_factors_gp
            else:
                raise ValueError(f"Invalid factor_prior: {self.factor_prior[group_name]}")

        self.sample_weights = {}
        for view_name in self.view_names:
            if self.weight_prior[view_name] is None:
                self.weight_prior[view_name] = "Normal"

            if self.weight_prior[view_name] == "Normal":
                self.sample_weights[view_name] = self._sample_weights_normal
            elif self.weight_prior[view_name] == "Laplace":
                self.sample_weights[view_name] = self._sample_weights_laplace
            elif self.weight_prior[view_name] == "Horseshoe":
                self.sample_weights[view_name] = self._sample_weights_horseshoe
            elif self.weight_prior[view_name] == "SnS":
                self.sample_weights[view_name] = self._sample_weights_sns
            else:
                raise ValueError(f"Invalid weight_prior: {self.weight_prior[view_name]}")

        self.dist_obs = {}
        for view_name in self.view_names:
            if self.likelihoods[view_name] is None:
                self.likelihoods[view_name] = "Normal"

            if self.likelihoods[view_name] == "Normal":
                self.dist_obs[view_name] = self._dist_obs_normal
            elif self.likelihoods[view_name] == "GammaPoisson":
                self.dist_obs[view_name] = self._dist_obs_gamma_poisson
            elif self.likelihoods[view_name] == "Bernoulli":
                self.dist_obs[view_name] = self._dist_obs_bernoulli
            elif self.likelihoods[view_name] == "BetaBinomial":
                self.dist_obs[view_name] = self._dist_obs_beta_binomial
            else:
                raise ValueError(f"Invalid likelihood: {self.likelihoods[view_name]}")

        self.sample_dispersion = self._sample_dispersion_gamma

    def _sample_factors_normal(self, group_name, plates, **kwargs):
        with plates["factors"], plates[f"samples_{group_name}"]:
            return pyro.sample(f"z_{group_name}", dist.Normal(self._zeros((1,)), self._ones((1,))))

    def _sample_factors_laplace(self, group_name, plates, **kwargs):
        with plates["factors"], plates[f"samples_{group_name}"]:
            return pyro.sample(f"z_{group_name}", dist.Laplace(self._zeros((1,)), self._ones((1,))))

    def _sample_factors_horseshoe(self, group_name, plates, **kwargs):
        global_scale = pyro.sample(f"global_scale_z_{group_name}", dist.HalfCauchy(self._ones((1,))))
        with plates["factors"]:
            inter_scale = pyro.sample(f"inter_scale_z_{group_name}", dist.HalfCauchy(self._ones((1,))))
            with plates[f"samples_{group_name}"]:
                local_scale = pyro.sample(f"local_scale_z_{group_name}", dist.HalfCauchy(self._ones((1,))))
                local_scale = local_scale * inter_scale * global_scale
                return pyro.sample(f"z_{group_name}", dist.Normal(self._zeros((1,)), self._ones((1,)) * local_scale))

    def _sample_factors_sns(self, group_name, plates, **kwargs):
        with plates["factors"]:
            alpha = pyro.sample(f"alpha_z_{group_name}", dist.Gamma(1e-3 * self._ones((1,)), 1e-3 * self._ones((1,))))
            theta = pyro.sample(f"theta_z_{group_name}", dist.Beta(self._ones((1,)), self._ones((1,))))
            with plates[f"samples_{group_name}"]:
                s = pyro.sample(f"s_z_{group_name}", dist.Bernoulli(theta))
                return pyro.sample(f"z_{group_name}", dist.Normal(0.0, 1.0 / (alpha + EPS))) * s

    def _sample_factors_gp(self, group_name, plates, **kwargs):
        gp = self.gps[group_name]
        pyro.module(f"gp_{group_name}", gp)

        # Inducing values p(u)
        prior_distribution = gp.variational_strategy.prior_distribution
        prior_distribution = prior_distribution.to_event(len(prior_distribution.batch_shape))
        pyro.sample(f"gp_{group_name}.u", prior_distribution)

        # Draw samples from p(f)
        f_dist = gp(kwargs.get("covariates"), prior=True)
        f_dist = dist.Normal(loc=f_dist.mean, scale=f_dist.stddev).to_event(len(f_dist.event_shape) - 1)

        with plates["gp_batch"], plates[f"samples_{group_name}"]:
            f = pyro.sample(f"gp_{group_name}.f", f_dist.mask(False)).unsqueeze(-2)

        eta = gp.covar_module.outputscale.reshape(-1, 1, 1)

        with plates["factors"], plates[f"samples_{group_name}"]:
            return pyro.sample(f"z_{group_name}", dist.Normal(f, (1 - eta).clamp(1e-3, 1 - 1e-3)))

    def _sample_weights_normal(self, view_name, plates, **kwargs):
        with plates["factors"], plates["features_" + view_name]:
            return pyro.sample(f"w_{view_name}", dist.Normal(self._zeros((1,)), self._ones((1,))))

    def _sample_weights_laplace(self, view_name, plates, **kwargs):
        with plates["factors"], plates["features_" + view_name]:
            return pyro.sample(f"w_{view_name}", dist.Laplace(self._zeros((1,)), self._ones((1,))))

    def _sample_weights_horseshoe(self, view_name, plates, **kwargs):
        regularized = kwargs.get("regularized", None)
        prior_scales = kwargs.get("prior_scales", None)

        regularized |= prior_scales is not None

        global_scale = pyro.sample(f"global_scale_w_{view_name}", dist.HalfCauchy(self._ones((1,))))
        with plates["factors"]:
            inter_scale = pyro.sample(f"inter_scale_w_{view_name}", dist.HalfCauchy(self._ones((1,))))
            with plates["features_" + view_name]:
                local_scale = pyro.sample(f"local_scale_w_{view_name}", dist.HalfCauchy(self._ones((1,))))
                local_scale = local_scale * inter_scale * global_scale

                if regularized:
                    caux = self._sample(
                        f"caux_w_{view_name}", dist.InverseGamma(self._ones((1,)) * 0.5, self._ones((1,)) * 0.5)
                    )
                    c = torch.sqrt(caux)
                    if prior_scales is not None:
                        c = c * prior_scales.unsqueeze(-1)
                    local_scale = (c * local_scale) / torch.sqrt(c**2 + local_scale**2)

                return pyro.sample(f"w_{view_name}", dist.Normal(self._zeros((1,)), self._ones((1,)) * local_scale))

    def _sample_weights_sns(self, view_name, plates, **kwargs):
        with plates["factors"]:
            alpha = pyro.sample(f"alpha_w_{view_name}", dist.Gamma(1e-3 * self._ones((1,)), 1e-3 * self._ones((1,))))
            theta = pyro.sample(f"theta_w_{view_name}", dist.Beta(self._ones((1,)), self._ones((1,))))
            with plates["features_" + view_name]:
                s = pyro.sample(f"s_w_{view_name}", dist.Bernoulli(theta))
                return pyro.sample(f"w_{view_name}", dist.Normal(0.0, 1.0 / (alpha + EPS))) * s

    def _sample_dispersion_gamma(self, view_name, plates, **kwargs):
        with plates["features_" + view_name]:
            return pyro.sample(
                f"dispersion_{view_name}", dist.Gamma(1e-10 * self._ones((1,)), 1e-10 * self._ones((1,)))
            )

    def _dist_obs_normal(self, loc, **kwargs):
        view_name = kwargs["view_name"]
        precision = self.sample_dict[f"dispersion_{view_name}"]
        return dist.Normal(loc * self._ones(1), self._ones(1) / (precision + EPS))

    def _dist_obs_gamma_poisson(self, loc, **kwargs):
        view_name = kwargs["view_name"]
        dispersion = self.sample_dict[f"dispersion_{view_name}"]
        rate = self.pos_transform(loc)
        return dist.GammaPoisson(1 / dispersion, 1 / (rate * dispersion + EPS))

    def _dist_obs_bernoulli(self, loc, **kwargs):
        return dist.Bernoulli(logits=loc)

    def _dist_obs_beta_binomial(self, loc, **kwargs):
        obs = kwargs["obs"]
        obs_mask = kwargs["obs_mask"]
        view_name = kwargs["view_name"]
        obs_reshaped = obs.reshape(obs.shape[0] // 2, 2, obs.shape[-1])
        obs_mask_reshaped = obs_mask.reshape(obs.shape[0] // 2, 2, obs_mask.shape[-1])
        obs_total = obs_reshaped.sum(dim=1)
        obs = obs_reshaped[:, 0, :]  # equals success counts

        # add feature to mask if any of paired features is masked
        obs_mask = obs_mask_reshaped.all(dim=1)

        dispersion = self.sample_dict[f"dispersion_{view_name}"]
        probs = torch.nn.Sigmoid()(loc)
        return dist.BetaBinomial(
            concentration1=(dispersion * probs).clamp(EPS),
            concentration0=(dispersion * (1 - probs)).clamp(EPS),
            total_count=obs_total,
        )

    def forward(self, data):
        current_group_names = list(data.keys())

        plates = self._get_plates()

        # sample factors and transform if non-negative is required
        for group_name in current_group_names:
            self.sample_dict[f"z_{group_name}"] = self.sample_factors[group_name](
                group_name, plates, covariates=data[group_name].get("covariates", None)
            )

            if self.nonnegative_factors[group_name]:
                self.sample_dict[f"z_{group_name}"] = self.pos_transform(self.sample_dict[f"z_{group_name}"])

        # sample weights and transform if non-negative is required
        for view_name in self.view_names:
            prior_scales = None
            if self.prior_scales is not None:
                prior_scales = self.prior_scales[view_name]

            self.sample_dict[f"w_{view_name}"] = self.sample_weights[view_name](
                view_name, plates, prior_scales=prior_scales
            )

            if self.nonnegative_weights[view_name]:
                self.sample_dict[f"w_{view_name}"] = self.pos_transform(self.sample_dict[f"w_{view_name}"])

            # sample dispersion parameter
            if self.likelihoods[view_name] in ["Normal", "GammaPoisson", "BetaBinomial"]:
                self.sample_dict[f"dispersion_{view_name}"] = self.sample_dispersion(view_name, plates)

        # sample observations
        for group_name in current_group_names:
            for view_name in self.view_names:
                view_obs = data[group_name][view_name]

                z = self.sample_dict[f"z_{group_name}"]
                w = self.sample_dict[f"w_{view_name}"]

                loc = torch.einsum("...kji,...kji->...ji", z, w)

                obs = view_obs.T
                obs_mask = torch.logical_not(torch.isnan(obs))
                obs = torch.nan_to_num(obs, nan=0)

                dist_parameterized = self.dist_obs[view_name](
                    loc, obs=obs, obs_mask=obs_mask, group_name=group_name, view_name=view_name
                )

                with (
                    pyro.poutine.mask(mask=obs_mask),
                    pyro.poutine.scale(scale=self.view_scales[view_name]),
                    plates[f"features_{view_name}"],
                    plates[f"samples_{group_name}"],
                ):
                    self.sample_dict[f"x_{group_name}_{view_name}"] = pyro.sample(
                        f"x_{group_name}_{view_name}", dist_parameterized, obs=obs
                    )

        return self.sample_dict


class Variational(PyroModule):
    def __init__(
        self,
        generative,
        z_init_tensor: dict = None,
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
        self.z_init_tensor = z_init_tensor

        self.device = self.generative.device
        self.to(self.device)

        self._setup_parameters()
        self._setup_distributions()

        self.sample_dict: dict[str, torch.Tensor] = {}

    # @torch.no_grad()
    # def expectation(self, site_name: str, covariates: torch.Tensor = None, n_gp_samples: int = 100):
    #     if self.site_to_dist[site_name] in ["Normal", "LogNormal"]:
    #         loc, scale = self._get_loc_and_scale(site_name)
    #         expectation = loc
    #         if self.site_to_dist[site_name] == "LogNormal":
    #             expectation = (loc - scale.square()).exp()

    #         for gn in self.generative.n_samples:
    #             if site_name == f"z_{gn}" and self.generative.nonnegative_factors[gn]:
    #                 expectation = self.generative.pos_transform(expectation)

    #         for vn in self.generative.n_features:
    #             if site_name == f"w_{vn}" and self.generative.nonnegative_weights[vn]:
    #                 expectation = self.generative.pos_transform(expectation)

    #         return expectation.clone()

    #     if self.site_to_dist[site_name] == "Gamma":
    #         shape, rate = self._get_shape_and_rate(site_name)
    #         return (shape / rate).clone()

    #     if self.site_to_dist[site_name] == "Bernoulli":
    #         prob = self._get_prob(site_name)
    #         expectation = prob
    #         return expectation.clone()

    #     if self.site_to_dist[site_name] == "Beta":
    #         alpha, beta = self._get_alpha_and_beta(site_name)
    #         expectation = (alpha - 1) / (alpha + beta - 2)
    #         return expectation.clone()

    #     if self.site_to_dist[site_name] == "GP":
    #         gp = self.generative.gps[site_name[2:]]
    #         gp.eval()

    #         with torch.no_grad():
    #             expectation = gp.eval()(covariates.to(self.device))(torch.Size([n_gp_samples])).mean(axis=0)
    #         return expectation.clone()

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

    def _setup_parameters(self):
        """Setup parameters."""
        n_samples = self.generative.n_samples
        n_features = self.generative.n_features
        n_factors = self.generative.n_factors

        # factors variational parameters
        for group_name in self.generative.group_names:
            # if z_init_tensor is provided, use it
            if self.z_init_tensor is not None:
                z_loc_val = self.z_init_tensor[group_name]["loc"].clone()
                z_scale_val = self.z_init_tensor[group_name]["scale"].clone()
            else:
                z_loc_val = self.init_loc * self._ones((n_factors, 1, n_samples[group_name]))
                z_scale_val = self.init_scale * self._ones((n_factors, 1, n_samples[group_name]))

            if self.generative.factor_prior[group_name] == "Normal":
                deep_setattr(self.locs, f"z_{group_name}", PyroParam(z_loc_val, constraint=constraints.real))
                deep_setattr(
                    self.scales, f"z_{group_name}", PyroParam(z_scale_val, constraint=constraints.softplus_positive)
                )

            if self.generative.factor_prior[group_name] == "Laplace":
                deep_setattr(self.locs, f"z_{group_name}", PyroParam(z_loc_val, constraint=constraints.real))
                deep_setattr(
                    self.scales, f"z_{group_name}", PyroParam(z_scale_val, constraint=constraints.softplus_positive)
                )

            if self.generative.factor_prior[group_name] == "Horseshoe":
                deep_setattr(
                    self.locs,
                    f"global_scale_z_{group_name}",
                    PyroParam(self.init_loc * self._ones(1), constraint=constraints.real),
                )
                deep_setattr(
                    self.scales,
                    f"global_scale_z_{group_name}",
                    PyroParam(self.init_scale * self._ones(1), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.locs,
                    f"inter_scale_z_{group_name}",
                    PyroParam(self.init_loc * self._ones((n_factors, 1, 1)), constraint=constraints.real),
                )
                deep_setattr(
                    self.scales,
                    f"inter_scale_z_{group_name}",
                    PyroParam(
                        self.init_scale * self._ones((n_factors, 1, 1)), constraint=constraints.softplus_positive
                    ),
                )

                deep_setattr(
                    self.locs,
                    f"local_scale_z_{group_name}",
                    PyroParam(
                        self.init_loc * self._ones((n_factors, 1, n_samples[group_name])), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"local_scale_z_{group_name}",
                    PyroParam(
                        self.init_scale * self._ones((n_factors, 1, n_samples[group_name])),
                        constraint=constraints.softplus_positive,
                    ),
                )

                deep_setattr(self.locs, f"z_{group_name}", PyroParam(z_loc_val, constraint=constraints.real))
                deep_setattr(
                    self.scales, f"z_{group_name}", PyroParam(z_scale_val, constraint=constraints.softplus_positive)
                )

            if self.generative.factor_prior[group_name] == "SnS":
                deep_setattr(
                    self.shapes,
                    f"alpha_z_{group_name}",
                    PyroParam(
                        self.init_shape * self._ones((n_factors, 1, 1)), constraint=constraints.softplus_positive
                    ),
                )
                deep_setattr(
                    self.rates,
                    f"alpha_z_{group_name}",
                    PyroParam(self.init_rate * self._ones((n_factors, 1, 1)), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.alphas,
                    f"theta_z_{group_name}",
                    PyroParam(
                        self.init_alpha * self._ones((n_factors, 1, 1)), constraint=constraints.softplus_positive
                    ),
                )
                deep_setattr(
                    self.betas,
                    f"theta_z_{group_name}",
                    PyroParam(self.init_beta * self._ones((n_factors, 1, 1)), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.probs,
                    f"s_z_{group_name}",
                    PyroParam(
                        self.init_prob * self._ones((n_factors, 1, n_samples[group_name])),
                        constraint=constraints.unit_interval,
                    ),
                )

                deep_setattr(self.locs, f"z_{group_name}", PyroParam(z_loc_val, constraint=constraints.real))
                deep_setattr(
                    self.scales, f"z_{group_name}", PyroParam(z_scale_val, constraint=constraints.softplus_positive)
                )

        # weights variational parameters
        for view_name in self.generative.view_names:
            if self.generative.weight_prior[view_name] == "Normal":
                deep_setattr(
                    self.locs,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_loc * self._ones((n_factors, n_features[view_name], 1)), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_scale * self._ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.softplus_positive,
                    ),
                )

            if self.generative.weight_prior[view_name] == "Laplace":
                deep_setattr(
                    self.locs,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_loc * self._ones((n_factors, n_features[view_name], 1)), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_scale * self._ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.softplus_positive,
                    ),
                )

            if self.generative.weight_prior[view_name] == "Horseshoe":
                deep_setattr(
                    self.locs,
                    f"global_scale_w_{view_name}",
                    PyroParam(self.init_loc * self._ones(1), constraint=constraints.real),
                )
                deep_setattr(
                    self.scales,
                    f"global_scale_w_{view_name}",
                    PyroParam(self.init_scale * self._ones(1), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.locs,
                    f"inter_scale_w_{view_name}",
                    PyroParam(self.init_loc * self._ones(n_factors, 1, 1), constraint=constraints.real),
                )
                deep_setattr(
                    self.scales,
                    f"inter_scale_w_{view_name}",
                    PyroParam(self.init_scale * self._ones(n_factors, 1, 1), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.locs,
                    f"local_scale_w_{view_name}",
                    PyroParam(
                        self.init_loc * self._ones(n_factors, n_features[view_name], 1), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"local_scale_w_{view_name}",
                    PyroParam(
                        self.init_scale * self._ones(n_factors, n_features[view_name], 1),
                        constraint=constraints.softplus_positive,
                    ),
                )

                deep_setattr(
                    self.locs,
                    f"caux_w_{view_name}",
                    PyroParam(
                        self.init_loc * self._ones(n_factors, n_features[view_name], 1), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"caux_w_{view_name}",
                    PyroParam(
                        self.init_scale * self._ones(n_factors, n_features[view_name], 1),
                        constraint=constraints.softplus_positive,
                    ),
                )

                deep_setattr(
                    self.locs,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_loc * self._ones((n_factors, n_features[view_name], 1)), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_scale * self._ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.softplus_positive,
                    ),
                )

            if self.generative.weight_prior[view_name] == "SnS":
                deep_setattr(
                    self.shapes,
                    f"alpha_w_{view_name}",
                    PyroParam(
                        self.init_shape * self._ones((n_factors, 1, 1)), constraint=constraints.softplus_positive
                    ),
                )
                deep_setattr(
                    self.rates,
                    f"alpha_w_{view_name}",
                    PyroParam(self.init_rate * self._ones((n_factors, 1, 1)), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.alphas,
                    f"theta_w_{view_name}",
                    PyroParam(
                        self.init_alpha * self._ones((n_factors, 1, 1)), constraint=constraints.softplus_positive
                    ),
                )
                deep_setattr(
                    self.betas,
                    f"theta_w_{view_name}",
                    PyroParam(self.init_beta * self._ones((n_factors, 1, 1)), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.probs,
                    f"s_w_{view_name}",
                    PyroParam(
                        self.init_prob * self._ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.unit_interval,
                    ),
                )

                deep_setattr(
                    self.locs,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_loc * self._ones((n_factors, n_features[view_name], 1)), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_scale * self._ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.softplus_positive,
                    ),
                )

        # dispersion variational parameters
        for view_name in self.generative.view_names:
            if self.generative.likelihoods[view_name] in ["Normal", "GammaPoisson", "BetaBinomial"]:
                deep_setattr(
                    self.locs,
                    f"dispersion_{view_name}",
                    PyroParam(self.init_loc * self._ones((1,)), constraint=constraints.real),
                )
                deep_setattr(
                    self.scales,
                    f"dispersion_{view_name}",
                    PyroParam(self.init_scale * self._ones((1,)), constraint=constraints.softplus_positive),
                )

    def _setup_distributions(self):
        # factor_prior
        self.sample_factors = {}
        for group_name in self.generative.group_names:
            if self.generative.factor_prior[group_name] == "Normal":
                self.sample_factors[group_name] = self._sample_factors_normal
            if self.generative.factor_prior[group_name] == "Laplace":
                self.sample_factors[group_name] = self._sample_factors_laplace
            if self.generative.factor_prior[group_name] == "Horseshoe":
                self.sample_factors[group_name] = self._sample_factors_horseshoe
            if self.generative.factor_prior[group_name] == "SnS":
                self.sample_factors[group_name] = self._sample_factors_sns
            if self.generative.factor_prior[group_name] == "GP":
                self.sample_factors[group_name] = self._sample_factors_gp

        # weight_prior
        self.sample_weights = {}
        for view_name in self.generative.view_names:
            if self.generative.weight_prior[view_name] == "Normal":
                self.sample_weights[view_name] = self._sample_weights_normal
            if self.generative.weight_prior[view_name] == "Laplace":
                self.sample_weights[view_name] = self._sample_weights_laplace
            if self.generative.weight_prior[view_name] == "Horseshoe":
                self.sample_weights[view_name] = self._sample_weights_horseshoe
            if self.generative.weight_prior[view_name] == "SnS":
                self.sample_weights[view_name] = self._sample_weights_sns

        # dispersion_prior
        self.sample_dispersion = self._sample_dispersion_lognormal

    def _sample_factors_normal(self, group_name, plates, **kwargs):
        z_loc, z_scale = self._get_loc_and_scale(f"z_{group_name}")
        with plates["factors"], plates[f"samples_{group_name}"] as index:
            if index is not None:
                z_loc = z_loc.index_select(-1, index)
                z_scale = z_scale.index_select(-1, index)
            return pyro.sample(f"z_{group_name}", dist.Normal(z_loc, z_scale))

    def _sample_factors_laplace(self, group_name, plates, **kwargs):
        z_loc, z_scale = self._get_loc_and_scale(f"z_{group_name}")
        with plates["factors"], plates[f"samples_{group_name}"] as index:
            if index is not None:
                z_loc = z_loc.index_select(-1, index)
                z_scale = z_scale.index_select(-1, index)
            return pyro.sample(f"z_{group_name}", dist.Laplace(z_loc, z_scale))

    def _sample_factors_horseshoe(self, group_name, plates, **kwargs):
        global_scale_loc, global_scale_scale = self._get_loc_and_scale(f"global_scale_z_{group_name}")
        pyro.sample(f"global_scale_z_{group_name}", dist.LogNormal(global_scale_loc, global_scale_scale))

        with plates["factors"]:
            inter_scale_loc, inter_scale_scale = self._get_loc_and_scale(f"inter_scale_z_{group_name}")
            pyro.sample(f"inter_scale_z_{group_name}", dist.LogNormal(inter_scale_loc, inter_scale_scale))

            with plates[f"samples_{group_name}"] as index:
                local_scale_loc, local_scale_scale = self._get_loc_and_scale(f"local_scale_z_{group_name}")
                if index is not None:
                    local_scale_loc = local_scale_loc.index_select(-1, index)
                    local_scale_scale = local_scale_scale.index_select(-1, index)
                pyro.sample(f"local_scale_z_{group_name}", dist.LogNormal(local_scale_loc, local_scale_scale))

                z_loc, z_scale = self._get_loc_and_scale(f"z_{group_name}")
                if index is not None:
                    z_loc = z_loc.index_select(-1, index)
                    z_scale = z_scale.index_select(-1, index)
                return pyro.sample(f"z_{group_name}", dist.Normal(z_loc, z_scale))

    def _sample_factors_sns(self, group_name, plates, **kwargs):
        with plates["factors"]:
            alpha_shape, alpha_rate = self._get_shape_and_rate(f"alpha_z_{group_name}")
            pyro.sample(f"alpha_z_{group_name}", dist.Gamma(alpha_shape, alpha_rate))

            theta_alpha, theta_beta = self._get_alpha_and_beta(f"theta_z_{group_name}")
            pyro.sample(f"theta_z_{group_name}", dist.Beta(theta_alpha, theta_beta))

            with plates[f"samples_{group_name}"] as index:
                prob = self._get_prob(f"s_z_{group_name}")
                if index is not None:
                    prob = prob.index_select(-1, index)
                pyro.sample(f"s_z_{group_name}", ReinMaxBernoulli(temperature=2.0, probs=prob))

                z_loc, z_scale = self._get_loc_and_scale(f"z_{group_name}")
                if index is not None:
                    z_loc = z_loc.index_select(-1, index)
                    z_scale = z_scale.index_select(-1, index)
                return pyro.sample(f"z_{group_name}", dist.Normal(z_loc, z_scale))

    def _sample_factors_gp(self, group_name, plates, **kwargs):
        gp = self.generative.gps[group_name]

        # Inducing values q(u)
        variational_distribution = gp.variational_strategy.variational_distribution
        variational_distribution = variational_distribution.to_event(len(variational_distribution.batch_shape))
        pyro.sample(f"gp_{group_name}.u", variational_distribution)

        with plates["gp_batch_plate"], plates[f"samples_{group_name}"] as index:
            # Draw samples from q(f)
            f_dist = gp(kwargs.get("covariates"), prior=False)
            f_dist = dist.Normal(f_dist.mean, f_dist.stddev).to_event(len(f_dist.event_shape) - 1)
            pyro.sample(f"gp_{group_name}.f", f_dist.mask(False))

        with plates["factors"], plates[f"samples_{group_name}"]:
            z_loc, z_scale = self._get_loc_and_scale(f"z_{group_name}")
            if index is not None:
                z_loc = z_loc.index_select(-1, index)
                z_scale = z_scale.index_select(-1, index)
            return pyro.sample(f"z_{group_name}", dist.Normal(z_loc, z_scale))

    def _sample_weights_normal(self, view_name, plates, **kwargs):
        w_loc, w_scale = self._get_loc_and_scale(f"w_{view_name}")
        with plates["factors"], plates[f"features_{view_name}"]:
            return pyro.sample(f"w_{view_name}", dist.Normal(w_loc, w_scale))

    def _sample_weights_laplace(self, view_name, plates, **kwargs):
        w_loc, w_scale = self._get_loc_and_scale(f"w_{view_name}")
        with plates["factors"], plates[f"features_{view_name}"]:
            return pyro.sample(f"w_{view_name}", dist.Laplace(w_loc, w_scale))

    def _sample_weights_horseshoe(self, view_name, plates, **kwargs):
        regularized = kwargs.get("regularized", True)
        regularized |= kwargs.get("prior_scales", None) is not None

        global_scale_loc, global_scale_scale = self._get_loc_and_scale(f"global_scale_w_{view_name}")
        pyro.sample(f"global_scale_w_{view_name}", dist.LogNormal(global_scale_loc, global_scale_scale))

        with plates["factors"]:
            inter_scale_loc, inter_scale_scale = self._get_loc_and_scale(f"inter_scale_w_{view_name}")
            pyro.sample(f"inter_scale_w_{view_name}", dist.LogNormal(inter_scale_loc, inter_scale_scale))

            with plates[f"features_{view_name}"]:
                local_scale_loc, local_scale_scale = self._get_loc_and_scale(f"local_scale_w_{view_name}")
                pyro.sample(f"local_scale_w_{view_name}", dist.LogNormal(local_scale_loc, local_scale_scale))

                if regularized:
                    caux_loc, caux_scale = self._get_loc_and_scale(f"caux_w_{view_name}")
                    pyro.sample(f"caux_w_{view_name}", dist.LogNormal(caux_loc, caux_scale))

                w_loc, w_scale = self._get_loc_and_scale(f"w_{view_name}")
                return pyro.sample(f"w_{view_name}", dist.Normal(w_loc, w_scale))

    def _sample_weights_sns(self, view_name, plates, **kwargs):
        with plates["factors"]:
            alpha_shape, alpha_rate = self._get_shape_and_rate(f"alpha_w_{view_name}")
            pyro.sample(f"alpha_w_{view_name}", dist.Gamma(alpha_shape, alpha_rate))

            theta_alpha, theta_beta = self._get_alpha_and_beta(f"theta_w_{view_name}")
            pyro.sample(f"theta_w_{view_name}", dist.Beta(theta_alpha, theta_beta))

            with plates[f"features_{view_name}"]:
                prob = self._get_prob(f"s_w_{view_name}")
                pyro.sample(f"s_w_{view_name}", ReinMaxBernoulli(temperature=2.0, probs=prob))

                w_loc, w_scale = self._get_loc_and_scale(f"w_{view_name}")
                return pyro.sample(f"w_{view_name}", dist.Normal(w_loc, w_scale))

    def _sample_dispersion_lognormal(self, view_name, plates, **kwargs):
        dispersion_loc, dispersion_scale = self._get_loc_and_scale(f"dispersion_{view_name}")
        with plates[f"features_{view_name}"]:
            return pyro.sample(f"dispersion_{view_name}", dist.LogNormal(dispersion_loc, dispersion_scale))

    def forward(self, data):
        current_group_names = list(data.keys())
        subsample = {group_name: None for group_name in current_group_names}
        for group_name in current_group_names:
            if "sample_idx" in data[group_name].keys():
                subsample[group_name] = data[group_name]["sample_idx"]

        plates = self.generative._get_plates(subsample=subsample)

        for group_name in current_group_names:
            self.sample_dict[f"z_{group_name}"] = self.sample_factors[group_name](
                group_name, plates, covariates=data[group_name].get("covariates", None)
            )

        for view_name in self.generative.view_names:
            self.sample_dict[f"w_{view_name}"] = self.sample_weights[view_name](view_name, plates)

            if self.generative.likelihoods[view_name] in ["Normal", "GammaPoisson", "BetaBinomial"]:
                self.sample_dict[f"dispersion_{view_name}"] = self.sample_dispersion(view_name, plates)

        return self.sample_dict
