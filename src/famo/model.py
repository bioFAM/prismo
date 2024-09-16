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
        n_views = len(n_features)
        self.view_scales = {vn: 1.0 for vn in n_features}
        if self.scale_elbo and n_views > 1:
            for vn, nf in n_features.items():
                self.view_scales[vn] = (n_views / (n_views - 1)) * (1.0 - nf / sum(n_features.values()))

        self.device = device
        self.to(self.device)

        self._setup_samplers()

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
        plates["gp_batch"] = pyro.plate("gp_batch", self.n_samples["gp"], dim=-2, device=self.device)

        return plates

    def _setup_samplers(self):
        group_names = list(self.n_samples.keys())
        view_names = list(self.n_features.keys())

        self.sample_factors = {}
        for group_name in group_names:
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
        for view_name in view_names:
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
        for view_name in view_names:
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

    def _sample_factors_normal(self, group_name, **kwargs):
        plates = self.get_plates()
        with plates["factors"], plates[f"samples_{group_name}"]:
            return pyro.sample(f"z_{group_name}", dist.Normal(self._zeros((1,)), self._ones((1,))))

    def _sample_factors_laplace(self, group_name, **kwargs):
        plates = self.get_plates()
        with plates["factors"], plates[f"samples_{group_name}"]:
            return pyro.sample(f"z_{group_name}", dist.Laplace(self._zeros((1,)), self._ones((1,))))

    def _sample_factors_horseshoe(self, group_name, **kwargs):
        plates = self.get_plates()
        global_scale = pyro.sample(f"global_scale_z_{group_name}", dist.HalfCauchy(self._ones((1,))))
        with plates["factors"]:
            inter_scale = pyro.sample(f"inter_scale_z_{group_name}", dist.HalfCauchy(self._ones((1,))))
            with plates[f"samples_{group_name}"]:
                local_scale = pyro.sample(f"local_scale_z_{group_name}", dist.HalfCauchy(self._ones((1,))))
                local_scale = local_scale * inter_scale * global_scale
                return pyro.sample(f"z_{group_name}", dist.Normal(self._zeros((1,)), self._ones((1,)) * local_scale))

    def _sample_factors_sns(self, group_name, **kwargs):
        plates = self.get_plates()
        with plates["factors"]:
            alpha = pyro.sample(f"alpha_z_{group_name}", dist.Gamma(1e-3 * self._ones((1,)), 1e-3 * self._ones((1,))))
            theta = pyro.sample(f"theta_z_{group_name}", dist.Beta(self._ones((1,)), self._ones((1,))))
            with plates[f"samples_{group_name}"]:
                s = pyro.sample(f"s_z_{group_name}", dist.Bernoulli(theta))
                return pyro.sample(f"z_{group_name}", dist.Normal(0.0, 1.0 / (alpha + EPS))) * s

    def _sample_factors_gp(self, group_name, **kwargs):
        plates = self.get_plates()
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

    def _sample_weights_normal(self, view_name, **kwargs):
        plates = self.get_plates()
        with plates["factors"], plates["features_" + view_name]:
            return pyro.sample(f"w_{view_name}", dist.Normal(self._zeros((1,)), self._ones((1,))))

    def _sample_weights_laplace(self, view_name, **kwargs):
        plates = self.get_plates()
        with plates["factors"], plates["features_" + view_name]:
            return pyro.sample(f"w_{view_name}", dist.Laplace(self._zeros((1,)), self._ones((1,))))

    def _sample_weights_horseshoe(self, view_name, **kwargs):
        plates = self.get_plates()
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

    def _sample_weights_sns(self, view_name, **kwargs):
        plates = self.get_plates()
        with plates["factors"]:
            alpha = pyro.sample(f"alpha_w_{view_name}", dist.Gamma(1e-3 * self._ones((1,)), 1e-3 * self._ones((1,))))
            theta = pyro.sample(f"theta_w_{view_name}", dist.Beta(self._ones((1,)), self._ones((1,))))
            with plates["features_" + view_name]:
                s = pyro.sample(f"s_w_{view_name}", dist.Bernoulli(theta))
                return pyro.sample(f"w_{view_name}", dist.Normal(0.0, 1.0 / (alpha + EPS))) * s

    def _sample_dispersion_gamma(self, view_name, **kwargs):
        plates = self.get_plates()
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
        plates = self.get_plates()
        group_names = list(data.keys())
        view_names = self.n_features.keys()

        # sample factors and transform if non-negative is required
        for group_name in group_names:
            self.sample_dict[f"z_{group_name}"] = self.sample_factors[group_name](
                group_name, covariates=data[group_name].get("covariates", None)
            )

            if self.nonnegative_factors[group_name]:
                self.sample_dict[f"z_{group_name}"] = self.pos_transform(self.sample_dict[f"z_{group_name}"])

        # sample weights and transform if non-negative is required
        for view_name in view_names:
            prior_scales = None
            if self.prior_scales is not None:
                prior_scales = self.prior_scales[view_name]

            self.sample_dict[f"w_{view_name}"] = self.sample_weights[view_name](view_name, prior_scales=prior_scales)

            if self.nonnegative_weights[view_name]:
                self.sample_dict[f"w_{view_name}"] = self.pos_transform(self.sample_dict[f"w_{view_name}"])

            # sample dispersion parameter
            if self.likelihoods[view_name] in ["Normal", "GammaPoisson", "BetaBinomial"]:
                self.sample_dict[f"dispersion_{view_name}"] = self.sample_dispersion(view_name)

        # sample observations
        for group_name in group_names:
            for view_name in view_names:
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
        gp_sites = []

        for gn in n_samples.keys():
            if self.generative.factor_prior[gn] == "Horseshoe":
                name_to_shape[f"global_scale_z_{gn}"] = (1,)
                name_to_shape[f"inter_scale_z_{gn}"] = (n_factors, 1, 1)
                name_to_shape[f"local_scale_z_{gn}"] = (n_factors, 1, n_samples[gn])
                name_to_shape[f"caux_z_{gn}"] = (n_factors, 1, n_samples[gn])
            if self.generative.factor_prior[gn] == "SnS":
                name_to_shape[f"alpha_z_{gn}"] = (n_factors, 1, 1)
                name_to_shape[f"theta_z_{gn}"] = (n_factors, 1, 1)
                name_to_shape[f"s_z_{gn}"] = (n_factors, 1, n_samples[gn])

            name_to_shape[f"z_{gn}"] = (n_factors, 1, n_samples[gn])

            normal_sites.extend([f"z_{gn}"])
            lognormal_sites.extend(
                [f"global_scale_z_{gn}", f"inter_scale_z_{gn}", f"local_scale_z_{gn}", f"caux_z_{gn}"]
            )
            gamma_sites.extend([f"alpha_z_{gn}"])
            bernoulli_sites.extend([f"s_z_{gn}"])
            beta_sites.extend([f"theta_z_{gn}"])
            gp_sites.extend([f"f_{gn}"])

        for vn in n_features.keys():
            if self.generative.weight_prior[vn] == "Horseshoe":
                name_to_shape[f"global_scale_w_{vn}"] = (1,)
                name_to_shape[f"inter_scale_w_{vn}"] = (n_factors, 1, 1)
                name_to_shape[f"local_scale_w_{vn}"] = (n_factors, n_features[vn], 1)
                name_to_shape[f"caux_w_{vn}"] = (n_factors, n_features[vn], 1)
            if self.generative.weight_prior[vn] == "SnS":
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
        site_to_dist.update({k: "GP" for k in gp_sites})

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
                deep_setattr(self.scales, site_name, PyroParam(scale_val, constraints.softplus_positive))
            elif site_name in bernoulli_sites:
                loc_prob_val = self.init_prob * torch.ones(site_shape)
                deep_setattr(self.probs, site_name, PyroParam(loc_prob_val, constraints.unit_interval))
            elif site_name in gamma_sites:
                shape_val = self.init_shape * torch.ones(site_shape)
                rate_val = self.init_rate * torch.ones(site_shape)
                deep_setattr(self.shapes, site_name, PyroParam(shape_val, constraints.softplus_positive))
                deep_setattr(self.rates, site_name, PyroParam(rate_val, constraints.softplus_positive))
            else:
                loc_alpha = self.init_alpha * torch.ones(site_shape)
                loc_beta = self.init_beta * torch.ones(site_shape)
                deep_setattr(self.alphas, site_name, PyroParam(loc_alpha, constraints.softplus_positive))
                deep_setattr(self.betas, site_name, PyroParam(loc_beta, constraints.softplus_positive))

        return site_to_dist

    @torch.no_grad()
    def expectation(self, site_name: str, covariates: torch.Tensor = None, n_gp_samples: int = 100):
        if self.site_to_dist[site_name] in ["Normal", "LogNormal"]:
            loc, scale = self._get_loc_and_scale(site_name)
            expectation = loc
            if self.site_to_dist[site_name] == "LogNormal":
                expectation = (loc - scale.square()).exp()

            for gn in self.generative.n_samples:
                if site_name == f"z_{gn}" and self.generative.nonnegative_factors[gn]:
                    expectation = self.generative.pos_transform(expectation)

            for vn in self.generative.n_features:
                if site_name == f"w_{vn}" and self.generative.nonnegative_weights[vn]:
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

        if self.site_to_dist[site_name] == "GP":
            gp = self.generative.gps[site_name[2:]]
            gp.eval()

            with torch.no_grad():
                expectation = gp.eval()(covariates.to(self.device))(torch.Size([n_gp_samples])).mean(axis=0)
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
        self, site_name, outer_plate, inner_plate, regularized=True, prior_scales=None, **kwargs
    ):
        regularized |= prior_scales is not None

        self._sample(f"global_scale_{site_name}")
        with outer_plate:
            self._sample(f"inter_scale_{site_name}")
            with inner_plate as index:
                self._sample(f"local_scale_{site_name}", index=index, dim=inner_plate.dim)

                if regularized:
                    self._sample(f"caux_{site_name}", index=index, dim=inner_plate.dim)

                return self._sample(site_name, index=index, dim=inner_plate.dim)

    def _sample_component_SnS(self, site_name, outer_plate, inner_plate, **kwargs):
        with outer_plate:
            self._sample(f"alpha_{site_name}")
            self._sample(f"theta_{site_name}")
            with inner_plate as index:
                self._sample(f"s_{site_name}", index=index, dim=inner_plate.dim)
                return self._sample(site_name, index=index, dim=inner_plate.dim)

    def _sample_component_gp(self, site_name, outer_plate, inner_plate, covariates: torch.Tensor = None, **kwargs):
        group_name = site_name[2:]
        gp = self.generative.gps[group_name]

        # Inducing values q(u)
        variational_distribution = gp.variational_strategy.variational_distribution
        variational_distribution = variational_distribution.to_event(len(variational_distribution.batch_shape))
        pyro.sample(f"gp_{group_name}.u", variational_distribution)

        with pyro.plate("gp_batch_plate", dim=-2), inner_plate:
            # Draw samples from q(f)
            f_dist = gp(covariates, prior=False)
            f_dist = dist.Normal(f_dist.mean, f_dist.stddev).to_event(len(f_dist.event_shape) - 1)

            pyro.sample(f"gp_{group_name}.f", f_dist.mask(False))

        z_loc, z_scale = self._get_loc_and_scale(site_name)
        with outer_plate, inner_plate as index:
            return pyro.sample(site_name, dist.Normal(z_loc[..., index], z_scale[..., index]))

    def _setup_samplers(self):
        # factor_prior
        self.sample_factors = {}
        for gn in self.generative.n_samples:
            self.sample_factors[gn] = self._sample_component
            if self.generative.factor_prior[gn] == "Horseshoe":
                self.sample_factors[gn] = self._sample_component_horseshoe
            if self.generative.factor_prior[gn] == "SnS":
                self.sample_factors[gn] = self._sample_component_SnS
            if self.generative.factor_prior[gn] == "GP":
                self.sample_factors[gn] = self._sample_component_gp

        # weight_prior
        self.sample_weights = {}
        for vn in self.generative.n_features:
            self.sample_weights[vn] = self._sample_component
            if self.generative.weight_prior[vn] == "Horseshoe":
                self.sample_weights[vn] = self._sample_component_horseshoe
            if self.generative.weight_prior[vn] == "SnS":
                self.sample_weights[vn] = self._sample_component_SnS

        # dispersion_prior
        self.sample_dispersion = self._sample_vector

    def forward(self, data):
        group_names = list(data.keys())
        subsample = {group_name: None for group_name in group_names}
        for group_name in group_names:
            if "sample_idx" in data[group_name].keys():
                subsample[group_name] = data[group_name]["sample_idx"]

        plates = self.generative.get_plates(subsample=subsample)

        for group_name in group_names:
            self.sample_dict[f"z_{group_name}"] = self.sample_factors[group_name](
                f"z_{group_name}",
                plates["factors"],
                plates[f"samples_{group_name}"],
                covariates=data[group_name].get("covariates", None),
            )

        for view_name in self.generative.n_features:
            self.sample_dict[f"w_{view_name}"] = self.sample_weights[view_name](
                f"w_{view_name}", plates["factors"], plates[f"features_{view_name}"]
            )

            if self.generative.likelihoods[view_name] in ["Normal", "GammaPoisson", "BetaBinomial"]:
                self.sample_dict[f"dispersion_{view_name}"] = self.sample_dispersion(
                    f"dispersion_{view_name}", plates[f"features_{view_name}"]
                )

        return self.sample_dict
