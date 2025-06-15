from operator import attrgetter

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.infer.autoguide.guides import deep_setattr
from pyro.nn import PyroModule, PyroParam

from .dist import ReinMaxBernoulli
from .gp import GP
from .likelihoods import Likelihood
from .pyro.likelihoods import PyroLikelihood
from .utils import FactorPrior, MeanStd, WeightPrior

EPS = 1e-8

PyroModuleDict = PyroModule[torch.nn.ModuleDict]


class Generative(PyroModule):
    def __init__(
        self,
        n_samples: dict[str, int],
        n_features: dict[str, int],
        n_factors: int,
        likelihoods: dict[str, Likelihood],
        guiding_vars_names: list[str] | None = None,
        guiding_vars_weight_priors: dict[str, str] | None = None,
        guiding_vars_likelihoods: dict[str, str] | None = None,
        guiding_vars_n_categories: dict[str, int] | None = None,
        guiding_vars_factors: dict[str, int] | None = None,
        guiding_vars_scales: dict[str, float] | None = None,
        prior_scales=None,
        factor_prior: dict[str, FactorPrior] | FactorPrior = "Normal",
        weight_prior: dict[str, WeightPrior] | WeightPrior = "Normal",
        nonnegative_weights: dict[str, bool] | bool = False,
        nonnegative_factors: dict[str, bool] | bool = False,
        feature_means: dict[dict[str, torch.Tensor]] = None,
        sample_means: dict[dict[str, torch.Tensor]] = None,
        gp: GP | None = None,
        gp_group_names: list[str] | None = None,
        **kwargs,
    ):
        super().__init__("Generative")

        self.group_names = tuple(n_samples.keys())
        self.view_names = tuple(n_features.keys())

        if isinstance(factor_prior, str):
            factor_prior = dict.fromkeys(self.group_names, factor_prior)

        if isinstance(weight_prior, str):
            weight_prior = dict.fromkeys(self.view_names, weight_prior)

        if isinstance(nonnegative_weights, bool):
            nonnegative_weights = dict.fromkeys(self.view_names, nonnegative_weights)

        if isinstance(nonnegative_factors, bool):
            nonnegative_factors = dict.fromkeys(self.group_names, nonnegative_factors)

        if isinstance(prior_scales, dict) and len(prior_scales):
            for vn, ps in prior_scales.items():
                self.register_buffer(f"prior_scales_{vn}", torch.as_tensor(ps), persistent=False)

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_factors = n_factors
        self.factor_prior = factor_prior
        self.weight_prior = weight_prior
        self.likelihoods = PyroModuleDict(
            {
                view_name: likelihood.pyro_likelihood(
                    view_name=view_name,
                    sample_dim=self._sample_plate_dim,
                    feature_dim=self._feature_plate_dim,
                    sample_means=sample_means,
                    feature_means=feature_means,
                    is_guiding_var=False,
                )
                for view_name, likelihood in likelihoods.items()
            }
        )
        self.guiding_vars_names = guiding_vars_names if guiding_vars_names is not None else []
        self.guiding_vars_weight_priors = guiding_vars_weight_priors
        self.guiding_vars_likelihoods = PyroModuleDict(
            {
                guiding_var_name: PyroLikelihood(
                    guiding_vars_likelihoods[guiding_var_name],
                    view_name=guiding_var_name,
                    sample_dim=self._sample_plate_dim,
                    feature_dim=self._feature_plate_dim,
                    sample_means=sample_means,
                    feature_means={"dummy_name": {guiding_var_name: torch.zeros(1, 1)}},
                )
                for guiding_var_name in self.guiding_vars_names
            }
        )
        self.guiding_vars_n_categories = guiding_vars_n_categories
        self.guiding_vars_factors = guiding_vars_factors
        self.nonnegative_weights = nonnegative_weights
        self.nonnegative_factors = nonnegative_factors
        self.guiding_vars_scales = guiding_vars_scales

        self.gp = gp
        if self.gp is not None:
            pyro.module("gp", self.gp)

        self.gp_group_names = gp_group_names if gp_group_names is not None else []
        for i, n in enumerate(self.gp_group_names):
            self.register_buffer(f"gp_group_{n}", torch.as_tensor(i))

        self.pos_transform = torch.nn.ReLU()

        self.scale_elbo = True
        n_views = len(self.view_names)
        self.view_scales = dict.fromkeys(self.view_names, 1.0)
        if self.scale_elbo and n_views > 1:
            for view_name, view_n_features in n_features.items():
                self.view_scales[view_name] = (n_views / (n_views - 1)) * (
                    1.0 - view_n_features / sum(n_features.values())
                )

        self._setup_distributions()

        self.sample_dict: dict[str, torch.Tensor] = {}

    def _get_prior_scale(self, view_name: str):
        return getattr(self, f"prior_scales_{view_name}", None)

    def get_gp_group_idx(self, group: str):
        return getattr(self, f"gp_group_{group}")

    _sample_plate_dim = -1
    _feature_plate_dim = -2

    def _get_plates(self, subsample=None):
        plates = {}

        for group_name in self.group_names:
            plates[f"samples_{group_name}"] = pyro.plate(
                f"plate_samples_{group_name}",
                self.n_samples[group_name],
                dim=self._sample_plate_dim,
                subsample=subsample[group_name],
            )

        if len(self.gp_group_names):
            offset = 0
            gp_subsample = []
            for g in self.gp_group_names:
                gp_subsample.append(subsample[g] + offset)
                offset += self.n_samples[g]
            gp_subsample = torch.cat(gp_subsample)

            plates["gp_samples"] = pyro.plate(
                "plate_gp_samples", offset, dim=self._sample_plate_dim, subsample=gp_subsample
            )

            # needs to be at dim=-2 to work with GPyTorch
            plates["gp_batch"] = pyro.plate("gp_batch", self.n_factors, dim=-2)

        for view_name in self.view_names:
            plates[f"features_{view_name}"] = pyro.plate(
                f"plate_features_{view_name}",
                self.n_features[view_name],
                subsample=torch.arange(  # workaround for https://github.com/pyro-ppl/pyro/pull/3405
                    self.n_features[view_name]
                ),
                dim=self._feature_plate_dim,
            )

        for guiding_var_name in self.guiding_vars_names:
            plates[f"guiding_vars_{guiding_var_name}"] = pyro.plate(
                f"plate_guiding_vars_{guiding_var_name}", 1, subsample=torch.arange(1), dim=self._feature_plate_dim
            )

        plates["factors"] = pyro.plate("plate_factors", self.n_factors, dim=-3)

        return plates

    def _setup_distributions(self):
        self.sample_factors = {}
        for group_name in self.group_names:
            if self.factor_prior[group_name] is None:
                self.factor_prior[group_name] = "Normal"

            if self.factor_prior[group_name] == "Normal":
                self.sample_factors[group_name] = self._sample_factors_normal
            elif self.factor_prior[group_name] == "Laplace":
                self.sample_factors[group_name] = self._sample_factors_laplace
            elif self.factor_prior[group_name] == "Horseshoe":
                self.sample_factors[group_name] = self._sample_factors_horseshoe
            elif self.factor_prior[group_name] == "SnS":
                self.sample_factors[group_name] = self._sample_factors_sns
            elif self.factor_prior[group_name] == "GP":
                pass
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

        self.sample_guiding_vars_weights = {}
        for guiding_var_name in self.guiding_vars_names:
            if self.guiding_vars_weight_priors[guiding_var_name] == "Normal":
                self.sample_guiding_vars_weights[guiding_var_name] = self._sample_guiding_vars_weights_normal
            else:
                raise ValueError(
                    f"Invalid guiding_vars_weight_prior: {self.guiding_vars_weight_priors[guiding_var_name]}"
                )

    def _sample_factors_normal(self, group_name, plates, **kwargs):
        with plates["factors"], plates[f"samples_{group_name}"]:
            return pyro.sample(f"z_{group_name}", dist.Normal(torch.zeros((1,)), torch.ones((1,))))

    def _sample_factors_laplace(self, group_name, plates, **kwargs):
        with plates["factors"], plates[f"samples_{group_name}"]:
            return pyro.sample(f"z_{group_name}", dist.Laplace(torch.zeros((1,)), torch.ones((1,))))

    def _sample_factors_horseshoe(self, group_name, plates, regularized=True, **kwargs):
        global_scale = pyro.sample(f"global_scale_z_{group_name}", dist.HalfCauchy(torch.ones((1,))))
        with plates["factors"]:
            inter_scale = pyro.sample(f"inter_scale_z_{group_name}", dist.HalfCauchy(torch.ones((1,))))
            with plates[f"samples_{group_name}"]:
                local_scale = pyro.sample(f"local_scale_z_{group_name}", dist.HalfCauchy(torch.ones((1,))))
                local_scale = local_scale * inter_scale * global_scale

                if regularized:
                    caux = pyro.sample(
                        f"caux_z_{group_name}", dist.InverseGamma(torch.ones((1,)) * 0.5, torch.ones((1,)) * 0.5)
                    )
                    c = torch.sqrt(caux)
                    local_scale = (c * local_scale) / torch.sqrt(c**2 + local_scale**2)

                return pyro.sample(f"z_{group_name}", dist.Normal(torch.zeros((1,)), local_scale))

    def _sample_factors_sns(self, group_name, plates, **kwargs):
        with plates["factors"]:
            alpha = pyro.sample(f"alpha_z_{group_name}", dist.Gamma(1e-3 * torch.ones((1,)), 1e-3 * torch.ones((1,))))
            theta = pyro.sample(f"theta_z_{group_name}", dist.Beta(torch.ones((1,)), torch.ones((1,))))
            with plates[f"samples_{group_name}"]:
                s = pyro.sample(f"s_z_{group_name}", dist.Bernoulli(theta))
                return pyro.sample(f"z_{group_name}", dist.Normal(torch.zeros(1), torch.ones(1) / (alpha + EPS))) * s

    def _sample_factors_gp(self, plates, group_idx, covariates, **kwargs):
        gp = self.gp

        # Inducing values p(u)
        prior_distribution = gp.variational_strategy.prior_distribution
        prior_distribution = prior_distribution.to_event(len(prior_distribution.batch_shape))
        pyro.sample("gp.u", prior_distribution)

        # Draw samples from p(f)
        f_dist = gp(group_idx[..., None], covariates, prior=True)
        f_dist = dist.Normal(loc=f_dist.mean, scale=f_dist.stddev).to_event(len(f_dist.event_shape) - 1)

        with plates["gp_batch"]:
            f = pyro.sample("gp.f", f_dist.mask(False)).unsqueeze(-2)

        outputscale = gp.outputscale.reshape(-1, 1, 1)

        with plates["factors"]:
            return pyro.sample("z", dist.Normal(f, (1 - outputscale).clamp(1e-3, 1 - 1e-3)))

    def _sample_weights_normal(self, view_name, plates, **kwargs):
        with plates["factors"], plates[f"features_{view_name}"]:
            return pyro.sample(f"w_{view_name}", dist.Normal(torch.zeros((1,)), torch.ones((1,))))

    def _sample_weights_laplace(self, view_name, plates, **kwargs):
        with plates["factors"], plates[f"features_{view_name}"]:
            return pyro.sample(f"w_{view_name}", dist.Laplace(torch.zeros((1,)), torch.ones((1,))))

    def _sample_weights_horseshoe(self, view_name, plates, regularized=True, prior_scales=None, **kwargs):
        regularized |= prior_scales is not None

        global_scale = pyro.sample(f"global_scale_w_{view_name}", dist.HalfCauchy(torch.ones((1,))))
        with plates["factors"]:
            inter_scale = pyro.sample(f"inter_scale_w_{view_name}", dist.HalfCauchy(torch.ones((1,))))
            with plates[f"features_{view_name}"]:
                local_scale = pyro.sample(f"local_scale_w_{view_name}", dist.HalfCauchy(torch.ones((1,))))
                local_scale = local_scale * inter_scale * global_scale

                if regularized:
                    caux = pyro.sample(
                        f"caux_w_{view_name}", dist.InverseGamma(torch.ones((1,)) * 0.5, torch.ones((1,)) * 0.5)
                    )
                    c = torch.sqrt(caux)
                    if prior_scales is not None:
                        c = c * prior_scales[..., None]
                    local_scale = (c * local_scale) / torch.sqrt(c**2 + local_scale**2)

                return pyro.sample(f"w_{view_name}", dist.Normal(torch.zeros((1,)), local_scale))

    def _sample_weights_sns(self, view_name, plates, **kwargs):
        with plates["factors"]:
            alpha = pyro.sample(f"alpha_w_{view_name}", dist.Gamma(1e-3 * torch.ones((1,)), 1e-3 * torch.ones((1,))))
            theta = pyro.sample(f"theta_w_{view_name}", dist.Beta(torch.ones((1,)), torch.ones((1,))))
            with plates[f"features_{view_name}"]:
                s = pyro.sample(f"s_w_{view_name}", dist.Bernoulli(theta))
                return pyro.sample(f"w_{view_name}", dist.Normal(0.0, 1.0 / (alpha + EPS))) * s

    def _sample_guiding_vars_weights_normal(self, guiding_var_name, **kwargs):
        weights_dim = self.guiding_vars_n_categories[guiding_var_name]
        return pyro.sample(
            f"guiding_vars_w_{guiding_var_name}",
            dist.Normal(torch.zeros(weights_dim, 2), torch.ones(weights_dim, 2)).to_event(
                2
            ),  # (categories, intercept & slope)
        )

    def forward(self, data, sample_idx, nonmissing_samples, nonmissing_features, covariates, guiding_vars):
        current_gp_groups = {g: self.get_gp_group_idx(g) for g in self.gp_group_names if g in data}
        current_group_names = tuple(k for k in data.keys() if k not in current_gp_groups)

        plates = self._get_plates(subsample=sample_idx)

        # sample non-GP factors
        for group_name in current_group_names:
            self.sample_dict[f"z_{group_name}"] = self.sample_factors[group_name](
                group_name, plates, covariates=covariates.get(group_name, None)
            )

        # sample GP factors
        if len(current_gp_groups):
            factors = self._sample_factors_gp(
                plates,
                covariates=torch.cat(tuple(covariates[g] for g in current_gp_groups.keys()), dim=0),
                group_idx=torch.cat(
                    tuple(torch.as_tensor(i).expand(covariates[g].shape[0]) for g, i in current_gp_groups.items()),
                    dim=0,
                ),
            )
            factors = torch.split(factors, tuple(covariates[g].shape[0] for g in current_gp_groups.keys()), dim=-1)
            for group_name, factor in zip(current_gp_groups.keys(), factors, strict=True):
                self.sample_dict[f"z_{group_name}"] = factor

        # non-negative transform
        for group_name in data.keys():
            if self.nonnegative_factors[group_name]:
                self.sample_dict[f"z_{group_name}"] = self.pos_transform(self.sample_dict[f"z_{group_name}"])

        # sample weights and transform if non-negative is required
        for view_name in self.view_names:
            prior_scales = self._get_prior_scale(view_name)
            self.sample_dict[f"w_{view_name}"] = self.sample_weights[view_name](
                view_name, plates, prior_scales=prior_scales
            )

            if self.nonnegative_weights[view_name]:
                self.sample_dict[f"w_{view_name}"] = self.pos_transform(self.sample_dict[f"w_{view_name}"])

        # sample guiding variable weights
        for guiding_var_name in self.guiding_vars_names:
            self.sample_dict[f"w_guiding_vars_{guiding_var_name}"] = self.sample_guiding_vars_weights[guiding_var_name](
                guiding_var_name
            )

        # sample observations
        for group_name, group in data.items():
            gnonmissing_samples = nonmissing_samples[group_name]
            gnonmissing_features = nonmissing_features[group_name]
            for view_name, view_obs in group.items():
                if view_obs.numel() == 0:  # can occur in the last batch of an epoch if the batch is small
                    continue

                vnonmissing_samples = gnonmissing_samples[view_name]
                vnonmissing_features = gnonmissing_features[view_name]

                z = self.sample_dict[f"z_{group_name}"][..., vnonmissing_samples]
                w = self.sample_dict[f"w_{view_name}"][..., vnonmissing_features, :]

                loc = torch.einsum("...ijk,...ilj->...jlk", z, w)

                obs = view_obs.T
                self.likelihoods[view_name].model(
                    data=obs,
                    estimate=loc,
                    group_name=group_name,
                    scale=self.view_scales[view_name],
                    sample_plate=plates[f"samples_{group_name}"],
                    feature_plate=plates[f"features_{view_name}"],
                    nonmissing_samples=vnonmissing_samples,
                    nonmissing_features=vnonmissing_features,
                )

            # guiding variables
            for guiding_var_name in self.guiding_vars_names:
                if group_name not in guiding_vars[guiding_var_name]:
                    continue

                z_guiding = self.sample_dict[f"z_{group_name}"][self.guiding_vars_factors[guiding_var_name], 0]
                w_guiding = self.sample_dict[f"w_guiding_vars_{guiding_var_name}"]

                # (n_cats, 1) + (n_cats, 1) * (n_samples,)
                loc = w_guiding[:, 0, None] + w_guiding[:, 1, None] * z_guiding  # (n_cats, n_samples)
                obs_guiding_vars = guiding_vars[guiding_var_name][group_name].squeeze(-1)

                self.guiding_vars_likelihoods[guiding_var_name].model(
                    data=obs_guiding_vars,
                    estimate=loc,
                    group_name=group_name,
                    scale=self.guiding_vars_scales[guiding_var_name],
                    sample_plate=plates[f"samples_{group_name}"],
                    feature_plate=plates[f"guiding_vars_{guiding_var_name}"],
                    nonmissing_samples=slice(None),
                    nonmissing_features=slice(None),
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
        init_shape: float = 10,
        init_rate: float = 10,
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

        self._setup_parameters()
        self._setup_distributions()

        self.sample_dict: dict[str, torch.Tensor] = {}

    def _get_loc_and_scale(self, site_name):
        site_loc = attrgetter(site_name)(self.locs)
        site_scale = attrgetter(site_name)(self.scales)
        return MeanStd(site_loc, site_scale)

    def _get_gp_loc_and_scale(self, group: str | None = None):
        if not len(self.generative.gp_group_names):
            return {}, {}

        loc = attrgetter("z_gp")(self.locs)
        scale = attrgetter("z_gp")(self.scales)

        gp_group_sizes = [self.generative.n_samples[g] for g in self.generative.gp_group_names]
        if group is not None:
            gp_group_offsets = torch.as_tensor([0] + gp_group_sizes).cumsum()
            group_idx = self.generative.get_gp_group_idx(group)
            offset = slice(gp_group_offsets[group_idx], gp_group_offsets[group_idx + 1])
            site_loc = offset
            site_scale = scale[..., offset]
        else:
            site_loc = dict(zip(self.generative.gp_group_names, torch.split(loc, gp_group_sizes, dim=-1), strict=False))
            site_scale = dict(
                zip(self.generative.gp_group_names, torch.split(scale, gp_group_sizes, dim=-1), strict=False)
            )

        return MeanStd(site_loc, site_scale)

    def _get_prob(self, site_name: str):
        site_prob = attrgetter(site_name)(self.probs)
        return site_prob

    def _get_alpha_and_beta(self, site_name: str):
        site_alpha = attrgetter(site_name)(self.alphas)
        site_beta = attrgetter(site_name)(self.betas)
        return site_alpha, site_beta

    def _get_shape_and_rate(self, site_name: str):
        site_shape = attrgetter(site_name)(self.shapes)
        site_rate = attrgetter(site_name)(self.rates)
        return site_shape, site_rate

    def _setup_parameters(self):
        """Setup parameters."""
        n_samples = self.generative.n_samples
        n_features = self.generative.n_features
        n_factors = self.generative.n_factors

        n_gp_samples = sum(n_samples[g] for g in self.generative.gp_group_names)

        gp_z_loc_val = self.init_loc * torch.ones((n_factors, 1, n_gp_samples))
        gp_z_scale_val = self.init_scale * torch.ones((n_factors, 1, n_gp_samples))

        if n_gp_samples:
            deep_setattr(self.locs, "z_gp", PyroParam(gp_z_loc_val, constraint=constraints.real))
            deep_setattr(self.scales, "z_gp", PyroParam(gp_z_scale_val, constraint=constraints.softplus_positive))

        # factors variational parameters
        for group_name in self.generative.group_names:
            if group_name in self.generative.gp_group_names:
                continue
            # if z_init_tensor is provided, use it
            if self.z_init_tensor is not None:
                z_loc_val = self.z_init_tensor[group_name]["loc"].clone()
                z_scale_val = self.z_init_tensor[group_name]["scale"].clone()
            else:
                z_loc_val = self.init_loc * torch.ones((n_factors, 1, n_samples[group_name]))
                z_scale_val = self.init_scale * torch.ones((n_factors, 1, n_samples[group_name]))

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
                    PyroParam(self.init_loc * torch.ones((1,)), constraint=constraints.real),
                )
                deep_setattr(
                    self.scales,
                    f"global_scale_z_{group_name}",
                    PyroParam(self.init_scale * torch.ones((1,)), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.locs,
                    f"inter_scale_z_{group_name}",
                    PyroParam(self.init_loc * torch.ones((n_factors, 1, 1)), constraint=constraints.real),
                )
                deep_setattr(
                    self.scales,
                    f"inter_scale_z_{group_name}",
                    PyroParam(
                        self.init_scale * torch.ones((n_factors, 1, 1)), constraint=constraints.softplus_positive
                    ),
                )

                deep_setattr(
                    self.locs,
                    f"local_scale_z_{group_name}",
                    PyroParam(
                        self.init_loc * torch.ones((n_factors, 1, n_samples[group_name])), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"local_scale_z_{group_name}",
                    PyroParam(
                        self.init_scale * torch.ones((n_factors, 1, n_samples[group_name])),
                        constraint=constraints.softplus_positive,
                    ),
                )

                deep_setattr(
                    self.locs,
                    f"caux_z_{group_name}",
                    PyroParam(
                        self.init_loc * torch.ones((n_factors, 1, n_samples[group_name])), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"caux_z_{group_name}",
                    PyroParam(
                        self.init_scale * torch.ones((n_factors, 1, n_samples[group_name])),
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
                        self.init_shape * torch.ones((n_factors, 1, 1)), constraint=constraints.softplus_positive
                    ),
                )
                deep_setattr(
                    self.rates,
                    f"alpha_z_{group_name}",
                    PyroParam(self.init_rate * torch.ones((n_factors, 1, 1)), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.alphas,
                    f"theta_z_{group_name}",
                    PyroParam(
                        self.init_alpha * torch.ones((n_factors, 1, 1)), constraint=constraints.softplus_positive
                    ),
                )
                deep_setattr(
                    self.betas,
                    f"theta_z_{group_name}",
                    PyroParam(self.init_beta * torch.ones((n_factors, 1, 1)), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.probs,
                    f"s_z_{group_name}",
                    PyroParam(
                        self.init_prob * torch.ones((n_factors, 1, n_samples[group_name])),
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
                        self.init_loc * torch.ones((n_factors, n_features[view_name], 1)), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_scale * torch.ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.softplus_positive,
                    ),
                )

            if self.generative.weight_prior[view_name] == "Laplace":
                deep_setattr(
                    self.locs,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_loc * torch.ones((n_factors, n_features[view_name], 1)), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_scale * torch.ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.softplus_positive,
                    ),
                )

            if self.generative.weight_prior[view_name] == "Horseshoe":
                deep_setattr(
                    self.locs,
                    f"global_scale_w_{view_name}",
                    PyroParam(self.init_loc * torch.ones((1,)), constraint=constraints.real),
                )
                deep_setattr(
                    self.scales,
                    f"global_scale_w_{view_name}",
                    PyroParam(self.init_scale * torch.ones((1,)), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.locs,
                    f"inter_scale_w_{view_name}",
                    PyroParam(self.init_loc * torch.ones((n_factors, 1, 1)), constraint=constraints.real),
                )
                deep_setattr(
                    self.scales,
                    f"inter_scale_w_{view_name}",
                    PyroParam(
                        self.init_scale * torch.ones((n_factors, 1, 1)), constraint=constraints.softplus_positive
                    ),
                )

                deep_setattr(
                    self.locs,
                    f"local_scale_w_{view_name}",
                    PyroParam(
                        self.init_loc * torch.ones((n_factors, n_features[view_name], 1)), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"local_scale_w_{view_name}",
                    PyroParam(
                        self.init_scale * torch.ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.softplus_positive,
                    ),
                )

                deep_setattr(
                    self.locs,
                    f"caux_w_{view_name}",
                    PyroParam(
                        self.init_loc * torch.ones((n_factors, n_features[view_name], 1)), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"caux_w_{view_name}",
                    PyroParam(
                        self.init_scale * torch.ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.softplus_positive,
                    ),
                )

                deep_setattr(
                    self.locs,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_loc * torch.ones((n_factors, n_features[view_name], 1)), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_scale * torch.ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.softplus_positive,
                    ),
                )

            if self.generative.weight_prior[view_name] == "SnS":
                deep_setattr(
                    self.shapes,
                    f"alpha_w_{view_name}",
                    PyroParam(
                        self.init_shape * torch.ones((n_factors, 1, 1)), constraint=constraints.softplus_positive
                    ),
                )
                deep_setattr(
                    self.rates,
                    f"alpha_w_{view_name}",
                    PyroParam(self.init_rate * torch.ones((n_factors, 1, 1)), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.alphas,
                    f"theta_w_{view_name}",
                    PyroParam(
                        self.init_alpha * torch.ones((n_factors, 1, 1)), constraint=constraints.softplus_positive
                    ),
                )
                deep_setattr(
                    self.betas,
                    f"theta_w_{view_name}",
                    PyroParam(self.init_beta * torch.ones((n_factors, 1, 1)), constraint=constraints.softplus_positive),
                )

                deep_setattr(
                    self.probs,
                    f"s_w_{view_name}",
                    PyroParam(
                        self.init_prob * torch.ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.unit_interval,
                    ),
                )

                deep_setattr(
                    self.locs,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_loc * torch.ones((n_factors, n_features[view_name], 1)), constraint=constraints.real
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"w_{view_name}",
                    PyroParam(
                        self.init_scale * torch.ones((n_factors, n_features[view_name], 1)),
                        constraint=constraints.softplus_positive,
                    ),
                )

        # guiding variables variational parameters
        for guiding_var_name in self.generative.guiding_vars_names:
            if self.generative.guiding_vars_weight_priors[guiding_var_name] == "Normal":
                deep_setattr(
                    self.locs,
                    f"guiding_vars_w_{guiding_var_name}",
                    PyroParam(
                        torch.full([self.generative.guiding_vars_n_categories[guiding_var_name], 2], self.init_loc),
                        constraint=constraints.real,
                    ),
                )
                deep_setattr(
                    self.scales,
                    f"guiding_vars_w_{guiding_var_name}",
                    PyroParam(
                        torch.full([self.generative.guiding_vars_n_categories[guiding_var_name], 2], self.init_scale),
                        constraint=constraints.softplus_positive,
                    ),
                )

            if self.generative.guiding_vars_likelihoods[guiding_var_name] == "Normal":
                deep_setattr(
                    self.locs,
                    f"guiding_vars_dispersion_{guiding_var_name}",
                    PyroParam(self.init_loc * torch.ones([1]), constraint=constraints.real),
                )

                deep_setattr(
                    self.scales,
                    f"guiding_vars_dispersion_{guiding_var_name}",
                    PyroParam(self.init_scale * torch.ones([1]), constraint=constraints.positive),
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

        # guiding variables
        self.sample_guiding_vars_weights = {}
        for guiding_var_name in self.generative.guiding_vars_names:
            if self.generative.guiding_vars_weight_priors[guiding_var_name] == "Normal":
                self.sample_guiding_vars_weights[guiding_var_name] = self._sample_guiding_vars_weights_normal

            if self.generative.guiding_vars_likelihoods[guiding_var_name] == "Normal":
                self.sample_guiding_vars_dispersion = self._sample_guiding_vars_dispersion

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

    def _sample_factors_horseshoe(self, group_name, plates, regularized=True, **kwargs):
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

                if regularized:
                    caux_loc, caux_scale = (arr[..., index] for arr in self._get_loc_and_scale(f"caux_z_{group_name}"))
                    pyro.sample(f"caux_z_{group_name}", dist.LogNormal(caux_loc, caux_scale))

                z_loc, z_scale = (arr[..., index] for arr in self._get_loc_and_scale(f"z_{group_name}"))
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

    def _sample_factors_gp(self, plates, group_idx, covariates, **kwargs):
        gp = self.generative.gp

        # Inducing values q(u)
        variational_distribution = gp.variational_strategy.variational_distribution
        variational_distribution = variational_distribution.to_event(len(variational_distribution.batch_shape))
        pyro.sample("gp.u", variational_distribution)

        with plates["gp_batch"]:
            # Draw samples from q(f)
            f_dist = gp(group_idx[..., None], covariates, prior=False)
            f_dist = dist.Normal(f_dist.mean, f_dist.stddev).to_event(len(f_dist.event_shape) - 1)
            pyro.sample("gp.f", f_dist.mask(False))

        with plates["factors"], plates["gp_samples"] as index:
            z_loc, z_scale = self._get_loc_and_scale("z_gp")
            if index is not None:
                z_loc = z_loc.index_select(-1, index)
                z_scale = z_scale.index_select(-1, index)
            return pyro.sample("z", dist.Normal(z_loc, z_scale))

    def _sample_weights_normal(self, view_name, plates, **kwargs):
        w_loc, w_scale = self._get_loc_and_scale(f"w_{view_name}")
        with plates["factors"], plates[f"features_{view_name}"]:
            return pyro.sample(f"w_{view_name}", dist.Normal(w_loc, w_scale))

    def _sample_weights_laplace(self, view_name, plates, **kwargs):
        w_loc, w_scale = self._get_loc_and_scale(f"w_{view_name}")
        with plates["factors"], plates[f"features_{view_name}"]:
            return pyro.sample(f"w_{view_name}", dist.Laplace(w_loc, w_scale))

    def _sample_weights_horseshoe(self, view_name, plates, regularized=True, prior_scales=None, **kwargs):
        regularized |= prior_scales is not None

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

    def _sample_guiding_vars_weights_normal(self, guiding_var_name, plates, **kwargs):
        w_loc, w_scale = self._get_loc_and_scale(f"guiding_vars_w_{guiding_var_name}")
        return pyro.sample(f"guiding_vars_w_{guiding_var_name}", dist.Normal(w_loc, w_scale).to_event(2))

    def _sample_guiding_vars_dispersion(self, guiding_var_name, plates, **kwargs):
        dispersion_loc, dispersion_scale = self._get_loc_and_scale(f"guiding_vars_dispersion_{guiding_var_name}")
        return pyro.sample(
            f"guiding_vars_dispersion_{guiding_var_name}", dist.LogNormal(dispersion_loc, dispersion_scale)
        )

    def forward(self, data, sample_idx, nonmissing_samples, nonmissing_features, covariates, guiding_vars):
        current_gp_groups = {
            g: self.generative.get_gp_group_idx(g) for g in self.generative.gp_group_names if g in data
        }
        current_group_names = tuple(k for k in data.keys() if k not in current_gp_groups)

        plates = self.generative._get_plates(subsample=sample_idx)

        for group_name in current_group_names:
            self.sample_dict[f"z_{group_name}"] = self.sample_factors[group_name](
                group_name, plates, covariates=covariates.get(group_name, None)
            )

        if len(current_gp_groups):
            factors = self._sample_factors_gp(
                plates,
                covariates=torch.cat(tuple(covariates[g] for g in current_gp_groups.keys()), dim=0),
                group_idx=torch.cat(
                    tuple(torch.as_tensor(i).expand(covariates[g].shape[0]) for g, i in current_gp_groups.items()),
                    dim=0,
                ),
            )
            factors = torch.split(factors, tuple(covariates[g].shape[0] for g in current_gp_groups.keys()), dim=-1)
            for group_name, factor in zip(current_gp_groups.keys(), factors, strict=True):
                self.sample_dict[f"z_{group_name}"] = factor

        for view_name in self.generative.view_names:
            self.sample_dict[f"w_{view_name}"] = self.sample_weights[view_name](view_name, plates)

        for guiding_var_name in self.generative.guiding_vars_names:
            self.sample_dict[f"guiding_vars_w_{guiding_var_name}"] = self.sample_guiding_vars_weights[guiding_var_name](
                guiding_var_name, plates
            )

        for group_name, group in data.items():
            for view_name in group.keys():
                self.generative.likelihoods[view_name].guide(
                    group_name, plates[f"samples_{group_name}"], plates[f"features_{view_name}"]
                )

            for guiding_var_name in self.generative.guiding_vars_names:
                if group_name in guiding_vars[guiding_var_name]:
                    self.generative.guiding_vars_likelihoods[guiding_var_name].guide(
                        group_name, plates[f"samples_{group_name}"], plates[f"guiding_vars_{guiding_var_name}"]
                    )

        return self.sample_dict

    def get_lr_func(self, base_lr: float, **kwargs):
        sns_params = {
            f"s_w_{view_name}" for view_name, view_prior in self.generative.weight_prior.items() if view_prior == "SnS"
        } | {
            f"s_z_{group_name}"
            for group_name, group_prior in self.generative.factor_prior.items()
            if group_prior == "SnS"
        }

        def lr_func(param_name):
            idx = param_name.rfind(".")
            if idx > -1:
                param_name = param_name[idx + 1 :]
            lr = base_lr
            if param_name in sns_params:
                lr *= 10
            return dict(lr=lr, **kwargs)

        return lr_func

    @torch.inference_mode()
    def get_factors(self):
        """Get all factor matrices, z_x."""
        factors = MeanStd({}, {})

        for lsidx, vals in enumerate(self._get_gp_loc_and_scale()):
            for group_name, fac in vals.items():
                factors[lsidx][group_name] = fac
        for group_name in self.generative.group_names:
            if group_name not in self.generative.gp_group_names:
                for lsidx, vals in enumerate(self._get_loc_and_scale(f"z_{group_name}")):
                    factors[lsidx][group_name] = vals

            if self.generative.nonnegative_factors[group_name]:
                factors.mean[group_name] = self.generative.pos_transform(factors.mean[group_name])
            for lsidx in range(2):
                factors[lsidx][group_name] = factors[lsidx][group_name].cpu().numpy().squeeze(1)

        return factors

    @torch.inference_mode()
    def get_sparse_factor_precisions(self):
        alphas = MeanStd({}, {})
        for group_name in self.generative.group_names:
            if self.generative.factor_prior[group_name] == "SnS":
                d = dist.Gamma(*self._get_shape_and_rate(f"alpha_z_{group_name}"))
                alphas.mean[group_name] = d.mean.cpu().numpy().squeeze(1)
                alphas.std[group_name] = d.stddev.cpu().numpy().squeeze(1)
        return alphas

    @torch.inference_mode()
    def get_sparse_factor_probabilities(self):
        probs = {}
        for group_name in self.generative.group_names:
            if self.generative.factor_prior[group_name] == "SnS":
                probs[group_name] = self._get_prob(f"s_z_{group_name}").cpu().numpy().squeeze(1)
        return probs

    @torch.inference_mode()
    def get_weights(self):
        """Get all weight matrices, w_x."""
        weights = MeanStd({}, {})
        for view_name in self.generative.view_names:
            for lsidx, vals in enumerate(self._get_loc_and_scale(f"w_{view_name}")):
                weights[lsidx][view_name] = vals

            if self.generative.nonnegative_weights[view_name]:
                weights.mean[view_name] = self.generative.pos_transform(weights.mean[view_name])
            for lsidx in range(2):
                weights[lsidx][view_name] = weights[lsidx][view_name].cpu().numpy().squeeze(-1)

        return weights

    @torch.inference_mode()
    def get_sparse_weight_precisions(self):
        alphas = MeanStd({}, {})
        for view_name in self.generative.view_names:
            if self.generative.weight_prior[view_name] == "SnS":
                d = dist.Gamma(*self._get_shape_and_rate(f"alpha_w_{view_name}"))
                alphas.mean[view_name] = d.mean.cpu().numpy().squeeze(-1)
                alphas.std[view_name] = d.stddev.cpu().numpy().squeeze(-1)
        return alphas

    @torch.inference_mode()
    def get_sparse_weight_probabilities(self):
        probs = {}
        for view_name in self.generative.view_names:
            if self.generative.weight_prior[view_name] == "SnS":
                probs[view_name] = self._get_prob(f"s_w_{view_name}").cpu().numpy().squeeze(-1)
        return probs

    @torch.inference_mode()
    def get_dispersion(self):
        """Get all dispersion vectors, dispersion_x."""
        dispersion = MeanStd({}, {})
        for view_name, likelihood in self.generative.likelihoods.items():
            try:
                disp = likelihood.dispersion
            except AttributeError:
                continue
            dispersion.mean[view_name] = disp.mean
            dispersion.std[view_name] = disp.std

        return dispersion
