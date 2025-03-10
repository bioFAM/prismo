"""Presets corresponding to different previously published factor analysis models.

These can be used by passing them to the :class:`PRISMO<prismo.PRISMO>` constructor: `PRISMO(*preset)`.

Attributes:
    MOFA: Options used to reproduce MOFA results in the PRISMO paper.
    MEFISTO: Options used to reproduce MEFISTO results in the PRISMO paper.
    NSF: Options used to reproduce NSF results in the PRISMO paper.
"""

from .prismo import ModelOptions, SmoothOptions, TrainingOptions

MOFA = (
    ModelOptions(n_factors=10, weight_prior="Horseshoe", factor_prior="Normal", likelihoods="Normal"),
    TrainingOptions(lr=0.05),
)

MEFISTO = (
    ModelOptions(n_factors=4, weight_prior="Horseshoe", factor_prior="GP", likelihoods="Normal"),
    TrainingOptions(lr=0.05),
    SmoothOptions(n_inducing=1000, kernel="RBF", mefisto_kernel=True),
)

NSF = (
    ModelOptions(
        n_factors=5,
        weight_prior="HorseShoe",
        factor_prior="GP",
        likelihoods="GammaPoisson",
        nonnegative_weights=True,
        nonnegative_factors=True,
    ),
    TrainingOptions(lr=0.05),
    SmoothOptions(n_inducing=800, kernel="Matern"),
)
