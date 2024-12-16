from collections import namedtuple
from typing import Literal, TypeAlias

WeightPrior: TypeAlias = Literal["Normal", "Laplace", "Horseshoe", "SnS", "GP"]
FactorPrior: TypeAlias = Literal["Normal", "Laplace", "Horseshoe", "SnS"]
Likelihood: TypeAlias = Literal["Normal", "GammaPoisson", "Bernoulli"]

MeanStd = namedtuple("MeanStd", ["mean", "std"])
