import logging

from . import pl, tl
from ._core import PRISMO, DataOptions, FeatureSet, FeatureSets, ModelOptions, SmoothOptions, TrainingOptions
from ._version import __version__, __version_tuple__

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
