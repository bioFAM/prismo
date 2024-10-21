import logging

from .core import CORE, DataOptions, ModelOptions, SmoothOptions, TrainingOptions

_logger = logging.getLogger(__name__)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

_logger.setLevel(logging.INFO)
_logger.addHandler(_h)
del _h
