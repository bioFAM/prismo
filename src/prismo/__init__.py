import logging

from .prismo import PRISMO, DataOptions, ModelOptions, SmoothOptions, TrainingOptions

_logger = logging.getLogger(__name__)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))

_logger.setLevel(logging.INFO)
_logger.addHandler(_h)
del _h
