import logging

from . import _pl as pl
from . import _tl as tl
from ._core import *  # noqa F403
from ._version import __version__, __version_tuple__

_logger = logging.getLogger(__name__)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))

_logger.setLevel(logging.INFO)
_logger.addHandler(_h)
del _h
