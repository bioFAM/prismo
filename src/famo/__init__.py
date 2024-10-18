import logging
import logging.config
from pathlib import Path

logger = logging.getLogger(__name__)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

logger.setLevel(logging.INFO)
logger.addHandler(_h)
del _h
