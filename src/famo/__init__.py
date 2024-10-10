import logging
import logging.config
from pathlib import Path

logging.config.fileConfig(Path(__file__).resolve().parent / "log.conf", disable_existing_loggers=False)
logging.getLogger().setLevel(logging.INFO)
