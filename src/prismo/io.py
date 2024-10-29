import logging
from io import BytesIO
from pathlib import Path

import dill
import h5py
import numpy as np
import pyro
import torch

logger = logging.getLogger(__name__)


def save_model(model, path: str | Path):
    dset_kwargs = {"compression": "gzip", "compression_opts": 9}

    path = Path(path)
    if path.exists():
        logger.warning("`path` already exists, overwriting")
    with h5py.File(path, "w") as f:
        prismogrp = f.create_group("prismo")

        paramspkl, modelpkl = BytesIO(), BytesIO()
        torch.save(pyro.get_param_store().get_state(), paramspkl, pickle_module=dill)
        torch.save(model, modelpkl, pickle_module=dill)

        prismogrp.create_dataset(
            "param_store", data=np.frombuffer(paramspkl.getbuffer(), dtype=np.uint8), **dset_kwargs
        )
        prismogrp.create_dataset("model", data=np.frombuffer(modelpkl.getbuffer(), dtype=np.uint8), **dset_kwargs)

    logger.info(f"Saved model to {path}")


def load_model(path: str | Path, with_params=True, map_location=None):
    path = Path(path)
    with h5py.File(path, "r") as f:
        prismogrp = f["prismo"]
        paramspkl = BytesIO(prismogrp["param_store"][()].tobytes())
        modelpkl = BytesIO(prismogrp["model"][()].tobytes())

        model = torch.load(modelpkl, map_location=map_location, pickle_module=dill)
        if with_params:
            pyro.get_param_store().set_state(torch.load(paramspkl, map_location=map_location, pickle_module=dill))

    logger.info(f"Loaded model from {path}")

    return model
