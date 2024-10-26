from pathlib import Path

import dill
import pyro
import torch


def save_model(model, dir_path="."):
    model_path = Path(dir_path) / "model.pkl"
    params_path = Path(dir_path) / "params.save"
    for pth in [model_path, params_path]:
        if pth.exists():
            print(f"`{pth}` already exists, overwriting.")

    Path(dir_path).mkdir(parents=True, exist_ok=True)

    # first, save results
    # TODO:

    # second, save parameters
    pyro.get_param_store().save(params_path)

    # third, save model
    with open(model_path, "wb") as f:
        torch.save(model, f, pickle_module=dill)

    print(f"- Model saved to {model_path}")
    print(f"- Parameters saved to {params_path}")


def load_model(dir_path=".", with_params=True, map_location=None):
    model_path = Path(dir_path) / "model.pkl"
    params_path = Path(dir_path) / "params.save"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file `{model_path}` not found.")
    if not params_path.exists():
        raise FileNotFoundError(f"Parameter file `{params_path}` not found.")

    # first, load model
    with open(model_path, "rb") as f:
        model = torch.load(f, map_location=map_location, pickle_module=dill)

    # second, load parameters
    if with_params:
        pyro.get_param_store().load(params_path, map_location=map_location)

    print(f"Model loaded from {model_path}")
    print(f"Parameters loaded from {params_path}")

    return model
