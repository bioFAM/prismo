import pytest
import os
import shutil
import torch
import pyro
from pathlib import Path
from famo.utils_io import load_model
from famo.core import CORE


@pytest.fixture
def setup_teardown():
    # Setup: Create a temporary directory for saving models
    temp_dir = Path("./temp_model_dir")
    temp_dir.mkdir(exist_ok=True)

    yield temp_dir

    # Teardown: Remove the temporary directory and its contents
    shutil.rmtree(temp_dir)


def test_save_load_model(setup_teardown):
    temp_dir = setup_teardown

    # Prepare dummy data
    data = {"group1": {"view1": torch.rand(3, 10), "view2": torch.rand(3, 5)}}

    # Create and train the CORE model for a single epoch
    model = CORE(device="cpu")
    model.fit(
        n_factors=2,
        data=data,
        likelihoods={
            "view1": "Normal",
            "view2": "Normal",
        },
        factor_prior="Normal",
        weight_prior="Normal",
        lr=0.001,
        scale_per_group=False,
        max_epochs=1,  # Train for a single epoch
        save=True,
        save_path=temp_dir,
    )

    # Check if files are saved
    assert os.path.exists(temp_dir / "model.pkl")
    assert os.path.exists(temp_dir / "params.save")

    # model params
    prev_generator_params = model.parameters()
    # Save pyro param store
    prev_param_store = pyro.get_param_store()

    # Wipe everything
    del model
    pyro.clear_param_store()
    # Check that everything is empty
    assert len(pyro.get_param_store().get_all_param_names()) == 0

    # Load the model and its parameters
    loaded_model = load_model(dir_path=temp_dir)

    # Check if the model's parameter is correctly loaded
    for original_param, loaded_param in zip(
        prev_generator_params, loaded_model.parameters()
    ):
        assert torch.equal(original_param, loaded_param), "Model parameter mismatch"

    # Check if the param store parameter is correctly loaded
    for name in pyro.get_param_store().get_all_param_names():
        loaded_param = pyro.param(name)
        original_param = prev_param_store[name]
        assert torch.equal(
            original_param, loaded_param
        ), f"Parameter store mismatch for {name}"
