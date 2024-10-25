import os
import shutil
from itertools import chain
from pathlib import Path

import pyro
import pytest
import torch

from famo import CORE, DataOptions, ModelOptions, TrainingOptions
from famo.utils_io import load_model


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
    model = CORE()
    model.fit(
        data,
        DataOptions(scale_per_group=False),
        ModelOptions(
            n_factors=2,
            likelihoods={"view1": "Normal", "view2": "Normal"},
            factor_prior="Normal",
            weight_prior="Normal",
        ),
        TrainingOptions(
            device="cpu",
            lr=0.001,
            max_epochs=1,  # Train for a single epoch
            save=True,
            save_path=temp_dir,
        ),
    )

    # Check if files are saved
    assert os.path.exists(temp_dir / "model.pkl")
    assert os.path.exists(temp_dir / "params.save")

    # model params
    prev_generator_params = chain(model.generative.parameters(), model.variational.parameters())
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
        prev_generator_params,
        chain(loaded_model.generative.parameters(), loaded_model.variational.parameters()),
        strict=False,
    ):
        assert torch.equal(original_param, loaded_param), "Model parameter mismatch"

    # Check if the param store parameter is correctly loaded
    for name in pyro.get_param_store().get_all_param_names():
        loaded_param = pyro.param(name)
        original_param = prev_param_store[name]
        assert torch.equal(original_param, loaded_param), f"Parameter store mismatch for {name}"
