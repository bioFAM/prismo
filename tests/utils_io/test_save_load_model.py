import os
import tempfile

import numpy as np
import pytest
import torch

from prismo import PRISMO, DataOptions, ModelOptions, TrainingOptions


@pytest.fixture
def setup_teardown():
    # Setup: Create a temporary directory for saving models
    temp_file = tempfile.mkstemp(suffix=".h5")
    os.close(temp_file[0])

    yield temp_file[1]

    # Teardown: Remove the temporary directory and its contents
    os.unlink(temp_file[1])


def test_save_load_model(setup_teardown):
    temp_file = setup_teardown

    # Prepare dummy data
    data = {"group1": {"view1": torch.rand(3, 10), "view2": torch.rand(3, 5)}}

    # Create and train the PRISMO model for a single epoch
    model = PRISMO(
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
            save_path=temp_file,
        ),
    )

    # Check if files are saved
    assert os.path.exists(temp_file)

    # Load the model and its parameters
    loaded_model = PRISMO.load(path=temp_file)
    # Check if the model's parameter is correctly loaded
    if model._gp is not None:  # TODO: test with GP
        for original_param, loaded_param in zip(model._gp.parameters(), loaded_model._gp.parameters(), strict=False):
            assert torch.equal(original_param, loaded_param), "Model parameter mismatch"

    for attr in (
        "group_names",
        "n_groups",
        "view_names",
        "n_views",
        "feature_names",
        "n_features",
        "sample_names",
        "n_samples",
        "n_samples_total",
        "n_factors",
        "n_dense_factors",
        "n_informed_factors",
        "factor_order",
        "factor_names",
        "warped_covariates",
        "covariates",
        "gp_lengthscale",
        "gp_scale",
        "gp_group_correlation",
    ):
        assert np.all(getattr(model, attr) == getattr(loaded_model, attr)), attr
