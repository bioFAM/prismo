import numpy as np

from prismo._core import PrismoDataset
from prismo._core.preprocessing import PrismoPreprocessor
from prismo._core.utils import sample_all_data_as_one_batch


def preprocess(data, likelihoods, scale_per_group=True, cast_to=np.float32):
    data = PrismoDataset(data, cast_to=cast_to)
    preprocessor = PrismoPreprocessor(
        data,
        likelihoods,
        {view_name: False for view_name in data.view_names},
        {group_name: False for group_name in data.group_names},
        scale_per_group=scale_per_group,
    )
    data.preprocessor = preprocessor

    return preprocessor, data.__getitems__(sample_all_data_as_one_batch(data))["data"]
