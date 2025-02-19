import numpy as np
from torch.utils.data import BatchSampler, SequentialSampler

from prismo._core import PrismoDataset
from prismo._core.preprocessing import PrismoPreprocessor


def one_batch_sample(data: PrismoDataset) -> dict[str, list[int]]:
    return {
        k: next(
            iter(BatchSampler(SequentialSampler(range(nsamples)), batch_size=data.n_samples_total, drop_last=False))
        )
        for k, nsamples in data.n_samples.items()
    }


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

    return preprocessor, data.__getitems__(one_batch_sample(data))["data"]
