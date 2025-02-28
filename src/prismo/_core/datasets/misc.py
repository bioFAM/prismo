import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import BatchSampler, Dataset, RandomSampler, Sampler, StackDataset

from .base import PrismoDataset


class PrismoBatchSampler(Sampler[dict[str, list[int]]]):
    """A sampler for dicts.

    Given a dict with arbitrary keys and values indicating the number of data points in
    individual atasets, creates dicts of indices, such that the largest dataset is
    sampled without replacement, while for the smaller datasets multiple permutations
    are concatenated to yield the length of the largest dataset.
    """

    def __init__(
        self, n_samples: dict[str, int], batch_size: int, drop_last: bool = False, generator: torch.Generator = None
    ):
        super().__init__()
        self._n_samples = n_samples
        self._largest_group = max(n_samples.values())
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._samplers = {
            k: BatchSampler(
                RandomSampler(range(nsamples), num_samples=self._largest_group, generator=generator),
                batch_size,
                drop_last,
            )
            for k, nsamples in self._n_samples.items()
        }

    def __len__(self):
        return (
            self._largest_group // self._batch_size
            if self._drop_last
            else (self._largest_group + self._batch_size - 1) // self._batch_size
        )

    def __iter__(self):
        iterators = {k: iter(sampler) for k, sampler in self._samplers.items()}
        for _ in range(len(self)):
            yield {k: next(sampler) for k, sampler in iterators.items()}


class CovariatesDataset(Dataset):
    def __init__(
        self,
        data: PrismoDataset,
        covariates_obs_key: dict[str, str] | None = None,
        covariates_obsm_key: dict[str, str] | None = None,
    ):
        super().__init__()

        self.covariates, self.covariates_names = data.get_covariates(covariates_obs_key, covariates_obsm_key)
        self.covariates = {
            group_name: np.nanmean(np.stack(tuple(group_covars.values()), axis=0), axis=0)
            for group_name, group_covars in self.covariates.items()
        }
        self._n_samples = max(data.n_samples.values())
        self._cast_to = data.cast_to

    def __len__(self):
        return self._n_samples

    def __getitem__(self, idx: dict[str, int | list[int]]) -> dict[str, NDArray]:
        return {
            group_name: self.covariates[group_name][group_idx, :].astype(self._cast_to)
            for group_name, group_idx in idx.items()
            if group_name in self.covariates
        }

    __getitems__ = __getitem__


class StackDataset(StackDataset):
    def __getitems__(self, idx: list | dict):
        if isinstance(idx, list):
            return super().__getitems__(idx)

        if isinstance(self.datasets, dict):
            return {k: self._get_items_from_dset(dataset, idx) for k, dataset in self.datasets.items()}
        else:
            return [self._get_items_from_dset(dataset, idx) for dataset in self.datasets]

    @staticmethod
    def _get_items_from_dset(dataset: Dataset, idx: dict) -> dict:
        if not callable(getattr(dataset, "__getitems__", None)):
            raise ValueError("Expected nested dataset to have a `__getitems__` method.")

        return dataset.__getitems__(idx)
