from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Concatenate, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from torch.utils.data import Dataset

T = TypeVar("T")
ApplyCallable: TypeAlias = Callable[Concatenate[AnnData, str, str, ...], T]


class Preprocessor:
    def __call__(self, arr: NDArray, group: str, view: str):
        return arr


class PrismoDataset(Dataset, ABC):
    def __init__(self, data, preprocessor: Preprocessor | None = None, cast_to: np.ScalarType = np.float32):
        super().__init__()

        self._data = data

        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = Preprocessor()

        self._cast_to = cast_to

    @property
    def preprocessor(self) -> Preprocessor:
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, preproc: Preprocessor):
        self._preprocessor = preproc

    @property
    def cast_to(self) -> np.ScalarType:
        return self._cast_to

    @property
    @abstractmethod
    def n_features(self) -> dict[str, int]:
        pass

    @property
    @abstractmethod
    def n_samples(self) -> dict[str, int]:
        pass

    @property
    @abstractmethod
    def n_samples_total(self) -> int:
        pass

    @property
    @abstractmethod
    def view_names(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def group_names(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def sample_names(self) -> dict[str, list[str]]:
        pass

    @property
    @abstractmethod
    def feature_names(self) -> dict[str, list[str]]:
        pass

    def __len__(self):
        return max(self.n_samples.values())

    @abstractmethod
    def __getitem__(self, idx: dict[str, int]) -> tuple[dict[str, dict[str, NDArray]], dict[str, int]]:
        pass

    @abstractmethod
    def __getitems__(self, idx: dict[str, list[int]]) -> tuple[dict[str, dict[str, NDArray]], dict[str, int]]:
        pass

    @abstractmethod
    def align_array_to_samples(
        self, arr: NDArray[T], view_name: str, group_name: str, axis: int = 0, fill_value: np.ScalarType = np.nan
    ) -> NDArray[T]:
        pass

    @abstractmethod
    def get_obs(self) -> dict[str, pd.DataFrame]:
        pass

    @abstractmethod
    def get_missing_obs(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_covariates(
        self, covariates_obs_key: dict[str, str] | None = None, covariates_obsm_key: dict[str, str] | None = None
    ) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
        pass

    @abstractmethod
    def get_annotations(self, varm_key: dict[str, str]) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
        pass

    @abstractmethod
    def apply(
        self,
        func: ApplyCallable[T],
        by_group: bool = True,
        by_view: bool = True,
        view_kwargs: dict[str, dict[str, Any]] | None = None,
        group_kwargs: dict[str, dict[str, Any]] | None = None,
        group_view_kwargs: dict[str, dict[str, dict[str, Any]]] | None = None,
        **kwargs,
    ) -> dict[str, dict[str, T]]:
        pass
