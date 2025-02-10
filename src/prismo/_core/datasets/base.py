from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Concatenate, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import sparray, spmatrix
from torch.utils.data import Dataset

T = TypeVar("T")
ApplyCallable: TypeAlias = Callable[Concatenate[AnnData, str, str, ...], T]


class Preprocessor:
    """Base class for data preprocessors."""

    def __call__(self, arr: NDArray | sparray | spmatrix, group: str, view: str) -> NDArray | sparray | spmatrix:
        """Will be called by subclasses of PrismoDataset on each minibatch.

        Args:
            arr: The data for one group and view.
            group: The group name.
            view: The view name.

        Returns:
            An array containing preprocessed data. In this implementation, returns the unmodified input array.
        """
        return arr


class PrismoDataset(Dataset, ABC):
    """Base class for PRISMO datasets, compatible with the PyTorch dataloader interface.

    Args:
        data: The data.
        preprocessor: A preprocessor. If None, will use the default preprocessor that does not apply any preprocessing.
        cast_to: Data type to cast the data to.
    """

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
        """The preprocessor."""
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, preproc: Preprocessor):
        self._preprocessor = preproc

    @property
    def cast_to(self) -> np.ScalarType:
        """The data type to cast to."""
        return self._cast_to

    @property
    @abstractmethod
    def n_features(self) -> dict[str, int]:
        """Number of features in each view."""
        pass

    @property
    @abstractmethod
    def n_samples(self) -> dict[str, int]:
        """Number of samples in each group."""
        pass

    @property
    @abstractmethod
    def n_samples_total(self) -> int:
        """Total number of samples."""
        pass

    @property
    @abstractmethod
    def view_names(self) -> list[str]:
        """View names."""
        pass

    @property
    @abstractmethod
    def group_names(self) -> list[str]:
        """Group names."""
        pass

    @property
    @abstractmethod
    def sample_names(self) -> dict[str, list[str]]:
        """Sample names for each group."""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> dict[str, list[str]]:
        """Feature names for each view."""
        pass

    def __len__(self):
        """Length of this dataset."""
        return max(self.n_samples.values())

    @abstractmethod
    def __getitem__(self, idx: dict[str, int]) -> tuple[dict[str, dict[str, NDArray]], dict[str, int]]:
        """Get one sample for each group.

        Args:
            idx: Sample indices for each group.

        Returns:
            A tuple. The first element is a nested dict with group names and view names as keys and observations
            as values, the second element is the sample index (the `idx` argument passed through). If the requested
            sample is missing in the respective view, the return array will consist of nans.
        """
        pass

    @abstractmethod
    def __getitems__(self, idx: dict[str, list[int]]) -> tuple[dict[str, dict[str, NDArray]], dict[str, int]]:
        """Get one minibatch for each group.

        Args:
            idx: Sample indices for each group.

        Returns:
            A tuple. The first element is a nested dict with group names and view names as keys and observations
            as values, the second element is the sample index (the `idx` argument passed through). If the requested
            sample is missing in the respective view, the return array will contain nans in the corresponding rows..
        """
        pass

    @abstractmethod
    def align_array_to_samples(
        self, arr: NDArray[T], group_name: str, view_name: str, axis: int = 0, fill_value: np.ScalarType = np.nan
    ) -> NDArray[T]:
        """Align an array corresponding to a view with potentially missing samples to global samples by inserting filler values for missing samples.

        Args:
            arr: The array to align.
            group_name: Group name.
            view_name: View name.
            axis: The axis to align along.
            fill_value: The value to insert for missing samples.
        """
        pass

    @abstractmethod
    def get_obs(self) -> dict[str, pd.DataFrame]:
        """Get observation metadata for each group."""
        pass

    @abstractmethod
    def get_missing_obs(self) -> pd.DataFrame:
        """Determine which observations are missing where.

        Returns: A dataframe with columns `view`, `group`, `obs_name`, and `missing`. `missing` is a boolean with value `True` if
            the observation `obs_name` is missing in view `view` and group `group`, and `False` otherwise.
        """
        pass

    @abstractmethod
    def get_covariates(
        self, covariates_obs_key: dict[str, str] | None = None, covariates_obsm_key: dict[str, str] | None = None
    ) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
        """Get the covariates for each group.

        Args:
            covariates_obs_key: Column in `.obs` for each group containing the covariate.
            covariates_obsm_key: Key in `.obsm` for each group containing the covariates.

        Returns:
            A tuple. The first element contains the covariates for each group, the second contains the covariate names for each group.
            If the covariate names could not be determined for a group, the corresponding entry is missing from the dict.
        """
        pass

    @abstractmethod
    def get_annotations(self, varm_key: dict[str, str]) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
        """Get the annotations for each view.

        Args:
            varm_key: Key in `.varm` for each view containing the annotations.

        Returns:
            A tuple. The first element contains the annotations for each view, the second contains the annotation names for each view.
            If the annotation names could not be determined for a view, the corresponding entry is missing from the dict.
        """
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
        """Apply a function to each group and/or view.

        Args:
            func: The function to apply. The function will be passed an `AnnData` object, the group name, and the view name as the first
                three arguments.
            by_group: Whether to apply the function to each group individually or to all groups at once.
            by_view: Whether to apply the function to each view individually or to all views at once.
            view_kwargs: Additional arguments to pass to `func` for each view. The outer dict contains the argument name as key, the inner
                dict contains the value of that argument for each view. If the inner dict is missing a view, `None` will be used as the
                value of that argument for the view.
            group_kwargs: Additional arguments to pass to `func` for each group. The outer dict contains the argument name as key, the inner
                dict contains the value of that argument for each group. If the inner dict is missing a group, `None` will be used as the
                value of that argument for the group.
            group_view_kwargs: Additional arguments to pass to `func` for each combination of group and view. The outer dict contains the
                argument name as key, the first inner dict has groups as keys and the second inner dict has views as keys. If a group is missing
                from the outer dict or a view is missing from the inner dict, `None` will be used as the value of that argument for all views
                in the group or for the view, respectively.
            **kwargs: Additional arguments to pass to `func`.

        Returns: Nested dict with the return value of `func` for each group and view.
        """
        pass
