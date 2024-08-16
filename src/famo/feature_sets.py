# Highly inspired by https://github.com/krassowski/gsea-api
import logging

from collections import Counter
from collections.abc import Collection
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances


logger = logging.getLogger(__name__)


class FeatureSet:
    def __init__(self, features: Collection[str], name: str, description: str = ""):
        self.name = name
        self.features = frozenset(features)
        self.description = description

        if self.empty:
            logger.warning(f"FeatureSet {name!r} is empty.")

        redundant_features = None

        if len(features) != len(self.features):
            redundant_features = {feature: count for feature, count in Counter(features).items() if count > 1}

            logger.warning(
                f"FeatureSet {name!r} received a non-unique "
                f"collection of features; redundant features: {redundant_features}"
            )

        self.redundant_features = redundant_features

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self.features)

    def __repr__(self) -> str:
        features = ": " + ", ".join(sorted(self.features)) if len(self.features) < 5 else ""
        return f"<FeatureSet {self.name!r} with {len(self)} features{features}>"

    def __iter__(self) -> Iterable[str]:
        return iter(self.features)

    def __eq__(self, other: "FeatureSet") -> bool:
        return self.features == other.features

    def __hash__(self) -> int:
        return hash(self.features)

    def __and__(self, other: "FeatureSet") -> "FeatureSet":
        return FeatureSet(self.features & other.features, name=f"{self.name}&{other.name}")

    def __or__(self, other: "FeatureSet") -> "FeatureSet":
        return FeatureSet(self.features | other.features, name=f"{self.name}|{other.name}")

    def __add__(self, other: "FeatureSet") -> "FeatureSet":
        return self.__or__(other)

    def subset(self, features: Iterable[str]) -> "FeatureSet":
        """Subset features from a feature set.

        Parameters
        ----------
        features : Iterable[str]
            Features to subset.

        Returns
        -------
        FeatureSet
            A new feature set with the subset of features.
        """
        return FeatureSet(self.features & features, name=self.name)


class FeatureSets:
    def __init__(self, feature_sets: Collection[FeatureSet], name: str = "UNL", remove_empty: bool = True):
        self.name = name

        if remove_empty:
            feature_sets = {feature_set for feature_set in feature_sets if not feature_set.empty}

        redundant_feature_sets = None

        if len(set(feature_sets)) != len(feature_sets):
            redundant_feature_sets = {
                feature_set: count for feature_set, count in Counter(feature_sets).items() if count > 1
            }

            logger.warning(
                f"FeatureSets {name!r} received a non-unique "
                "collection of feature sets; redundant feature sets: "
                f"{redundant_feature_sets}"
            )
        self.feature_sets = frozenset(feature_sets)
        self.redundant_feature_sets = redundant_feature_sets

    @property
    def empty(self):
        return len(self) == 0

    @property
    def median_size(self) -> int:
        return int(np.median([len(fs) for fs in self.feature_sets]))

    @property
    def features(self) -> frozenset:
        return frozenset.union(*[fs.features for fs in self.feature_sets])

    @property
    def feature_set_by_name(self) -> dict:
        return {feature_set.name: feature_set for feature_set in self.feature_sets}

    def __getitem__(self, name: str) -> FeatureSet:
        return self.feature_set_by_name[name]

    def __len__(self):
        return len(self.feature_sets)

    def __iter__(self) -> Iterable[FeatureSet]:
        return iter(self.feature_sets)

    def __repr__(self) -> str:
        feature_sets = (
            ": " + ", ".join(sorted({fs.name for fs in self.feature_sets})) if len(self.feature_sets) < 5 else ""
        )
        return f"<FeatureSets {self.name!r} with {len(self)} " + f"feature sets{feature_sets}>"

    def __eq__(self, other: "FeatureSets") -> bool:
        return self.feature_sets == other.feature_sets

    def __hash__(self) -> int:
        return hash(self.feature_sets)

    def __and__(self, other: "FeatureSets") -> "FeatureSets":
        return FeatureSets(name=f"{self.name}&{other.name}", feature_sets=self.feature_sets & other.feature_sets)

    def __or__(self, other: "FeatureSets") -> "FeatureSets":
        return FeatureSets(name=f"{self.name}|{other.name}", feature_sets=self.feature_sets | other.feature_sets)

    def __add__(self, other: "FeatureSets") -> "FeatureSets":
        return self.__or__(other)

    def find(self, partial_name: str):
        """Perform a simple search given a (partial) feature set name.

        Parameters
        ----------
        partial_name : str
            Feature set (partial) name to search for.

        Returns
        -------
        FeatureSets
            Search results.
        """
        return FeatureSets(
            {feature_set for feature_set in self.feature_sets if partial_name in feature_set.name},
            name=f"{self.name}:{partial_name}",
        )

    def remove(self, names: Iterable[str]):
        """Remove feature sets by name.

        Parameters
        ----------
        names : Iterable[str]
            Collection of feature set names.
        """
        return FeatureSets(
            {feature_set for feature_set in self.feature_sets if feature_set.name not in names}, name=self.name
        )

    def keep(self, names: Iterable[str]):
        """Keep feature sets by name.

        Parameters
        ----------
        names : Iterable[str]
            Collection of feature set names.
        """
        return FeatureSets(
            {feature_set for feature_set in self.feature_sets if feature_set.name in names}, name=self.name
        )

    def trim(self, min_count: int = 1, max_count: Optional[int] = None):
        """Trim feature sets by min/max size.

        Parameters
        ----------
        min_count : int, optional
            Minimum number of features, by default 1.
        max_count : int, optional
            Maximum number of features, by default None.
        """
        return FeatureSets(
            {
                feature_set
                for feature_set in self.feature_sets
                if min_count <= len(feature_set.features) <= (max_count or len(feature_set.features))
            },
            name=self.name,
        )

    def subset(self, features: Iterable[str]):
        """Subset feature sets by features.

        Parameters
        ----------
        features : Iterable[str]
            Collection of features.
        """
        return FeatureSets({feature_set.subset(set(features)) for feature_set in self.feature_sets}, name=self.name)

    def filter(
        self,
        features: Iterable[str],
        min_fraction: float = 0.5,
        min_count: int = 5,
        max_count: Optional[int] = None,
        keep: Optional[Iterable[str]] = None,
        subset: bool = True,
    ):
        """Filter feature sets.

        Parameters
        ----------
        features : Iterable[str]
            Features to filter.
        min_fraction : float, optional
            Mininimum portion of the feature set to be present in `features`,
            by default 0.5
        min_count : int, optional
            Minimum size of the intersection set
            between a feature set and the set of `features`,
            by default 5
        max_count : int, optional
            Maximum size of the intersection set
            between a feature set and the set of `features`,
            by default None
        keep : Iterable[str], optional
            Feature sets to keep regardless of the filter conditions,
            by default None
        subset : bool, optional
            Whether to subset the resulting feature sets based on `features`,
            by default True

        Returns
        -------
        FeatureSets
            Filtered feature sets.
        """
        features = set(features)

        if keep is None:
            keep = set()

        feature_set_subset = set()

        for feature_set in self.feature_sets:
            if feature_set.name in keep:
                feature_set_subset.add(feature_set)
                continue
            intersection = features & feature_set.features
            count = len(intersection)
            fraction = count / len(feature_set)
            if count >= min_count and fraction >= min_fraction and (max_count is None or count <= max_count):
                feature_set_subset.add(feature_set)

        filtered_feature_sets = FeatureSets(feature_set_subset, name=self.name)
        if subset:
            filtered_feature_sets = filtered_feature_sets.subset(features)
        return filtered_feature_sets

    def to_mask(self, features: Optional[Iterable[str]] = None, sort: bool = True) -> pd.DataFrame:
        """Convert feature sets to a mask.

        Parameters
        ----------
        features : Iterable[str], optional
            Collection of features, by default None.
        sort : bool, optional
            Sort feature sets alphabetically, by default True.

        Returns
        -------
        pd.DataFrame
            Mask of features.
        """
        features = features or self.features
        features_list = list(features)
        feature_sets_list = list(self.feature_sets)
        if sort:
            feature_sets_list = sorted(feature_sets_list, key=lambda fs: fs.name)
        return pd.DataFrame(
            [[feature in feature_set.features for feature in features_list] for feature_set in feature_sets_list],
            index=[feature_set.name for feature_set in feature_sets_list],
            columns=features_list,
        )

    def similarity_to_feature_sets(self, other: "FeatureSets" = None, metric: str = "jaccard") -> pd.DataFrame:
        """Compute similarity matrix between feature sets.

        Parameters
        ----------
        other : FeatureSets, optional
            Other feature set collection, by default None.
        metric : str, optional
            Similarity metric, by default "jaccard".

        Returns
        -------
        pd.DataFrame
            Similarity matrix as 1 minus distance matrix,
            may lead to negative values for some distance metrics.
        """

        if metric not in ["jaccard", "cosine"]:
            logger.warning(
                f"Similarity matrix for `{metric}` might be negative. " "Recommended metrics are `jaccard` or `cosine`."
            )

        self_mask = self.to_mask()
        other_mask = other.to_mask() if other else self_mask
        return 1 - pd.DataFrame(
            pairwise_distances(self_mask.to_numpy(), other_mask.to_numpy(), metric=metric),
            index=self_mask.index,
            columns=other_mask.index,
        )

    def similarity_to_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Compute similarity matrix between feature sets.

        Parameters
        ----------
        observations : pd.DataFrame
            Dataframe of observations.

        Returns
        -------
        pd.DataFrame
            Similarity matrix as correlation matrix.
        """
        obs_mean = observations.mean(axis=1)

        dist_to_mean_dict = {}
        for feature_set in self.feature_sets:
            col_subset = [col for col in observations.columns if col in feature_set.features]
            if len(col_subset) == 0:
                dist_to_mean_dict[feature_set.name] = pd.Series(np.nan, index=observations.index)
                continue
            dist_to_mean_dict[feature_set.name] = observations.loc[:, col_subset].mean(axis=1) - obs_mean

        return pd.DataFrame(dist_to_mean_dict).corr()

    def _find_similar_pairs(self, sim_matrix: pd.DataFrame, similarity_threshold: float) -> set[tuple[str, str]]:
        """Find similar pairs of feature sets.

        Parameters
        ----------
        sim_matrix : pd.DataFrame
            Similarity matrix.
        similarity_threshold : float
            Similarity threshold to consider similar pairs.

        Returns
        -------
        set[tuple[str, str]]
            Similar pairs of feature sets.
        """

        pairs = set()
        visited = set()

        row_offset = 0
        for current_fs, row in sim_matrix.iterrows():
            row_offset += 1
            if row_offset >= len(row):
                break
            if current_fs in visited:
                continue
            visited.add(current_fs)
            closest_fs = row.iloc[row_offset:].idxmax()
            similarity = row[closest_fs]
            if similarity >= similarity_threshold and closest_fs not in visited:
                pairs.add((current_fs, closest_fs, similarity))
                visited.add(closest_fs)

        return pairs

    def find_similar_pairs(
        self, observations: pd.DataFrame = None, metric: Optional[str] = None, similarity_threshold: float = 0.8
    ) -> set[tuple[str, str]]:
        """Find similar pairs of feature sets.

        Parameters
        ----------
        observations : pd.DataFrame, optional
            Dataframe of observations, if provided, the similarity between feature sets
            is computed based on the correlation of the similarity from the mean
            of the observations in the feature set, by default None.
        metric : str, optional
            Similarity metric, by default "jaccard" if observations not provided.
        similarity_threshold : float, optional
            Similarity threshold to consider similar pairs,
            by default 0.8.

        Returns
        -------
        set[tuple[str, str]]
            Similar pairs of feature sets.
        """
        if observations is None and metric is None:
            logger.warning("Neither observations nor metric is provided," " using `metric=jaccard` as default.")
            metric = "jaccard"

        sim_matrix = []
        if observations is not None:
            sim_matrix.append(self.similarity_to_observations(observations))
        if metric is not None:
            sim_matrix.append(self.similarity_to_feature_sets(metric=metric))

        if observations is not None and metric is not None:
            sim_matrix[0][sim_matrix[0] < 0] = 0.0
            sim_matrix[1][sim_matrix[1] < 0] = 0.0
            sim_matrix = (2 * sim_matrix[0] * sim_matrix[1]) / (sim_matrix[0] + sim_matrix[1])
        else:
            sim_matrix = sim_matrix[0]
        return self._find_similar_pairs(sim_matrix.fillna(0.0), similarity_threshold)

    def merge_pairs(self, pairs: Iterable[tuple[str, str]]):
        """Merge pairs of feature sets.

        Parameters
        ----------
        pairs : Iterable[tuple[str, str]]
            Pairs of feature sets.

        Returns
        -------
        FeatureSets
            Merged feature sets.
        """
        names_to_remove = set()
        merged_feature_sets = set()
        for pair in pairs:
            merged_feature_sets.add(self[pair[0]] | self[pair[1]])
            names_to_remove |= {pair[0], pair[1]}

        # remove merged feature sets
        feature_sets = self.remove(names_to_remove)
        # then add merged feature sets
        feature_sets |= FeatureSets(merged_feature_sets)
        feature_sets.name = self.name
        return feature_sets

    def merge_similar(
        self,
        observations: pd.DataFrame = None,
        metric: Optional[str] = None,
        similarity_threshold: float = 0.8,
        iteratively: bool = True,
    ):
        """Merge similar feature sets.


        Parameters
        ----------
        observations : pd.DataFrame, optional
            Dataframe of observations, if provided, the similarity between feature sets
            is computed based on the correlation of the similarity from the mean
            of the observations in the feature set, by default None.
        metric : str, optional
            Similarity metric, by default "jaccard" if observations not provided.
        similarity_threshold : float, optional
            Similarity threshold to consider similar pairs,
            by default 0.8.
        iteratively : bool, optional
            Whether to merge iteratively, by default True

        Returns
        -------
        FeatureSets
            Merged feature sets.
        """

        feature_sets = self
        while True:
            pairs = {
                (name1, name2)
                for name1, name2, _ in feature_sets.find_similar_pairs(
                    observations=observations, metric=metric, similarity_threshold=similarity_threshold
                )
            }
            stopping = ""
            if len(pairs) == 0 and iteratively:
                stopping = " Stopping..."
            logger.info(f"Found {len(pairs)} pairs to merge.{stopping}")
            feature_sets = feature_sets.merge_pairs(pairs)
            if len(pairs) == 0 or not iteratively:
                break
        return feature_sets

    def to_gmt(self, path: Path):
        """Write this feature set collection to a GMT file.

        Parameters
        ----------
        path : Path
            Path to the output file.
        """
        with open(path, "w") as f:
            for feature_set in self.feature_sets:
                f.write(
                    feature_set.name + "\t" + feature_set.description + "\t" + "\t".join(feature_set.features) + "\n"
                )

    def to_dict(self) -> dict[str, Iterable[str]]:
        """Convert this feature set collection to a dictionary.

        Returns
        -------
        dict[str, Iterable[str]]
            Dictionary of feature sets.
        """
        return {fs.name: fs.features for fs in self.feature_sets}


def from_gmt(path: Path, name: Optional[str] = None, **kwargs) -> FeatureSets:
    """Create a FeatureSets object from a GMT file.

    Parameters
    ----------
    path : Path
        Path to the GMT file.
    name : str, optional
        Name of the collection, by default None.

    Returns
    -------
    FeatureSets
    """
    feature_sets = set()
    with open(path) as f:
        for line in f:
            fs_name, description, *features = line.strip().split("\t")
            feature_sets.add(FeatureSet(features, name=fs_name, description=description))
    return FeatureSets(feature_sets, name=name or Path(path).name, **kwargs)


def from_dict(d: dict[str, Iterable[str]], name: Optional[str] = None, **kwargs) -> FeatureSets:
    """Create a FeatureSets object from a dictionary.

    Parameters
    ----------
    d : dict[str, Iterable[str]]
        Dictionary of feature sets.
    name : str, optional
        Name of the collection, by default None.

    Returns
    -------
    FeatureSets
    """
    feature_sets = set()
    for fs_name, features in d.items():
        feature_sets.add(FeatureSet(features, name=fs_name))
    return FeatureSets(feature_sets, name=name, **kwargs)


def from_dataframe(
    df: pd.DataFrame,
    name: Optional[str] = None,
    name_col: str = "name",
    features_col: str = "features",
    desc_col: Optional[str] = None,
    **kwargs,
) -> FeatureSets:
    """Create a FeatureSets object from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of feature sets.
    name : str, optional
        Name of the collection, by default None.
    name_col : str, optional
        Name of the column containing feature set names, by default "name".
    features_col : str, optional
        Name of the column containing feature set features, by default "features".
    desc_col : str, optional
        Name of the column containing feature set descriptions, by default None.

    Returns
    -------
    FeatureSets
    """
    feature_sets = set()
    for _, row in df.iterrows():
        feature_sets.add(
            FeatureSet(row[features_col], name=row[name_col], description=desc_col is not None and row[desc_col])
        )
    return FeatureSets(feature_sets, name=name, **kwargs)
