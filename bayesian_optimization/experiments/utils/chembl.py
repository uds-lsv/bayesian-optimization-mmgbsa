from functools import wraps
from typing import Callable, List

import pandas as pd


def dataframe_length_change(func: Callable) -> Callable:
    """
    A decorator that prints the length of a DataFrame before and after
    the execution of the decorated function.

    :param func: A function that takes a DataFrame and returns a DataFrame.
    :return: The decorated function.

    Example:
    >>> @dataframe_length_change
    ... def remove_first_row(df: pd.DataFrame) -> pd.DataFrame:
    ...     return df.iloc[1:]
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> remove_first_row(df)
    Length before: 3
    Length after: 2
    """

    @wraps(func)
    def wrapper(df: pd.DataFrame) -> pd.DataFrame:
        length_before = len(df)
        result = func(df)
        length_after = len(result)
        print(
            f"Discarded {length_before - length_after} datapoints using {func.__name__}. Keeping {length_after} values."
        )
        return result

    return wrapper


def load_chembl(path):
    data = pd.read_csv(path, index_col=False)
    data = data.rename(
        columns={
            "molecule_dictionary.chembl_id": "molecule_chembl_id",
            "compound_structures.canonical_smiles": "smiles",
            "target_dictionary.chembl_id": "protein_chembl_id",
            "activities.standard_type": "measurement_type",
            "activities.standard_value": "value",
        }
    )
    return data[
        [
            "protein_chembl_id",
            "measurement_type",
            "molecule_chembl_id",
            "smiles",
            "value",
        ]
    ]


def top_k(data: pd.DataFrame, k: int) -> List[pd.DataFrame]:
    """
    Selects the top k proteins and their measured affinities
    :param data: DataFrame as returned by `load_chembl`
    :param k: Number of top proteins to select
    :return: List with a dataframe for each of top k proteins
    """
    counts = data["protein_chembl_id"].value_counts()
    return [
        data[data["protein_chembl_id"] == chembl_id]
        for chembl_id in counts.nlargest(k).index
    ]


@dataframe_length_change
def keep_only_highest_value(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["value"], ascending=False).drop_duplicates(
        ["protein_chembl_id", "molecule_chembl_id"]
    )
    return data


@dataframe_length_change
def keep_only_most_common_measurement(data: pd.DataFrame) -> pd.DataFrame:
    measurement_counts = data["measurement_type"].value_counts()
    most_common_measurement = measurement_counts.idxmax()
    data = data[data["measurement_type"] == most_common_measurement]
    return data


@dataframe_length_change
def keep_only_non_nan(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna()


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = keep_only_highest_value(data)
    data = keep_only_most_common_measurement(data)
    data = keep_only_non_nan(data)  # For some proteins, some smiles are NaN !?
    return data
