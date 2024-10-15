from typing import List
import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold


def proportional_sampling(df: pd.DataFrame, group_column: str, total_samples: int):
    """
    Samples proportionally the first k_i entries from each group, where k_i is determined
    by the size of the group and such that sum k_i = total_samples.
    """
    group_sizes = df[group_column].value_counts()

    if total_samples > len(df):
        raise ValueError(
            f"total_samples ({total_samples}) cannot exceed the number of rows in the dataframe ({len(df)})"
        )

    proportions = group_sizes / group_sizes.sum()
    # Exact proportions
    sample_sizes = (proportions * total_samples).astype(float)

    # Approximate proportion
    int_samples = np.floor(sample_sizes).astype(int)
    remainders = sample_sizes - int_samples

    # Does not yet sum up to total_samples
    # we will distribute the remaining samples over the
    # groups with the largest remainders
    remaining = total_samples - int_samples.sum()
    indices = indices = remainders.nlargest(int(remaining)).index
    int_samples.loc[indices] += 1
    assert int_samples.sum() == total_samples, int_samples.sum()

    result = df.groupby(group_column, group_keys=False)[
        df.columns
    ].apply(  # [df.columns] to retain the group columns
        lambda group: group.head(int_samples[group.name])
    )

    assert (
        len(result) == total_samples
    ), f"Result length ({len(result)}) does not match total_samples ({total_samples})"
    return result


def sample_dataset(molecules: List[str], size: int, scaffolds=None):
    """ """

    if scaffolds is None:
        scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(smi) for smi in molecules]

    df = pd.DataFrame({"molecules": molecules, "scaffolds": scaffolds})

    return proportional_sampling(df, "scaffolds", size)
