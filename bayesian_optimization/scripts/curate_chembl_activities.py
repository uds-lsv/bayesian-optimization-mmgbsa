"""
ChEMBL preprocessing as described in
https://github.com/openkinome/kinodata/blob/2169fdd8bc356b63cc03f3ed24db76946ad97bb2/kinase-bioactivities-in-chembl/kinase-bioactivities-in-chembl.ipynb
"""
import pathlib
import sqlite3 as sql
import urllib.request

import numpy as np
import pandas as pd


def get_uniprot_mapping(version: int) -> pd.DataFrame:
    url = rf"ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{version}/chembl_uniprot_mapping.txt"
    with urllib.request.urlopen(url) as response:
        return pd.read_csv(
            response,
            sep="\t",
            skiprows=[0],
            names=["UniprotID", "chembl_targets", "description", "type"],
        )


def create_human_kinases_and_chembl_targets(
    kinases_path: pathlib.Path, out_path: pathlib.Path, version=33
) -> pd.DataFrame:
    uniprot_map = get_uniprot_mapping(version)
    kinases = pd.read_csv(kinases_path, index_col=0)
    # At least one source
    kinases_subset = kinases[
        kinases[["kinhub", "klifs", "pkinfam", "dunbrack_msa"]].sum(axis=1) > 0
    ]
    kinases_subset["origin"] = kinases_subset.apply(  # Store sources in big string
        lambda s: "|".join(
            [
                k
                for k in [
                    "kinhub",
                    "klifs",
                    "pkinfam",
                    "reviewed_uniprot",
                    "dunbrack_msa",
                ]
                if s[k]
            ]
        ),
        axis=1,
    )
    # Merge kinases and chembl on uniprotid
    merged = pd.merge(
        kinases_subset[["UniprotID", "Name", "origin"]],
        uniprot_map[["UniprotID", "chembl_targets", "type"]],
        how="inner",
        on="UniprotID",
    )
    merged = merged[["UniprotID", "Name", "chembl_targets", "type", "origin"]]

    # Keep only single protein measurements
    merged[merged.type == "SINGLE PROTEIN"].to_csv(
        out_path / f"human_kinases_and_chembl_targets.chembl_{version}.csv",
        index=False,
    )
    return merged[merged.type == "SINGLE PROTEIN"]


def load_chembl_data(args):
    select_these = [
        "activities.activity_id",
        "assays.chembl_id",
        "target_dictionary.chembl_id",
        "molecule_dictionary.chembl_id",
        "molecule_dictionary.max_phase",
        "activities.standard_type",
        "activities.standard_value",
        "activities.standard_units",
        "compound_structures.canonical_smiles",
        "compound_structures.standard_inchi",
        "component_sequences.sequence",
        "assays.confidence_score",
        "docs.chembl_id",
        "docs.year",
        "docs.authors",
    ]

    conn = sql.connect(args.chembl_path, isolation_level=None)
    kinases = create_human_kinases_and_chembl_targets(args.out_path, args.kinases_path)
    kinases = kinases[kinases.type == "SINGLE PROTEIN"].drop("type", axis=1)
    targets = set(kinases.chembl_targets.tolist())
    q = f"""
    SELECT
        {', '.join(select_these)}
    FROM
        activities
        LEFT JOIN assays ON assays.assay_id=activities.assay_id
        LEFT JOIN target_dictionary ON target_dictionary.tid=assays.tid
        LEFT JOIN compound_structures ON activities.molregno=compound_structures.molregno
        LEFT JOIN molecule_dictionary ON activities.molregno=molecule_dictionary.molregno
        LEFT JOIN target_components ON target_dictionary.tid=target_components.tid
        LEFT JOIN component_sequences ON target_components.component_id=component_sequences.component_id
        LEFT JOIN docs ON docs.doc_id=activities.doc_id
    WHERE
        target_dictionary.chembl_id IN ({', '.join([f'"{x}"' for x in targets])})
    AND
        activities.standard_relation="="
    AND
        assays.assay_type="B"
    AND
        activities.standard_type in ("IC50", "Ki", "Kd")
    AND
        assays.confidence_score > 0
    """
    activities_sql = pd.read_sql_query(q, conn)
    activities_sql.columns = select_these
    # Merge sql result with kinases
    activities = pd.merge(
        activities_sql,
        kinases[["chembl_targets", "UniprotID"]],
        left_on="target_dictionary.chembl_id",
        right_on="chembl_targets",
        how="left",
    ).drop(columns=["chembl_targets"])

    # Log transform all values to follow normal distribution with mean 7?
    nm_activities = activities.query("`activities.standard_units` == 'nM'")
    with pd.option_context("chained_assignment", None):
        nm_activities.loc[:, "activities.standard_value"] = nm_activities[
            "activities.standard_value"
        ].apply(lambda x: 9 - (np.log(x) / np.log(10)))
        nm_activities.loc[:, "activities.standard_type"] = nm_activities[
            "activities.standard_type"
        ].apply("p{}".format)
    nm_activities.head()

    return nm_activities


def remove_dummy(activities: pd.DataFrame) -> pd.DataFrame:
    return activities.query("'CHEMBL612545' not in `target_dictionary.chembl_id`")


def remove_extreme(activities: pd.DataFrame) -> pd.DataFrame:
    return activities.query("1 <= `activities.standard_value` <= 15")


def remove_duplicates(activities: pd.DataFrame) -> pd.DataFrame:
    # Only keep the highest value from the same publication
    activities = activities.sort_values(
        "activities.standard_value", ascending=False
    ).drop_duplicates(
        [
            "target_dictionary.chembl_id",
            "molecule_dictionary.chembl_id",
            "docs.chembl_id",
        ]
    )

    # Remove exact duplicates
    activities = activities.drop_duplicates(
        [
            "target_dictionary.chembl_id",
            "molecule_dictionary.chembl_id",
            "activities.standard_value",
        ]
    )

    # Remove duplicates with rounding error
    return (
        activities.assign(
            activities_standard_value_rounded=lambda x: x[
                "activities.standard_value"
            ].round(2)
        )
        .drop_duplicates(
            [
                "target_dictionary.chembl_id",
                "molecule_dictionary.chembl_id",
                "activities_standard_value_rounded",
            ]
        )
        .drop(columns=["activities_standard_value_rounded"])
    )


def remove_same_author(activities: pd.DataFrame) -> pd.DataFrame:
    def shared_authors(group):
        """
        Return True if authors are not shared and we should keep this group
        :param group:
        :return:
        """
        if group.shape[0] == 1:
            return [True]
        authors_per_entry = [
            (set() if entry is None else set(entry.split(", ")))
            for entry in group.values
        ]
        return [
            any(a.intersection(b) for b in authors_per_entry if a != b)
            for a in authors_per_entry
        ]

    # Remove measurement if different publications share at least one author
    no_shared_authors = activities.groupby(
        ["target_dictionary.chembl_id", "molecule_dictionary.chembl_id"]
    )["docs.authors"].transform(shared_authors)
    return activities[no_shared_authors]


def curate_chembl(args):
    activities = load_chembl_data(args)
    print(f"Loaded data:        {activities.shape}")
    activities = remove_dummy(activities)
    print(f"Removed dummies:    {activities.shape}")
    activities = remove_extreme(activities)
    print(f"Removed extremes:   {activities.shape}")
    activities = remove_duplicates(activities)
    print(f"Removed duplicates: {activities.shape}")
    activities = remove_same_author(activities)
    print(f"Removed same author:{activities.shape}")
    activities.to_csv(args.out_path / "activities-chembl33.csv", index=False)
    print(f"Saved to {args.out_path / 'activities-chembl33.csv'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "chembl_path",
        type=pathlib.Path,
        help="Path to the local export of the ChEMBL database in sqlite format.",
    )
    parser.add_argument(
        "kinases_path",
        type=pathlib.Path,
        help="Path to the https://github.com/openkinome/kinodata/blob/master/data/human_kinases.aggregated.csv file",
    )
    parser.add_argument(
        "out_path", type=pathlib.Path, help="Output path of the processed data."
    )

    curate_chembl(parser.parse_args())
