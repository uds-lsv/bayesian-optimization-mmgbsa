# Takes the average of dg_en_gb and finds the corresponding smiles string for each molecule
import argparse
from pathlib import Path

import pandas as pd

# Extracted from https://github.com/openforcefield/protein-ligand-benchmark/blob/main/data/mcl1/00_data/ligands.yml
BENCHMARK_LOOKUP = {'lig_27': '[H]c1c(c(c(c(c1[H])[H])OC([H])([H])C([H])([H])C([H])([H])C2=C(N(c3c2c(c(c(c3[H])[H])[H])[H])[H])C(=O)[O-])[H])[H]',
 'lig_28': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3C([H])([H])[H])[H])[H])[H])[H])[H])[H]',
 'lig_30': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])[H])[H])[H])[H])[H]',
 'lig_31': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C(F)(F)F)[H])[H])[H])[H])[H]',
 'lig_32': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])[H])C([H])([H])[H])[H])[H])[H])[H]',
 'lig_33': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])[H])Cl)[H])[H])[H])[H]',
 'lig_34': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])[H])C(F)(F)F)[H])[H])[H])[H]',
 'lig_35': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)[H])[H])[H])[H]',
 'lig_36': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])C([H])([H])[H])Cl)[H])[H])[H])[H]',
 'lig_37': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)C([H])([H])[H])[H])[H])[H]',
 'lig_43': '[H]c1c(c(c2c(c(c(c(c2c1[H])[H])[H])OC([H])([H])C([H])([H])C([H])([H])C3=C(N(c4c3c(c(c(c4[H])[H])[H])[H])[H])C(=O)[O-])[H])[H])[H]',
 'lig_46': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c4c(c3[H])C(C(C4([H])[H])([H])[H])([H])[H])[H])[H])[H])[H]',
 'lig_47': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c4c(c3[H])c(c(c(n4)[H])[H])[H])[H])[H])[H])[H]',
 'lig_48': '[H]c1c(c(c2c(c1[H])C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c4c3C(=C(N4[H])[H])[H])[H])[H])[H])[H])[H]',
 'lig_49': '[H]c1c(c2c(c(c1[H])Cl)C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)[H])[H])[H]',
 'lig_50': '[H]c1c(c2c(c(c1[H])Cl)C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)C([H])([H])[H])[H])[H]',
 'lig_52': '[H]c1c(c(c(c2c1C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)[H])[H])[H])Cl)[H]',
 'lig_53': '[H]c1c(c(c(c2c1C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)C([H])([H])[H])[H])[H])Cl)[H]',
 'lig_56': '[H]c1c(c(c2c(c1[H])C(=C(N2C([H])([H])[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)C([H])([H])[H])[H])[H])[H]',
 'lig_58': '[H]c1c(c(c(c2c1C(=C(N2C([H])([H])[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)C([H])([H])[H])[H])[H])Cl)[H]',
 'lig_60': '[H]c1c(c(c2c(c1[H])C(=C(S2)C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)C([H])([H])[H])[H])[H])[H]',
 'lig_61': '[H]c1c(c2c(c(c1[H])Cl)C(=C(S2)C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)C([H])([H])[H])[H])[H]',
 'lig_63': '[H]c1c(c(c(c2c1C(=C(S2)C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)C([H])([H])[H])[H])[H])Cl)[H]',
 'lig_65': '[H]c1c(c2c(c(c1[H])Cl)SC(=C2C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)C([H])([H])[H])[H])C(=O)[O-])[H]',
 'lig_67': '[H]c1c(c(c2c(c1[H])C(=C(O2)C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)C([H])([H])[H])[H])[H])[H]'}


def map_with_error(series, mapping):
    """
    Map values in a Series using a dictionary, raising an error if any values are missing.
    """
    try:
        return series.map(mapping)
    except KeyError as e:
        unmapped_values = series[~series.isin(mapping.keys())].unique()
        raise ValueError(f"The following values could not be mapped: {', '.join(unmapped_values)}") from e



def process(csvfile: str, column: str, lookup_smiles: bool, lookup_benchmark: bool):
    df = pd.read_csv(csvfile)
    if column not in df.columns:
        raise argparse.ArgumentError(f"{column} is not a valid column.")

    df = pd.read_csv(csvfile)
    avg = df.groupby("name")[column].mean()
    avg = avg.reset_index().rename(columns={column: "target"})

    if lookup_smiles:
        # To find the smiles string we use the molecules.smi file created after screening
        smiles_data = pd.read_csv(
            "docking/results.csv", names=["zincid", "smiles", "_score"],
        )
        smiles_lookup = dict(
            zip(smiles_data["zincid"].to_list(), smiles_data["smiles"].to_list())
        )
        avg["name"] = avg["name"].str.removeprefix("ligand_")
        avg["smiles"] = map_with_error(avg["name"], smiles_lookup)

    elif lookup_benchmark:
        avg["name"] = avg["name"].str.removeprefix("ligand_benchmark_")
        avg["smiles"] = map_with_error(avg["name"], BENCHMARK_LOOKUP)

    out = f"{Path(csvfile).stem}_{column}_avg.csv"
    print(f"Writing result to '{out}'.")
    avg.to_csv(out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        help="Raw CSV file containing measurements of all samples for each molecule",
    )
    parser.add_argument("--column", default="dg_en_pb", help="Which column to average")
    parser.add_argument(
        "--lookup-smiles",
        action="store_true",
        default=False,
        help="Converts ligand_zinc_xxxx back to it's SMILES string",
    )
    parser.add_argument(
        "--lookup-benchmark",
        action="store_true",
        default=False,
        help="Converts ligand_benchmark_lig_xx back to it's SMILES string",
    )

    args = parser.parse_args()

    process(csvfile=args.source, column=args.column, lookup_smiles=args.lookup_smiles, lookup_benchmark=args.lookup_benchmark)
