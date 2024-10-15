import argparse
from collections import defaultdict
import csv
import pprint
import sqlite3
import pathlib

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "-db",
    "--databases",
    required=True,
    help="EasyDock Database from which the molecule should be extracted",
    type=pathlib.Path,
    nargs="+",
)
parser.add_argument(
    "-o",
    "--outputdir",
    help="Output directory",
    type=pathlib.Path,
    default=pathlib.Path(__file__).parent.parent / "mmpbsa" / "ligands",
)
args = parser.parse_args()


if not args.outputdir.exists():
    args.outputdir.mkdir()


stats = defaultdict(lambda: 0)
results = []
print(f"Extracting results from {len(args.databases)} databases.")

with tqdm(total=2500 * len(args.databases)) as pbar:
    for db in args.databases:
        pbar.set_description(str(db))

        conn = sqlite3.connect(db)
        cur = conn.cursor()

        res = cur.execute("SELECT id, smi, mol_block, docking_score FROM mols")
        rows = res.fetchall()

        for i, (id_, smi, mol, score) in enumerate(rows):
            pbar.update(1)
            pbar.set_postfix_str(id_)
            if mol is None:
                # warnings.warn(f"No .mol block availabe for {id_}!")
                stats[db] += 1
                continue
            elif score is None:
                # warnings.warn(f"No score availabe for {id_}!")
                continue

            filename = args.outputdir / f"ligand_{id_}.mol"
            # print(f"({i:>6}/{len(rows)})  Writing to {filename}")
            with open(filename, "w") as f:
                f.write(mol)

            results.append((id_, smi, score))


pprint.pprint(stats)

with open("results.csv", "w") as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(results)
