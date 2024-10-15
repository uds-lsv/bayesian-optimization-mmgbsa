import argparse
import gzip
import multiprocessing as mp
from functools import partial
from typing import Iterator, Literal, Optional
import warnings

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdFMCS
from tqdm import tqdm
import pathlib


def mcs_similarity(a: Chem.Mol, b: Chem.Mol):
    """
    Most common substructure similarity between two molecules as described in
    https://pubmed.ncbi.nlm.nih.gov/26419860/
    """
    mcs = rdFMCS.FindMCS((a, b), ringMatchesRingOnly=True, completeRingsOnly=True)
    sim = mcs.numBonds / (a.GetNumBonds() + b.GetNumBonds() - mcs.numBonds)
    return sim


def tanimoto_similarity(a: Chem.Mol, b: Chem.Mol):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    fp1 = generator.GetSparseFingerprint(a)
    fp2 = generator.GetSparseFingerprint(b)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def process_line(
    molecule: str, threshold: float, query: Chem.Mol, similarity_func
) -> Optional[float]:
    mol = Chem.MolFromSmiles(molecule)

    if mol is None:
        warnings.warn(f"Failed to load molecule {molecule}.")
        return None, float("-inf")

    similarity = similarity_func(mol, query)
    return molecule if similarity > threshold else None, similarity


def read_gz_file(file_path: str) -> Iterator[str]:
    """
    Generator function to read lines from a gzipped file.

    Args:
        file_path (str): Path to the gzipped file.

    Yields:
        str: Each line from the file, decoded and stripped.
    """
    with gzip.open(file_path, "rt") as f:
        for line in f:
            yield line.strip()


def process_gz_file(
    input_path: str,
    output_path: str,
    threshold: float,
    query: str,
    similarity: Literal["mcs", "tanimoto"],
    chunk_size: int = 10_000,
):
    """
    Process a gzipped file in parallel, writing results to an output file.

    Args:
        input_path (str): Path to the input .gz file.
        output_path (str): Path to the output file.
        threshold (float): Similarity threshold for filtering processed strings.
        query (str): Query molecule as SMILES
        chunk_size (int): Number of lines to process in each chunk.
    """

    if similarity == "mcs":
        similarity_func = mcs_similarity
    elif similarity == "tanimoto":
        similarity_func = tanimoto_similarity
    else:
        raise ValueError(f"Invalid similarity measure {similarity}.")

    query = Chem.MolFromSmiles(query)
    func = partial(
        process_line, threshold=threshold, query=query, similarity_func=similarity_func
    )
    with mp.Pool() as pool, open(output_path, "w") as f_out:
        for result, similarity in tqdm(
            pool.imap_unordered(func, read_gz_file(input_path), chunksize=chunk_size)
        ):
            if result:
                f_out.write(f"{result},{similarity}\n")
                f_out.flush()  # Ensure data is written to disk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Screen ZINC22 based on maximum common substructure"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=False,
        help="Input query molecules as SMILES string.",
        default="[H]c1c(c(c(c2c1C(=C(N2[H])C(=O)[O-])C([H])([H])C([H])([H])C([H])([H])Oc3c(c(c(c(c3[H])C([H])([H])[H])Cl)C([H])([H])[H])[H])[H])Cl)[H]",
    )
    parser.add_argument(
        "-l",
        "--ligands",
        required=True,
        help="Gzip compressed input ligand file with SMILES strings.",
        nargs="+",
        type=pathlib.Path,
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "-s",
        "--similarity",
        required=False,
        choices=["mcs", "tanimoto"],
        help="Which similarity measure to use.",
        default="mcs",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        required=False,
        help="Similarity threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-b", "--batch-size", default=100_000, help="Batch size", type=int
    )

    args = parser.parse_args()
    for ligand_file in tqdm(args.ligands):
        try:
            process_gz_file(
                ligand_file,
                f"{args.output}/{ligand_file.stem}.csv",
                args.threshold,
                args.input,
                args.similarity,
                args.batch_size,
            )
        except:
            pass
