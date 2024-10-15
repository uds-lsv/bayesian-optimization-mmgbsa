import argparse
import gzip
import multiprocessing as mp
import pathlib
from functools import partial
from typing import Iterator, Optional, List, Set, Tuple

from tqdm import tqdm


def process_line(zinc_id: str, queries: Set[str]) -> Optional[Tuple[bool, str]]:
    return zinc_id in queries, zinc_id


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
            yield line.strip().split("\t")[1]


def process_gz_file(
    input_path: str,
    output_path: str,
    queries: List[str],
    chunk_size: int = 10_000,
):
    """
    Process a gzipped file in parallel, writing results to an output file.

    Args:
        input_path (str): Path to the input .gz file.
        output_path (str): Path to the output file.
        queries (str): Query molecules as ZINC22 IDs
        chunk_size (int): Number of lines to process in each chunk.
    """

    func = partial(process_line, queries=set(queries))
    with mp.Pool() as pool, open(output_path, "w") as f_out:
        for contained, zid in tqdm(
            pool.imap_unordered(func, read_gz_file(input_path), chunksize=chunk_size)
        ):
            if contained:
                f_out.write(f"{zid}, {input_path}\n")
                f_out.flush()  # Ensure data is written to disk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search compressed ZINC22 for a ZINC ID's"
    )
    parser.add_argument(
        "-i", "--inputs", required=True, help="ZINC ids of query molecules.", nargs="+"
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
        "-b", "--batch-size", default=100_000, help="Batch size", type=int
    )

    args = parser.parse_args()
    for ligand_file in tqdm(args.ligands):
        process_gz_file(
            ligand_file,
            f"{args.output}/{ligand_file.stem}.csv",
            args.inputs,
            args.batch_size,
        )
