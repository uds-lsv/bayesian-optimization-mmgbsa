"""UMAP Visualization of Molecular Embeddings with Init Point Highlighting.

Extends umap_embeddings.py to overlay initialisation molecules (iteration == -1)
from one or more Bayesian optimisation run databases onto the UMAP projections.

Produces a single figure with one row per init set (sorted by label) and one
column per embedding model.
"""

import argparse
import logging
import re
import sqlite3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from matplotlib.gridspec import GridSpec
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


def embed_huggingface(
    smiles_list: list[str],
    model_name: str,
    batch_size: int = BATCH_SIZE,
    pool: str = "cls",
) -> np.ndarray:
    """Embed SMILES using a HuggingFace transformer model.

    :param list smiles_list: List of SMILES strings
    :param str model_name: HuggingFace model identifier
    :param int batch_size: Number of molecules per batch
    :param str pool: Pooling strategy — 'cls' for [CLS] token, 'pooler' for pooler_output
    :return: (N, d) embedding matrix
    :rtype: np.ndarray
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    for param in model.parameters():
        param.requires_grad = False
    model.to(DEVICE).eval()

    embeddings = []
    for start in range(0, len(smiles_list), batch_size):
        batch = smiles_list[start : start + batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**tokens)
        if pool == "cls":
            vec = out.last_hidden_state[:, 0, :].cpu().numpy()
        else:
            vec = out.pooler_output.cpu().numpy()
        embeddings.append(vec)

    del model
    return np.vstack(embeddings)


def embed_morgan(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Embed SMILES as Morgan fingerprints.

    :param list smiles_list: List of SMILES strings
    :param int radius: Morgan radius
    :param int n_bits: Fingerprint length
    :return: (N, n_bits) binary fingerprint matrix
    :rtype: np.ndarray
    """
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = np.zeros((len(smiles_list), n_bits))
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        fp = fpgen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, fps[i])
    return fps


def load_init_smiles(db_path: Path) -> dict[str, set[str]]:
    """Load init molecules (iteration == -1) grouped by embedding_model.

    Each database typically contains one experiment per embedding model.
    Returning a dict allows callers to pick the molecules appropriate for
    each embedding column rather than pooling across models.

    :param Path db_path: Path to the SQLite run database
    :return: Mapping of embedding_model → set of SMILES
    :rtype: dict[str, set[str]]
    """
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        """SELECT e.embedding_model, qr.molecule
           FROM query_result qr
           JOIN experiment e ON qr.experiment_id = e.id
           WHERE qr.iteration == -1""",
        con,
    )
    con.close()
    return {
        model: set(group["molecule"].tolist())
        for model, group in df.groupby("embedding_model")
    }


def _sort_key(label: str) -> tuple:
    """Sort key: alphabetical method name, then numeric init size.

    :param str label: Init set label, e.g. 'Diverse (n=50)'
    :return: (method_name, init_size) tuple for sorting
    :rtype: tuple
    """
    m = re.search(r"n=(\d+)", label)
    size = int(m.group(1)) if m else 0
    name = re.sub(r"\s*\(.*\)", "", label).strip()
    return (name, size)


def plot_umap_grid(
    projections: dict,
    targets: np.ndarray,
    smiles_list: list[str],
    init_sets: list[tuple[str, dict[str, set[str]]]],
    out_path: Path,
) -> None:
    """Create and save a grid UMAP figure: rows = init sets, columns = embeddings.

    Background points in each cell are coloured by MM-GBSA score. Init molecules
    for that row/column are overlaid in a distinct colour — each column uses only
    the init molecules from the experiment run with that embedding model.

    :param dict projections: Mapping of embedding name to (N, 2) UMAP projection
    :param np.ndarray targets: MM-GBSA scores for each molecule
    :param list smiles_list: SMILES strings corresponding to rows in projections
    :param list init_sets: List of (label, {embedding_name: smiles_set}) pairs
    :param Path out_path: Output path stem (.pdf and .svg are written)
    """
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.4)
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    # Sort rows by method name, then init size
    init_sets = sorted(init_sets, key=lambda t: _sort_key(t[0]))

    smiles_arr = np.array(smiles_list)
    embedding_names = list(projections.keys())
    n_rows = len(init_sets)
    n_cols = len(embedding_names)

    cmap = "plasma_r"
    norm = mpl.colors.Normalize(vmin=targets.min(), vmax=targets.max())
    highlight_color = "tab:green"

    # Layout: n_rows × (n_cols + 1 colorbar column)
    fig = plt.figure(figsize=(4.5 * n_cols + 0.5, 4 * n_rows))
    gs = GridSpec(
        n_rows, n_cols + 1,
        figure=fig,
        width_ratios=[1] * n_cols + [0.04],
        wspace=0.05,
        hspace=0.3,
    )
    cax = fig.add_subplot(gs[:, n_cols])  # colorbar spans all rows

    for row, (label, init_smiles_by_col) in enumerate(init_sets):
        for col, name in enumerate(embedding_names):
            init_smiles = init_smiles_by_col.get(name, set())
            init_mask = np.isin(smiles_arr, list(init_smiles))
            logger.info(
                "Row %d col '%s' '%s': highlighting %d molecules",
                row, name, label, init_mask.sum(),
            )

            proj = projections[name]
            ax = fig.add_subplot(gs[row, col])

            ax.scatter(
                proj[:, 0],
                proj[:, 1],
                c=targets,
                cmap=cmap,
                norm=norm,
                s=6,
                alpha=0.4,
                linewidths=0,
                rasterized=True,
            )
            ax.scatter(
                proj[init_mask, 0],
                proj[init_mask, 1],
                c=highlight_color,
                marker="o",
                s=30,
                linewidths=0.4,
                edgecolors="white",
                zorder=3,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(name, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=11)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("MM-GBSA score", labelpad=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg"):
        path = out_path.with_suffix(f".{fmt}")
        plt.savefig(path, format=fmt, bbox_inches="tight")
        logger.info("Saved %s", path)

    plt.close(fig)


def main() -> None:
    """Entry point: compute embeddings, run UMAP, produce grid figure."""
    parser = argparse.ArgumentParser(
        description="UMAP visualization with initialisation points highlighted"
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent.parent
        / "bayesian_optimization/data/processed/MCL1-mmgbsa.csv",
        help="Path to MCL1-mmgbsa CSV file",
    )
    data_group.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "figures",
        help="Directory for output figures",
    )
    data_group.add_argument(
        "--init-db",
        type=Path,
        action="append",
        dest="init_dbs",
        metavar="DB_PATH",
        help="SQLite run database to extract init molecules from (can be repeated)",
    )
    data_group.add_argument(
        "--init-label",
        type=str,
        action="append",
        dest="init_labels",
        metavar="LABEL",
        help="Display label for each --init-db (must match count of --init-db); "
             "defaults to the database stem name",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for transformer inference",
    )
    model_group.add_argument(
        "--umap-seed",
        type=int,
        default=42,
        help="Random seed for UMAP",
    )
    model_group.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors (default 30)",
    )
    model_group.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.05,
        help="UMAP min_dist (default 0.05)",
    )
    model_group.add_argument(
        "--umap-init",
        type=str,
        default="pca",
        choices=["pca", "spectral", "random"],
        help="UMAP initialization method (default: pca)",
    )
    args = parser.parse_args()

    # Validate and pair init-db / init-label
    init_dbs = args.init_dbs or []
    init_labels = args.init_labels or []
    if init_labels and len(init_labels) != len(init_dbs):
        parser.error("--init-label count must match --init-db count")
    if not init_labels:
        init_labels = [db.stem for db in init_dbs]

    logger.info("Using device: %s", DEVICE)

    # Load molecule data
    df = pd.read_csv(args.data)
    smiles_list = df["smiles"].tolist()
    targets = df["target"].values
    logger.info("Loaded %d molecules from %s", len(smiles_list), args.data)

    # Maps plot display name → database embedding_model value
    embedding_configs = [
        ("ChemBERTa", "chemberta-mtr", lambda: embed_huggingface(
            smiles_list, "DeepChem/ChemBERTa-77M-MTR", batch_size=args.batch_size, pool="cls"
        )),
        ("MolFormer", "molformer", lambda: embed_huggingface(
            smiles_list, "ibm/MoLFormer-XL-both-10pct", batch_size=args.batch_size, pool="pooler"
        )),
        ("Morgan Fingerprint", "fingerprint", lambda: embed_morgan(smiles_list)),
    ]
    display_to_db_model = {name: db_model for name, db_model, _ in embedding_configs}

    # Load init sets from databases; index by plot display name
    expected_db_models = set(display_to_db_model.values())
    init_sets = []
    for db_path, label in zip(init_dbs, init_labels):
        smiles_by_db_model = load_init_smiles(db_path)
        missing = expected_db_models - smiles_by_db_model.keys()
        assert not missing, (
            f"Database '{db_path}' is missing experiments for: {missing}. "
            f"Found: {set(smiles_by_db_model.keys())}"
        )
        init_smiles_by_col = {
            plot_name: smiles_by_db_model.get(db_model, set())
            for plot_name, db_model in display_to_db_model.items()
        }
        for plot_name, smiles_set in init_smiles_by_col.items():
            logger.info(
                "Loaded %d init molecules for '%s' / %s (%s)",
                len(smiles_set), label, plot_name, db_path,
            )
        init_sets.append((label, init_smiles_by_col))

    projections = {}
    for name, _db_model, compute_emb in embedding_configs:
        logger.info("Computing %s embeddings...", name)
        emb = compute_emb()
        logger.info("  shape: %s", emb.shape)
        logger.info("Running UMAP on %s...", name)
        reducer = umap.UMAP(
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            init=args.umap_init,
            random_state=args.umap_seed,
        )
        projections[name] = reducer.fit_transform(emb)
        del emb

    logger.info("Plotting grid figure (%d rows × %d cols)...", len(init_sets), len(projections))
    out_path = args.out_dir / "umap_init_highlight"
    plot_umap_grid(projections, targets, smiles_list, init_sets, out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
