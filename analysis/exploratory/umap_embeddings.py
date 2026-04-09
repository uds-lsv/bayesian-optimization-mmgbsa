"""UMAP Visualization of Molecular Embeddings.

Compare ChemBERTa, MolFormer, and Morgan fingerprint embeddings of MCL1-mmgbsa molecules.
"""

import argparse
import logging
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
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


def plot_umap(
    projections: dict,
    targets: np.ndarray,
    out_dir: Path,
) -> None:
    """Create and save a 3-panel UMAP figure coloured by MM-GBSA score.

    :param dict projections: Mapping of embedding name to (N, 2) UMAP projection
    :param np.ndarray targets: MM-GBSA scores for each molecule
    :param Path out_dir: Directory where figures are saved
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

    cmap = "plasma_r"
    norm = mpl.colors.Normalize(vmin=targets.min(), vmax=targets.max())

    fig = plt.figure(figsize=(14, 4))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.04], wspace=0.1)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in range(1, 3)]
    cax = fig.add_subplot(gs[0, 3])

    for i, (ax, (name, proj)) in enumerate(zip(axes, projections.items())):
        ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=targets,
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=0.6,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("UMAP 2")

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("MM-GBSA score", labelpad=8)

    fig.suptitle(
        "UMAP of MCL1 MM-GBSA molecules", fontsize=14, fontweight="bold", y=1.02
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg"):
        path = out_dir / f"umap_embeddings.{fmt}"
        plt.savefig(path, format=fmt, bbox_inches="tight")
        logger.info("Saved %s", path)

    plt.close(fig)


def main() -> None:
    """Entry point: compute embeddings, run UMAP, save figures."""
    parser = argparse.ArgumentParser(
        description="UMAP visualization of molecular embeddings"
    )
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent.parent
        / "data/MCL1-mmgbsa.csv",
        help="Path to MCL1-mmgbsa CSV file",
    )
    data_group.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "figures",
        help="Directory for output figures",
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
        help="UMAP n_neighbors: controls local vs global structure (default 30)",
    )
    model_group.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.05,
        help="UMAP min_dist: controls point packing tightness (default 0.05)",
    )
    model_group.add_argument(
        "--umap-init",
        type=str,
        default="pca",
        choices=["pca", "spectral", "random"],
        help="UMAP initialization method (default: pca, more stable than spectral)",
    )
    args = parser.parse_args()

    logger.info("Using device: %s", DEVICE)

    # Load data
    df = pd.read_csv(args.data)
    smiles_list = df["smiles"].tolist()
    targets = df["target"].values
    logger.info("Loaded %d molecules from %s", len(smiles_list), args.data)

    # Compute embeddings and project to 2D one at a time to minimise peak memory
    embedding_configs = [
        ("ChemBERTa", lambda: embed_huggingface(
            smiles_list, "DeepChem/ChemBERTa-77M-MTR", batch_size=args.batch_size, pool="cls"
        )),
        ("MolFormer", lambda: embed_huggingface(
            smiles_list, "ibm/MoLFormer-XL-both-10pct", batch_size=args.batch_size, pool="pooler"
        )),
        ("Morgan Fingerprint", lambda: embed_morgan(smiles_list)),
    ]

    projections = {}
    for name, compute_emb in embedding_configs:
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

    logger.info("Plotting...")
    plot_umap(projections, targets, args.out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
