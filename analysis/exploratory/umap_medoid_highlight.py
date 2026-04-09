"""UMAP Visualization of Molecular Embeddings with Medoid Init Highlighting.

Projects all MCL1-mmgbsa molecules into 2-D UMAP space (one panel per embedding
model) and overlays the single initialisation molecule (iteration == -1) from each
xth-closest-to-centroid experiment, colour-coded by xth value.

Usage example
-------------
cd bayesian_optimization && export PYTHONPATH="."
python ../notebooks/umap_medoid_highlight.py \\
    --medoid-db ../../bayesian_optimization_results/mcl1_mmgbsa_medoid_chemberta-mtr_shuffled.sqlite \\
    --medoid-db ../../bayesian_optimization_results/mcl1_mmgbsa_medoid_molformer_shuffled.sqlite
"""

import argparse
import logging
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

# Canonical order of xth values from the ablation
XTH_ORDER = [0, 1, 5, 25, 50, 100, 500]


# ---------------------------------------------------------------------------
# Embedding helpers (identical to umap_init_highlight.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------

def load_medoid_init_smiles(db_paths: list[Path]) -> dict[str, dict[int, str]]:
    """Load init molecules (iteration == -1) grouped by embedding_model and init_batch_size.

    Pools across all provided databases and de-duplicates by taking the unique
    molecule per (embedding_model, init_batch_size) pair. The xth-closest
    selection is deterministic, so multiple repeat runs yield the same molecule.

    :param list db_paths: Paths to SQLite run databases
    :return: Mapping of embedding_model → {init_batch_size → smiles}
    :rtype: dict[str, dict[int, str]]
    """
    frames = []
    for db_path in db_paths:
        con = sqlite3.connect(db_path)
        df = pd.read_sql(
            """SELECT DISTINCT e.embedding_model, e.init_batch_size, qr.molecule
               FROM query_result qr
               JOIN experiment e ON qr.experiment_id = e.id
               WHERE qr.iteration == -1""",
            con,
        )
        con.close()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["embedding_model", "init_batch_size"]
    )

    result: dict[str, dict[int, str]] = {}
    for (emb_model, xth), group in combined.groupby(["embedding_model", "init_batch_size"]):
        result.setdefault(emb_model, {})[int(xth)] = group["molecule"].iloc[0]

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_umap_medoid(
    projections: dict[str, np.ndarray],
    targets: np.ndarray,
    smiles_list: list[str],
    init_by_model: dict[str, dict[int, str]],
    xth_values: list[int],
    out_path: Path,
) -> None:
    """Save a 1-row UMAP figure with medoid init points colour-coded by xth.

    Each column is one embedding model. Background points are coloured by
    MM-GBSA score. The single initialisation molecule for each xth value is
    overlaid as a larger marker whose colour encodes the xth rank.

    :param dict projections: Mapping of embedding name to (N, 2) UMAP projection
    :param np.ndarray targets: MM-GBSA scores, one per molecule
    :param list smiles_list: SMILES strings corresponding to rows in projections
    :param dict init_by_model: embedding_model → {xth → smiles}
    :param list xth_values: Ordered list of xth values to plot and legend
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

    smiles_arr = np.array(smiles_list)
    embedding_names = list(projections.keys())
    n_cols = len(embedding_names)

    bg_cmap = "plasma_r"
    bg_norm = mpl.colors.Normalize(vmin=targets.min(), vmax=targets.max())

    # Discrete colormap for xth values — winter (blue→green) contrasts against plasma_r's warm tones
    n_xth = len(xth_values)
    xth_cmap = mpl.colormaps.get_cmap("winter").resampled(n_xth)
    xth_colors = {xth: xth_cmap(i / max(n_xth - 1, 1)) for i, xth in enumerate(xth_values)}

    # Layout: 1 row × (n_cols + colorbar_xth)
    fig = plt.figure(figsize=(4.5 * n_cols + 1.0, 4.5))
    gs = GridSpec(
        1, n_cols + 1,
        figure=fig,
        width_ratios=[1] * n_cols + [0.04],
        wspace=0.05,
    )
    cax_xth = fig.add_subplot(gs[0, n_cols])

    for col, name in enumerate(embedding_names):
        proj = projections[name]
        ax = fig.add_subplot(gs[0, col])

        # Background: all molecules coloured by MM-GBSA
        ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=targets,
            cmap=bg_cmap,
            norm=bg_norm,
            s=6,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
        )

        # Overlay: one point per xth value
        db_model_key = {
            "ChemBERTa": "chemberta-mtr",
            "MolFormer": "molformer",
            "Morgan Fingerprint": "fingerprint",
        }.get(name, name)
        xth_map = init_by_model.get(db_model_key, {})

        for xth in xth_values:
            smi = xth_map.get(xth)
            if smi is None:
                logger.warning("No init molecule found for embedding '%s', xth=%d", name, xth)
                continue
            mask = smiles_arr == smi
            if not mask.any():
                logger.warning("Init SMILES not found in dataset for xth=%d", xth)
                continue
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                color=xth_colors[xth],
                marker="o",
                s=30,
                linewidths=0.4,
                edgecolors="white",
                zorder=4,
                label=f"xth={xth}",
            )
            logger.info("Plotted init point for '%s' xth=%d", name, xth)

        ax.set_title(name, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        if col == 0:
            ax.set_ylabel("UMAP 2")

    # Colorbar: xth value
    xth_norm = mpl.colors.BoundaryNorm(
        boundaries=[-0.5] + [i + 0.5 for i in range(n_xth)],
        ncolors=n_xth,
    )
    sm_xth = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=-0.5, vmax=n_xth - 0.5),
        cmap=xth_cmap,
    )
    cbar_xth = fig.colorbar(sm_xth, cax=cax_xth)
    cbar_xth.set_ticks(range(n_xth))
    cbar_xth.set_ticklabels([str(v) for v in xth_values])
    cbar_xth.set_label("xth closest", labelpad=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png", "svg"):
        path = out_path.with_suffix(f".{fmt}")
        plt.savefig(path, format=fmt, bbox_inches="tight")
        logger.info("Saved %s", path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Compute embeddings, run UMAP, produce medoid-highlight figure."""
    parser = argparse.ArgumentParser(
        description="UMAP visualization with medoid initialisation points highlighted"
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
    data_group.add_argument(
        "--medoid-db",
        type=Path,
        action="append",
        dest="medoid_dbs",
        metavar="DB_PATH",
        help="SQLite run database(s) from the medoid ablation (can be repeated)",
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

    medoid_dbs = args.medoid_dbs or []
    if not medoid_dbs:
        parser.error("At least one --medoid-db is required")

    logger.info("Using device: %s", DEVICE)

    # Load molecule data
    df = pd.read_csv(args.data)
    smiles_list = df["smiles"].tolist()
    targets = df["target"].values
    logger.info("Loaded %d molecules from %s", len(smiles_list), args.data)

    # Load init molecules from all provided databases
    init_by_model = load_medoid_init_smiles(medoid_dbs)
    xth_values = sorted(
        {xth for xth_map in init_by_model.values() for xth in xth_map}
    )
    logger.info("Found xth values: %s", xth_values)

    # Maps plot display name → database embedding_model value
    embedding_configs = [
        ("ChemBERTa", lambda: embed_huggingface(
            smiles_list, "DeepChem/ChemBERTa-77M-MTR", batch_size=args.batch_size, pool="cls"
        )),
        ("MolFormer", lambda: embed_huggingface(
            smiles_list, "ibm/MoLFormer-XL-both-10pct", batch_size=args.batch_size, pool="pooler"
        )),
        ("Morgan Fingerprint", lambda: embed_morgan(smiles_list)),
    ]

    projections: dict[str, np.ndarray] = {}
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
            metric="euclidean",
        )
        projections[name] = reducer.fit_transform(emb)
        del emb

    logger.info("Plotting...")
    out_path = args.out_dir / "umap_medoid_highlight"
    plot_umap_medoid(projections, targets, smiles_list, init_by_model, xth_values, out_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
