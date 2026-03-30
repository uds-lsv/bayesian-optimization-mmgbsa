#!/usr/bin/env python3
"""ROGI analysis of molecular datasets.

Computes the Roughness Index (ROGI) for the MCL1 dataset using Morgan
fingerprints and MolFormer embeddings, with MM-GBSA and Vina scores as targets.

Requires: pip install rogi
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rogi import RoughnessIndex

# Import embedding functions from the existing notebook utility
sys.path.insert(0, str(Path(__file__).parent))
from umap_embeddings import BATCH_SIZE, embed_huggingface  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "bayesian_optimization/data/processed"


def load_data(mmgbsa_path: Path, vina_path: Path, n_samples) -> pd.DataFrame:
    """Load and join MM-GBSA and Vina datasets on SMILES.

    :param Path mmgbsa_path: Path to MCL1-mmgbsa.csv
    :param Path vina_path: Path to MCL1-vina.csv
    :param int or None n_samples: If set, subsample this many molecules
    :return: DataFrame with columns [smiles, mmgbsa, vina]
    :rtype: pd.DataFrame
    """
    mmgbsa = pd.read_csv(mmgbsa_path).rename(columns={"target": "mmgbsa"})
    vina = pd.read_csv(vina_path).rename(columns={"target": "vina"})
    df = mmgbsa.merge(vina[["smiles", "vina"]], on="smiles", how="inner")
    logger.info("Merged dataset: %d molecules", len(df))

    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        logger.info("Subsampled to %d molecules", len(df))

    return df


def main() -> None:
    """Entry point: compute ROGI scores for all (embedding, target) combinations."""
    parser = argparse.ArgumentParser(
        description="Compute ROGI landscape roughness for MCL1 dataset"
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--mmgbsa",
        type=Path,
        default=DATA_DIR / "MCL1-mmgbsa.csv",
    )
    data_group.add_argument(
        "--vina",
        type=Path,
        default=DATA_DIR / "MCL1-vina.csv",
    )
    data_group.add_argument(
        "--n-samples",
        type=int,
        default=5_000,
        help="Molecules to subsample for testing (0 = full dataset)",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for MolFormer inference",
    )
    model_group.add_argument(
        "--nboots",
        type=int,
        default=3,
        help="Bootstrap resamples for ROGI uncertainty (default: 3)",
    )
    model_group.add_argument(
        "--skip-molformer",
        action="store_true",
        help="Skip MolFormer (faster for quick sanity checks)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for ROGI curve plots (skipped if not set)",
    )

    args = parser.parse_args()
    n_samples = args.n_samples if args.n_samples > 0 else None

    df = load_data(args.mmgbsa, args.vina, n_samples)
    smiles_list = df["smiles"].tolist()
    targets = {"MM-GBSA": df["mmgbsa"].values, "Vina": df["vina"].values}

    # --- Embeddings: (display_name, X_or_fps_kwarg, metric) ---
    embedding_configs = []

    embedding_configs.append(("Morgan FP", {"smiles": smiles_list}, "tanimoto"))

    if not args.skip_molformer:
        logger.info("Computing MolFormer embeddings...")
        molformer = embed_huggingface(
            smiles_list,
            "ibm/MoLFormer-XL-both-10pct",
            batch_size=args.batch_size,
            pool="pooler",
        )
        logger.info("  shape: %s", molformer.shape)
        embedding_configs.append(("MolFormer", {"X": molformer}, "euclidean"))

    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for emb_name, X_kwargs, metric in embedding_configs:
        for target_name, y in targets.items():
            logger.info("Computing ROGI: %s / %s ...", emb_name, target_name)
            ri = RoughnessIndex(y, norm_Y=True, metric=metric, verbose=False, **X_kwargs)
            result = ri.compute_index(nboots=args.nboots)
            score, uncertainty = result if isinstance(result, tuple) else (result, float("nan"))
            logger.info("  ROGI = %.4f ± %.4f", score, uncertainty)

            if args.plot_dir is not None:
                fig, ax = plt.subplots()
                ax.plot(ri.thresholds, ri.cg_sds)
                ax.set_xlabel("Threshold")
                ax.set_ylabel("CG SD")
                ax.set_title(f"ROGI curve — {emb_name} / {target_name}\nROGI = {score:.4f} ± {uncertainty:.4f}")
                safe = f"{emb_name}_{target_name}".replace(" ", "_").replace("/", "-")
                plot_path = args.plot_dir / f"rogi_{safe}.pdf"
                fig.savefig(plot_path, bbox_inches="tight")
                plt.close(fig)
                logger.info("  Saved curve → %s", plot_path)

            results.append(
                {
                    "Embedding":   emb_name,
                    "Target":      target_name,
                    "ROGI":        round(score, 4),
                    "Uncertainty": round(uncertainty, 4),
                }
            )

    print("\n=== ROGI Results ===")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
