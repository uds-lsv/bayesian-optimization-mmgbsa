#!/usr/bin/env python3
"""ROGI-XD analysis of molecular datasets.

Computes the ROGI-XD Roughness Index for the MCL1 dataset using Morgan
fingerprints and MolFormer embeddings, with MM-GBSA and Vina scores as targets.

The ROGI-XD formulation integrates over 1 - log(N_clusters) / log(N) instead
of the raw distance threshold, making the index independent of the
representational scale.

Usage (inside the rogi-xd Docker image):
    python rogi_xd_analysis.py
    python rogi_xd_analysis.py --skip-molformer   # fingerprints only
    python rogi_xd_analysis.py --n-samples 0      # full dataset

Reference:
    rogi_xd.rogi.rogi — https://github.com/coleygroup/rogi-xd
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rogi_xd.rogi import rogi
from rogi_xd.utils.rogi import IntegrationDomain

from transformers import AutoModel, AutoTokenizer

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
    # MolFormer's forward() calls this method unconditionally, but it was only
    # added to PreTrainedModel in a later transformers release than is installed in the rogi-xd environment.
    if not hasattr(model, "warn_if_padding_and_no_attention_mask"):
        model.warn_if_padding_and_no_attention_mask = lambda *a, **kw: None
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


def calc_rogi_xd(
    xs: np.ndarray,
    y: np.ndarray,
    label: str,
    metric: str,
    nboots: int,
    plot_path: Path = None,
) -> dict:
    """Compute ROGI-XD for a single (representation, target) pair.

    Uses IntegrationDomain.LOG_CLUSTER_RATIO, which is the ROGI-XD formulation.

    :param xs: Either a (N, d) embedding matrix or a list of SMILES strings.
               SMILES strings trigger automatic Morgan FP + Tanimoto distance.
    :param np.ndarray y: Target values
    :param str label: Display name for this representation
    :param str or None metric: Distance metric (None = auto-select per type)
    :param int nboots: Bootstrap resamples for uncertainty estimation
    :param Path plot_path: If set, save ROGI curve to this path
    :return: Dict with keys Embedding, ROGI-XD, Uncertainty
    :rtype: dict
    """
    logger.info("Computing ROGI-XD: %s ...", label)
    result = rogi(
        xs,
        y,
        metric=metric,
        domain=IntegrationDomain.LOG_CLUSTER_RATIO,
        nboots=nboots,
    )
    score = result.rogi
    uncertainty = result.uncertainty if result.uncertainty is not None else float("nan")
    logger.info("  ROGI-XD = %.4f ± %.4f", score, uncertainty)

    if plot_path is not None:
        fig, ax = plt.subplots()
        ax.plot(result.thresholds, result.cg_sds)
        ax.set_xlabel("Threshold (log cluster ratio)")
        ax.set_ylabel("CG SD")
        ax.set_title(f"ROGI-XD curve — {label}\nROGI-XD = {score:.4f} ± {uncertainty:.4f}")
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved curve → %s", plot_path)

    return {"Embedding": label, "ROGI-XD": round(score, 4), "Uncertainty": round(uncertainty, 4)}


def main() -> None:
    """Entry point: compute ROGI-XD scores for all (embedding, target) combinations."""
    parser = argparse.ArgumentParser(
        description="Compute ROGI-XD landscape roughness for MCL1 dataset"
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
        help="Molecules to subsample (0 = full dataset)",
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
        help="Bootstrap resamples for uncertainty estimation (default: 3)",
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

    # --- Embeddings: (label, xs, metric) ---
    # Passing SMILES strings to rogi() triggers automatic Morgan FP computation
    # followed by Tanimoto distance — equivalent to the 'morgan' featurizer in
    # the rogi_xd CLI (see rogi_xd/rogi.py::calc_distance_matrix).
    embedding_configs = []
    embedding_configs.append(("Morgan FP", smiles_list, "tanimoto"))

    if not args.skip_molformer:
        logger.info("Computing MolFormer embeddings...")
        molformer = embed_huggingface(
            smiles_list,
            "ibm/MoLFormer-XL-both-10pct",
            batch_size=args.batch_size,
            pool="pooler",
        )
        logger.info("  shape: %s", molformer.shape)
        # np.ndarray input → euclidean distance by default
        embedding_configs.append(("MolFormer", molformer, "euclidean"))

    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for target_name, y in targets.items():
        for emb_label, xs, metric in embedding_configs:
            label = f"{emb_label} / {target_name}"
            plot_path = None
            if args.plot_dir is not None:
                safe = label.replace(" ", "_").replace("/", "-")
                plot_path = args.plot_dir / f"rogi_xd_{safe}.pdf"
            record = calc_rogi_xd(xs, y, label, metric, args.nboots, plot_path=plot_path)
            record["Target"] = target_name
            record["Embedding"] = emb_label
            results.append(record)

    print("\n=== ROGI-XD Results ===")
    cols = ["Embedding", "Target", "ROGI-XD", "Uncertainty"]
    print(pd.DataFrame(results)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
