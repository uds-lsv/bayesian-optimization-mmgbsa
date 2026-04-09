#!/usr/bin/env python3
"""ROGI and ROGI-XD analysis of molecular datasets.

Computes ROGI and/or ROGI-XD for the MCL1 dataset using Morgan
fingerprints and MolFormer embeddings, with MM-GBSA and Vina scores as targets.

Usage (inside the mmgbsa-bo-rogi Docker image):
    python rogi_analysis.py                        # both methods (default)
    python rogi_analysis.py --method rogi          # ROGI only
    python rogi_analysis.py --method rogi-xd       # ROGI-XD only
    python rogi_analysis.py --skip-molformer       # fingerprints only
    python rogi_analysis.py --n-samples 0          # full dataset
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
import torch
from rogi_xd.rogi import rogi
from rogi_xd.utils.rogi import IntegrationDomain
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Method configuration: name → (IntegrationDomain, x-axis label, output file prefix)
METHOD_CONFIG = {
    "rogi":    (IntegrationDomain.THRESHOLD,         "Distance threshold",            "rogi_"),
    "rogi-xd": (IntegrationDomain.LOG_CLUSTER_RATIO, "Threshold (log cluster ratio)", "rogi_xd_"),
}


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
    # added to PreTrainedModel in a later transformers release than is installed here.
    if not hasattr(model, "warn_if_padding_and_no_attention_mask"):
        model.warn_if_padding_and_no_attention_mask = lambda *_args, **_kwargs: None
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


def calc_rogi(
    xs: np.ndarray,
    y: np.ndarray,
    label: str,
    metric: str,
    method: str,
    nboots: int,
    plot_path: Path = None,
) -> dict:
    """Compute ROGI or ROGI-XD for a single (representation, target) pair.

    :param xs: (N, d) embedding matrix or list of SMILES strings
               (SMILES trigger automatic Morgan FP + Tanimoto distance).
    :param np.ndarray y: Target values.
    :param str label: Display label formatted as "Embedding / Target".
    :param str metric: Distance metric (e.g. 'tanimoto', 'euclidean').
    :param str method: One of 'rogi' or 'rogi-xd'.
    :param int nboots: Bootstrap resamples for uncertainty estimation.
    :param Path plot_path: If set, save ROGI curve plot and JSON data here.
    :return: Dict with keys Method, Embedding, Target, Score, Uncertainty.
    :rtype: dict
    """
    domain, x_label, _ = METHOD_CONFIG[method]
    method_upper = method.upper()
    logger.info("Computing %s: %s ...", method_upper, label)
    result = rogi(xs, y, metric=metric, domain=domain, nboots=nboots)
    score = result.rogi
    uncertainty = result.uncertainty if result.uncertainty is not None else float("nan")
    logger.info("  %s = %.4f ± %.4f", method_upper, score, uncertainty)

    parts = label.split(" / ", 1)
    emb_name, target_name = (parts[0], parts[1]) if len(parts) == 2 else (label, "")

    if plot_path is not None:
        disp_loss = 2 * (result.cg_sds[0] - result.cg_sds)
        auc = trapezoid(disp_loss, result.thresholds)
        fig, ax = plt.subplots()
        ax.plot(result.thresholds, disp_loss)
        ax.fill_between(result.thresholds, 0, disp_loss, alpha=0.1, label=f"AUC = {auc:.4f}")
        ax.legend()
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$2\,(\sigma_0 - \sigma_t)$")
        ax.set_title(f"{method_upper} curve — {label}\n{method_upper} = {score:.4f} ± {uncertainty:.4f}")
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved curve → %s", plot_path)

        json_path = plot_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "method": method_upper,
                    "embedding": emb_name,
                    "target": target_name,
                    "score": score,
                    "uncertainty": uncertainty,
                    "auc": float(auc),
                    "thresholds": np.asarray(result.thresholds).tolist(),
                    "disp_loss": np.asarray(disp_loss).tolist(),
                },
                f,
            )
        logger.info("  Saved curve data → %s", json_path)

    return {
        "Method":      method_upper,
        "Embedding":   emb_name,
        "Target":      target_name,
        "Score":       round(score, 4),
        "Uncertainty": round(uncertainty, 4),
    }


def main() -> None:
    """Entry point: compute ROGI/ROGI-XD scores for all (method, embedding, target) combinations."""
    parser = argparse.ArgumentParser(
        description="Compute ROGI / ROGI-XD landscape roughness for MCL1 dataset"
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
        "--method",
        nargs="+",
        choices=["rogi", "rogi-xd"],
        default=["rogi", "rogi-xd"],
        metavar="METHOD",
        help="Roughness index variant(s) to compute: rogi, rogi-xd (default: both)",
    )
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

    # --- Embeddings: (display_name, xs, metric) ---
    embedding_configs = [("Morgan FP", smiles_list, "tanimoto")]

    if not args.skip_molformer:
        logger.info("Computing MolFormer embeddings...")
        molformer = embed_huggingface(
            smiles_list,
            "ibm/MoLFormer-XL-both-10pct",
            batch_size=args.batch_size,
            pool="pooler",
        )
        logger.info("  shape: %s", molformer.shape)
        embedding_configs.append(("MolFormer", molformer, "euclidean"))

    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for method in args.method:
        _, _, file_prefix = METHOD_CONFIG[method]
        for target_name, y in targets.items():
            for emb_name, xs, metric in embedding_configs:
                label = f"{emb_name} / {target_name}"
                plot_path = None
                if args.plot_dir is not None:
                    safe = label.replace(" ", "_").replace("/", "-")
                    plot_path = args.plot_dir / f"{file_prefix}{safe}.pdf"
                record = calc_rogi(xs, y, label, metric, method, args.nboots, plot_path=plot_path)
                results.append(record)

    print("\n=== Results ===")
    print(pd.DataFrame(results)[["Method", "Embedding", "Target", "Score", "Uncertainty"]].to_string(index=False))


if __name__ == "__main__":
    main()
