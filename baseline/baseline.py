import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sample_dataset import sample_dataset
from sklearn.utils import gen_batches
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=f"baseline.log",
    filemode="w",
)

logger = logging.getLogger(__file__)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=(1024, 1024, 512, 256)):
        super(MLPRegressor, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layer_sizes[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, device="cuda"):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        loss_fn = nn.HuberLoss()

        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        _train_curve = []
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                logger.debug(
                    f"[{epoch:>3}] ({i:>3}/{len(train_loader)}) {loss.item():.3f}"
                )

            avg_train_loss = train_loss / len(train_loader)
            _train_curve.append(avg_train_loss)

        return _train_curve

    @torch.no_grad()
    def predict(self, X: np.ndarray):
        X = torch.from_numpy(X).float()
        loader = DataLoader(X, batch_size=256, shuffle=False)

        predictions = []
        self.eval()
        for batch_X in loader:
            predictions.append(self(batch_X.to("cuda")).cpu())

        return torch.cat(predictions).numpy()

    def loss(self, X, y):
        p = torch.from_numpy(self.predict(X)).float()
        y = torch.from_numpy(y).float()
        return nn.functional.huber_loss(p, y)


def featurize(smiles: list[str], kind):
    molecules = list(map(Chem.MolFromSmiles, smiles))
    if kind == "fingerprint":
        mpfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return np.vstack(list(map(mpfgen.GetFingerprintAsNumPy, molecules)))
    else:
        from transformers import AutoModel, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if kind == "chemberta":
            model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR").to(device)
            tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        elif kind == "molformer":
            model = AutoModel.from_pretrained(
                "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
            )

        pad_to = max(map(len, smiles))

        embeddings = []
        model.eval()
        for batch_idx in gen_batches(len(smiles), 256):
            model_inputs = tokenizer(
                smiles[batch_idx],
                padding="max_length",
                max_length=pad_to,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                output = model(**model_inputs)

                if kind == "chemberta":
                    embeddings.append(output.last_hidden_state[:, 0, :].cpu())
                else:
                    embeddings.append(output.pooler_output.cpu())

        return torch.cat(embeddings, dim=0).numpy()


def run_baseline_experiments(df):
    results = []

    nice_feature_names = {
        "chemberta": "ChemBERTa-2",
        "molformer": "MolFormer",
        "fingerprint": "Morgan Fingerprint",
    }

    for size, epochs in [(1000, 150), (5000, 150), (10_000, 200)]:
        subset = sample_dataset(df["smiles"].to_list(), size, scaffolds=df["scaffolds"])
        for feature_type in ["fingerprint", "chemberta", "molformer"]:
            logger.info(
                f"Training a model based on '{feature_type}' for {epochs} epochs with {size} samples."
            )
            X = featurize(df["smiles"].to_list(), feature_type)
            y = df["target"].values

            train_idx = df["smiles"].isin(subset["molecules"])

            train_X = X[train_idx]
            test_X = X[~train_idx]

            train_y = y[train_idx]
            test_y = y[~train_idx]

            model = MLPRegressor(X.shape[1], [1024, 1024, 512, 256])
            model = model.to("cuda")
            model.fit(train_X, train_y, epochs=epochs)
            logger.info(f"Training done.")

            logger.info(
                f"[{size:>5}] {feature_type} \t Train loss: {model.loss(train_X, train_y):.3f} \t Test loss {model.loss(test_X, test_y):.3f}"
            )
            logger.info(f"Calculating metrics...")

            # Calculate EF and Recall
            predictions = model.predict(X)

            for q in [0.01, 0.05, 0.1]:
                quantile = df["target"].quantile(q)
                hits_true = df["target"] < quantile
                hits_pred = predictions < quantile

                overlap = (hits_true & hits_pred).sum()
                overlap_ratio = overlap / size
                pool_ratio = hits_true.sum() / len(df)

                data_ratio = len(df) / size
                recall = overlap / hits_true.sum()

                q_results = {
                    "Training samples": size,
                    "Quantile Size": hits_true.sum(),
                    "Quantile": q,
                    "Recall": recall,
                    "Enrichment Factor": recall * data_ratio,
                    "overlap_ratio": overlap_ratio,
                    "pool_ratio": pool_ratio,
                    "max_enrichment_factor": data_ratio,
                    "Feature": nice_feature_names[feature_type],
                    "Overlap": overlap,
                    "Ratio_D": data_ratio,
                    "Ratio_R": overlap / hits_true.sum(),
                }

                results.append(q_results)

            logger.info(f"Done.")

    results = pd.DataFrame(results)
    return results


def load_data():
    logger.info(f"Loading data...")
    docking_df = pd.read_csv(
        "../docking/results.csv", names=["zincid", "smiles", "docking_score"]
    )
    docking_df = docking_df.set_index("zincid")
    mmpbsa_df = pd.read_csv("../mmpbsa_dg_en_gb_avg.csv")
    mmpbsa_df = mmpbsa_df.set_index("name")
    df = mmpbsa_df.join(docking_df, how="inner", lsuffix="_mmpbsa", rsuffix="_docking")
    assert (df["smiles_mmpbsa"] == df["smiles_docking"]).all()
    df["smiles"] = df["smiles_mmpbsa"]
    df = df.drop(["smiles_mmpbsa", "smiles_docking"], axis="columns")

    logger.info(f"Calculating scaffolds...")
    df["scaffolds"] = df["smiles"].map(MurckoScaffold.MurckoScaffoldSmiles)
    return df


def main():
    df = load_data()
    logger.info(f"Data preprocessing")
    results = run_baseline_experiments(df)
    results.to_csv("results.csv")


if __name__ == "__main__":
    main()
