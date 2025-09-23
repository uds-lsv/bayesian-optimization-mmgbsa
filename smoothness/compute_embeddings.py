import argparse
import pathlib
import abc
from typing import List

import numpy as np
import torch.cuda
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from transformers import AutoModel, AutoTokenizer

from transformers import pipeline
import pandas as pd


def embed(model, molecules, batch_size = 256):
    max_length = max(map(len, molecules))
    
    # Setup model specific tokenization and encoding
    if model == "chemberta":
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

        to_embedding = lambda tokens: model(**tokens).last_hidden_state[:, 0, :].cpu()
        to_tokens = lambda batch: tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
    elif model == "molformer":
        tokenizer = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
        )

        to_embedding = lambda tokens: model(**tokens).pooler_output.cpu()
        to_tokens = lambda batch: tokenizer.batch_encode_plus(
            batch,
            padding=True,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt",
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(_device)
    model.eval()

    # Encode batch 
    embeddings = None
    for start in range(0, len(molecules), batch_size):
        batch_subset = slice(start, start + batch_size)
        batch = molecules[batch_subset]
        
        tokens = to_tokens(batch)
        tokens = tokens.to(_device)
        with torch.no_grad():
            result = to_embedding(tokens)

        if embeddings is None:
            # We infer the shape from the first batch
            embeddings = np.zeros((len(molecules), result.shape[1]))
        embeddings[batch_subset] = result.numpy()

    return embeddings



parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["chemberta", "molformer"])
parser.add_argument("outdir", type=pathlib.Path)

args = parser.parse_args()

# Load smiles
smiles = pd.read_csv("../data/MCL1-vina.csv")["smiles"]

embeddings = embed(args.model, smiles.tolist())
np.save(args.outdir / f"{args.model}.npy", embeddings)
