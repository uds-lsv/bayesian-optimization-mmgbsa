import abc
from typing import List

import numpy as np
import torch.cuda
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from transformers import AutoModel, AutoTokenizer

from bayesian_protein.types import VALID_EMBEDDING_MODELS, EmbeddingModel


class BaseEmbedder(abc.ABC):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __call__(self, molecules: List[str]):
        max_length = max(map(len, molecules))
        embeddings = None
        for start in range(0, len(molecules), self.batch_size):
            batch_subset = slice(start, start + self.batch_size)
            batch = molecules[batch_subset]
            res = self.embed_batch(batch, max_length)
            if embeddings is None:
                # We infer the shape from the first batch
                embeddings = np.zeros((len(molecules), res.shape[1]))
            embeddings[batch_subset] = res

        return embeddings

    @abc.abstractmethod
    def embed_batch(self, batch: List[str], max_length: int) -> np.ndarray:
        """
        Embeds the list of molecules
        :param batch: N List of SMILES strings
        :param max_length: Longest string in the dataset (not batch)
        :return: (N, d) numpy array where the ith row is the embedding of the ith molecule in the batch
        """
        raise not NotImplementedError()


class HuggingfaceEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = False

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self._device)
        self.model.eval()

    def tokenize(self, batch: List[str], pad_to: int):
        return self.tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            max_length=pad_to,
            truncation=True,
        )

    def embed_batch(self, batch: List[str], max_length) -> np.ndarray:
        tokens = self.tokenize(batch, pad_to=max_length)
        tokens = tokens.to(self._device)
        with torch.no_grad():
            result = self.model(**tokens)
            return result.last_hidden_state[:, 0, :].cpu()


class MolformerEmbedder(BaseEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self._device)
        self.model.eval()

    def tokenize(self, batch: List[str], pad_to: int):
        return self.tokenizer.batch_encode_plus(
            batch,
            padding=True,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt",
        )

    def embed_batch(self, batch: List[str], max_length) -> np.ndarray:
        tokens = self.tokenize(batch, pad_to=max_length)
        tokens = tokens.to(self._device)
        with torch.no_grad():
            result = self.model(**tokens)
            return result.pooler_output.cpu()


class FingerprintEmbedder(BaseEmbedder):
    def embed_batch(self, batch: List[str], max_length: int) -> np.ndarray:
        fps = np.zeros((len(batch), 2048))
        fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        for i, smi in enumerate(batch):
            mol = Chem.MolFromSmiles(smi)
            fp = fpgen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, fps[i, :])
        return fps


class ConcatEmbedder(BaseEmbedder):
    def __init__(self, model: str, batch_size: int):
        super().__init__(batch_size)
        self.fingerprint = FingerprintEmbedder(batch_size)
        self.hf_embedder = HuggingfaceEmbedder(model, batch_size=batch_size)

    def embed_batch(self, batch: List[str], max_length: int) -> np.ndarray:
        fps = self.fingerprint.embed_batch(batch, max_length=max_length)
        hf = self.hf_embedder.embed_batch(batch, max_length=max_length)
        assert fps.shape[0] == hf.shape[0]
        return np.hstack([hf, fps])


def embedding_factory(embedding_model: EmbeddingModel, batch_size: int) -> BaseEmbedder:
    if embedding_model == "fingerprint":
        return FingerprintEmbedder(batch_size)
    elif embedding_model == "chemberta-mlm":
        return HuggingfaceEmbedder("DeepChem/ChemBERTa-77M-MLM", batch_size)
    elif embedding_model == "chemberta-mtr":
        return HuggingfaceEmbedder("DeepChem/ChemBERTa-77M-MTR", batch_size)
    elif embedding_model == "molformer":
        return MolformerEmbedder(batch_size)
    elif embedding_model == "fingerprint-mtr-concat":
        return ConcatEmbedder("DeepChem/ChemBERTa-77M-MTR", batch_size)
    else:
        raise ValueError(
            f"Invalid embedding model {embedding_model}. Valid choices are {', '.join(VALID_EMBEDDING_MODELS)}"
        )
