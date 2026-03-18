import typing

EmbeddingModel = typing.Literal[
    "chemberta-mlm",
    "chemberta-mtr",
    "fingerprint",
    "fingerprint-mtr-concat",
    "molformer",
]
Simulator = typing.Literal["smina"]
Surrogate = typing.Literal[
    "linear-prior", "linear-empirical", "gp", "constant", "mlp", "molformer", "rf"
]
Sampler = typing.Literal[
    "random", "closest", "greedy", "expected-improvement", "explore", "xth-closest"
]
VALID_EMBEDDING_MODELS: typing.List[EmbeddingModel] = list(
    typing.get_args(EmbeddingModel)
)
VALID_SIMULATOR: typing.List[Simulator] = list(typing.get_args(Simulator))
VALID_SURROGATE_MODELS: typing.List[Surrogate] = list(typing.get_args(Surrogate))
VALID_SAMPLERS: typing.List[Sampler] = list(typing.get_args(Sampler))
MANDATORY_COLUMNS = ["smiles", "target"]
