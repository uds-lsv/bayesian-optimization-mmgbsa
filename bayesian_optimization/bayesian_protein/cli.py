import argparse
import itertools
import logging
import pathlib
import random
import warnings
from dataclasses import dataclass
from typing import Optional, List, Callable, Tuple, Literal

import pandas as pd

from bayesian_protein.simulation import SimulatorBase
from bayesian_protein.types import (
    VALID_EMBEDDING_MODELS,
    VALID_SAMPLERS,
    VALID_SIMULATOR,
    VALID_SURROGATE_MODELS,
    EmbeddingModel,
    Sampler,
    Simulator,
    Surrogate,
)


@dataclass
class CommandLineArgs:
    """
    Dataclass to store command line arguments.

    :param n_iter: Number of queries
    :param embedding: The type of embedding to use.
    :param out: Output file path.
    :param data: Path to dataset.
    :param simulate: Name of the simulator.
    :param sampler: Type of sampler to use.
    :param surrogate: Surrogate for optimization.
    :param cluster: Number of clusters.
    :param confidence: Confidence level for multi-armed bandit.
    :param validate: Boolean flag for validation.
    """

    n_iter: int
    embedding: EmbeddingModel
    data: pathlib.Path
    sampler: Sampler
    surrogate: Surrogate
    protein: str
    init_sampler: Literal["closest", "random", "xth-closest"]
    init_sample_size: int = 1
    cluster: Optional[int] = 1
    out: Optional[pathlib.Path] = None
    simulate: Optional[Simulator] = None
    sim_center: Optional[Tuple[float, float, float]] = None
    sim_box: Optional[Tuple[float, float, float]] = None
    confidence: Optional[float] = 1.0
    validate: Optional[bool] = False
    seed: Optional[int] = None
    sample_size: Optional[int] = 1
    # For internal use only
    _simulator: Optional[Callable[[pd.DataFrame], SimulatorBase]] = None

    def __post_init__(self):
        if isinstance(self.out, str):
            self.out = pathlib.Path(self.out)
            if "~" in str(self.out):
                self.out = self.out.expanduser()

        if isinstance(self.data, str):
            self.data = pathlib.Path(self.data)
            if "~" in str(self.data):
                self.data = self.data.expanduser()

        self._check_paths()

        if self.simulate is not None and self.validate:
            raise ValueError(
                "--validate cannot be used with a simulator. "
                "A dataset with measured affinities is required."
            )

        if self.simulate is not None and self.out is None:
            raise ValueError("The simulator requires a (valid) output directory.")

        if self.simulate is not None and any(
            arg is None for arg in [self.sim_center, self.sim_box]
        ):
            raise ValueError("Simulation requires --sim-center and --sim-box.")

        if self.cluster > 1 and self.confidence is None:
            raise ValueError("--confidence is required when --cluster is specified")
        elif self.cluster > 1 and self.confidence == 0.0:
            warnings.warn("--confidence is set to 0 while --cluster is specified")

        if self.seed is None:
            self.seed = int(random.random() * 100_000)

    def _check_paths(self):
        if not self.data.exists():
            raise ValueError(f"The dataset path '{self.data}' does not exist.")

    def set_simulator(self, simulator_cls: Callable[[pd.DataFrame], SimulatorBase]):
        """
        Sets the simulator to use to this simulator instance.
        :param simulator_cls: A function that takes the data and returns a simulator
        """
        self._simulator = simulator_cls

    def get_simulator(self, data):
        # Simulator has been altered by hand by the user
        if self._simulator is not None:
            return self._simulator(data)

        if self.simulate == "smina":
            # TODO: fix if protein is chembl id
            #  if args.simulate == "smina", but the args.protein is a chembl id
            #  an error will be thrown
            #  The user should be made aware that the simulator expects a PDB id
            from bayesian_protein.simulation import SminaSimulator

            return SminaSimulator(
                data, self.out, self.protein, self.seed, self.sim_center, self.sim_box
            )
        elif self.simulate is None:
            from bayesian_protein.simulation import DatasetSimulator

            return DatasetSimulator(data)


def parse_args() -> List[CommandLineArgs]:
    """
    Parses command line arguments and returns an instance of CommandLineArgs.

    :return: CommandLineArgs instance with parsed values.
    """
    parser = argparse.ArgumentParser(
        description="Command Line Interface to the ligand optimization process"
    )
    parser.add_argument(
        "--n-iter", type=int, required=True, help="Number of queries to make."
    )
    parser.add_argument(
        "--embedding",
        type=str,
        choices=VALID_EMBEDDING_MODELS,
        required=True,
        help="Specifies the type of embedding to use.",
        nargs="+",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Path to the output directory where results will be saved. Only required if a simulator is used.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the csv dataset. If no simulator is specified, i.e. --simulate is not set, the data set will "
        "be expected to contain measured binding affinities. "
        "The csv file has to contain at least a 'smiles' and 'target' column. The 'target' "
        "column should contain the *negative* binding affinities. "
        "If --simulate is used, only the 'smiles' column is required.",
    )
    parser.add_argument(
        "--simulate",
        type=str,
        choices=VALID_SIMULATOR,
        help="Specifies the simulator to use. Optional if --data is provided.",
    )
    parser.add_argument(
        "--sim-center",
        type=float,
        nargs=3,
        default=None,
        help="Center of the pocket. Must be provided if using simulate. --sim-center x y z",
    )
    parser.add_argument(
        "--sim-box",
        type=float,
        nargs=3,
        default=None,
        help="Size of the bounding box around the pocket. --sim-box sizex sizey sizez",
    )

    parser.add_argument(
        "--protein",
        type=str,
        required=True,
        help="Specified the docking target protein. Needs to be a valid PDB id if used with the simulator."
        "Otherwise is just stored for information.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=VALID_SAMPLERS,
        required=True,
        help="Type of sampler to use.",
        nargs="+",
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        choices=VALID_SURROGATE_MODELS,
        required=True,
        help="Surrogate model for optimization.",
        nargs="+",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=1,
        help="Number of k-means clusters to use. If set, --confidence must also be provided.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        help="Confidence level for the multi-armed bandit. Required if --cluster is specified.",
        default=1.0,
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Boolean flag to trigger validation on the rest of the dataset. Cannot be used with a simulator.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Sets the seed for numpy/torch etc. If no seed is set, a random value will be generated and saved in the database.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of samples to query in each iteration.",
        default=1,
    )

    parser.add_argument(
        "--init-sampler",
        type=str,
        choices=["random", "closest", "xth-closest"],
        help="How to choose the first sample. 'random' simply chooses one random sample. 'closest' chooses the cluster "
             "centroid in the embedding space. xth-closest chooses the xth closest input to the centroid, can be set "
             "using --init-sample-size",
        default="closest"
    )

    parser.add_argument(
        "--init-sample-size",
        type=int,
        default=1,
        help="How many molecule to label before the first iteration. If --init-sampler='xth-closest' this determines"
             "which input to choose, i.e. if --init-sample-size=5 we will choose the 5th closest input."
    )

    parser.add_argument(
        "--verbose",
        help="Output debugging logs",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    delattr(args, "loglevel")

    cli_args = set()
    for combination in itertools.product(args.embedding, args.surrogate, args.sampler):
        embedding, surrogate, sampler = combination
        if sampler in ("random", "closest") and surrogate != "constant":
            warnings.warn(
                f"Discarding combination {combination} as {sampler} sampling does not depend on the surrogate."
                f"Therefore the choice of the surrogate model is arbitrary. Using constant function as surrogate model instead"
            )
            surrogate = "constant"

        as_dict = vars(args)
        as_dict.update(
            {"embedding": embedding, "surrogate": surrogate, "sampler": sampler}
        )

        # Creates unhashable lists by default
        as_dict["sim_center"] = (
            tuple(as_dict["sim_center"]) if as_dict.get("sim_center") else None
        )
        as_dict["sim_box"] = (
            tuple(as_dict["sim_box"]) if as_dict.get("sim_box") else None
        )
        cli_args.add(tuple(as_dict.items()))

    return [CommandLineArgs(**dict(args)) for args in cli_args]
