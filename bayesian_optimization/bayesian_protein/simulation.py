import abc
import contextlib
import pathlib
import subprocess
import sys
import warnings
from hashlib import md5
from typing import Optional, Tuple, List

import pandas as pd
from openbabel import pybel
from openmm.app import PDBFile
from pdbfixer import PDBFixer

from bayesian_protein.types import MANDATORY_COLUMNS
from bayesian_protein.utils import is_temporary_directory

ResultT = Tuple[float, Optional[pathlib.Path]]


class SimulatorBase(contextlib.AbstractContextManager):
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self._check_data()

    @abc.abstractmethod
    def _check_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, smiles: str) -> ResultT: ...

    def simulate_many(
        self, smiles: List[str]
    ) -> Tuple[List[float], Optional[List[pathlib.Path]]]:
        results = [self(smi) for smi in smiles]
        return tuple(zip(*results))

    def simulate(self, smiles: str) -> ResultT:
        return self(smiles)


class DatasetSimulator(SimulatorBase):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.data = self.data.set_index("smiles")

    def _check_data(self):
        if any(col not in self.data.columns for col in MANDATORY_COLUMNS):
            missing_columns = set(MANDATORY_COLUMNS) - set(self.data.columns)
            raise ValueError(
                f"Dataset is missing mandatory columns: {', '.join(missing_columns)}."
            )

        if self.data[MANDATORY_COLUMNS].isnull().values.any():
            raise ValueError("The dataset contains NaN entries.")

        if (self.data["target"] >= 0).any():
            warnings.warn("The 'target' column contains non-negative entries.")

    def __exit__(self, __exc_type, __exc_value, __traceback):
        super().__exit__(__exc_type, __exc_value, __traceback)

    def __call__(self, smiles: str):
        value = self.data.loc[smiles]["target"]

        if isinstance(value, pd.Series):
            warnings.warn(
                f"Queried '{smiles}' that has {len(value)} entries. Returning the smallest value."
            )
            value = value.min()

        assert isinstance(value, float)
        return value, None


# TODO save output files in out dir
#   Think of a good file name structure
class SminaSimulator(SimulatorBase):
    def __init__(
        self,
        data: pd.DataFrame,
        out: pathlib.Path,
        pdb_id,
        seed: int,
        center: Tuple[float, float, float],
        box: Tuple[float, float, float],
    ):
        super().__init__(data)
        self.out = out
        self.center = center
        self.box = box

        if not self.out.exists() or not self.out.is_dir():
            raise ValueError(f"{self.out} is not a valid output directory")

        elif is_temporary_directory(self.out):
            warnings.warn(
                f"{self.out} appears to be a temporary directory. Output files may be lost."
            )

        self.pdb_id = pdb_id
        self.pdb_file = self.out / f"protein_{self.pdb_id}.pdb"
        self.protein_file = self.out / f"protein_{self.pdb_id}.pdbqt"
        self.seed = seed

    # For some reason self.temp_dir does no longer exist at this point
    # def __del__(self):
    #     self.temp_dir.cleanup()

    def _check_data(self):
        if "smiles" not in self.data.columns:
            raise ValueError("Dataset is missing mandatory columns: smiles.")

        if self.data["smiles"].isnull().values.any():
            raise ValueError("The dataset contains NaN entries.")

    def get_ligand_file(self, smiles: str) -> pathlib.Path:
        """
        Creates a unique file name for the given smiles string.
        The filename consists of ligand_{unique}.pdbqt where unique is the md5
        encoded SMILES string.
        Using the unique file name for each ligand, we can run multiple simulations in
        parallel.
        :param smiles: SMILES string of the ligand file
        :return: Path/Name, where the ligand file will be stored, for smina to access
        """
        md5sum = md5(smiles.encode()).hexdigest()
        return self.out / f"ligand_{md5sum}.pdbqt"

    def get_out_file(self, smiles: str) -> pathlib.Path:
        """
        Creates a unique file name for the given smiles string.
        The filename consists of ligand_{unique}.sdf where unique is the md5
        encoded SMILES string.
        Using the unique file name for each ligand, we can run multiple simulations in
        parallel. The file will be stored in the self.out directory.

        :param smiles: SMILES string of the ligand file
        :return: Path/Name, where the ligand file will be stored, for smina to write to
        """
        md5sum = md5(smiles.encode()).hexdigest()
        return self.out / f"result_{md5sum}.sdf"

    def __enter__(self):
        """
        Fetches the protein from PDB using its pdb, converts it to PDBQT format and
        stores it in a temporary directory.
        """
        super().__enter__()
        fixer = PDBFixer(pdbid=self.pdb_id)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)

        with open(self.pdb_file, "w") as fid:
            PDBFile.writeFile(fixer.topology, fixer.positions, fid)

        self.pdb_to_pdbqt(self.pdb_file, self.protein_file)
        return self

    def __exit__(self, __exc_type, __exc_value, __traceback): ...

    def __call__(self, smiles: str):
        ligand_file = self.get_ligand_file(smiles)
        out_file = self.get_out_file(smiles)
        self.smiles_to_pdbqt(smiles, ligand_file)
        affinity = self.run_smina(
            ligand_file,
            self.protein_file,
            out_file,
            self.center,
            self.box,
            seed=self.seed,
        )
        return affinity, out_file

    @staticmethod
    def pdb_to_pdbqt(pdb_path: pathlib.Path, pdbqt_path: pathlib.Path):
        """
        Converts pdb to PDBQT and stores the output ad pdbqt_path.
        Requires `prepare_receptor` from ADFR suite to be installed.
        See https://ccsb.scripps.edu/adfr/downloads/ or README.md
        :param pdb_path: File of the pdb file
        :param pdbqt_path: Output path
        """
        try:
            out = subprocess.run(
                ["prepare_receptor", "-r", pdb_path, "-o", pdbqt_path],
                stderr=subprocess.PIPE,
            )
            out.check_returncode()
        except subprocess.SubprocessError:
            raise ValueError(
                f'Error while preparing receptor: {out.stderr.decode("utf-8")}'
            )

    @staticmethod
    def smiles_to_pdbqt(
        smiles: str, pdbqt_path: pathlib.Path, pH: Optional[float] = 7.0
    ):
        """
        Converts smiles string to PDBQT format
        :param smiles: Smiles string to convert
        :param pdbqt_path: Output path
        :param pH: pH used for correction
        """
        molecule = pybel.readstring("smi", smiles)
        if pH is not None:
            # add hydrogens at given pH
            molecule.OBMol.CorrectForPH(pH)
            molecule.addh()
        # generate 3D coordinates
        molecule.make3D(forcefield="mmff94s", steps=10000)
        # add partial charges to each atom
        for atom in molecule.atoms:
            atom.OBAtom.GetPartialCharge()
        molecule.write("pdbqt", str(pdbqt_path), overwrite=True)

    @staticmethod
    def run_smina(
        ligand_path: pathlib.Path,
        protein_path: pathlib.Path,
        out_path: pathlib.Path,
        center: Tuple[float, float, float],
        box: Tuple[float, float, float],
        num_modes=1,
        max_retries=5,
        seed=None,
    ):
        """
        Runs smina
        :param ligand_path: Path to PDBQT file of the ligand
        :param protein_path: Path to PDBQT file of the protein
        :param out_path: Path to the output file
        :param num_modes: Number of binding modes to check
        :param max_retries: How many retries before we exit with an error
        :param seed: Seed passed to smina
        :return: Affinity as float
        """
        cmd = [
            "smina",
            "--ligand",
            str(ligand_path),
            "--receptor",
            str(protein_path),
            "--out",
            str(out_path),
            "--center_x",
            str(center[0]),
            "--center_y",
            str(center[1]),
            "--center_z",
            str(center[2]),
            "--size_x",
            str(box[0]),
            "--size_y",
            str(box[1]),
            "--size_z",
            str(box[2]),
            "--num_modes",
            str(num_modes),
        ]

        if seed is not None:
            cmd.append("--seed")
            cmd.append(str(seed))

        for i in range(max_retries):
            try:
                if i > 0:
                    warnings.warn(f"Retrying simulation ({i + 1}/{max_retries})")

                output = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,  # needed to capture output text
                    check=True,
                )
                results = output.stdout.splitlines()[29:-2]
                affinities = []
                for line in results:
                    affinity = line.split()[1]
                    affinities.append(float(affinity))
                return affinities[0]
            except subprocess.CalledProcessError as e:
                warnings.warn(
                    f"Simulation {i + 1} failed with exit code {e.returncode}"
                )
                warnings.warn(f"Output:\n {e.output}")
                warnings.warn(f"Error message:\n {e.stderr}")

                if "is not a valid AutoDock type" in e.stderr:
                    atom = e.stderr.splitlines()[-1].split()[-1]
                    raise InvalidAtomError(
                        f"{atom} is not supported by smina out of the box."
                        "Manual intervention is needed to support this atom for the forcefield calculation."
                    )

        with open(ligand_path, "r") as f:
            mol = next(pybel.readstring("pdbqt", f.read()))
            sys.exit(
                f"Simulation failed after {max_retries} attempts. Ligand was: {mol.write('smi').strip()}"
            )


class InvalidAtomError(BaseException): ...
