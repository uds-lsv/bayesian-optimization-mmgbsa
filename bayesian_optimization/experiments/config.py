import os
import pathlib

this = pathlib.Path(__file__)
STORAGE_DIR = pathlib.Path(os.environ.get("STORAGE_DIR", this.parent.parent))
DATA_DIR = STORAGE_DIR / "data"
FIGURE_DIR = STORAGE_DIR / "figures"
RESULTS_DIR = STORAGE_DIR / "results"

if not DATA_DIR.exists():
    raise RuntimeError(f"{DATA_DIR} does not exist.")

if not FIGURE_DIR.exists():
    FIGURE_DIR.mkdir(parents=True)

if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)
