# Using the Bayesian Optimization Framework

This guide covers how to run the Bayesian optimization (BO) loop on your own molecular dataset or
with a custom scoring oracle.

---

## Table of Contents

1. [Quick Start — Pre-scored Dataset](#1-quick-start--pre-scored-dataset)
2. [Custom Oracle via Bash Script](#2-custom-oracle-via-bash-script)
3. [CLI Reference](#3-cli-reference)
4. [Choosing Embeddings, Surrogates, and Samplers](#4-choosing-embeddings-surrogates-and-samplers)
5. [Reading Results](#5-reading-results)

---

## 1. Quick Start — Pre-scored Dataset

Use this mode for retrospective testing: provide a CSV where all molecules have known scores
(e.g. from a previous docking or MM/GBSA run). The optimizer iteratively queries molecules from
the pool, so you can evaluate how efficiently BO recovers top compounds compared to random
screening — without running any new simulations.

### Data format

Your CSV must contain at least:

| column   | type   | description                                           |
|----------|--------|-------------------------------------------------------|
| `smiles` | string | SMILES string of the molecule                         |
| `target` | float  | Binding affinity (**negative**, e.g. −8.5 kcal/mol)  |

```csv
smiles,target
CC(=O)Nc1ccc(cc1)O,-7.2
c1ccc(cc1)N,-5.8
...
```

> **Sign convention**: lower (more negative) = better binder. This matches standard docking score
> conventions.

### Running the optimizer

```bash
cd bayesian_optimization
export PYTHONPATH="."

python run.py \
  --data   path/to/dataset.csv \
  --protein MY_TARGET \
  --n-iter 50 \
  --sample-size 10 \
  --embedding molformer \
  --surrogate linear-empirical \
  --sampler expected-improvement \
  --out my_target_run
```

Results are written to `../bayesian_optimization_results/my_target_run.sqlite`.

---

## 2. Custom Oracle via Bash Script

Use this mode when you want the optimizer to score molecules on-the-fly — for example, by calling
a docking program, or any scoring pipeline you control.

### Writing the oracle script

Your script must:
- Accept a single SMILES string as its first positional argument (`$1`)
- Print a single float to stdout (negative = better, e.g. −8.5)
- Exit with code 0 on success, non-zero on failure

```bash
#!/bin/bash
# oracle.sh — minimal example
SMILES="$1"
echo "-8.5"   # replace with your actual scoring call
```

A realistic example wrapping a docking tool:

```bash
#!/bin/bash
SMILES="$1"

# Write SMILES to temp file, call docking tool, parse output
TMPDIR=$(mktemp -d)
echo "$SMILES" > "$TMPDIR/ligand.smi"

my_docking_tool \
  --ligand "$TMPDIR/ligand.smi" \
  --receptor protein.pdbqt \
  --out "$TMPDIR/result.txt"

grep "^SCORE" "$TMPDIR/result.txt" | awk '{print $2}'
rm -rf "$TMPDIR"
```

Make the script executable:

```bash
chmod +x oracle.sh
```

### Pool CSV format

When using a custom oracle, the `target` column is **not required**. Provide only the candidate
molecules you want to screen:

```csv
smiles
CC(=O)Nc1ccc(cc1)O
c1ccc(cc1)N
...
```

### Running the optimizer with a custom oracle

```bash
cd bayesian_optimization
export PYTHONPATH="."

python run.py \
  --oracle-script ../oracle.sh \
  --data   path/to/pool.csv \
  --protein MY_TARGET \
  --n-iter 50 \
  --sample-size 10 \
  --embedding molformer \
  --surrogate linear-empirical \
  --sampler expected-improvement \
  --out my_target_run
```

When `--oracle-script` is provided, `--data` only needs a `smiles` column — no pre-computed
scores are required.

---

## 3. CLI Reference

All arguments to `bayesian_optimization/run.py`:

### Required

| argument | description |
|----------|-------------|
| `--data PATH` | Path to CSV dataset (see format above) |
| `--protein NAME` | Protein identifier (any string; used for logging). Must be a valid PDB ID if using `--simulate smina`. |
| `--n-iter N` | Number of BO iterations |
| `--embedding MODEL` | Molecular embedding model. Accepts multiple values (runs all combinations). |
| `--surrogate MODEL` | Surrogate model. Accepts multiple values. |
| `--sampler NAME` | Acquisition function. Accepts multiple values. |

### Acquisition / batch size

| argument | default | description |
|----------|---------|-------------|
| `--sample-size N` | 1 | Molecules to query per iteration (batch acquisition) |
| `--init-sampler NAME` | `closest` | Strategy for the first query (before any labels exist) |
| `--init-sample-size N` | 1 | How many molecules to label before iteration 1. When `--init-sampler xth-closest`, selects the Nth closest to the cluster centroid. |

### Output / reproducibility

| argument | default | description |
|----------|---------|-------------|
| `--out NAME` | — | Output filename stem (without `.sqlite`). Written to `bayesian_optimization_results/`. |
| `--seed INT` | random | Random seed for reproducibility |
| `--validate` | off | After each iteration, evaluate surrogate on unlabeled molecules and log MAE/RMSE/R². Cannot be combined with `--simulate` or `--oracle-script`. |
| `--verbose` | off | Enable debug logging |

### Built-in simulators (optional)

| argument | description |
|----------|-------------|
| `--simulate smina` | Use SMINA for real-time docking instead of a pre-scored dataset |
| `--sim-center X Y Z` | Binding pocket center (required with `--simulate`) |
| `--sim-box SX SY SZ` | Bounding box size around pocket (required with `--simulate`) |

### Custom oracle

| argument | description |
|----------|-------------|
| `--oracle-script PATH` | Path to a bash script that scores a molecule (see [section 2](#2-custom-oracle-via-bash-script)) |

---

## 4. Choosing Embeddings, Surrogates, and Samplers

### Embeddings

| name | description | notes |
|------|-------------|-------|
| `molformer` | IBM MolFormer-XL | Best performance in our experiments |
| `chemberta-mtr` | ChemBERTa (multi-task regression) | Good balance of speed and quality |
| `chemberta-mlm` | ChemBERTa (masked language model) | Slightly weaker than MTR variant |
| `fingerprint` | Morgan fingerprints (radius 2, 2048 bits) | Fast; no model loading required |

### Surrogate models

| name | description |
|------|-------------|
| `linear-empirical` | Bayesian Ridge regression |
| `rf` | Random Forest |
| `linear-prior` | Bayesian linear with Student-t prior |
| `gp` | Gaussian Process (RBF kernel) |
| `mlp` | Neural net with MC-Dropout uncertainty | 
| `molformer` | End-to-end fine-tuned MolFormer |
| `constant` | Dummy (constant prediction,  used internally with random/closest samplers) |

### Acquisition functions (samplers)

| name | description |
|------|-------------|
| `expected-improvement` | Classic EI acquisition |
| `ucb` | Upper confidence bound |
| `greedy` | Minimize predicted score |
| `closest` | Closest molecule to cluster centroid |
| `random` | Uniform random selection |
| `diverse` | Maximally spread selection (MaxMin) |
| `explore` | Maximize distance to already-labeled molecules |
| `stochastic-closest` | Random draw from top-K nearest to centroid |
| `xth-closest` | Select the Nth closest to centroid |

> The CLI automatically discards `(random, non-constant-surrogate)` and
> `(closest, non-constant-surrogate)` combinations, since those samplers ignore the surrogate.
> You can safely pass multiple values for each argument and all valid combinations will run.

---

## 5. Reading Results

Results are stored as SQLite databases in `bayesian_optimization_results/`.

### Python

```python
import sqlite3
import pandas as pd

con = sqlite3.connect("bayesian_optimization_results/my_target_run.sqlite")

# Per-iteration query results
queries = pd.read_sql("SELECT * FROM queries", con)
print(queries.columns)
# iteration, smiles, target (true score), prediction, experiment_id

# Experiment metadata
experiments = pd.read_sql("SELECT * FROM experiments", con)
print(experiments.columns)
# embedding, surrogate, sampler, n_iter, sample_size, seed, protein, ...
```

### Analysis notebooks

The `analysis/` directory contains Jupyter notebooks that load results and produce publication
figures. Use `analysis/utils.py` as a starting point:

```python
from utils import load_data

# Load top-k recovery curves from a results DB
df = load_data("../bayesian_optimization_results/my_target_run.sqlite", k_ratio=0.01)
```

`load_data` returns a DataFrame with per-iteration top-k recovery rates, grouped by
embedding/surrogate/sampler combination.
