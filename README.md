# Bayesian Optimization with MM/GBSA Scoring

This repository accompanies the paper:

> **Bayesian Optimization for Structure-Based Drug Design with High-Fidelity MM/GBSA Scoring**
> [https://doi.org/10.1101/2025.06.22.660936](https://doi.org/10.1101/2025.06.22.660936)

We study active learning / Bayesian optimization (BO) for structure-based drug design, using MM/GBSA
as a high-fidelity oracle in place of fast docking scores. The main finding is that BO with
molecular language model embeddings efficiently recovers top-scoring compounds with far fewer
oracle evaluations than random screening.

![Top-k recovery across iterations](analysis/figures/top_k_iter_both_Linear.png)

For a guide on running the optimizer on your own data or with a custom scoring oracle, see
[USAGE.md](USAGE.md).

---

## Reproducing Results

### Setup

All steps except docking use the base conda environment:

```bash
conda env create -f docker/environment.yml
conda activate base
```

The **docking step requires a separate environment** due to Python version incompatibilities with
the `vina` and `easydock` packages:

```bash
conda env create -f docker/environment-docking.yml
```

Docker images are also provided (`docker/Dockerfile.base`, `docker/Dockerfile.docking`) for
fully reproducible containerized execution.

### Prepare the data

Run the preprocessing notebook to produce the required CSV files:
```
data/process_data.ipynb
```
Copy the outputs to `bayesian_optimization/data/processed/`. Expected files:
- `MCL1-mmgbsa.csv` — MCL1 MM/GBSA scores
- `MCL1-vina.csv` — MCL1 Vina docking scores
- `Enamine10k_scores.csv`
- `Enamine50k_scores.csv`

### Run Bayesian Optimization

```bash
bash run_bayesian_optimization.sh
```

Results are written to `bayesian_optimization_results/`. To run ablations:

```bash
bash run_seeds.sh           # Initialization robustness
bash run_batch_size.sh      # batch acquisition size
bash run_diverse_init.sh    # diverse initialization
bash run_ucb_experiment.sh  # UCB acquisition function
bash run_ucb_kappa.sh       # UCB kappa sweep
```

### Generate figures

Open the notebooks in `analysis/` corresponding to the experiment of interest (e.g.
`analysis/mcl1_mmgbsa.ipynb` for the main MM/GBSA results). Figures are saved to
`analysis/figures/`.

---

## Full Pipeline (from scratch)

If you want to reproduce the full pipeline including data collection:

1. **Screening** — ZINC22 similarity screening; see `screening/README.md`
2. **Docking** — AutoDock Vina via easydock (docking environment); see `docking/README.md`
3. **MD simulation + MM/GBSA** — GROMACS + gmx_MMPBSA; see `simulation_configuration/README.md`
4. **Preprocessing** — `data/process_data.ipynb`
5. **BO experiments** — `run_bayesian_optimization.sh`
6. **Analysis** — notebooks in `analysis/`

---

## Repository Structure

```
bayesian-optimization-mmgbsa/
│
├── docker/                      # Conda environment files and Docker images
│   ├── environment.yml          # Base environment (BO + analysis)
│   ├── environment-docking.yml  # Docking environment (separate due to version constraints)
│   ├── Dockerfile.base          # Docker image for base environment
│   └── Dockerfile.docking       # Docker image for docking environment
│
├── screening/                   # Step 1: ZINC22 similarity screening
│   ├── similarity.py            # MCS-based screening script
│   └── README.md
│
├── docking/                     # Step 2: AutoDock Vina docking via easydock
│   ├── start_docking            # Docking launcher script
│   ├── extract_mols.py          # Post-processing: extract mol blocks from .db files
│   └── README.md
│
├── simulation_configuration/    # Step 3: GROMACS + gmx_MMPBSA setup
│   ├── *.mdp                    # GROMACS MD parameter files
│   ├── mmpbsa.in                # gmx_MMPBSA input
│   └── README.md
│
├── data/                        # Step 4: Data preprocessing
│   ├── process_data.ipynb       # Deduplication and formatting
│   └── README.md
│
├── bayesian_optimization/       # Step 5: Active learning experiments
│   ├── run.py                   # Main entry point
│   ├── bayesian_protein/        # Core BO library (surrogate models, embeddings, samplers)
│   └── data/processed/          # Preprocessed datasets (populated by data/process_data.ipynb)
│
├── bayesian_optimization_results/  # Output: BO run results (populated at runtime)
│
├── analysis/                    # Step 6: Analysis notebooks and figures
│   ├── figures/                 # Publication figures
│   ├── mcl1_mmgbsa.ipynb        # Main MCL1 MM/GBSA results
│   ├── mcl1_vina.ipynb          # MCL1 Vina results
│   ├── mcl1_vina_docking_in_one.ipynb  # Combined Vina analysis
│   ├── mcl1_batch_size.ipynb    # Effect of batch acquisition size
│   ├── mcl1_medoid_effect.ipynb # Effect of initialization strategy (medoid)
│   ├── mcl1_mmgbsa_diverse_init.ipynb  # Diverse initialization ablation
│   ├── mcl1_mmgbsa_ucb.ipynb    # UCB acquisition function experiments
│   ├── enamine10k.ipynb         # Enamine 10k benchmark
│   ├── enamine50k.ipynb         # Enamine 50k benchmark
│   ├── seeds_error_bars.ipynb   # Seed robustness / error bars
│   ├── rogi_curves.ipynb        # ROGI landscape analysis
│   └── utils.py                 # Shared plotting utilities
│
├── notebooks/                   # Exploratory dataset analysis
│   ├── score_analysis.ipynb     # Docking and MM/GBSA score distributions
│   ├── exp_corr.ipynb           # Docking vs MM/GBSA correlation on experimental binders
│   ├── screening.ipynb          # Screening result analysis
│   └── README.md
│
├── smoothness/                  # Landscape smoothness analysis (ROGI)
│
├── run_bayesian_optimization.sh # Runs all main BO experiments
├── run_seeds.sh                 # Seed robustness sweep
├── run_batch_size.sh            # Batch size ablation
├── run_diverse_init.sh          # Diverse initialization ablation
├── run_ucb_experiment.sh        # UCB acquisition ablation
└── run_ucb_kappa.sh             # UCB kappa hyperparameter sweep
```
