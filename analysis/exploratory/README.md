# Exploratory Analysis

Exploratory notebooks and scripts for dataset characterisation.
The rogi analysis requires the environment specified in `docker/environment.rogi-xd.yml`, all other files can be executed using the base environment.

```
├── score_analysis.ipynb          # Score distributions for docking and MM/GBSA
├── exp_corr.ipynb                # Docking vs MM/GBSA correlation on experimental binders
├── screening.ipynb               # Screening result analysis
├── umap_embeddings.py            # UMAP projections of molecular embeddings
├── umap_medoid_highlight.py      # Overlay medoid initialisation points on UMAP
├── umap_init_highlight.py        # Overlay diverse initialisation points on UMAP
└── rogi_analysis.py              # ROGI landscape smoothness analysis
```