#!/bin/bash

set -e

[ -f ~/.netrc ] || cp /nethome/mrdupont/.netrc ~/

source /nethome/mrdupont/.bashrc

cd /nethome/mrdupont/bayesian-optimization-mmgbsa/notebooks

python umap_embeddings.py \
    --data /nethome/mrdupont/bayesian-optimization-mmgbsa/bayesian_optimization/data/processed/MCL1-mmgbsa.csv \
    --out-dir /data/users/mrdupont/bayesian-optimization-mmgbsa/figures \
    --umap-n-neighbors 30 \
    --umap-min-dist 0.05 \
    --umap-init pca

echo "Completed at $(date)"
