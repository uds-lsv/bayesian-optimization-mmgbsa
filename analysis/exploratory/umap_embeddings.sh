#!/bin/bash

set -e

[ -f ~/.netrc ] || cp /nethome/mrdupont/.netrc ~/

source /nethome/mrdupont/.bashrc

# Assume we execute from repo root
cd analysis/exploratory

python umap_embeddings.py \
    --data ../../data/MCL1-mmgbsa.csv \
    --out-dir /data/users/mrdupont/bayesian-optimization-mmgbsa/figures \
    --umap-n-neighbors 30 \
    --umap-min-dist 0.05 \
    --umap-init pca

python umap_init_highlight.py \
    --data ../../data/MCL1-mmgbsa.csv \
    --out-dir /data/users/mrdupont/bayesian-optimization-mmgbsa/figures \
    --umap-n-neighbors 30 \
    --umap-min-dist 0.05 \
    --umap-init pca \
    --init-db ../../bayesian_optimization_results/mcl1_mmgbsa_ei_diverse_init10.sqlite  --init-label "Diverse (n=10)" \
    --init-db ../../bayesian_optimization_results/mcl1_mmgbsa_ei_diverse_init50.sqlite  --init-label "Diverse (n=50)" \
    --init-db ../../bayesian_optimization_results/mcl1_mmgbsa_ei_diverse_init100.sqlite --init-label "Diverse (n=100)" \
python umap_medoid_highlight.py \
    --data ../../data/MCL1-mmgbsa.csv \
    --out-dir /data/users/mrdupont/bayesian-optimization-mmgbsa/figures \
    --umap-n-neighbors 30 \
    --umap-min-dist 0.05 \
    --umap-init pca \
    --medoid-db ../../bayesian_optimization_results/mcl1_mmgbsa_medoid_chemberta-mtr.sqlite \
    --medoid-db ../../bayesian_optimization_results/mcl1_mmgbsa_medoid_molformer.sqlite \
    --medoid-db ../../bayesian_optimization_results/mcl1_mmgbsa_medoid_fingerprint.sqlite

echo "Completed at $(date)"
