#!/bin/bash
set -e

SEED=$1

if [ -z "$SEED" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

[ -f ~/.netrc ] || cp /nethome/mrdupont/.netrc ~/

echo "$(which python)"
echo "$PYTHONPATH"

cd /nethome/mrdupont/bayesian-optimization-mmgbsa/bayesian_optimization
export PYTHONPATH="."

python run.py \
    --out mcl1_mmgbsa_seeds_${SEED} \
    --data ./data/processed/MCL1-mmgbsa.csv \
    --surrogate linear-empirical \
    --embedding chemberta-mtr molformer fingerprint \
    --sampler expected-improvement \
    --init-sampler random \
    --n-iter 3600 \
    --protein MCL1 \
    --sample-size 1 \
    --seed "$SEED"

python run.py \
    --out mcl1_mmgbsa_stochastic_closest_${SEED} \
    --data ./data/processed/MCL1-mmgbsa.csv \
    --surrogate linear-empirical \
    --embedding chemberta-mtr molformer fingerprint \
    --sampler expected-improvement \
    --init-sampler stochastic-closest \
    --init-sample-size 50 \
    --n-iter 3600 \
    --protein MCL1 \
    --sample-size 1 \
    --seed "$SEED"

echo "Seed ${SEED} completed at $(date)"
