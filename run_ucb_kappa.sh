#!/bin/bash
set -e

KAPPA=$1

if [ -z "$KAPPA" ]; then
    echo "Usage: $0 <kappa>"
    exit 1
fi

[ -f ~/.netrc ] || cp /nethome/mrdupont/.netrc ~/

echo "$(which python)"
echo "$PYTHONPATH"

cd /nethome/mrdupont/bayesian-optimization-mmgbsa/bayesian_optimization
export PYTHONPATH="."

python run.py \
    --out mcl1_mmgbsa_ucb_kappa${KAPPA} \
    --data ./data/processed/MCL1-mmgbsa.csv \
    --surrogate linear-empirical \
    --embedding chemberta-mtr molformer fingerprint \
    --sampler ucb \
    --n-iter 3600 \
    --protein MCL1 \
    --sample-size 1 \
    --kappa $KAPPA

echo "UCB kappa=${KAPPA} completed at $(date)"
