#!/bin/bash
set -e

INIT_SAMPLER=$1
INIT_SIZE=$2

if [ -z "$INIT_SAMPLER" ] || [ -z "$INIT_SIZE" ]; then
    echo "Usage: $0 <init_sampler> <init_size>"
    echo "  init_sampler: diverse | random"
    echo "  init_size:    1 | 10 | 50 | 100 | 600"
    exit 1
fi

[ -f ~/.netrc ] || cp /nethome/mrdupont/.netrc ~/

echo "$(which python)"
echo "$PYTHONPATH"

cd /nethome/mrdupont/bayesian-optimization-mmgbsa/bayesian_optimization
export PYTHONPATH="."

python run.py \
    --out mcl1_mmgbsa_ei_${INIT_SAMPLER}_init${INIT_SIZE} \
    --data ../data/MCL1-mmgbsa.csv \
    --surrogate linear-empirical \
    --embedding chemberta-mtr molformer fingerprint \
    --sampler expected-improvement \
    --init-sampler ${INIT_SAMPLER} \
    --init-sample-size ${INIT_SIZE} \
    --n-iter 3600 \
    --protein MCL1 \
    --sample-size 1

echo "EI ${INIT_SAMPLER} init=${INIT_SIZE} completed at $(date)"
