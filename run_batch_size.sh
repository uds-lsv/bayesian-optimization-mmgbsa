#!/bin/bash
set -e

DATASET=$1
BATCH_SIZE=$2

if [ -z "$DATASET" ] || [ -z "$BATCH_SIZE" ]; then
    echo "Usage: $0 <dataset> <batch_size>"
    exit 1
fi

[ -f ~/.netrc ] || cp /nethome/mrdupont/.netrc ~/

echo "$(which python)"
echo "$PYTHONPATH"

cd /nethome/mrdupont/bayesian-optimization-mmgbsa/bayesian_optimization
export PYTHONPATH="."

# Always acquire 3600 datapoints total
N_ITER=$((3600 / BATCH_SIZE))

# Derive a short dataset tag for the output name (e.g. MCL1-mmgbsa -> mcl1_mmgbsa)
DATASET_TAG=$(basename "$DATASET" .csv | tr '[:upper:]' '[:lower:]' | tr '-' '_')

python run.py \
    --out "${DATASET_TAG}_batch${BATCH_SIZE}" \
    --data "$DATASET" \
    --surrogate linear-empirical \
    --embedding chemberta-mtr molformer fingerprint \
    --sampler expected-improvement \
    --n-iter "$N_ITER" \
    --protein MCL1 \
    --sample-size "$BATCH_SIZE"

echo "Batch size=${BATCH_SIZE} on ${DATASET_TAG} completed at $(date)"
