echo $(which python)
echo $PYTHONPATH
cd bayesian_optimization
export PYTHONPATH="."

# Batch acquisition
python run.py --out mcl1_mmpbsa_gb_batch --data  ../data/MCL1-mmgbsa.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  6 --protein  MCL1 --sample-size  600
python run.py --out mcl1_vina_batch --data  ../data/MCL1-vina.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  6 --protein  MCL1 --sample-size  600
python run.py --out enamine_10k_batch --data  ../data/Enamine10k_scores.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  6 --protein  4HW3 --sample-size  100
python run.py --out enamine_50k_batch --data  ../data/Enamine50k_scores.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  6 --protein  4HW3 --sample-size  500

# Single acquisition
python run.py --out enamine_10k --data  ../data/Enamine10k_scores.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  600 --protein  4HW3 --sample-size  1
python run.py --out enamine_50k --data ../data/Enamine50k_scores.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter 3000 --protein 4HW3 --sample-size 1
python run.py --out mcl1_vina --data  ../data/MCL1-vina.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  3600 --protein  MCL1 --sample-size  1
python run.py --out mcl1_mmpbsa_gb --data  ../data/MCL1-mmgbsa.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  3600 --protein  MCL1 --sample-size  1


# Abolation on the effect of the medoid
XTH=(0 1 5 25 50 100 500)
for xth in "${XTH[@]}"; do
    echo "($1) Running experiment $xth closest to centroid as starting point"

    for i in {1..5}; do
        python run.py \
            --out mcl1_mmgbsa_medoid_$1_shuffled \
            --data ../data/MCL1-mmgbsa.csv \
            --surrogate linear-empirical \
            --embedding chemberta-mtr molformer fingerprint \
            --sampler expected-improvement \
            --n-iter "3600" \
            --sample-size  "1" \
            --init-sampler xth-closest \
            --protein  MCL1 \
            --init-sample-size "$xth"   # Argument is overloaded to represent the xth closest sample.
    done
done