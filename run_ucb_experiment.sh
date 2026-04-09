echo $(which python)
echo $PYTHONPATH
cd bayesian_optimization
export PYTHONPATH="."

# UCB acquisition — linear-empirical surrogate, MCL1 MMGBSA
# Sweeping kappa to characterize explore/exploit tradeoff
for kappa in 0.5 1.0 2.0 5.0; do
    python run.py \
        --out mcl1_mmgbsa_ucb_kappa${kappa} \
        --data ../data/MCL1-mmgbsa.csv \
        --surrogate linear-empirical \
        --embedding chemberta-mtr molformer fingerprint \
        --sampler ucb \
        --n-iter 3600 \
        --protein MCL1 \
        --sample-size 1 \
        --kappa $kappa
done
