echo $(which python)
echo $PYTHONPATH
cd bayesian_optimization
export PYTHONPATH="."

python experiments/run.py --out mcl1_mmpbsa_gb_batch --data  ./data/processed/MCL1-MCL1-mmgbsa.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  6 --protein  MCL1 --sample-size  600
python experiments/run.py --out mcl1_vina_batch --data  ./data/processed/MCL1-vina.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  6 --protein  MCL1 --sample-size  600
python experiments/run.py --out enamine_10k_batch --data  ./data/processed/Enamine10k_scores.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  6 --protein  4HW3 --sample-size  100
python experiments/run.py --out enamine_50k_batch --data  ./data/processed/Enamine50k_scores.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  6 --protein  4HW3 --sample-size  500
python experiments/run.py --out enamine_10k --data  ./data/processed/Enamine10k_scores.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  600 --protein  4HW3 --sample-size  1
python experiments/run.py --out enamine_50k --data ./data/processed/Enamine50k_scores.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter 3000 --protein 4HW3 --sample-size 1
python experiments/run.py --out mcl1_vina --data  ./data/processed/MCL1-vina.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  3600 --protein  MCL1 --sample-size  1
python experiments/run.py --out mcl1_mmpbsa_gb --data  ./data/processed/MCL1-mmgbsa.csv --surrogate linear-empirical rf --embedding chemberta-mtr molformer fingerprint --sampler expected-improvement --n-iter  3600 --protein  MCL1 --sample-size  1
