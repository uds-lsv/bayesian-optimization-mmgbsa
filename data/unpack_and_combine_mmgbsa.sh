#! /bin/sh

tar -I pigz -xf results_benchmark.tar.gz --skip-old-files -C ../results/benchmark --strip-components=1
# Write header
find ../results/benchmark -type f -name "*.csv" -exec head -n 1 {} \; -quit > benchmark.csv
# Write content of all other files
find ../results/benchmark -type f -name "*.csv" -exec awk 'FNR > 1' {} + >> benchmark.csv

# Avg samples of each molecule
python process_results.py benchmark.csv --column='dg_en_gb' --lookup-benchmark


# Unpack all csv files in the md results to ./result/
tar -I pigz -xf ./results.tar.gz --skip-old-files -C ../results/zinc --strip-components=2
find ../results/zinc -type f -name "*.csv" -exec head -n 1 {} \; -quit > mmgbsa.csv
find ../results/zinc -type f -name "*.csv" -exec awk 'FNR > 1' {} + >> mmgbsa.csv

python process_results.py mmgbsa.csv --column='dg_en_gb' --lookup-smiles
