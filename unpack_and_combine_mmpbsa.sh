#! /bin/sh

tar -I pigz -xf results_benchmark.tar.gz --skip-old-files -C ./results/benchmark --strip-components=1
# Write header
find ./results/benchmark -type f -name "*.csv" -exec head -n 1 {} \; -quit > benchmark.csv
# Write content of all other files
find ./results/benchmark -type f -name "*.csv" -exec awk 'FNR > 1' {} + >> benchmark.csv

# Avg samples of each molecule
python process_mmbsa_results.py benchmark.csv --column='dg_en_gb' --lookup-benchmark


# Unpack all csv files in the mmpbsa results to ./result/
tar -I pigz -xf ./csv_files.tar.gz --skip-old-files -C ./results/zinc --strip-components=2
find ./results/zinc -type f -name "*.csv" -exec head -n 1 {} \; -quit > mmpbsa.csv
find ./results/zinc -type f -name "*.csv" -exec awk 'FNR > 1' {} + >> mmpbsa.csv

python process_mmbsa_results.py mmpbsa.csv --column='dg_en_gb' --lookup-smiles
