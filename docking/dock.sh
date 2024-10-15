#! /bin/bash
#SBATCH --partition=uds-hub
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --nodes=1-1
#SBATCH --nodelist=gpu[119-148]

echo $@
mamba run -n easydock run_dock -i $2 -o $1 --program vina --config vina_config.yml -c 4 --sdf