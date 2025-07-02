For each step of the workflow we have a single directory with more detailed instructions on how to reproduce each step.

1. Screening of ZINC for similar molecules: `screening`
2. Docking of the screened molecules: `docking`
3. Simulation using MMGBSA: An example configuration of the system for can be found in `simulation_configuration`
4. Before running Bayesian Optimization/Active learning we preprocess (e.g. deduplication) the data. See `data/process_data.ipynb`


## Active Learning

### Setup 
The project uses Python 3.10. All requirements for the different steps in the workflow can be found in the respective directory.

To setup Bayesian Optimization follow the instructions [here](https://github.com/uds-lsv/chemical-lm-active-learning).
Once setup preprocess using the `data/process_data.ipynb` notebook and copy the resulting `Enamine10k_scores.csv`, `Enamine50k_scores.csv`, `MCL1-vina.csv`, `MCL1-mmgbsa.csv`
to `bayesian_optimization/data/processed`.
To run all setups execute the `run_bayesian_optimization.sh` script. Once this is finished the results can be found in `bayesian_optimization/runs`.
