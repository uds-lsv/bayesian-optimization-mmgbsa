papermill make_plots.ipynb mcl1_mmpbsa_shuffled.ipynb -p db mcl1_mmpbsa_gb_shuffled.sqlite -p TITLE MCL1-MMPBSA
papermill make_plots.ipynb mcl1_mmpbsa_batch.ipynb -p db mcl1_mmpbsa_gb_batch.sqlite -p TITLE MCL1-MMPBSA

papermill make_plots.ipynb mcl1_vina_shuffled.ipynb -p db mcl1_vina_shuffled.sqlite -p TITLE MCL1-VINA
papermill make_plots.ipynb mcl1_vina_batch.ipynb -p db mcl1_vina_batch.sqlite -p TITLE MCL1-VINA

papermill make_plots.ipynb enamine10k_shuffled.ipynb -p db enamine_10k_shuffled.sqlite -p TITLE Enamine10k
papermill make_plots.ipynb enamine10k_batch.ipynb -p db enamine_10k_batch.sqlite -p TITLE Enamine10k

papermill make_plots.ipynb enamine50k_shuffled.ipynb -p db enamine_50k_shuffled.sqlite -p TITLE Enamine50k
papermill make_plots.ipynb enamine50k_batch.ipynb -p db enamine_50k_batch.sqlite -p TITLE Enamine50k
