echo $(which mamba)
echo $PYTHONPATH
cd /nethome/mrdupont/bayesian-optimization-mmgbsa/smoothness
export PYTHONPATH=".."


DATA_DIR=/data/users/mrdupont/mcl1

# Running this script produces roughly 60GB of disk space

# Store embeddings
# python compute_embeddings.py chemberta $DATA_DIR
# python compute_embeddings.py molformer $DATA_DIR

# # Compute pairwise distances for each embeddings
# python compute_distance_matrix.py "${DATA_DIR}/chemberta.npy" $DATA_DIR
# python compute_distance_matrix.py "${DATA_DIR}/molformer.npy" $DATA_DIR

# # Compute absolute activity difference for Docking and MMGBSA
# python compute_activity_distance_matrix.py vina $DATA_DIR
# python compute_activity_distance_matrix.py mmgbsa $DATA_DIR


# Compute ratio and plot histogram
python create_histogram1d.py /data/users/mrdupont/mcl1/molformer_distance.npy /data/users/mrdupont/mcl1/
python create_histogram1d.py /data/users/mrdupont/mcl1/chemberta_distance.npy /data/users/mrdupont/mcl1/