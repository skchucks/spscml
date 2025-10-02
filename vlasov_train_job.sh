#!/bin/bash -l
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J vlasov_train_sims
#SBATCH -A m4490
#SBATCH -t 2:30:00

module load conda
conda activate dpp_stuff

# Run from scratch (fast I/O)
cd $SCRATCH
python /global/homes/s/sriyakc/spscml/scripts/generate_train_vlasov_dataset.py

# Copy results back home
cp vlasov_traindata_results_newbounds_*.csv /global/homes/s/sriyakc/spscml/