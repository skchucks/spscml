#!/bin/bash -l
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J vlasov_test_sims
#SBATCH -A m4490
#SBATCH -t 2:30:00

module load conda
conda activate dpp_stuff

# Copy input CSV to scratch
scp /global/homes/s/sriyakc/spscml/data/new_testingparams/test_parameters_mixed.csv $SCRATCH/

# Run from scratch (fast I/O)
cd $SCRATCH
python /global/homes/s/sriyakc/spscml/scripts/generate_test_vlasov_dataset.py

# Copy results back home
cp vlasov_testdata_results_newbounds_*.csv /global/homes/s/sriyakc/spscml