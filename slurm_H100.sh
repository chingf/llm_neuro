#!/bin/bash
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100          # Partition to submit to
#SBATCH --account=kempner_krajan_lab
#SBATCH -c 24               # Number of cores (-c)
#SBATCH --mem=375G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --mail-user=ching_fang@hms.harvard.edu

source activate base
conda activate sae

# Verify the Python interpreter (these are for debugging)
which python
python --version
echo $CONDA_DEFAULT_ENV

# Brain region to analyze
REGION="ac"

# Loop through layers (0, 3, 6, ..., 24)
for LAYER in {0..24..3}; do
    echo "Running layer ${LAYER} with SAE features..."
    python fit_features_to_brain.py --brain_region ${REGION} --model_layer ${LAYER}
done