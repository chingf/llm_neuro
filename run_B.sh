#!/bin/bash

# Brain region to analyze
REGION="broca"

# Loop through layers (0, 3, 6, ..., 24)
for LAYER in {0..24..3}; do
    echo "Running layer ${LAYER} with SAE features..."
    python fit_features_to_brain.py --brain_region ${REGION} --model_layer ${LAYER}
done



# Brain region to analyze
REGION="ac"

# Loop through layers (0, 3, 6, ..., 24)
for LAYER in {0..24..3}; do
    echo "Running layer ${LAYER} with SAE features..."
    python fit_features_to_brain.py --brain_region ${REGION} --model_layer ${LAYER}
done
