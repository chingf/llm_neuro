#!/bin/bash

# Brain region to analyze
REGION="ac"

# Loop through layers (0, 3, 6, ..., 24)
for LAYER in {0..24..3}; do
    # Run without SAE
    #echo "Running layer ${LAYER} without SAE..."
    #python fit_llm_to_brain.py --brain_region ${REGION} --model_layer ${LAYER}
    
    # Run with SAE
    echo "Running layer ${LAYER} with SAE..."
    python fit_llm_to_brain.py --brain_region ${REGION} --model_layer ${LAYER} --use_sae
done