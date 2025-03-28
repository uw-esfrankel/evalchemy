#!/bin/bash

# Set your Hugging Face username here
HF_USERNAME="esfrankel17"

# Directory to store logs
LOGS_DIR="eval_logs"
mkdir -p $LOGS_DIR

# Get all models for the user
echo "Fetching models from Hugging Face..."
MODELS_JSON=$(curl -s "https://huggingface.co/api/models?author=$HF_USERNAME")

# Extract model names starting with "dpo"
echo "Finding DPO models..."
DPO_MODELS=$(echo $MODELS_JSON | jq -r '.[] | .id | select(endswith("/dpo") or contains("/dpo-"))')

# Check if we found any models
if [ -z "$DPO_MODELS" ]; then
    echo "No DPO models found for user $HF_USERNAME"
    exit 1
fi

# Display found models
echo "Found the following DPO models:"
echo "$DPO_MODELS"
echo ""

# Run evaluation for each model
for model_name in $DPO_MODELS; do
    echo "====================================="
    echo "Evaluating model: $model_name"
    echo "====================================="
    
    # Create a unique log directory for this model
    model_short_name=$(basename "$model_name")
    model_log_dir="$LOGS_DIR/$model_short_name"
    mkdir -p "$model_log_dir"
    
    # Run the evaluation
    accelerate launch --num-processes 8 --num-machines 1 \
        --multi-gpu -m eval.eval \
        --model hf \
        --tasks alpaca_eval \
        --model_args "pretrained=${model_name}" \
        --batch_size 8 \
        --output_path "$model_log_dir"
    
    # Check if evaluation was successful
    if [ $? -eq 0 ]; then
        echo "Evaluation completed for $model_name"
    else
        echo "Error evaluating $model_name"
    fi
    
    echo ""
done

echo "All evaluations complete. Results are in the $LOGS_DIR directory."