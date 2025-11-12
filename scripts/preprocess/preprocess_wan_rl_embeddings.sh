#!/bin/bash
# Preprocess T5 embeddings for WAN SRPO training
# Usage: bash scripts/preprocess/preprocess_wan_rl_embeddings.sh

# Configuration
WAN_MODEL_PATH="./data/wan2.2-t2v-a14b"  # Path to WAN model
INPUT_JSON="./data/captions.json"        # Input JSON with captions
OUTPUT_DIR="./data/rl_embeddings"        # Output directory for embeddings
GPU_ID=0

echo "=========================================="
echo "WAN T5 Embedding Preprocessing"
echo "=========================================="
echo "Model Path: ${WAN_MODEL_PATH}"
echo "Input JSON: ${INPUT_JSON}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "GPU ID: ${GPU_ID}"
echo "=========================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run preprocessing
python fastvideo/data_preprocess/preprocess_wan_embedding.py \
    --model_path ${WAN_MODEL_PATH} \
    --input_json ${INPUT_JSON} \
    --output_dir ${OUTPUT_DIR} \
    --gpu_id ${GPU_ID}

echo "Preprocessing complete!"
echo "Embeddings saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "1. Copy the output JSON to your training data directory:"
echo "   cp ${OUTPUT_DIR}/embeddings.json ./data/rl_embeddings/"
echo "2. Run training with:"
echo "   bash scripts/finetune/SRPO_training_wan_t2i.sh"
