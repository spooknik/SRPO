#!/bin/bash
# SRPO Training Script for WAN Low-Noise Expert (Text-to-Image)
# OPTIMIZED FOR SINGLE GPU (RTX 6000 Pro 96GB or similar)
# Usage: bash scripts/finetune/SRPO_training_wan_t2i_single_gpu.sh

# =============================================================================
# Single GPU Configuration (RTX 6000 Pro 96GB)
# =============================================================================
export NNODES=1
export NPROC_PER_NODE=1                 # Single GPU
export NODE_RANK=0
export CHIEF_IP=localhost

# Model paths
WAN_MODEL_PATH="./data/wan2.2-t2v-a14b"
DATA_JSON_PATH="./data/rl_embeddings/embeddings.json"
OUTPUT_DIR="./outputs/wan_srpo_t2i_single_gpu"

# =============================================================================
# Training Hyperparameters (Optimized for 96GB VRAM)
# =============================================================================
TRAIN_BATCH_SIZE=2                      # Increased for 96GB
GRADIENT_ACCUMULATION_STEPS=4           # Effective batch size = 2×4 = 8
LEARNING_RATE=5e-6
MAX_TRAIN_STEPS=100

# =============================================================================
# Memory Optimization (Critical for Single GPU)
# =============================================================================
MASTER_WEIGHT_TYPE="bf16"              # Saves ~55GB memory vs fp32
CPU_OFFLOAD=false                      # Not needed with 96GB + bf16

# Image settings
TRAIN_HEIGHT=512
TRAIN_WIDTH=512
VIS_SIZE=1024

# SRPO settings
SAMPLING_STEPS=25
VIS_SAMPLING_STEPS=40
TRAIN_TIMESTEP_START=5
TRAIN_TIMESTEP_END=25
DISCOUNT_POS_START=0.1
DISCOUNT_POS_END=0.25
DISCOUNT_INV_START=0.3
DISCOUNT_INV_END=0.01
SHIFT=1.0
TRAIN_GUIDANCE=3.5
VIS_GUIDANCE=5.0

# Reward model
REWARD_MODEL="HPS"

# Parallelism
SP_SIZE=1
FSDP_SHARDING="full"

echo "=========================================="
echo "WAN SRPO Single GPU Training"
echo "=========================================="
echo "GPU: Single GPU (NPROC_PER_NODE=1)"
echo "Model Path: ${WAN_MODEL_PATH}"
echo "Data Path: ${DATA_JSON_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Batch Size: ${TRAIN_BATCH_SIZE}"
echo "Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective Batch Size: $((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Max Steps: ${MAX_TRAIN_STEPS}"
echo "Master Weight Type: ${MASTER_WEIGHT_TYPE}"
echo "CPU Offload: ${CPU_OFFLOAD}"
echo "Resolution: ${TRAIN_HEIGHT}x${TRAIN_WIDTH}"
echo "Reward Model: ${REWARD_MODEL}"
echo "Expected Memory Usage: ~70-75GB"
echo "=========================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ./images/wan_t2i_single_gpu

# Build CPU offload flag
CPU_OFFLOAD_FLAG=""
if [ "${CPU_OFFLOAD}" = "true" ]; then
    CPU_OFFLOAD_FLAG="--use_cpu_offload"
    echo "Enabling CPU offload for optimizer states"
fi

# Run training
torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --rdzv_endpoint=${CHIEF_IP}:29501 \
    --rdzv_id=123 \
    fastvideo/SRPO_wan.py \
    --pretrained_model_name_or_path ${WAN_MODEL_PATH} \
    --data_json_path ${DATA_JSON_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --image_p wan_t2i_single_gpu \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --max_train_steps ${MAX_TRAIN_STEPS} \
    --checkpointing_steps 20 \
    --dataloader_num_workers 4 \
    --h ${TRAIN_HEIGHT} \
    --w ${TRAIN_WIDTH} \
    --sampling_steps ${SAMPLING_STEPS} \
    --vis_sampling_step ${VIS_SAMPLING_STEPS} \
    --vis_size ${VIS_SIZE} \
    --shift ${SHIFT} \
    --timestep_length 100 \
    --groundtruth_ratio 0.9 \
    --train_timestep ${TRAIN_TIMESTEP_START} ${TRAIN_TIMESTEP_END} \
    --discount_pos ${DISCOUNT_POS_START} ${DISCOUNT_POS_END} \
    --discount_inv ${DISCOUNT_INV_START} ${DISCOUNT_INV_END} \
    --train_guidance ${TRAIN_GUIDANCE} \
    --vis_guidance ${VIS_GUIDANCE} \
    --reward_model ${REWARD_MODEL} \
    --gradient_checkpointing \
    --selective_checkpointing 1.0 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 10 \
    --weight_decay 0.01 \
    --max_grad_norm 2.0 \
    --sp_size ${SP_SIZE} \
    --train_sp_batch_size 1 \
    --fsdp_sharding_strategy ${FSDP_SHARDING} \
    --master_weight_type ${MASTER_WEIGHT_TYPE} \
    ${CPU_OFFLOAD_FLAG} \
    --seed 42 \
    --sampler_seed 42

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "Training images: ./images/wan_t2i_single_gpu/"
echo "Logs: ${OUTPUT_DIR}/train.log"
