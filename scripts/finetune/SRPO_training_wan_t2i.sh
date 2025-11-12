#!/bin/bash
# SRPO Training Script for WAN Low-Noise Expert (Text-to-Image)
# Usage: bash scripts/finetune/SRPO_training_wan_t2i.sh

# Configuration
export NNODES=${NNODES:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NODE_RANK=${NODE_RANK:-0}
export CHIEF_IP=${CHIEF_IP:-localhost}

# Model paths
WAN_MODEL_PATH="./data/wan2.2-t2v-a14b"  # Update this to your WAN model path
DATA_JSON_PATH="./data/rl_embeddings/embeddings.json"  # Pre-extracted T5 embeddings
OUTPUT_DIR="./outputs/wan_srpo_t2i"

# Training hyperparameters
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=5e-6
MAX_TRAIN_STEPS=100

# Image settings
TRAIN_HEIGHT=512
TRAIN_WIDTH=512
VIS_SIZE=1024

# SRPO settings
SAMPLING_STEPS=25
VIS_SAMPLING_STEPS=40  # WAN typically uses 40 steps
TRAIN_TIMESTEP_START=5
TRAIN_TIMESTEP_END=25
DISCOUNT_POS_START=0.1
DISCOUNT_POS_END=0.25
DISCOUNT_INV_START=0.3
DISCOUNT_INV_END=0.01
SHIFT=1.0  # Timestep shift for WAN
TRAIN_GUIDANCE=3.5
VIS_GUIDANCE=5.0

# Reward model
REWARD_MODEL="HPS"  # Options: HPS, CLIP, PickScore

# Parallelism
SP_SIZE=1
FSDP_SHARDING="full"

echo "=========================================="
echo "WAN SRPO Training Configuration"
echo "=========================================="
echo "Model Path: ${WAN_MODEL_PATH}"
echo "Data Path: ${DATA_JSON_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Batch Size: ${TRAIN_BATCH_SIZE}"
echo "Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Max Steps: ${MAX_TRAIN_STEPS}"
echo "Reward Model: ${REWARD_MODEL}"
echo "=========================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ./images/wan_t2i

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
    --image_p wan_t2i \
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
    --master_weight_type fp32 \
    --seed 42 \
    --sampler_seed 42

echo "Training complete!"
