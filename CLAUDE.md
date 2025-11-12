# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SRPO (Structured Reward Policy Optimization) is a research project for fine-tuning diffusion models (specifically FLUX.1-dev) using reinforcement learning with human preference feedback. The method introduces a novel "Direct-Align" sampling strategy that achieves faster training (under 10 minutes for FLUX.1.dev) while avoiding reward hacking issues.

This codebase builds on [FastVideo](https://github.com/hao-ai-lab/FastVideo) and [DanceGRPO](https://github.com/XueZeyue/DanceGRPO).

## Environment Setup

```bash
# Create environment
conda create -n SRPO python=3.10.16 -y
conda activate SRPO

# Install dependencies
bash ./env_setup.sh
```

The setup script installs:
- PyTorch 2.6.0 with CUDA 12.4
- Flash Attention 2.7.0
- Diffusers 0.32.0
- HPSv2 (cloned and installed from GitHub)
- All dependencies from pyproject.toml

## Key Commands

### Data Preprocessing
Pre-extract text embeddings for training (improves efficiency):
```bash
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
cp videos2caption2.json ./data/rl_embeddings
```

### Training
Train FLUX.1-dev with HPS v2.1 reward model:
```bash
bash scripts/finetune/SRPO_training_hpsv2.sh
```

Train with PickScore reward model (experimental):
```bash
bash scripts/finetune/SRPO_training_ps.sh
```

### Inference
Using the trained model:
```bash
torchrun --nnodes=1 --nproc_per_node=8 \
    --node_rank 0 \
    --rdzv_endpoint $CHIEF_IP:29502 \
    --rdzv_id 456 \
    vis.py
```

## Architecture

### Core Training Pipeline (`fastvideo/SRPO.py`)

The main training script implements the SRPO algorithm with three key components:

1. **Direct-Align Three-Step Process** (lines 544-614):
   - Step 1: Inject noise at specific timesteps
   - Step 2: Apply inversion or denoising step
   - Step 3: Recover clean image for reward calculation

2. **Reward Models** (lines 92-246):
   - `HPS`: Human Preference Score v2.1 (primary)
   - `CLIP`/`PickScore`: Alternative reward models
   - Both implement `SRP_cfg()` method for style-controllable reward preference

3. **Controllable Text Conditions** (lines 514-517):
   - Dynamic positive/negative control words for style guidance
   - Realism adjectives: "Natural-lighting", "Detail", "Detailed", "Real"
   - Anti-oily adjectives: "Concept art", "Painting", "Anime", "Flat", "Oil"

### Directory Structure

- `fastvideo/`: Main package
  - `SRPO.py`: Core training implementation
  - `dataset/`: Dataset loaders for latent RL training
    - `latent_flux_rl_datasets.py`: FLUX-specific RL datasets
    - `latent_rl_datasets.py`: Generic RL dataset handling
  - `data_preprocess/`: Text embedding preprocessing
    - `preprocess_flux_embedding.py`: Extract FLUX text embeddings
  - `models/`: Model architectures
    - `flux_hf/`: FLUX HuggingFace integration
    - `hunyuan/`, `mochi_hf/`: Alternative model support
  - `utils/`: Training utilities
    - `communications_flux.py`: Sequence parallel communication
    - `fsdp_util.py`: FSDP (Fully Sharded Data Parallel) helpers
    - `checkpoint.py`: Model checkpointing

- `scripts/`: Shell scripts for training and preprocessing
- `comfyui/`: ComfyUI workflow integration

## Training Configuration

Key hyperparameters in `SRPO_training_hpsv2.sh`:

- **Batch size**: 1 per GPU (with `gradient_accumulation_steps=2`)
- **Learning rate**: 5e-6
- **Image resolution**: 720x720 (training), 1024x1024 (visualization)
- **Sampling steps**: 25 (training), 50 (inference)
- **Timestep training range**: `--train_timestep 5 25` (early-to-middle diffusion stages)
- **Discount factors**:
  - `--discount_inv 0.3 0.01`: Inversion branch (preserve structure)
  - `--discount_pos 0.1 0.25`: Denoising branch (avoid color oversaturation)
- **Shift**: 3 (timestep schedule shift for FLUX)
- **Checkpoint frequency**: Every 20 steps

## Model Downloads

Required models are stored in `./data/`:

1. **FLUX.1-dev** (`./data/flux`):
   ```bash
   huggingface-cli download --resume-download black-forest-labs/FLUX.1-dev --local-dir ./data/flux
   ```

2. **HPS v2.1 Reward Model** (`./data/hps_ckpt`):
   ```bash
   huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
   huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt
   ```

3. **PickScore (Optional)** (`./data/ps`):
   ```bash
   python ./scripts/huggingface/download_hf.py --repo_id yuvalkirstain/PickScore_v1 --local_dir ./data/ps
   ```

## Distributed Training

The codebase uses PyTorch's torchrun with FSDP for distributed training:

- **Sequence Parallel**: Controlled by `--sp_size` (default: 1)
- **FSDP Sharding**: Full sharding strategy by default
- **Gradient Checkpointing**: Enabled with `--gradient_checkpointing`
- **Mixed Precision**: bfloat16 recommended

Environment variables used:
- `NNODES`, `NPROC_PER_NODE`: Cluster configuration
- `NODE_RANK`, `CHIEF_IP`: Multi-node coordination
- `HOST_NUM`, `HOST_GPU_NUM`, `INDEX`: Node-specific settings

## Important Implementation Details

1. **Timestep Selection** (line 534):
   - Training focuses on `sigma_schedule[5:25]` to avoid early structural distortions and late-stage reward hacking
   - Adjust `--vis_sampling_step` to match model's regular inference steps

2. **Image Recovery Formula** (line 608):
   - `latents = (latents - gt_vector) / (1 - sigmast)`
   - Critical for Direct-Align to restore clean images from noisy latents

3. **Reward Threshold** (line 623):
   - `loss = F.relu(-outputs + 0.7)` follows ReFL approach
   - Only backprop when reward < 0.7

4. **Control Word Rotation** (lines 514-517):
   - Cycles through adjectives based on training step
   - Prevents overfitting to specific style prompts

5. **Latent Broadcasting** (line 473):
   - Ensures all devices generate with identical latents for consistency

## Adapting to Custom Models

From README section "How to Support Custom Models":

1. Modify `preprocess_flux_embedding.py` and `latent_flux_rl_datasets.py` for custom text embedding extraction
2. Adjust `--vis_sampling_step` to match your model's inference steps
3. Enable VAE gradient checkpointing before reward calculation to reduce memory
4. Disable inversion branch initially to check for reward hacking
5. Pure Direct-Align works for SRPO-unsupported tasks (OCR, image editing) with minimal changes

## Training Tips

From README "Hyperparameter Recommendations":

- **Batch size**: Larger improves quality (32 recommended for FLUX.1.dev)
- **Learning rate**: 1e-5 to 1e-6 for most models
- **Train_timestep**: Early-to-middle stages; too early (sigma>0.99) causes distortions, too late encourages color hacking
- **Discount balance**: Set `discount_pos[0]` slightly > `discount_inv[1]` to preserve structure; set `discount_inv[0]` slightly > `discount_pos[1]` to fix oversaturation

## Visualization and Validation

- Training generates images in `./images/{args.image_p}/` directory
- Images saved after step `--checkpointing_steps` for validation
- Use `vis.py` for batch inference with trained checkpoints
- ComfyUI workflow available in `comfyui/SRPO-workflow.json`
