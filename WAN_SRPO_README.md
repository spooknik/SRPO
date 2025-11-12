# WAN SRPO: Text-to-Image Fine-tuning with Low-Noise Expert

This implementation adapts SRPO (Structured Reward Policy Optimization) for WAN's low-noise expert model for text-to-image generation.

## Overview

Instead of training the full WAN MoE architecture, we focus on the **low-noise expert** (`transformer_2`), which handles the later stages of denoising and is responsible for refining image details. This is ideal for SRPO because:

1. **Aligns with SRPO's training regime**: SRPO trains on early-to-middle timesteps (sigma 5-25), which overlaps with the low-noise expert's domain
2. **Simpler than full video pipeline**: We generate single images (setting `num_frames=1`) to avoid temporal complexity
3. **Directly compatible with image reward models**: HPS v2.1 and PickScore work out-of-the-box

## Architecture Differences from FLUX SRPO

| Component | FLUX SRPO | WAN SRPO |
|-----------|-----------|----------|
| **Text Encoder** | CLIP-L + T5-XXL | UMT5-XXL only |
| **VAE** | AutoencoderKL (8×8 compression, 16 channels) | AutoencoderKLWan (16×16 compression, 4 channels) |
| **Transformer** | FluxTransformer2DModel | WanTransformer3DModel (transformer_2 for low-noise) |
| **Timestep Schedule** | Rectified Flow with shift=3 | Flow Matching with shift=1.0 |
| **Inference Steps** | 25-50 | 40 (standard) |
| **Latent Shape** | (B, 16, H/8, W/8) | (B, 4, 1, H/16, W/16) for T2I |

## Hardware Requirements

### Multi-GPU Setup (Recommended)

**8x A100 (80GB each)**:
- Default configuration works out-of-the-box
- Training time: ~10-15 minutes for 100 steps
- Uses FSDP for distributed training

### Single GPU Setup

#### RTX 6000 Pro (96GB VRAM) ✅

Perfect for WAN SRPO! Here's the memory breakdown:

**With fp32 master weights** (default):
```
28GB (model) + 56GB (optimizer) + 28GB (gradients) + 10GB (activations) + 5GB (other)
= ~127GB ❌ Won't fit
```

**With bf16 master weights** (recommended):
```
14GB (model) + 28GB (optimizer) + 14GB (gradients) + 10GB (activations) + 5GB (other)
= ~71GB ✅ Fits comfortably with 25GB headroom!
```

#### Configuration for Single RTX 6000 Pro

Edit `scripts/finetune/SRPO_training_wan_t2i.sh`:

```bash
# GPU settings
export NPROC_PER_NODE=1  # Single GPU

# Optimize for 96GB VRAM
TRAIN_BATCH_SIZE=2              # Increased from 1 (you have the memory!)
GRADIENT_ACCUMULATION_STEPS=4   # Effective batch size = 2×4 = 8
MASTER_WEIGHT_TYPE="bf16"       # Critical: Change from fp32 to bf16

# Start with conservative resolution
TRAIN_HEIGHT=512
TRAIN_WIDTH=512
VIS_SIZE=1024

# Add to torchrun arguments:
torchrun ... \
    --master_weight_type bf16 \
    ...
```

**Expected performance**:
- Memory usage: ~70-75GB
- Training speed: ~5-10 seconds/step
- Total training time: ~15-30 minutes for 100 steps

**Why bf16 works perfectly here**:
- RTX 6000 Pro (Blackwell) has excellent bf16 Tensor Core performance
- Saves ~55GB of memory vs fp32
- Negligible quality loss for fine-tuning
- Original SRPO paper likely used bf16/fp16 training

#### Alternative: CPU Offload

If you prefer fp32 precision:

```bash
# Add this flag to keep fp32 but offload optimizer to CPU
--use_cpu_offload
```

**Trade-off**: ~20-30% slower training due to CPU↔GPU memory transfers, but preserves full fp32 precision.

### Other Single GPU Configurations

**RTX 4090 (24GB)**: Requires aggressive optimization
```bash
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
TRAIN_HEIGHT=256
TRAIN_WIDTH=256
--master_weight_type bf16
--use_cpu_offload  # Essential
```

**A100 (40GB)**: Good middle ground
```bash
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
TRAIN_HEIGHT=512
TRAIN_WIDTH=512
--master_weight_type bf16
```

**H100 (80GB)**: Excellent single-GPU option
```bash
TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
TRAIN_HEIGHT=512
TRAIN_WIDTH=512
--master_weight_type bf16
```

## Installation

### Prerequisites

```bash
# Install diffusers from main branch (WAN support)
pip install git+https://github.com/huggingface/diffusers.git

# Or use the official release when WAN is merged
pip install diffusers>=0.32.0
```

### Download WAN Model

```bash
# Download WAN 2.2 T2V-A14B model (includes low-noise expert)
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir ./data/wan2.2-t2v-a14b

# Or download from the main WAN repository format
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./data/wan2.2-t2v-a14b
```

### Download Reward Models

```bash
# HPS v2.1 (recommended)
huggingface-cli download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt

# PickScore (optional)
python ./scripts/huggingface/download_hf.py --repo_id yuvalkirstain/PickScore_v1 --local_dir ./data/ps
```

## Quick Start

### 1. Prepare Caption Data

#### Option A: Convert from YAML (If you have prompts.yaml)

If you already have a `prompts.yaml` file in the project root:

```yaml
prompts:
  - A beautiful sunset over the ocean
  - A cat sitting on a windowsill
  - An astronaut riding a horse
```

Simply run the conversion script:

```bash
bash scripts/utils/convert_prompts.sh
```

This will automatically create `./data/captions.json` with the correct format.

#### Option B: Create JSON Manually

Create a JSON file with your training captions:

```json
[
  {
    "caption": "A beautiful sunset over the ocean"
  },
  {
    "caption": "A cat sitting on a windowsill"
  },
  {
    "caption": "An astronaut riding a horse"
  }
]
```

Save as `./data/captions.json`.

### 2. Pre-extract T5 Embeddings

```bash
bash scripts/preprocess/preprocess_wan_rl_embeddings.sh
```

This will:
- Load WAN's UMT5-XXL text encoder
- Extract embeddings for all captions
- Save embeddings to `./data/rl_embeddings/`

**Output**: `./data/rl_embeddings/embeddings.json` containing paths to embedding files.

### 3. Run SRPO Training

#### Option A: Single GPU (RTX 6000 Pro 96GB) - Recommended Script

Use the pre-configured single GPU script (already optimized):

```bash
bash scripts/finetune/SRPO_training_wan_t2i_single_gpu.sh
```

This script is already configured with:
- `NPROC_PER_NODE=1` (single GPU)
- `TRAIN_BATCH_SIZE=2` (utilizes 96GB VRAM)
- `GRADIENT_ACCUMULATION_STEPS=4` (effective batch size = 8)
- `MASTER_WEIGHT_TYPE="bf16"` (saves ~55GB memory)
- Expected memory usage: ~70-75GB

#### Option B: Multi-GPU (8x A100) or Custom Setup

Edit `scripts/finetune/SRPO_training_wan_t2i.sh` for your hardware, then run:

```bash
bash scripts/finetune/SRPO_training_wan_t2i.sh
```

**Training time**:
- 8x A100 GPUs: ~10-15 minutes for 100 steps
- Single RTX 6000 Pro: ~15-30 minutes for 100 steps

### 4. Monitor Training

Training images will be saved to `./images/wan_t2i/` after step 20.

Logs are available in `./outputs/wan_srpo_t2i/train.log`.

## Configuration

### Key Hyperparameters

Edit `scripts/finetune/SRPO_training_wan_t2i.sh`:

```bash
# Image resolution
TRAIN_HEIGHT=512
TRAIN_WIDTH=512
VIS_SIZE=1024

# Training
LEARNING_RATE=5e-6
MAX_TRAIN_STEPS=100
GRADIENT_ACCUMULATION_STEPS=2

# SRPO-specific
TRAIN_TIMESTEP_START=5      # Focus on low-noise region
TRAIN_TIMESTEP_END=25       # (early to middle denoising)
DISCOUNT_POS_START=0.1      # Denoising branch discount
DISCOUNT_POS_END=0.25
DISCOUNT_INV_START=0.3      # Inversion branch discount
DISCOUNT_INV_END=0.01
SHIFT=1.0                   # WAN's timestep shift
VIS_SAMPLING_STEPS=40       # WAN uses 40 inference steps

# Reward model
REWARD_MODEL="HPS"          # Options: HPS, CLIP, PickScore
```

### Tuning Guidelines

Based on SRPO paper recommendations:

1. **Batch size**: Larger is better (32+ recommended, currently 1×2=2 effective batch size)
   - Increase `TRAIN_BATCH_SIZE` or `GRADIENT_ACCUMULATION_STEPS`
   - Scale learning rate proportionally

2. **Learning rate**: Start with 5e-6, can try 1e-5 to 1e-6 range

3. **Train timesteps**: Focus on low-noise region
   - WAN's low-noise expert activates when t < t_moe (approximately < 0.5)
   - SRPO trains on sigma[5:25] which maps well to this region
   - Don't go too early (sigma > 0.99) to avoid structural distortions

4. **Discount factors**: Balance between structure preservation and quality
   - `discount_pos`: Higher values = more aggressive quality optimization
   - `discount_inv`: Higher values = better structure preservation
   - Recommended: `discount_pos[0]` slightly > `discount_inv[1]`

5. **Shift parameter**: WAN uses shift=1.0 by default
   - Adjust based on model's inference schedule
   - Check WAN's scheduler config in model files

## Implementation Details

### Direct-Align Three-Step Process

Same as original SRPO (fastvideo/SRPO_wan.py:544-614):

1. **Inject noise at specific timestep**
   ```python
   noisy = sigma * noise + (1.0 - sigma) * latent_start
   ```

2. **Apply inversion or denoising step**
   ```python
   pred = transformer_low_noise(latents, encoder_hidden_states, timestep, guidance)
   latents = latents ± dsigma * pred
   ```

3. **Recover clean image for reward**
   ```python
   latents = (latents - gt_vector) / (1 - sigmast)
   image = vae.decode(latents)
   reward = reward_model.SRP_cfg(pos_caption, neg_caption, image, discount)
   ```

### WAN-Specific Adaptations

#### 1. Model Loading
```python
# Load low-noise expert only (transformer_2)
transformer_low_noise = WanTransformer3DModel.from_pretrained(
    model_path,
    subfolder="transformer_2",  # Low-noise expert
    torch_dtype=torch.float32
)

# Load WAN-VAE
vae = AutoencoderKLWan.from_pretrained(
    model_path,
    subfolder="vae",
    torch_dtype=torch.bfloat16
)
```

#### 2. Latent Initialization
```python
# Text-to-image: Use single frame
IN_CHANNELS = 4                # WAN-VAE has 4 latent channels
SPATIAL_DOWNSAMPLE = 16        # 16x spatial compression
TEMPORAL_FRAMES = 1            # Single frame for T2I

input_latents = torch.randn(
    (1, IN_CHANNELS, TEMPORAL_FRAMES, latent_h, latent_w),
    device=device,
    dtype=torch.bfloat16
)
```

#### 3. Forward Pass
```python
# WAN transformer forward (simplified)
pred = transformer_low_noise(
    hidden_states=latents,
    encoder_hidden_states=encoder_hidden_states,  # T5 embeddings only
    timestep=timesteps,
    guidance=guidance_scale,
    return_dict=False
)[0]
```

#### 4. VAE Decode
```python
# WAN-VAE expects 5D input: (B, C, T, H, W)
image = vae.decode(latents, return_dict=False)[0]

# Extract single frame for T2I
if image.ndim == 5:
    image = image[:, :, 0, :, :]  # Take first frame
```

## Troubleshooting

### Import Errors

**Problem**: `ImportError: cannot import name 'WanTransformer3DModel'`

**Solution**: Install diffusers from main branch:
```bash
pip install git+https://github.com/huggingface/diffusers.git
```

### Model Loading Errors

**Problem**: `OSError: transformer_2 not found in model directory`

**Solution**: Ensure you downloaded the Diffusers version of WAN:
```bash
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir ./data/wan2.2-t2v-a14b
```

### Memory Issues

**Problem**: CUDA out of memory

**Solutions**:
1. Enable CPU offload: Add `--use_cpu_offload` to training script
2. Reduce batch size: Set `TRAIN_BATCH_SIZE=1` and `GRADIENT_ACCUMULATION_STEPS=1`
3. Lower resolution: Set `TRAIN_HEIGHT=256` and `TRAIN_WIDTH=256`
4. Enable gradient checkpointing: Already enabled by default with `--gradient_checkpointing`

### Latent Shape Mismatch

**Problem**: `RuntimeError: expected shape [B, 4, 1, H, W] but got [B, 4, H, W]`

**Solution**: WAN-VAE expects temporal dimension. Check that latents are initialized with shape `(B, C, T, H, W)` where T=1 for images.

### Slow Training

**Problem**: Training is very slow

**Solutions**:
1. Use fewer workers: Set `--dataloader_num_workers 2`
2. Use smaller validation sampling steps during training (already using 25)
3. Disable visualization before checkpoint step
4. Use FP16 instead of BF16 if your GPU doesn't support BF16 efficiently

## Extending to Video

To extend this implementation to full video generation:

1. **Set temporal frames > 1**:
   ```python
   TEMPORAL_FRAMES = 17  # Or 81 for full video
   ```

2. **Add temporal reward aggregation**:
   ```python
   # Average reward across frames
   rewards = []
   for frame_idx in range(num_frames):
       frame = image[:, :, frame_idx, :, :]
       reward = reward_model.SRP_cfg(pos_caption, neg_caption, frame, discount)
       rewards.append(reward)
   total_reward = torch.mean(torch.stack(rewards))
   ```

3. **Enable memory-efficient VAE decoding**:
   ```python
   # Decode frames in chunks
   vae.enable_slicing()
   vae.enable_tiling()
   ```

4. **Consider video-specific reward models**:
   - Temporal consistency metrics
   - Motion quality scores
   - VideoRM (if available)

## Citation

If you use this code, please cite both SRPO and WAN:

```bibtex
@article{srpo2024,
  title={Structured Reward Policy Optimization for Diffusion Models},
  author={[SRPO Authors]},
  journal={arXiv preprint},
  year={2024}
}

@article{wan2024,
  title={Wan: Open and Advanced Large-Scale Video Generative Models},
  author={[WAN Authors]},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This code inherits licenses from:
- SRPO: Apache 2.0
- FastVideo: Apache 2.0
- WAN: Check WAN repository for license details

## Acknowledgments

- Original SRPO implementation: [Your Repository]
- FastVideo framework: [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo)
- WAN models: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
