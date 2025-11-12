# Copyright (c) [2025] [FastVideo Team]
# SRPO Training Script for WAN Low-Noise Expert (Text-to-Image)
# Modified from SRPO.py to support WAN's architecture

import sys
import pdb
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = None
import logging
from loguru import logger
import argparse
import os
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
)
import random
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
from torch.utils.data import DataLoader
import torch
torch.autograd.set_detect_anomaly(True)

from torch.utils.data.distributed import DistributedSampler
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from diffusers.optimization import get_scheduler

# Disable bitsandbytes check (not needed for bf16 training)
import os
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

from diffusers.utils import check_min_version
from fastvideo.dataset.latent_wan_rl_datasets import LatentDataset, latent_collate_function
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import save_checkpoint
from fastvideo.utils.logging_ import main_print
from diffusers.image_processor import VaeImageProcessor
from transformers import AutoProcessor, AutoModel, UMT5EncoderModel
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

check_min_version("0.31.0")
import time
from collections import deque
import torch.distributed as dist
from torch.nn import functional as F

# Import WAN-specific models
try:
    from diffusers import WanTransformer3DModel, AutoencoderKLWan
except ImportError:
    print("WARNING: WAN models not found in diffusers. You may need to use transformers from main branch.")
    # Fallback imports if needed
    WanTransformer3DModel = None
    AutoencoderKLWan = None


# Control words for reward preference (same as original SRPO)
def get_random_cg_oily_adjective(index=0):
    cg_oily_adjectives = [
        "Concept art",
        "Painting",
        "Anime",
        "Flat",
        "Oil"
    ]
    return cg_oily_adjectives[index % len(cg_oily_adjectives)]


def get_random_realism_adjective(index=0):
    realism_adjectives = [
        "Natural-lighting", "Detail", "Detailed", "Real"
    ]
    return realism_adjectives[index % len(realism_adjectives)]


# Reward models (HPS, CLIP, PickScore) - same as original
class CLIP(torch.nn.Module):
    def __init__(self, is_pickscore=True, device="cuda", dtype=torch.float32):
        super().__init__()
        processor_path = "./data/clip"
        model_path = "./data/ps"
        self.device = device
        self.dtype = dtype
        if is_pickscore:
            self.processor = AutoProcessor.from_pretrained(processor_path)
            self.model = AutoModel.from_pretrained(model_path).eval().to(device)
        else:
            self.processor = AutoProcessor.from_pretrained(processor_path)
            self.model = AutoModel.from_pretrained(processor_path).eval().to(device)
        self.model = self.model.to(dtype=dtype)
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        crop_size = 224
        resize_size = 224
        def _transform():
            transform = Compose([
                Resize(resize_size, interpolation=BICUBIC),
                CenterCrop(crop_size),
                Normalize(std=image_std, mean=image_mean),
            ])
            return transform
        self.v_pre = _transform()

    def SRP_cfg(self, prompt, neg_prompt, image_inputs, k):
        image_inputs = self.v_pre(image_inputs)
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        neg_text_input = self.processor(
            text=neg_prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        neg_text_input = {k: v.to(device=self.device) for k, v in neg_text_input.items()}
        image_embs = self.model.get_image_features(pixel_values=image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        text_embs_neg = self.model.get_text_features(**neg_text_input)
        text_embs_neg = text_embs_neg / text_embs_neg.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * ((k+1)*text_embs-text_embs_neg) @ image_embs.T
        scores = scores.diag()
        scores = scores/20
        return scores


class HPS(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        hpsv2_model, hpsv2_token, hpsv2_pre = self.build_reward_model()
        self.model = hpsv2_model.to(dtype=dtype)
        self.token = hpsv2_token

        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        crop_size = 224
        resize_size = 224
        def _transform():
            transform = Compose([
                Resize(resize_size, interpolation=BICUBIC),
                CenterCrop(crop_size),
                Normalize(std=image_std, mean=image_mean),
            ])
            return transform
        self.vis_pre = _transform()
        self.device = device

    def build_reward_model(self):
        model, preprocess_train, reprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        if isinstance(self.device, int):
            ml_device = str(self.device)
        else:
            ml_device = self.device

        if not ml_device.startswith('cuda'):
            ml_device = f'cuda:{ml_device}' if ml_device.isdigit() else ml_device

        checkpoint = torch.load('./data/hps_ckpt/HPS_v2.1_compressed.pt', map_location=ml_device)
        model.load_state_dict(checkpoint['state_dict'])
        text_processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(self.device)
        reward_model.eval()

        return reward_model, text_processor, preprocess_train

    def SRP_cfg(self, prompt, neg_prompt, images, k):
        image = self.vis_pre(images.squeeze(0)).unsqueeze(0).to(device=self.device, non_blocking=True)
        text = self.token(prompt).to(device=self.device, non_blocking=True)
        neg_text = self.token(neg_prompt).to(device=self.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(text, normalize=True)
            text_features_neg = self.model.encode_text(neg_text, normalize=True)

            logits_per_image = image_features @ ((1+k)*text_features.T-text_features_neg.T)
            hps_score = torch.diagonal(logits_per_image)
        return hps_score


def empty_logger():
    logger = logging.getLogger("hymm_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def setup_logger(exp_dir):
    if int(os.environ["RANK"]) <= 0:
        logger.add(os.path.join(exp_dir, "train.log"), level="DEBUG", colorize=False, backtrace=True,
                   diagnose=True, encoding="utf-8", filter=lambda record: record["extra"].get("name") == "train")
        logger.add(os.path.join(exp_dir, "val.log"), level="DEBUG", colorize=False, backtrace=True,
                   diagnose=True, encoding="utf-8", filter=lambda record: record["extra"].get("name") == "val")
        train_logger = logger.bind(name="train")
        val_logger = logger.bind(name="val")
    else:
        val_logger = train_logger = empty_logger()

    train_logger.info(f"Experiment directory created at: {exp_dir}")
    return train_logger, val_logger


# WAN-specific timestep shift (using Flow Matching framework)
def wan_time_shift(shift, t):
    """Time shift function for WAN's Flow Matching"""
    return (shift * t) / (1 + (shift - 1) * t)


def wan_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
):
    """Single denoising step for WAN model"""
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output
    return prev_sample_mean


def main_print(content):
    if int(os.environ["RANK"]) <= 0:
        print(content)


def run_sample_step_wan(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer_low_noise,  # WAN's low-noise expert (transformer_2)
        encoder_hidden_states,
        guidance_scale
    ):
    """
    Run sampling using WAN's low-noise expert only
    For text-to-image, we use num_frames=1
    """
    for i in progress_bar:
        sigma = sigma_schedule[i]
        timestep_value = sigma  # WAN uses sigma directly as timestep

        timesteps = torch.full([z.shape[0]], timestep_value, device=z.device, dtype=torch.float32)
        transformer_low_noise.eval()

        with torch.autocast("cuda", torch.bfloat16):
            # WAN transformer forward pass (conditional)
            # Note: For T2I, we set num_frames=1
            pred_cond = transformer_low_noise(
                hidden_states=z,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                return_dict=False,
            )[0]

            # Apply classifier-free guidance if guidance_scale != 1.0
            if guidance_scale != 1.0:
                # Create unconditional embeddings (zeros)
                uncond_embeddings = torch.zeros_like(encoder_hidden_states)
                pred_uncond = transformer_low_noise(
                    hidden_states=z,
                    encoder_hidden_states=uncond_embeddings,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
                # CFG formula: output = uncond + scale * (cond - uncond)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_cond

        z = wan_step(pred, z.to(torch.float32), sigmas=sigma_schedule, index=i)
        z = z.to(torch.bfloat16)
    return z


def SRPO_train_wan(
    args,
    device,
    transformer_low_noise,  # WAN's low-noise expert
    vae,
    encoder_hidden_states,
    reward_model,
    caption,
    mid_timestep,
    step,
    fp,
    visualization_step
):
    """
    SRPO training adapted for WAN low-noise expert (text-to-image)
    """
    timestep_length = args.timestep_length
    discount = torch.linspace(args.discount_pos[0], args.discount_pos[1], timestep_length).to(device)
    discount_inversion = torch.linspace(args.discount_inv[0], args.discount_inv[1], timestep_length).to(device)
    w, h = args.w, args.h
    shift = args.shift

    # Sampling parameters
    if not visualization_step:
        guidance_scale = args.train_guidance
        sample_steps = args.sampling_steps
        gradient_accumulation_steps = args.gradient_accumulation_steps
    else:
        guidance_scale = args.vis_guidance
        gradient_accumulation_steps = 1
        sample_steps = args.vis_sampling_step
        h = w = args.vis_size

    sigma_schedule = torch.linspace(1, 0, sample_steps + 1)
    sigma_schedule = wan_time_shift(shift, sigma_schedule)

    image_processor = VaeImageProcessor(16)  # WAN uses 16x downsampling
    B = encoder_hidden_states.shape[0]

    # WAN-specific settings
    SPATIAL_DOWNSAMPLE = 16  # WAN-VAE uses 16x spatial downsampling
    IN_CHANNELS = 4  # WAN-VAE latent channels
    TEMPORAL_FRAMES = 1  # For T2I, we use single frame

    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    # Initialize latents for text-to-image (single frame)
    input_latents = torch.randn(
        (1, IN_CHANNELS, TEMPORAL_FRAMES, latent_h, latent_w),  # (B, C, T, H, W)
        device=device,
        dtype=torch.bfloat16,
    )

    # Broadcast latents to all devices
    dist.broadcast(input_latents, src=0)

    progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")

    # Rollout one image using low-noise expert
    with torch.no_grad():
        latent_start = run_sample_step_wan(
            args,
            input_latents,
            progress_bar,
            sigma_schedule,
            transformer_low_noise,
            encoder_hidden_states,
            guidance_scale
        )

    # Visualization
    if visualization_step and step < 40:
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # WAN-VAE decode
                # Note: AutoencoderKLWan expects (B, C, T, H, W)
                image = vae.decode(latent_start, return_dict=False)[0]
                # Extract single frame if needed
                if image.ndim == 5:  # (B, C, T, H, W)
                    image = image[:, :, 0, :, :]  # Take first frame
                image = (image / 2 + 0.5).clamp(0, 1)
                decoded_image = image_processor.postprocess(image)
        try:
            decoded_image[0].save(fp + f"/{step}_{dist.get_rank()}.png")
        except:
            pass
        return 0

    # Add control words
    pos_control = get_random_realism_adjective(step)
    neg_control = get_random_cg_oily_adjective(step)
    pos_caption = pos_control + '. ' + caption[0]
    neg_caption = neg_control + '. ' + caption[0]

    k = int((1 - args.groundtruth_ratio) * timestep_length) + 1
    noise = input_latents

    num_inference_steps = timestep_length
    sigma_schedule = torch.linspace(1, 0, args.vis_sampling_step + 1)
    sigma_schedule = wan_time_shift(shift, sigma_schedule)
    timestep_select = mid_timestep

    k = min(min(timestep_length - mid_timestep, k), mid_timestep)

    for i in reversed(range(gradient_accumulation_steps)):
        inversion = i % 2

        # Timestep for training (focus on low-noise region)
        sigmas_l = torch.linspace(
            sigma_schedule[args.train_timestep[0]],
            sigma_schedule[args.train_timestep[1]],
            num_inference_steps
        ).to(device)

        if inversion == 0:
            mid_timestep = max(timestep_select - k, 1)
        else:
            mid_timestep = timestep_select
        start = min(mid_timestep + k, num_inference_steps)
        t_base = num_inference_steps - mid_timestep
        t_start = num_inference_steps - start

        # Direct-Align Step 1: Inject noise
        with torch.no_grad():
            if inversion == 0:
                sigmas = sigmas_l[t_base]
                noisy = sigmas * noise + (1.0 - sigmas) * latent_start
                sigmast = sigmas_l[t_start]
            else:
                sigmas = sigmas_l[t_start]
                noisy = sigmas * noise + (1.0 - sigmas) * latent_start
                sigmast = sigmas_l[t_base]
            gt_vector = sigmast * noise

        latents = noisy.detach()
        transformer_low_noise.train()
        latents = latents.requires_grad_(True)

        # Direct-Align Step 2: Inverse/Denoise one step
        with torch.autocast("cuda", torch.bfloat16):
            if inversion == 0:
                sigma = sigmas_l[t_base]
            else:
                sigma = sigmas_l[t_start]

            timesteps = torch.full([latents.shape[0]], sigma, device=latents.device, dtype=torch.float32)

            # During training, we skip CFG to save memory and computation
            # The model is trained with conditional embeddings directly
            pred = transformer_low_noise(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                return_dict=False,
            )[0]

            if inversion == 0:
                dsigma = sigma - sigmas_l[t_start]
                latents = latents.to(torch.float32) - dsigma * pred
            else:
                dsigma = sigmas_l[t_base] - sigma
                latents = latents.to(torch.float32) + dsigma * pred

        # Direct-Align Step 3: Recover image
        latents = (latents - gt_vector) / (1 - sigmast)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # VAE decode
            image = vae.decode(latents, return_dict=False)[0]
            if image.ndim == 5:  # (B, C, T, H, W)
                image = image[:, :, 0, :, :]  # Take first frame
            image = (image / 2 + 0.5).clamp(0, 1)

        # Calculate reward
        with torch.amp.autocast('cuda'):
            if inversion == 1:
                outputs = reward_model.SRP_cfg([pos_caption], [neg_caption], image, discount[mid_timestep])
            else:
                outputs = reward_model.SRP_cfg([neg_caption], [pos_caption], image, discount_inversion[mid_timestep])

        # Loss with threshold (ReFL approach)
        loss = F.relu(-outputs + 0.7) / gradient_accumulation_steps
        loss = loss.mean()
        loss.backward()

    return loss


def train_one_step(
    args,
    device,
    transformer_low_noise,
    vae,
    reward_model,
    optimizer,
    lr_scheduler,
    loader,
    mid_timestep,
    step,
    fp,
    visualization_step
):
    total_loss = 0.0
    optimizer.zero_grad()

    # Get data from loader
    # WAN dataset returns: (encoder_hidden_states, captions)
    data_item = next(loader)

    # Handle both wrapped and unwrapped dataloaders
    if len(data_item) == 2:
        # Direct from WAN dataset
        encoder_hidden_states, caption = data_item
    elif len(data_item) == 4:
        # From FLUX-style wrapper (shouldn't happen with sp_size=1)
        encoder_hidden_states, _, _, caption = data_item
    else:
        raise ValueError(f"Unexpected data format: {len(data_item)} items")

    loss = SRPO_train_wan(
        args,
        device,
        transformer_low_noise,
        vae,
        encoder_hidden_states,
        reward_model,
        caption,
        mid_timestep,
        step,
        fp,
        visualization_step
    )

    if visualization_step:
        return 0, 0

    grad_norm = transformer_low_noise.clip_grad_norm_(args.max_grad_norm)
    grad_norm = torch.tensor(grad_norm, device=loss.device, dtype=loss.dtype)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    avg_loss = loss.detach().clone()
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    total_loss += avg_loss.item()

    if dist.get_rank() % 8 == 0:
        print("final loss", loss.item())
    dist.barrier()

    return total_loss, grad_norm.item()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    initialize_sequence_parallel_state(args.sp_size)

    if args.seed is not None:
        set_seed(args.seed + rank)

    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    supported_models = ["HPS", "CLIP", "PickScore"]
    logger, _ = setup_logger(args.output_dir)

    if rank <= 0:
        log_dir = os.path.join(args.output_dir, "logs")
        tb_writer = SummaryWriter(log_dir=log_dir)

    # Initialize reward model
    if args.reward_model == "HPS":
        print(f"Initializing {args.reward_model} reward model...")
        reward_model = HPS().to(device)
    elif args.reward_model == "CLIP":
        print(f"Initializing {args.reward_model} reward model...")
        reward_model = CLIP(is_pickscore=False).to(device)
    elif args.reward_model == "PickScore":
        reward_model = CLIP(is_pickscore=True).to(device)
        print(f"Initializing {args.reward_model} reward model...")
    else:
        raise ValueError(
            f"Unsupported reward model: '{args.reward_model}'. "
            f"Please choose from: {supported_models}"
        )
    reward_model.eval()

    main_print(f"--> Loading WAN model from {args.pretrained_model_name_or_path}")

    # Load WAN's low-noise expert (transformer_2)
    # Note: We load only the low-noise expert for efficiency
    if WanTransformer3DModel is None:
        raise ImportError("WanTransformer3DModel not found. Please install diffusers from main branch or use the correct version.")

    transformer_low_noise = WanTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer_2",  # Low-noise expert
        torch_dtype=torch.float32
    )

    # Setup FSDP
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer_low_noise,
        args.fsdp_sharding_strategy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )

    transformer_low_noise = FSDP(transformer_low_noise, **fsdp_kwargs)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer_low_noise, no_split_modules, args.selective_checkpointing
        )

    # Load WAN-VAE
    if AutoencoderKLWan is None:
        raise ImportError("AutoencoderKLWan not found. Please install diffusers from main branch.")

    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)

    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_strategy}")
    main_print(f"--> Model loaded")

    transformer_low_noise.train()
    reward_model.requires_grad_(False)
    reward_model.eval()

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer_low_noise.parameters()))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, 0)  # No temporal frames for T2I
    sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=False, seed=args.sampler_seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    dist.barrier()
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )

    main_print("***** Running WAN SRPO Training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer_low_noise.parameters() if p.requires_grad) / 1e9} B"
    )
    main_print(f"  Master weight dtype: {next(transformer_low_noise.parameters()).dtype}")

    dir_path = './images'
    fp = os.path.join(dir_path, args.image_p)
    os.makedirs(fp, exist_ok=True)

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        disable=local_rank > 0,
    )

    # Use simple iterator when sp_size=1 (no sequence parallelism)
    # WAN dataset returns (encoder_hidden_states, captions) which is incompatible with FLUX wrapper
    if args.sp_size == 1:
        data_loader = iter(train_dataloader)
    else:
        # For sequence parallelism (sp_size > 1), use the wrapper
        data_loader = sp_parallel_dataloader_wrapper(
            train_dataloader,
            device,
            args.train_batch_size,
            args.sp_size,
            args.train_sp_batch_size,
        )

    step_times = deque(maxlen=100)

    for epoch in range(1000000):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for step in range(init_steps + 1, args.max_train_steps + 1):
            start_time = time.time()

            if step == args.checkpointing_steps:
                save_checkpoint(transformer_low_noise, rank, args.output_dir, step, epoch)

            mid_timestep_tensor = torch.tensor([random.randint(5, args.timestep_length - 5)], device=device)
            visualization_step = step > (args.checkpointing_steps - 1)

            loss, grad_norm = train_one_step(
                args,
                device,
                transformer_low_noise,
                vae,
                reward_model,
                optimizer,
                lr_scheduler,
                data_loader,
                mid_timestep_tensor,
                step,
                fp,
                visualization_step
            )

            loss_type = 'loss_hps'
            step_time = time.time() - start_time
            step_times.append(step_time)

            progress_bar.update(1)
            if rank <= 0 and not visualization_step:
                progress_info = {
                    loss_type: f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
                logger.info(f"Progress: {step}/{args.max_train_steps} | Details: {progress_info}")

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)

    # Model paths
    parser.add_argument("--image_p", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # Training
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument("--use_cpu_offload", action="store_true")

    # Parallel
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--train_sp_batch_size", type=int, default=1)
    parser.add_argument("--fsdp_sharding_strategy", default="full")

    # LR scheduler
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--master_weight_type", type=str, default="fp32")

    # SRPO settings
    parser.add_argument("--h", type=int, default=None, help="image height")
    parser.add_argument("--w", type=int, default=None, help="image width")
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--sampler_seed", type=int, default=None)
    parser.add_argument("--shift", type=float, default=1.0, help="timestep shift")
    parser.add_argument("--timestep_length", type=int, default=100)
    parser.add_argument("--groundtruth_ratio", type=float, default=0.9)
    parser.add_argument('--train_timestep', type=int, nargs=2, default=[5, 25])
    parser.add_argument('--discount_pos', type=float, nargs=2, default=[0.1, 0.25])
    parser.add_argument('--discount_inv', type=float, nargs=2, default=[0.3, 0.01])
    parser.add_argument('--vis_guidance', type=float, default=3.5)
    parser.add_argument('--train_guidance', type=float, default=3.5)
    parser.add_argument('--vis_sampling_step', type=int, default=50)
    parser.add_argument('--vis_size', type=int, default=1024)
    parser.add_argument('--reward_model', type=str, default='HPS')

    args = parser.parse_args()
    main(args)
