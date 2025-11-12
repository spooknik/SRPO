"""
Pre-extract T5 text embeddings for WAN SRPO training
Modified from preprocess_flux_embedding.py to support WAN's T5-only encoder
"""

import argparse
import json
import os
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel


def load_t5_encoder(model_path, device="cuda"):
    """
    Load WAN's T5 encoder (UMT5-XXL)

    Args:
        model_path: Path to WAN model directory or HuggingFace model ID
        device: Device to load model on

    Returns:
        tokenizer, text_encoder
    """
    print(f"Loading T5 encoder from {model_path}...")

    # Load tokenizer and encoder
    # WAN uses UMT5-XXL
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer"
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16
        ).to(device)
    except:
        # Fallback: try loading directly
        print("Warning: Could not load from subfolder, trying direct load...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to(device)

    text_encoder.eval()
    print("T5 encoder loaded successfully!")

    return tokenizer, text_encoder


def encode_prompt(caption, tokenizer, text_encoder, device="cuda"):
    """
    Encode a text prompt using T5

    Args:
        caption: Text prompt
        tokenizer: T5 tokenizer
        text_encoder: T5 encoder model
        device: Device

    Returns:
        encoder_hidden_states: T5 text embeddings
    """
    # Tokenize
    text_inputs = tokenizer(
        caption,
        padding="max_length",
        max_length=256,  # WAN uses 256 max length
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Encode
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoder_output = text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                return_dict=True
            )
            encoder_hidden_states = encoder_output.last_hidden_state

    return encoder_hidden_states


def preprocess_embeddings(args):
    """
    Pre-extract T5 embeddings for all captions
    """
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load T5 encoder
    tokenizer, text_encoder = load_t5_encoder(args.model_path, device)

    # Load caption data
    with open(args.input_json, 'r') as f:
        data = json.load(f)

    print(f"Processing {len(data)} captions...")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each caption
    output_data = []
    for idx, item in enumerate(tqdm(data, desc="Extracting embeddings")):
        caption = item.get('caption', '')

        if not caption:
            print(f"Warning: Empty caption at index {idx}, skipping...")
            continue

        # Encode prompt
        encoder_hidden_states = encode_prompt(caption, tokenizer, text_encoder, device)

        # Move to CPU and save
        encoder_hidden_states = encoder_hidden_states.cpu()

        # Save embedding
        embedding_filename = f"embedding_{idx:06d}.pt"
        embedding_path = output_dir / embedding_filename

        torch.save({
            'encoder_hidden_states': encoder_hidden_states,
            'caption': caption
        }, embedding_path)

        # Update output data
        output_item = {
            'caption': caption,
            'embedding_path': str(embedding_path)
        }
        # Copy other fields if present
        for key, value in item.items():
            if key not in ['caption', 'embedding_path']:
                output_item[key] = value

        output_data.append(output_item)

    # Save output JSON
    output_json = output_dir / "embeddings.json"
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nPreprocessing complete!")
    print(f"Saved {len(output_data)} embeddings to {output_dir}")
    print(f"Output JSON: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Pre-extract T5 embeddings for WAN SRPO training")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to WAN model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Input JSON file with captions"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use"
    )

    args = parser.parse_args()
    preprocess_embeddings(args)


if __name__ == "__main__":
    main()
