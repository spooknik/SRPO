#!/usr/bin/env python3
"""
Convert prompts.yaml to captions.json format for WAN SRPO preprocessing
Usage: python scripts/utils/convert_yaml_to_json.py
"""

import yaml
import json
from pathlib import Path


def convert_yaml_to_json(yaml_path, json_path):
    """
    Convert YAML prompts file to JSON format expected by preprocessing script

    Input YAML format:
        prompts:
          - "prompt 1"
          - "prompt 2"

    Output JSON format:
        [
          {"caption": "prompt 1"},
          {"caption": "prompt 2"}
        ]
    """
    print(f"Reading prompts from: {yaml_path}")

    # Read YAML file
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Extract prompts
    if isinstance(data, dict) and 'prompts' in data:
        prompts = data['prompts']
    elif isinstance(data, list):
        prompts = data
    else:
        raise ValueError("YAML file must contain 'prompts' key with a list of prompts")

    # Convert to expected JSON format
    captions = []
    for prompt in prompts:
        if prompt:  # Skip empty prompts
            captions.append({"caption": prompt.strip()})

    # Create output directory
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    print(f"✓ Converted {len(captions)} prompts")
    print(f"✓ Saved to: {json_path}")

    return captions


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    yaml_path = project_root / "prompts.yaml"
    json_path = project_root / "data" / "captions.json"

    # Check if YAML file exists
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found!")
        print("Please create prompts.yaml in the project root with your prompts.")
        return 1

    # Convert
    try:
        captions = convert_yaml_to_json(yaml_path, json_path)

        print("\n" + "="*50)
        print("Conversion complete!")
        print("="*50)
        print(f"Total prompts: {len(captions)}")
        print(f"\nFirst 3 prompts:")
        for i, caption in enumerate(captions[:3], 1):
            print(f"  {i}. {caption['caption'][:80]}{'...' if len(caption['caption']) > 80 else ''}")

        print(f"\nNext step:")
        print(f"  bash scripts/preprocess/preprocess_wan_rl_embeddings.sh")

        return 0

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
