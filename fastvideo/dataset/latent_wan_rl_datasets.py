# Dataset loader for WAN SRPO training
# Modified from latent_flux_rl_datasets.py to support WAN's T5-only embeddings

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path


class LatentDataset(Dataset):
    """
    Dataset for loading pre-extracted T5 embeddings for WAN SRPO training
    """
    def __init__(self, data_json_path, temporal_frames=0):
        """
        Args:
            data_json_path: Path to JSON file containing caption paths
            temporal_frames: Number of temporal frames (0 for T2I, >0 for video)
        """
        super().__init__()
        self.data_json_path = data_json_path
        self.temporal_frames = temporal_frames

        # Load data paths
        with open(data_json_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            encoder_hidden_states: T5 text embeddings (shape: [seq_len, hidden_dim])
            caption: Original text prompt
        """
        item = self.data[idx]

        # Load pre-extracted T5 embeddings
        embedding_path = item.get('embedding_path', None)

        if embedding_path is None:
            raise ValueError(f"No embedding_path found for item {idx}")

        # Load the embedding file (.pt or .pth)
        embedding_data = torch.load(embedding_path, map_location='cpu')

        # Extract T5 embeddings
        # Expected format: {'encoder_hidden_states': tensor, 'caption': str}
        if isinstance(embedding_data, dict):
            encoder_hidden_states = embedding_data['encoder_hidden_states']
            caption = embedding_data.get('caption', item.get('caption', ''))
        else:
            # If it's just a tensor
            encoder_hidden_states = embedding_data
            caption = item.get('caption', '')

        return {
            'encoder_hidden_states': encoder_hidden_states,
            'caption': caption
        }


def latent_collate_function(batch):
    """
    Collate function for WAN SRPO training

    Args:
        batch: List of dictionaries from __getitem__

    Returns:
        encoder_hidden_states: Batched T5 embeddings
        captions: List of captions
    """
    encoder_hidden_states = []
    captions = []

    for item in batch:
        encoder_hidden_states.append(item['encoder_hidden_states'])
        captions.append(item['caption'])

    # Stack embeddings
    # T5 embeddings shape: [batch, seq_len, hidden_dim]
    encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)

    return encoder_hidden_states, captions
