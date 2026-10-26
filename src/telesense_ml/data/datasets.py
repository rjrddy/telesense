"""
Dataset loaders for multimodal emotion recognition datasets.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple
import json

from .preprocessing import AudioPreprocessor, VideoPreprocessor, TextPreprocessor


class MultimodalDataset(Dataset):
    """
    Generic multimodal dataset loader.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        audio_preprocessor: Optional[AudioPreprocessor] = None,
        video_preprocessor: Optional[VideoPreprocessor] = None,
        text_preprocessor: Optional[TextPreprocessor] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the MultimodalDataset.
        Input:
            data_dir - root directory of the dataset
            split - dataset split to load ('train', 'val' or 'test')
            audio_preprocessor - AudioPreprocessor instance (default settings when None)
            video_preprocessor - VideoPreprocessor instance (default settings when None)
            text_preprocessor - TextPreprocessor instance (default settings when None)
            config - optional configuration dictionary
        """
        self.data_dir = data_dir
        self.split = split
        self.config = config or {}
        
        # Initialize preprocessors
        self.audio_preprocessor = audio_preprocessor or AudioPreprocessor(
            sample_rate=self.config.get('audio', {}).get('sample_rate', 16000),
            max_length=self.config.get('audio', {}).get('max_length', 10),
            normalize=self.config.get('audio', {}).get('normalize', True)
        )
        
        self.video_preprocessor = video_preprocessor or VideoPreprocessor(
            fps=self.config.get('video', {}).get('fps', 30),
            frame_size=self.config.get('video', {}).get('frame_size', 224),
            max_frames=self.config.get('video', {}).get('max_frames', 300),
            face_detection=self.config.get('video', {}).get('face_detection', True)
        )
        
        self.text_preprocessor = text_preprocessor or TextPreprocessor(
            max_length=self.config.get('text', {}).get('max_length', 512),
            lowercase=self.config.get('text', {}).get('lowercase', True)
        )
        
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> pd.DataFrame:
        """
        Load dataset metadata.
        Input:
            None

        Output:
            DataFrame with dataset information
        """
        # Try to load from metadata directory
        metadata_path = os.path.join(self.data_dir, "metadata", f"{self.split}.csv")
        
        if os.path.exists(metadata_path):
            return pd.read_csv(metadata_path)
        
        # No metadata for this split: return an empty dataset (no samples are invented)
        return pd.DataFrame({
            'audio_path': [],
            'video_path': [],
            'text': [],
            'label': []
        })
    
    def __len__(self) -> int:
        """
        Number of samples in the split.
        Input:
            None

        Output:
            Number of samples
        """
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        Input:
            idx - Sample index

        Output:
            Dictionary with audio, video, text, and label
        """
        row = self.metadata.iloc[idx]
        
        sample = {}
        
        # Load audio if available
        if 'audio_path' in row and pd.notna(row['audio_path']):
            audio_path = os.path.join(self.data_dir, row['audio_path'])
            if os.path.exists(audio_path):
                sample['audio'] = self.audio_preprocessor(audio_path)
                sample['audio_mask'] = self.audio_preprocessor.create_attention_mask(
                    sample['audio']
                )
        
        # Load video if available
        if 'video_path' in row and pd.notna(row['video_path']):
            video_path = os.path.join(self.data_dir, row['video_path'])
            if os.path.exists(video_path):
                sample['video'] = self.video_preprocessor(video_path)
        
        # Load text if available
        if 'text' in row and pd.notna(row['text']):
            text = self.text_preprocessor(str(row['text']))
            sample['text'] = text
        
        # Load label
        if 'label' in row:
            sample['label'] = torch.tensor(int(row['label']), dtype=torch.long)
        elif 'emotion' in row:
            sample['label'] = torch.tensor(int(row['emotion']), dtype=torch.long)
        else:
            # Default label if not found
            sample['label'] = torch.tensor(0, dtype=torch.long)
        
        return sample


def get_dataset(
    dataset_name: str,
    data_dir: str,
    split: str = "train",
    config: Optional[Dict] = None
) -> MultimodalDataset:
    """
    Get dataset by name.
    Input:
        dataset_name - Name of dataset (cmu_mosei, crema_d, meld, mer2024)
        data_dir - Root data directory
        split - Dataset split (train, val, test)
        config - Configuration dictionary

    Output:
        MultimodalDataset instance
    """
    dataset_path = os.path.join(data_dir, "raw", dataset_name)
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset not found at {dataset_path}")
    
    return MultimodalDataset(
        data_dir=dataset_path,
        split=split,
        config=config
    )

