"""
Custom collators for batching multimodal data.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List
from transformers import DistilBertTokenizer


class MultimodalCollator:
    """
    Collator for multimodal batches.
    """
    
    def __init__(self, text_tokenizer=None, max_length=512):
        """
        Initialize the MultimodalCollator.
        Input:
            text_tokenizer - HuggingFace tokenizer for text (DistilBERT tokenizer when None)
            max_length - maximum number of text tokens
        """
        self.text_tokenizer = text_tokenizer or DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        Input:
            batch - List of sample dictionaries

        Output:
            Batched dictionary
        """
        batched = {}
        
        # Collect all available modalities
        audio_samples = [s['audio'] for s in batch if 'audio' in s]
        video_samples = [s['video'] for s in batch if 'video' in s]
        text_samples = [s['text'] for s in batch if 'text' in s]
        labels = [s['label'] for s in batch]
        
        # Batch audio
        if audio_samples:
            # Pad audio sequences
            audio_batch = pad_sequence(
                audio_samples,
                batch_first=True,
                padding_value=0.0
            )
            batched['audio_waveform'] = audio_batch
            
            # Create attention masks
            audio_masks = []
            for audio in audio_samples:
                mask = torch.ones(len(audio), dtype=torch.bool)
                audio_masks.append(mask)
            audio_mask_batch = pad_sequence(
                audio_masks,
                batch_first=True,
                padding_value=False
            )
            batched['audio_mask'] = audio_mask_batch
        
        # Batch video
        if video_samples:
            # All videos should have same number of frames after preprocessing
            # Stack them
            video_batch = torch.stack(video_samples)  # (batch, num_frames, C, H, W)
            batched['video_frames'] = video_batch
        
        # Batch text
        if text_samples:
            # Tokenize texts
            tokenized = self.text_tokenizer(
                text_samples,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            batched['text_input_ids'] = tokenized['input_ids']
            batched['text_mask'] = tokenized['attention_mask']
        
        # Batch labels
        batched['labels'] = torch.stack(labels)
        
        return batched

