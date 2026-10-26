"""
Complete trainer for unsupervised temporal localization.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
from .vae_trainer import VAETrainer


class UnsupervisedTrainer:
    """
    Complete trainer for unsupervised model (contrastive + VAE).
    """
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the UnsupervisedTrainer.
        Input:
            model - model to train
            train_loader - DataLoader for the training split
            val_loader - DataLoader for the validation split (may be None)
            config - configuration dictionary (training section is read)
            device - torch device to train on (auto-selected when None)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        
        # Device setup
        if device is None:
            device_type = self.config.get('device', {}).get('type', 'mps')
            if device_type == 'mps' and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif device_type == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Training phases
        self.contrastive_epochs = self.config.get('training', {}).get('contrastive_epochs', 10)
        self.vae_epochs = self.config.get('training', {}).get('vae_epochs', 50)
        
        # Use VAE trainer for VAE phase
        self.vae_trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
    
    def train_contrastive_epoch(self, epoch: int):
        """
        Train contrastive learning for one epoch.
        Input:
            epoch - current epoch number

        Output:
            Dictionary with the average contrastive loss
        """
        self.model.train()
        
        from torch.optim import AdamW
        optimizer = AdamW(
            self.model.contrastive_encoder.parameters(),
            lr=self.config.get('training', {}).get('learning_rate', 1e-4)
        )
        
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Contrastive Epoch {epoch+1}/{self.contrastive_epochs}")
        
        for step, batch in enumerate(progress_bar):
            batch = self._move_to_device(batch)
            
            # Get projected features
            projected, features = self.model(
                video_frames=batch.get('video_frames'),
                audio_waveform=batch.get('audio_waveform'),
                audio_mask=batch.get('audio_mask'),
                text_input_ids=batch.get('text_input_ids'),
                text_mask=batch.get('text_mask'),
                mode='contrastive'
            )
            
            # Create positive and negative pairs
            # Positive: temporally close frames
            # Negative: temporally distant frames
            batch_size, num_frames, _ = projected.shape
            
            # Sample anchor, positive, negative
            anchor_idx = torch.randint(0, num_frames - 2, (batch_size,))
            positive_idx = anchor_idx + torch.randint(1, 3, (batch_size,))  # 1-2 frames away
            negative_idx = torch.randint(0, num_frames, (batch_size,))
            # Ensure negative is far from anchor
            negative_idx = torch.where(
                torch.abs(negative_idx - anchor_idx) < 5,
                (anchor_idx + 5) % num_frames,
                negative_idx
            )
            
            anchor = projected[torch.arange(batch_size), anchor_idx]
            positive = projected[torch.arange(batch_size), positive_idx]
            negative = projected[torch.arange(batch_size), negative_idx]
            
            # Compute contrastive loss
            loss = self.model.contrastive_encoder.contrastive_loss(anchor, positive, negative)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f'{total_loss/(step+1):.4f}'})
        
        return {'contrastive_loss': total_loss / len(self.train_loader)}
    
    def train_contrastive(self):
        """
        Train contrastive learning phase.
        Input:
            None

        Output:
            None
        """
        print("Phase 1: Contrastive Pre-training")
        for epoch in range(self.contrastive_epochs):
            metrics = self.train_contrastive_epoch(epoch)
            print(f"Contrastive Epoch {epoch+1}/{self.contrastive_epochs}, Loss: {metrics['contrastive_loss']:.4f}")
    
    def train_vae(self):
        """
        Train VAE phase.
        Input:
            None

        Output:
            None
        """
        print("\nPhase 2: VAE Training")
        self.vae_trainer.train()
    
    def train(self):
        """
        Complete training pipeline.
        Input:
            None

        Output:
            None
        """
        # Phase 1: Contrastive pre-training
        if self.contrastive_epochs > 0:
            self.train_contrastive()
        
        # Phase 2: VAE training
        self.train_vae()
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """
        Move batch to device.
        Input:
            batch - batch dictionary of tensors and metadata

        Output:
            Batch dictionary with tensors moved to self.device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

