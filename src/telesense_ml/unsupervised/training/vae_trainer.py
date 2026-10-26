"""
Trainer for Temporal VAE.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
from typing import Dict, Optional


class VAETrainer:
    """
    Trainer for Temporal VAE.
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
        Initialize the VAETrainer.
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
        
        # Loss weights
        self.reconstruction_weight = self.config.get('training', {}).get('reconstruction_weight', 1.0)
        self.kl_weight = self.config.get('training', {}).get('kl_weight', 0.01)
        
        # Optimizer
        training_config = self.config.get('training', {})
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        if training_config.get('scheduler') == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('num_epochs', 50)
            )
        else:
            self.scheduler = None
        
        # Training settings
        self.num_epochs = training_config.get('num_epochs', 50)
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = training_config.get('max_grad_norm', 1.0)
        
        # Checkpointing
        self.save_dir = training_config.get('save_dir', 'checkpoints_unsupervised')
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_every = training_config.get('save_every', 5)
        self.eval_every = training_config.get('eval_every', 1)
    
    def vae_loss(self, reconstructed, original, mu, logvar):
        """
        Compute VAE loss.
        Input:
            reconstructed - reconstructed features
            original - original features
            mu - latent mean
            logvar - latent log-variance

        Output:
            Tuple (total_loss, reconstruction_loss, kl_loss)
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        # Total loss
        total_loss = self.reconstruction_weight * recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_epoch(self, epoch: int):
        """
        Train for one epoch.
        Input:
            epoch - current epoch number

        Output:
            Dictionary with average total, reconstruction and KL training losses
        """
        self.model.train()
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            batch = self._move_to_device(batch)
            
            # Forward pass
            reconstructed, mu, logvar, z, features = self.model(
                video_frames=batch.get('video_frames'),
                audio_waveform=batch.get('audio_waveform'),
                audio_mask=batch.get('audio_mask'),
                text_input_ids=batch.get('text_input_ids'),
                text_mask=batch.get('text_mask'),
                mode='vae'
            )
            
            # Compute loss
            loss, recon_loss, kl_loss = self.vae_loss(reconstructed, features, mu, logvar)
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'recon': f'{total_recon/(step+1):.4f}',
                'kl': f'{total_kl/(step+1):.4f}'
            })
        
        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_recon': total_recon / len(self.train_loader),
            'train_kl': total_kl / len(self.train_loader)
        }
    
    def validate(self):
        """
        Validate the model.
        Input:
            None

        Output:
            Dictionary with average validation losses (empty when there is no val_loader)
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = self._move_to_device(batch)
                
                reconstructed, mu, logvar, z, features = self.model(
                    video_frames=batch.get('video_frames'),
                    audio_waveform=batch.get('audio_waveform'),
                    audio_mask=batch.get('audio_mask'),
                    text_input_ids=batch.get('text_input_ids'),
                    text_mask=batch.get('text_mask'),
                    mode='vae'
                )
                
                loss, recon_loss, kl_loss = self.vae_loss(reconstructed, features, mu, logvar)
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_recon': total_recon / len(self.val_loader),
            'val_kl': total_kl / len(self.val_loader)
        }
    
    def train(self):
        """
        Full training loop.
        Input:
            None

        Output:
            None
        """
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if self.val_loader is not None and (epoch + 1) % self.eval_every == 0:
                val_metrics = self.validate()
                
                print(f"\nEpoch {epoch+1}/{self.num_epochs}")
                print(f"Train Loss: {train_metrics['train_loss']:.4f}")
                print(f"Val Loss: {val_metrics['val_loss']:.4f}")
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
    
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
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        Input:
            epoch - current epoch number
            is_best - also save as vae_best_model.pt when True

        Output:
            None
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.save_dir, f'vae_checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'vae_best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best VAE model to {best_path}")

