"""
Training utilities for multimodal model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm
import os
from typing import Dict, Optional
import yaml

from ..models import MultimodalBehavioralModel
from .losses import MultiTaskLoss
from .metrics import Accuracy, F1Score, ConfusionMatrix


class Trainer:
    """
    Trainer for multimodal behavioral model.
    """
    
    def __init__(
        self,
        model: MultimodalBehavioralModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the Trainer.
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
        
        # Loss function
        loss_config = self.config.get('training', {}).get('loss_weights', {})
        self.criterion = MultiTaskLoss(
            classification_weight=loss_config.get('classification', 1.0),
            regression_weight=loss_config.get('regression', 0.5)
        )
        
        # Optimizer
        training_config = self.config.get('training', {})
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        scheduler_type = training_config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('num_epochs', 50)
            )
        else:
            self.scheduler = None
        
        # Training settings
        self.num_epochs = training_config.get('num_epochs', 50)
        self.gradient_accumulation_steps = training_config.get(
            'gradient_accumulation_steps', 1
        )
        self.max_grad_norm = training_config.get('max_grad_norm', 1.0)
        self.mixed_precision = training_config.get('mixed_precision', False)
        
        # Metrics
        num_classes = self.config.get('model', {}).get('classifier', {}).get('num_classes', 7)
        self.train_metrics = {
            'accuracy': Accuracy(),
            'f1': F1Score(num_classes=num_classes)
        }
        self.val_metrics = {
            'accuracy': Accuracy(),
            'f1': F1Score(num_classes=num_classes),
            'confusion_matrix': ConfusionMatrix(num_classes=num_classes)
        }
        
        # Checkpointing
        self.save_dir = training_config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_every = training_config.get('save_every', 5)
        self.eval_every = training_config.get('eval_every', 1)
        
        # Scaler for mixed precision (only for CUDA, MPS doesn't support autocast)
        if self.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            # Disable mixed precision for MPS/CPU
            if self.device.type != 'cuda':
                self.mixed_precision = False
    
    def train_epoch(self, epoch: int):
        """
        Train for one epoch.
        Input:
            epoch - Current epoch number

        Output:
            Dictionary of averaged training metrics
        """
        self.model.train()
        
        # Reset metrics
        for metric in self.train_metrics.values():
            metric.reset()
        
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            if self.mixed_precision and self.scaler is not None:
                # Only use autocast for CUDA
                with torch.cuda.amp.autocast():
                    logits = self._forward_pass(batch)
                    loss = self.criterion(logits, batch['labels'])
                    loss = loss / self.gradient_accumulation_steps
            else:
                logits = self._forward_pass(batch)
                loss = self.criterion(logits, batch['labels'])
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update metrics
            self._update_metrics(self.train_metrics, logits, batch['labels'])
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar
            avg_loss = total_loss / (step + 1)
            metrics_str = self._format_metrics(self.train_metrics)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                **metrics_str
            })
        
        # Compute final metrics
        final_metrics = {}
        for name, metric in self.train_metrics.items():
            final_metrics.update({f'train_{k}': v for k, v in metric.get_results().items()})
        
        final_metrics['train_loss'] = total_loss / len(self.train_loader)
        
        return final_metrics
    
    def validate(self):
        """
        Validate the model.
        Input:
            None

        Output:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        # Reset metrics
        for metric in self.val_metrics.values():
            metric.reset()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                logits = self._forward_pass(batch)
                loss = self.criterion(logits, batch['labels'])
                
                # Update metrics
                self._update_metrics(self.val_metrics, logits, batch['labels'])
                
                total_loss += loss.item()
        
        # Compute final metrics
        final_metrics = {}
        for name, metric in self.val_metrics.items():
            if name == 'confusion_matrix':
                final_metrics['val_confusion_matrix'] = metric.compute()
            else:
                final_metrics.update({f'val_{k}': v for k, v in metric.get_results().items()})
        
        final_metrics['val_loss'] = total_loss / len(self.val_loader)
        
        return final_metrics
    
    def train(self):
        """
        Full training loop.
        Input:
            None

        Output:
            None
        """
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if self.val_loader is not None and (epoch + 1) % self.eval_every == 0:
                val_metrics = self.validate()
                
                # Print metrics
                print(f"\nEpoch {epoch+1}/{self.num_epochs}")
                print(f"Train Loss: {train_metrics['train_loss']:.4f}")
                print(f"Train Accuracy: {train_metrics.get('train_accuracy', 0):.4f}")
                if val_metrics:
                    print(f"Val Loss: {val_metrics['val_loss']:.4f}")
                    print(f"Val Accuracy: {val_metrics.get('val_accuracy', 0):.4f}")
                
                # Save best model
                val_acc = val_metrics.get('val_accuracy', 0)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
    
    def _forward_pass(self, batch: Dict) -> torch.Tensor:
        """
        Forward pass through model.
        Input:
            batch - Input batch

        Output:
            Model logits
        """
        return self.model(
            audio_waveform=batch.get('audio_waveform'),
            audio_mask=batch.get('audio_mask'),
            video_frames=batch.get('video_frames'),
            text_input_ids=batch.get('text_input_ids'),
            text_mask=batch.get('text_mask')
        )
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """
        Move batch to device.
        Input:
            batch - Input batch

        Output:
            Batch on device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _update_metrics(self, metrics: Dict, logits: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics with predictions.
        Input:
            metrics - Dictionary of metrics
            logits - Model predictions
            labels - Ground truth labels

        Output:
            None
        """
        for metric in metrics.values():
            if isinstance(metric, ConfusionMatrix):
                metric.update(logits, labels)
            else:
                metric.update(logits, labels)
    
    def _format_metrics(self, metrics: Dict) -> Dict:
        """
        Format metrics for display.
        Input:
            metrics - Dictionary of metrics

        Output:
            Formatted metrics dictionary
        """
        formatted = {}
        for name, metric in metrics.items():
            if not isinstance(metric, ConfusionMatrix):
                results = metric.get_results()
                for k, v in results.items():
                    formatted[k] = f'{v:.4f}'
        return formatted
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        Input:
            epoch - Current epoch
            is_best - Whether this is the best model

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
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

