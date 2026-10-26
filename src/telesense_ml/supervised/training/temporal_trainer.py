"""
Trainer for temporal localization models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
from typing import Dict, Optional

from ..models import TemporalMultimodalBehavioralModel
from .temporal_losses import TemporalLocalizationLoss
from .metrics import Accuracy, F1Score


class TemporalTrainer:
    """
    Trainer for temporal localization models.
    """
    
    def __init__(
        self,
        model: TemporalMultimodalBehavioralModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the TemporalTrainer.
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
        loss_config = self.config.get('training', {}).get('loss', {})
        self.criterion = TemporalLocalizationLoss(
            loss_type=loss_config.get('type', 'focal'),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            class_weights=loss_config.get('class_weights'),
            temporal_consistency_weight=0.1
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
        
        # Metrics (frame-level)
        num_classes = self.config.get('model', {}).get('temporal', {}).get('num_concern_types', 7)
        self.train_metrics = {
            'accuracy': Accuracy(),
            'f1': F1Score(num_classes=num_classes)
        }
        self.val_metrics = {
            'accuracy': Accuracy(),
            'f1': F1Score(num_classes=num_classes)
        }
        
        # Checkpointing
        self.save_dir = training_config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_every = training_config.get('save_every', 5)
        self.eval_every = training_config.get('eval_every', 1)
        
        # Scaler for mixed precision
        if self.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            if self.device.type != 'cuda':
                self.mixed_precision = False
    
    def train_epoch(self, epoch: int):
        """
        Train for one epoch.
        Input:
            epoch - current epoch number

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
                with torch.cuda.amp.autocast():
                    frame_scores = self._forward_pass(batch)
                    # Convert targets to frame-level if needed
                    targets = self._prepare_targets(batch)
                    loss = self.criterion(frame_scores, targets)
                    loss = loss / self.gradient_accumulation_steps
            else:
                frame_scores = self._forward_pass(batch)
                targets = self._prepare_targets(batch)
                loss = self.criterion(frame_scores, targets)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update metrics (use frame-level predictions)
            self._update_metrics(self.train_metrics, frame_scores, targets)
            
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
            Dictionary of averaged validation metrics (empty when there is no val_loader)
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
                batch = self._move_to_device(batch)
                
                frame_scores = self._forward_pass(batch)
                targets = self._prepare_targets(batch)
                loss = self.criterion(frame_scores, targets)
                
                self._update_metrics(self.val_metrics, frame_scores, targets)
                total_loss += loss.item()
        
        # Compute final metrics
        final_metrics = {}
        for name, metric in self.val_metrics.items():
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
            batch - batch dictionary already on self.device

        Output:
            Frame-level concern scores (batch, time, num_concern_types)
        """
        return self.model(
            audio_waveform=batch.get('audio_waveform'),
            audio_mask=batch.get('audio_mask'),
            video_frames=batch.get('video_frames'),
            text_input_ids=batch.get('text_input_ids'),
            text_mask=batch.get('text_mask')
        )
    
    def _prepare_targets(self, batch: Dict) -> torch.Tensor:
        """
        Prepare targets for temporal localization.
        Input:
            batch - Input batch

        Output:
            Targets of shape (batch, time, num_classes)
        """
        if 'frame_labels' in batch:
            # Frame-level labels already provided
            return batch['frame_labels']
        elif 'temporal_segments' in batch:
            # Convert segments to frame-level labels
            return self._segments_to_frames(
                batch['temporal_segments'],
                batch.get('video_frames') or batch.get('audio_waveform')
            )
        elif 'labels' in batch:
            # Expand single label to all frames (fallback)
            labels = batch['labels']
            num_frames = batch.get('video_frames', batch.get('audio_waveform')).shape[1]
            num_classes = self.config.get('model', {}).get('temporal', {}).get('num_concern_types', 7)
            # Convert to one-hot
            frame_labels = torch.zeros(
                labels.shape[0], num_frames, num_classes,
                device=labels.device
            )
            frame_labels.scatter_(2, labels.unsqueeze(1).expand(-1, num_frames, -1), 1.0)
            return frame_labels
        else:
            raise ValueError("No valid targets found in batch")
    
    def _segments_to_frames(self, segments, video_or_audio):
        """
        Convert temporal segments to frame-level labels.
        Input:
            segments - per-sample lists of (start_frame, end_frame, concern_type)
            video_or_audio - input tensor used for the batch size, frame count and device

        Output:
            Frame-level multi-hot labels (batch, num_frames, num_concern_types)
        """
        # This is a placeholder - implement based on your annotation format
        batch_size = video_or_audio.shape[0]
        num_frames = video_or_audio.shape[1]
        num_classes = self.config.get('model', {}).get('temporal', {}).get('num_concern_types', 7)
        
        frame_labels = torch.zeros(batch_size, num_frames, num_classes, device=video_or_audio.device)
        
        for b, segs in enumerate(segments):
            for start, end, concern_type in segs:
                if start < num_frames and end < num_frames:
                    frame_labels[b, start:end+1, concern_type] = 1.0
        
        return frame_labels
    
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
    
    def _update_metrics(self, metrics: Dict, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with predictions.
        Input:
            metrics - dictionary of running metric values (updated in place)
            predictions - predicted frame scores
            targets - frame-level labels

        Output:
            None
        """
        # Flatten temporal dimension for metrics
        batch_size, num_frames, num_classes = predictions.shape
        predictions_flat = predictions.view(batch_size * num_frames, num_classes)
        
        if targets.dim() == 3:
            targets_flat = targets.view(batch_size * num_frames, num_classes)
            # Convert to class indices
            targets_flat = torch.argmax(targets_flat, dim=-1)
        else:
            targets_flat = targets.view(batch_size * num_frames)
        
        for metric in metrics.values():
            metric.update(predictions_flat, targets_flat)
    
    def _format_metrics(self, metrics: Dict) -> Dict:
        """
        Format metrics for display.
        Input:
            metrics - dictionary of metric values

        Output:
            Metrics formatted as a display string
        """
        formatted = {}
        for name, metric in metrics.items():
            results = metric.get_results()
            for k, v in results.items():
                formatted[k] = f'{v:.4f}'
        return formatted
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        Input:
            epoch - current epoch number
            is_best - also save as best_model.pt when True

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

