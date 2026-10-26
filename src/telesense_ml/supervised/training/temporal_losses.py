"""
Loss functions for temporal localization tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in temporal localization.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize the FocalLoss.
        Input:
            alpha - weight for the positive class
            gamma - focusing parameter (higher down-weights easy frames)
            reduction - 'mean', 'sum' or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute the focal loss.
        Input:
            inputs - Predictions of shape (batch, time, num_classes)
            targets - Ground truth of shape (batch, time, num_classes) or (batch, time)

        Output:
            Loss tensor (mean, sum or per-element depending on reduction)
        """
        if targets.dim() == 2:
            # Convert to one-hot if needed
            num_classes = inputs.shape[-1]
            targets_one_hot = F.one_hot(targets, num_classes).float()
        else:
            targets_one_hot = targets
        
        # Compute BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets_one_hot, reduction='none'
        )
        
        # Compute focal weight
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TemporalBCELoss(nn.Module):
    """
    Binary cross-entropy loss for temporal localization.
    """
    
    def __init__(self, class_weights=None, reduction='mean'):
        """
        Initialize the TemporalBCELoss.
        Input:
            class_weights - optional per-class weights
            reduction - 'mean', 'sum' or 'none'
        """
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
        
        if class_weights is not None:
            self.register_buffer('weights', torch.tensor(class_weights))
        else:
            self.weights = None
    
    def forward(self, inputs, targets):
        """
        Compute the weighted binary cross-entropy loss.
        Input:
            inputs - Predictions of shape (batch, time, num_classes)
            targets - Ground truth of shape (batch, time, num_classes) or (batch, time)

        Output:
            Loss tensor (mean, sum or per-element depending on reduction)
        """
        if targets.dim() == 2:
            # Convert to one-hot if needed
            num_classes = inputs.shape[-1]
            targets_one_hot = F.one_hot(targets, num_classes).float()
        else:
            targets_one_hot = targets
        
        # Compute BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets_one_hot, reduction='none', weight=self.weights
        )
        
        if self.reduction == 'mean':
            return bce_loss.mean()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        else:
            return bce_loss


class TemporalLocalizationLoss(nn.Module):
    """
    Combined loss for temporal localization.
    """
    
    def __init__(
        self,
        loss_type='focal',
        focal_alpha=0.25,
        focal_gamma=2.0,
        class_weights=None,
        temporal_consistency_weight=0.1
    ):
        """
        Initialize the TemporalLocalizationLoss.
        Input:
            loss_type - frame loss to use ('focal' or 'bce')
            focal_alpha - focal loss alpha
            focal_gamma - focal loss gamma
            class_weights - optional per-class weights
            temporal_consistency_weight - weight of the penalty on frame-to-frame changes
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.temporal_consistency_weight = temporal_consistency_weight
        
        if loss_type == 'focal':
            self.main_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_type == 'bce':
            self.main_loss = TemporalBCELoss(class_weights=class_weights)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def temporal_consistency_loss(self, predictions):
        """
        Encourage smooth predictions over time.
        Input:
            predictions - Predictions of shape (batch, time, num_classes)

        Output:
            Consistency loss
        """
        # Compute difference between consecutive frames
        diff = predictions[:, 1:] - predictions[:, :-1]
        # L2 penalty on differences
        consistency_loss = torch.mean(diff ** 2)
        return consistency_loss
    
    def forward(self, predictions, targets):
        """
        Forward.
        Input:
            predictions - Model predictions of shape (batch, time, num_classes)
            targets - Ground truth of shape (batch, time, num_classes) or (batch, time)

        Output:
            Total loss
        """
        # Main classification loss
        main_loss = self.main_loss(predictions, targets)
        
        # Temporal consistency loss
        if self.temporal_consistency_weight > 0:
            consistency_loss = self.temporal_consistency_loss(predictions)
            total_loss = main_loss + self.temporal_consistency_weight * consistency_loss
        else:
            total_loss = main_loss
        
        return total_loss

