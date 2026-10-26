"""
Loss functions for multimodal training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Standard classification loss.
    """
    
    def __init__(self, num_classes=7, class_weights=None):
        """
        Initialize the ClassificationLoss.
        Input:
            num_classes - number of classes
            class_weights - optional per-class weights for imbalance
        """
        super().__init__()
        self.num_classes = num_classes
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
        else:
            self.class_weights = None
    
    def forward(self, logits, labels):
        """
        Compute classification loss.
        Input:
            logits - Model predictions of shape (batch, num_classes)
            labels - Ground truth labels of shape (batch,)

        Output:
            Loss value
        """
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        return criterion(logits, labels)


class RegressionLoss(nn.Module):
    """
    Regression loss for continuous emotion dimensions.
    """
    
    def __init__(self):
        """
        Initialize the RegressionLoss.
        Input:
            None
        """
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Compute regression loss.
        Input:
            predictions - Model predictions
            targets - Ground truth targets

        Output:
            Loss value
        """
        return self.criterion(predictions, targets)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification and regression.
    """
    
    def __init__(self, classification_weight=1.0, regression_weight=0.5):
        """
        Initialize the MultiTaskLoss.
        Input:
            classification_weight - weight of the classification term
            regression_weight - weight of the regression term
        """
        super().__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        
        self.classification_loss = ClassificationLoss()
        self.regression_loss = RegressionLoss()
    
    def forward(
        self,
        classification_logits,
        classification_labels,
        regression_predictions=None,
        regression_targets=None
    ):
        """
        Compute multi-task loss.
        Input:
            classification_logits - Classification predictions
            classification_labels - Classification ground truth
            regression_predictions - Optional regression predictions
            regression_targets - Optional regression ground truth

        Output:
            Total loss value
        """
        # Classification loss
        cls_loss = self.classification_loss(classification_logits, classification_labels)
        
        total_loss = self.classification_weight * cls_loss
        
        # Regression loss (if provided)
        if regression_predictions is not None and regression_targets is not None:
            reg_loss = self.regression_loss(regression_predictions, regression_targets)
            total_loss += self.regression_weight * reg_loss
        
        return total_loss

