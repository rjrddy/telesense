"""
Evaluation metrics for multimodal models.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Dict, List


class Accuracy:
    """
    Accuracy metric.
    """
    
    def __init__(self):
        """
        Initialize the Accuracy.
        Input:
            None
        """
        self.reset()
    
    def reset(self):
        """
        Reset metric state.
        Input:
            None

        Output:
            None
        """
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metric with new predictions.
        Input:
            predictions - Model predictions (logits or class indices)
            targets - Ground truth labels

        Output:
            None
        """
        if predictions.dim() > 1:
            # If logits, get class indices
            predictions = torch.argmax(predictions, dim=1)
        
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self) -> float:
        """
        Compute accuracy.
        Input:
            None

        Output:
            Accuracy score
        """
        if len(self.predictions) == 0:
            return 0.0
        
        return accuracy_score(self.targets, self.predictions)
    
    def get_results(self) -> Dict:
        """
        Get metric results.
        Input:
            None

        Output:
            Dictionary with accuracy
        """
        return {"accuracy": self.compute()}


class F1Score:
    """
    F1 score metric.
    """
    
    def __init__(self, average="weighted", num_classes=None):
        """
        Initialize the F1Score.
        Input:
            average - averaging mode passed to sklearn ('weighted', 'macro' or 'micro')
            num_classes - number of classes (optional)
        """
        self.average = average
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """
        Reset metric state.
        Input:
            None

        Output:
            None
        """
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metric with new predictions.
        Input:
            predictions - Model predictions (logits or class indices)
            targets - Ground truth labels

        Output:
            None
        """
        if predictions.dim() > 1:
            # If logits, get class indices
            predictions = torch.argmax(predictions, dim=1)
        
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self) -> float:
        """
        Compute F1 score.
        Input:
            None

        Output:
            F1 score
        """
        if len(self.predictions) == 0:
            return 0.0
        
        return f1_score(
            self.targets,
            self.predictions,
            average=self.average,
            zero_division=0
        )
    
    def get_results(self) -> Dict:
        """
        Get metric results.
        Input:
            None

        Output:
            Dictionary with F1 scores
        """
        f1 = self.compute()
        results = {"f1_score": f1}
        
        # Also compute per-class F1 if applicable
        if self.num_classes is not None:
            per_class_f1 = f1_score(
                self.targets,
                self.predictions,
                average=None,
                labels=list(range(self.num_classes)),
                zero_division=0
            )
            for i, score in enumerate(per_class_f1):
                results[f"f1_class_{i}"] = float(score)
        
        return results


class ConfusionMatrix:
    """
    Confusion matrix metric.
    """
    
    def __init__(self, num_classes=None):
        """
        Initialize the ConfusionMatrix.
        Input:
            num_classes - number of classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """
        Reset metric state.
        Input:
            None

        Output:
            None
        """
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metric with new predictions.
        Input:
            predictions - Model predictions (logits or class indices)
            targets - Ground truth labels

        Output:
            None
        """
        if predictions.dim() > 1:
            # If logits, get class indices
            predictions = torch.argmax(predictions, dim=1)
        
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self) -> np.ndarray:
        """
        Compute confusion matrix.
        Input:
            None

        Output:
            Confusion matrix as numpy array
        """
        if len(self.predictions) == 0:
            return np.zeros((self.num_classes or 1, self.num_classes or 1))
        
        return confusion_matrix(
            self.targets,
            self.predictions,
            labels=list(range(self.num_classes)) if self.num_classes else None
        )
    
    def get_results(self) -> Dict:
        """
        Get metric results.
        Input:
            None

        Output:
            Dictionary with confusion matrix
        """
        cm = self.compute()
        return {"confusion_matrix": cm.tolist()}

