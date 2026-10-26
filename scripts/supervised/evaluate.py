"""
Evaluation script for multimodal behavioral model.
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from telesense_ml.supervised.models import MultimodalBehavioralModel
from telesense_ml.data import get_dataset, MultimodalCollator
from telesense_ml.supervised.training.metrics import Accuracy, F1Score, ConfusionMatrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    Input:
        config_path - path to the YAML configuration file

    Output:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main.
    Input:
        None

    Output:
        None
    """
    parser = argparse.ArgumentParser(description='Evaluate multimodal behavioral model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/supervised/model_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='datasets',
        help='Root directory for datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cmu_mosei',
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'config' in checkpoint:
        config = checkpoint['config']
    
    # Device setup
    device_type = config.get('device', {}).get('type', 'mps')
    if device_type == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load model
    model = MultimodalBehavioralModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load dataset
    try:
        test_dataset = get_dataset(
            args.dataset,
            args.data_dir,
            split=args.split,
            config=config.get('data', {})
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    collator = MultimodalCollator(
        max_length=config.get('data', {}).get('text', {}).get('max_length', 512)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('training', {}).get('batch_size', 8),
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    # Initialize metrics
    num_classes = config.get('model', {}).get('classifier', {}).get('num_classes', 7)
    accuracy = Accuracy()
    f1 = F1Score(num_classes=num_classes)
    confusion_matrix = ConfusionMatrix(num_classes=num_classes)
    
    # Evaluate
    print(f"Evaluating on {args.dataset} {args.split} split...")
    
    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            logits = model(
                audio_waveform=batch.get('audio_waveform'),
                audio_mask=batch.get('audio_mask'),
                video_frames=batch.get('video_frames'),
                text_input_ids=batch.get('text_input_ids'),
                text_mask=batch.get('text_mask')
            )
            
            # Update metrics
            accuracy.update(logits, batch['labels'])
            f1.update(logits, batch['labels'])
            confusion_matrix.update(logits, batch['labels'])
    
    # Compute results
    acc = accuracy.compute()
    f1_score = f1.compute()
    cm = confusion_matrix.compute()
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()

