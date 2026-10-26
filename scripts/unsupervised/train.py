"""
Training script for unsupervised temporal localization model.
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import os

from telesense_ml.unsupervised.models import UnsupervisedTemporalModel
from telesense_ml.data import get_dataset, MultimodalCollator
from telesense_ml.unsupervised.training import UnsupervisedTrainer


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
    parser = argparse.ArgumentParser(description='Train unsupervised temporal localization model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/unsupervised/unsupervised_config.yaml',
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
        nargs='+',
        default=['cmu_mosei', 'crema_d', 'meld', 'mer2024'],
        help='Datasets to use for training'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create data loaders
    train_loaders = []
    val_loaders = []
    
    for dataset_name in args.dataset:
        try:
            train_dataset = get_dataset(
                dataset_name,
                args.data_dir,
                split='train',
                config=config.get('data', {})
            )
            val_dataset = get_dataset(
                dataset_name,
                args.data_dir,
                split='val',
                config=config.get('data', {})
            )
            
            collator = MultimodalCollator(
                max_length=config.get('data', {}).get('text', {}).get('max_length', 512)
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.get('training', {}).get('batch_size', 4),
                shuffle=True,
                collate_fn=collator,
                num_workers=0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.get('training', {}).get('batch_size', 4),
                shuffle=False,
                collate_fn=collator,
                num_workers=0
            )
            
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
            
            print(f"Loaded {dataset_name}: {len(train_dataset)} train, {len(val_dataset)} val samples")
            
        except Exception as e:
            print(f"Warning: Could not load {dataset_name}: {e}")
            continue
    
    if not train_loaders:
        print("Error: No datasets loaded. Please check dataset paths.")
        return
    
    # Use first loader
    train_loader = train_loaders[0]
    val_loader = val_loaders[0] if val_loaders else None
    
    # Initialize model
    model = UnsupervisedTemporalModel(config)
    
    # Initialize trainer
    trainer = UnsupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Train
    print("Starting unsupervised temporal localization training...")
    trainer.train()
    print("Training completed!")


if __name__ == '__main__':
    main()

