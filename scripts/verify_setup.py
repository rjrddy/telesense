"""
Quick verification script to check if the setup is correct.
"""

import sys
import torch

def check_imports():
    """
    Check if all required modules can be imported.
    Input:
        None

    Output:
        True if every module imported, otherwise False
    """
    print("Checking imports...")
    try:
        from telesense_ml.supervised.models import MultimodalBehavioralModel
        from telesense_ml.data import MultimodalDataset, AudioPreprocessor, VideoPreprocessor, TextPreprocessor
        from telesense_ml.supervised.training import Trainer
        from telesense_ml.supervised.inference import Predictor
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_device():
    """
    Check available device.
    Input:
        None

    Output:
        Best available device name ('mps', 'cuda' or 'cpu')
    """
    print("\nChecking device...")
    if torch.backends.mps.is_available():
        print("✓ MPS (Metal) available")
        return "mps"
    elif torch.cuda.is_available():
        print("✓ CUDA available")
        return "cuda"
    else:
        print("✓ CPU available")
        return "cpu"

def check_dependencies():
    """
    Check if key dependencies are installed.
    Input:
        None

    Output:
        True if all key dependencies are installed, otherwise False
    """
    print("\nChecking dependencies...")
    dependencies = [
        'torch',
        'torchaudio',
        'torchvision',
        'transformers',
        'numpy',
        'pandas',
        'cv2',
        'librosa'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'librosa':
                import librosa
            else:
                __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} not found")
            missing.append(dep)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    return True

def main():
    """
    Main.
    Input:
        None

    Output:
        None
    """
    print("="*50)
    print("Setup Verification")
    print("="*50)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check imports
    imports_ok = check_imports()
    
    # Check device
    device = check_device()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    print("\n" + "="*50)
    if imports_ok and deps_ok:
        print("✓ Setup verification passed!")
        print(f"Ready to use {device.upper()} device")
    else:
        print("✗ Setup verification failed")
        print("Please install missing dependencies and fix import errors")
    print("="*50)

if __name__ == '__main__':
    main()

