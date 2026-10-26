"""
Quick verification script to check if the setup is correct.
"""

import sys
import importlib
import shutil

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
    import torch
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
        'librosa',
        'fastapi', 'uvicorn', 'multipart', 'websockets', 'soundfile',
        'scipy', 'sklearn', 'yaml', 'omegaconf', 'datasets',
        'moviepy.editor', 'transformers.pipelines.audio_utils'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'librosa':
                import librosa
            else:
                importlib.import_module(dep)
            print(f"✓ {dep}")
        except Exception as error:
            print(f"✗ {dep}: {error}")
            missing.append(dep)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install them with: python -m pip install -r app/backend/requirements.txt")
        return False
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        print("✗ ffmpeg/ffprobe missing; install system FFmpeg")
        return False
    print("✓ ffmpeg and ffprobe")
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from app.backend.main import app
        from app.backend.models.transcriber import TRANSFORMERS_AVAILABLE
        from app.backend.models.video_transcriber import TRANSFORMERS_AVAILABLE as VIDEO_AVAILABLE
        assert TRANSFORMERS_AVAILABLE and VIDEO_AVAILABLE, "Whisper imports unavailable"
        print("✓ Backend app, Whisper, and emotion model classes")
    except Exception as error:
        print(f"✗ Backend import: {error}")
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
    
    # Check dependencies before importing application modules.
    deps_ok = check_dependencies()
    if not deps_ok:
        sys.exit(1)
    imports_ok = check_imports()
    
    # Check device
    device = check_device()
    
    
    print("\n" + "="*50)
    if imports_ok and deps_ok:
        print("✓ Setup verification passed!")
        print(f"Ready to use {device.upper()} device")
    else:
        print("✗ Setup verification failed")
        print("Please install missing dependencies and fix import errors")
        sys.exit(1)
    print("="*50)

if __name__ == '__main__':
    # Direct script execution otherwise only puts scripts/ on sys.path.
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()

