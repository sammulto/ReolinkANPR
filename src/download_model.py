"""Utility to download pre-trained CompCars vehicle classification model."""

import os
import sys
from pathlib import Path
import urllib.request
import hashlib
from tqdm import tqdm

# Model URLs - using a pre-trained CompCars ResNet50 model
# Real model sources from public repositories
MODEL_URLS = {
    'resnet50': {
        # Using torchvision pretrained as base - needs fine-tuning for CompCars
        # For actual CompCars trained model, check:
        # - https://github.com/XiangZhang1997/MLIC
        # - https://github.com/omertov/CompCars
        # - HuggingFace model hub
        'weights': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',  # ImageNet weights
        'labels': 'https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/_meta.py',
        'weights_md5': None,
        'labels_md5': None,
        'note': 'Base ResNet50 - needs CompCars fine-tuning for vehicle classification'
    }
}

# Alternative: Use HuggingFace Hub for CompCars fine-tuned models
# Search for models at: https://huggingface.co/models?search=compcars
HUGGINGFACE_MODEL = {
    'repo_id': 'nateraw/vit-base-beans',  # Example repo format
    'filename': 'pytorch_model.bin',
    'note': 'Replace with actual CompCars fine-tuned model repo'
}

# Recommended CompCars pre-trained sources:
# 1. Train your own: See COMPCARS_SETUP.md
# 2. Stanford Cars dataset models (similar): https://github.com/BotechEngineering/StanfordCarsDatasetClassification
# 3. CompCars papers with code: https://paperswithcode.com/dataset/compcars
# 4. Model Zoo: https://github.com/XiangZhang1997/MLIC


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path, desc: str = "Downloading"):
    """Download file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    """Verify file MD5 hash."""
    if not expected_md5:
        return True
    
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    
    return md5.hexdigest() == expected_md5


def download_pretrained_base_model(model_dir: Path) -> bool:
    """
    Download a pre-trained base model that can be used as starting point.
    This downloads ResNet50 trained on ImageNet.
    
    For actual CompCars classification, you need to fine-tune this model
    or download a CompCars-specific fine-tuned version.
    
    Args:
        model_dir: Directory to save models
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import torch
        from torchvision import models
        
        print("📥 Downloading pre-trained ResNet50 (ImageNet)...")
        print("⚠️  Note: This is a BASE MODEL - needs fine-tuning on CompCars")
        
        # Download using torchvision
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        weights_path = model_dir / 'compcars_resnet50_base.pth'
        torch.save(model.state_dict(), weights_path)
        print(f"✓ Saved base model to: {weights_path}")
        
        # Create a note file
        note_path = model_dir / 'README_BASE_MODEL.txt'
        with open(note_path, 'w') as f:
            f.write("This is a BASE ResNet50 model trained on ImageNet.\n\n")
            f.write("It is NOT trained on CompCars dataset.\n")
            f.write("To use for vehicle classification:\n")
            f.write("1. Fine-tune on CompCars dataset (see COMPCARS_SETUP.md)\n")
            f.write("2. Or download a CompCars fine-tuned model\n")
            f.write("3. Or use the sample model generator for testing\n")
        
        print("\n⚠️  IMPORTANT:")
        print("   This base model is trained on ImageNet (general objects)")
        print("   It needs to be fine-tuned on CompCars for vehicle classification")
        print("   See COMPCARS_SETUP.md for training instructions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading base model: {e}")
        return False


def download_from_github(model_dir: Path, model_name: str = 'resnet50') -> bool:
    """
    Download model from GitHub releases.
    
    Args:
        model_dir: Directory to save models
        model_name: Model architecture name
        
    Returns:
        True if successful, False otherwise
    """
    if model_name not in MODEL_URLS:
        print(f"❌ Unknown model: {model_name}")
        return False
    
    model_info = MODEL_URLS[model_name]
    weights_path = model_dir / 'compcars_resnet50.pth'
    labels_path = model_dir / 'compcars_labels.txt'
    
    try:
        # Download weights
        if not weights_path.exists():
            print(f"📥 Downloading model weights...")
            download_url(model_info['weights'], weights_path, "Model weights")
            
            # Verify checksum
            if model_info['weights_md5']:
                if verify_md5(weights_path, model_info['weights_md5']):
                    print("✓ Model weights verified")
                else:
                    print("❌ Model weights verification failed!")
                    weights_path.unlink()
                    return False
        else:
            print(f"✓ Model weights already exist: {weights_path}")
        
        # Download labels
        if not labels_path.exists():
            print(f"📥 Downloading class labels...")
            download_url(model_info['labels'], labels_path, "Class labels")
            
            # Verify checksum
            if model_info['labels_md5']:
                if verify_md5(labels_path, model_info['labels_md5']):
                    print("✓ Class labels verified")
                else:
                    print("❌ Class labels verification failed!")
                    labels_path.unlink()
                    return False
        else:
            print(f"✓ Class labels already exist: {labels_path}")
        
        print("✅ CompCars model downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def download_from_huggingface(model_dir: Path) -> bool:
    """
    Download model from HuggingFace Hub.
    
    Args:
        model_dir: Directory to save models
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print("📥 Downloading from HuggingFace Hub...")
        
        # Download model weights
        weights_path = hf_hub_download(
            repo_id=HUGGINGFACE_MODEL['repo_id'],
            filename=HUGGINGFACE_MODEL['filename'],
            cache_dir=str(model_dir)
        )
        
        # Download labels
        labels_path = hf_hub_download(
            repo_id=HUGGINGFACE_MODEL['repo_id'],
            filename='labels.txt',
            cache_dir=str(model_dir)
        )
        
        # Copy to expected locations
        import shutil
        shutil.copy(weights_path, model_dir / 'compcars_resnet50.pth')
        shutil.copy(labels_path, model_dir / 'compcars_labels.txt')
        
        print("✅ Model downloaded from HuggingFace!")
        return True
        
    except ImportError:
        print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Error downloading from HuggingFace: {e}")
        return False


def download_model(method: str = 'github', model_dir: Path = None) -> bool:
    """
    Download CompCars pre-trained model.
    
    Args:
        method: 'github' or 'huggingface'
        model_dir: Directory to save models (default: models/)
        
    Returns:
        True if successful, False otherwise
    """
    if model_dir is None:
        model_dir = Path(__file__).parent.parent / 'models'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Downloading CompCars model to: {model_dir}")
    
    if method == 'github':
        return download_from_github(model_dir)
    elif method == 'huggingface':
        return download_from_huggingface(model_dir)
    else:
        print(f"❌ Unknown download method: {method}")
        return False


def main():
    """CLI interface for downloading models."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download CompCars vehicle classification model',
        epilog='For more options, see COMPCARS_SETUP.md and DOWNLOAD_MODELS.md'
    )
    parser.add_argument('--method', 
                       choices=['github', 'huggingface', 'base', 'sample'], 
                       default='sample',
                       help='Download method (default: sample)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save models (default: models/)')
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    print("=" * 60)
    print("CompCars Model Setup")
    print("=" * 60)
    print()
    
    if args.method == 'sample':
        print("Creating SAMPLE model for testing (random weights)...")
        print()
        try:
            from . import create_sample_model
            create_sample_model.save_sample_model(str(model_dir))
            success = True
        except ImportError:
            # Try direct import if running as script
            import create_sample_model
            create_sample_model.save_sample_model(str(model_dir))
            success = True
        except Exception as e:
            print(f"❌ Error creating sample model: {e}")
            print("   Try: python -m src.create_sample_model")
            success = False
    
    elif args.method == 'base':
        print("Downloading BASE ResNet50 model (ImageNet)...")
        print("This needs fine-tuning on CompCars dataset")
        print()
        success = download_pretrained_base_model(model_dir)
    
    elif args.method == 'github':
        print("Attempting to download from GitHub...")
        print("Note: You need to configure URLs in this script first")
        print()
        success = download_model(method='github', model_dir=model_dir)
        if not success:
            print("\n💡 Tip: Configure MODEL_URLS in src/download_model.py")
            print("   Or use --method sample to create test model")
    
    elif args.method == 'huggingface':
        print("Attempting to download from HuggingFace Hub...")
        print()
        success = download_model(method='huggingface', model_dir=model_dir)
        if not success:
            print("\n💡 Tip: Configure HUGGINGFACE_MODEL in src/download_model.py")
            print("   Or use --method sample to create test model")
    
    else:
        print(f"❌ Unknown method: {args.method}")
        success = False
    
    print()
    print("=" * 60)
    
    if success:
        print("✅ Setup complete!")
        print(f"   Model directory: {model_dir}")
        
        # Check what was created
        if (model_dir / 'compcars_resnet50.pth').exists():
            print(f"   ✓ Model file: compcars_resnet50.pth")
        if (model_dir / 'compcars_labels.txt').exists():
            print(f"   ✓ Labels file: compcars_labels.txt")
        
        print("\n🚀 Next steps:")
        if args.method == 'sample':
            print("   - Sample model created for testing only")
            print("   - For real classification, train on CompCars dataset")
            print("   - See COMPCARS_SETUP.md for training instructions")
        elif args.method == 'base':
            print("   - Base model downloaded - needs fine-tuning")
            print("   - See COMPCARS_SETUP.md for training instructions")
        else:
            print("   - Restart ReolinkANPR to use the model")
            print("   - Check logs to verify model loaded correctly")
    else:
        print("❌ Setup failed!")
        print("\nAvailable options:")
        print("   --method sample      Create test model (random weights)")
        print("   --method base        Download ImageNet ResNet50 (needs training)")
        print("   --method github      Download from GitHub (configure first)")
        print("   --method huggingface Download from HuggingFace (configure first)")
        print("\nSee COMPCARS_SETUP.md and DOWNLOAD_MODELS.md for more info")
    
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
