# CompCars Model Download Tools

This directory contains utilities to download and create CompCars vehicle classification models.

## Available Tools

### 1. Model Downloader (`download_model.py`)

Automatically downloads pre-trained CompCars models from GitHub or HuggingFace.

**Usage:**
```bash
# Download from GitHub (default)
python -m src.download_model

# Download from HuggingFace Hub
python -m src.download_model --method huggingface

# Specify custom output directory
python -m src.download_model --model-dir /path/to/models
```

**Configuration:**

Edit `src/download_model.py` to configure your model sources:

```python
# For GitHub Releases
MODEL_URLS = {
    'resnet50': {
        'weights': 'https://github.com/your-user/repo/releases/download/v1.0/compcars_resnet50.pth',
        'labels': 'https://github.com/your-user/repo/releases/download/v1.0/compcars_labels.txt',
    }
}

# For HuggingFace Hub
HUGGINGFACE_MODEL = {
    'repo_id': 'your-username/compcars-resnet50',
    'filename': 'pytorch_model.bin'
}
```

### 2. Sample Model Generator (`create_sample_model.py`)

Creates a sample model structure for testing purposes (NOT trained).

**Usage:**
```bash
# Create sample model in default location
python -m src.create_sample_model

# Specify custom output directory
python -m src.create_sample_model --output-dir /path/to/models
```

**⚠️ Important:** This creates a model with random weights. Use only for:
- Testing code structure
- Development without real model
- Verifying model loading works

**DO NOT use for real classification!** The predictions will be meaningless.

### 3. Automatic Download (Built-in)

The vehicle recognizer automatically attempts to download models on first run if configured.

Control this behavior in `config.yaml`:
```yaml
alpr:
  auto_download_vehicle_model: true  # Set to false to disable
```

## Where Models Are Saved

Default location: `models/`
```
models/
├── compcars_resnet50.pth    # PyTorch model weights
└── compcars_labels.txt       # Class labels (one per line)
```

## Model Hosting Options

### Option A: GitHub Releases

1. Train or obtain a CompCars model
2. Create a GitHub release
3. Upload model files as release assets
4. Update URLs in `download_model.py`

### Option B: HuggingFace Hub

1. Create HuggingFace account
2. Create a new model repository
3. Upload model using `huggingface-cli`:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload your-username/compcars-resnet50 models/compcars_resnet50.pth
   huggingface-cli upload your-username/compcars-resnet50 models/compcars_labels.txt
   ```
4. Update repo ID in `download_model.py`

### Option C: Direct URL

Host the model files anywhere and provide direct download URLs in `download_model.py`.

## Training Your Own Model

See [COMPCARS_SETUP.md](../COMPCARS_SETUP.md) for complete training instructions.

Quick steps:
1. Download CompCars dataset
2. Run training script (provided in COMPCARS_SETUP.md)
3. Model saved to `models/compcars_resnet50.pth`
4. Labels saved to `models/compcars_labels.txt`

## Verification

After downloading/creating a model, verify it loads:

```bash
# Check model files exist
ls -lh models/

# Start ReolinkANPR and check logs
python run.py
```

Expected log output:
```
✓ Loaded CompCars fine-tuned model from models/compcars_resnet50.pth
✓ Loaded 431 vehicle classes from CompCars
✓ CompCars model loaded successfully with 431 classes
```

## Troubleshooting

### Download fails
- Check internet connection
- Verify URLs in `download_model.py`
- Try manual download instead

### Model won't load
- Verify file paths are correct
- Check PyTorch version compatibility
- Ensure model format matches expected structure

### Random predictions (sample model)
- Sample model has random weights - train a real model
- See COMPCARS_SETUP.md for training instructions

## Dependencies

Required packages:
```bash
pip install torch torchvision tqdm
pip install huggingface_hub  # Optional, for HuggingFace downloads
```

All included in `requirements.txt`.
