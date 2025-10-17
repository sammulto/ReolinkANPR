# Vehicle Classification Models

This directory should contain trained models for vehicle make/model classification.

## Required Files

1. **compcars_resnet50.pth** - Fine-tuned ResNet50 model on CompCars dataset
2. **compcars_labels.txt** - Class labels file (one class per line)

## Setup

See [COMPCARS_SETUP.md](../COMPCARS_SETUP.md) for detailed instructions on:
- Downloading pre-trained models
- Training your own CompCars model
- Model format specifications

## Quick Download (Example)

If you have a pre-trained model:

```bash
# Place your model files here
models/
├── compcars_resnet50.pth    # PyTorch model weights
└── compcars_labels.txt       # Class labels
```

## Status

Currently, the system will check for models on startup:
- ✅ If models found: Make/model detection enabled
- ⚠️ If models missing: Returns 'unknown' for make/model (color detection still works)

Check application logs for model loading status.
