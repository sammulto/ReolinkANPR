# Quick Start: Vehicle Make/Model Recognition

Get vehicle make/model recognition working in **under 2 minutes**!

## TL;DR

```bash
# 1. Install the dependency
pip install huggingface-hub

# 2. Download the pre-trained model
python -m src.download_model

# 3. Restart your app - done!
```

## What You Get

- **Real Pre-trained Model**: Stanford Cars ResNet50 (not random weights!)
- **196 Vehicle Classes**: Includes make/model/year for common vehicles
- **Instant Results**: Download takes ~1-2 minutes on good connection
- **No Training Required**: Model is ready to use immediately

## Sample Vehicle Classes

The model can recognize vehicles like:
- Acura TL Sedan 2012
- Audi A4 Sedan 2012
- BMW 3 Series Sedan 2012
- Chevrolet Silverado 1500 Hybrid Crew Cab 2012
- Ford F-150 Regular Cab 2012
- Honda Accord Sedan 2012
- Tesla Model S Sedan 2012
- Toyota Camry Sedan 2012
- And 188+ more...

## Detailed Steps

### Step 1: Install Requirements

```bash
pip install huggingface-hub
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Download Model

```bash
python -m src.download_model
```

This will:
1. Download ResNet50 weights (~100MB) from HuggingFace
2. Download class labels (196 vehicle types)
3. Save everything to `models/` directory

Expected output:
```
[Downloading] Stanford Cars ResNet50 from HuggingFace Hub...
   Repository: anonauthors/stanford_cars-resnet50
   Downloading model weights...
   Downloading config...
   [SUCCESS] Saved model weights to: models/compcars_resnet50.pth
   [SUCCESS] Saved 196 class labels to: models/compcars_labels.txt

[SUCCESS] Stanford Cars model downloaded successfully!
   Classes: 196 vehicle types
```

### Step 3: Restart Your App

```bash
python run.py
```

Check the logs for confirmation:
```
[OK] Loaded CompCars fine-tuned model from models/compcars_resnet50.pth
[OK] Loaded 196 vehicle classes from CompCars
[OK] CompCars model loaded successfully with 196 classes
```

## Verification

After setup, vehicle detections will include:

```json
{
  "color": "silver",
  "make": "toyota",
  "model": "camry sedan 2012",
  "confidence": 0.87
}
```

Instead of:
```json
{
  "color": "silver",
  "make": "unknown",
  "model": "unknown",
  "confidence": 0.0
}
```

## Alternative Methods

### Method 1: Sample Model (Testing Only)

If you just want to test the code structure without downloading:

```bash
python -m src.download_model --method sample
```

**WARNING**: Creates random weights - predictions will be meaningless!

### Method 2: Train Your Own

For better accuracy on specific vehicle types, train your own model:

See [COMPCARS_SETUP.md](COMPCARS_SETUP.md) for detailed training instructions.

## Troubleshooting

### "huggingface_hub not installed"

```bash
pip install huggingface-hub
```

### Download fails

- Check internet connection
- Try again (HuggingFace might be temporarily down)
- Use sample model for testing: `python -m src.download_model --method sample`

### Model not loading

- Verify files exist:
  - `models/compcars_resnet50.pth`
  - `models/compcars_labels.txt`
- Check file permissions
- Look for errors in `logs/anpr.log`

### Poor predictions

The Stanford Cars model is trained on:
- Primarily US market vehicles
- Cars from 2008-2012 era
- Clean, well-lit images

For better accuracy:
- Use good quality camera images
- Ensure vehicles are well-lit
- Consider training a custom model for your region
- See [COMPCARS_SETUP.md](COMPCARS_SETUP.md) for training guide

## Performance

- **Download Size**: ~100MB
- **Download Time**: 1-2 minutes (fast connection)
- **Inference Speed**: 
  - GPU: ~50-100ms per vehicle
  - CPU: ~200-500ms per vehicle
- **Accuracy**: 70-85% top-1 on Stanford Cars test set

## What's Next?

1. Install dependencies
2. Download model
3. Test with your cameras
4. Monitor accuracy in your environment
5. Consider training custom model if needed

## Need Help?

- Check [COMPCARS_SETUP.md](COMPCARS_SETUP.md) for full documentation
- Review [VEHICLE_RECOGNITION.md](VEHICLE_RECOGNITION.md) for technical details
- See logs in `logs/anpr.log` for errors

---

**Note**: The Stanford Cars dataset is similar to CompCars and provides good general vehicle classification. For production use with specific requirements, consider training on your own dataset.
