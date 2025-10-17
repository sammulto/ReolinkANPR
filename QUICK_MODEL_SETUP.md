# Quick Model Setup Guide

Choose the option that fits your needs:

## Option 1: Testing/Development (Fastest) ⚡

**Create a sample model with random weights (NOT for real use)**

```bash
python -m src.download_model --method sample
```

**Result:** Model structure created instantly. Good for:
- Testing the code works
- Development without real model
- Understanding model format

**⚠️ Warning:** Predictions will be random/meaningless!

---

## Option 2: Download Base Model 📥

**Download pre-trained ResNet50 (ImageNet)**

```bash
python -m src.download_model --method base
```

**Result:** 
- Base model downloaded
- Needs fine-tuning on CompCars dataset
- See `COMPCARS_SETUP.md` for training

---

## Option 3: Train Your Own 🎓

**Best option for real vehicle classification**

1. Download CompCars dataset
2. Follow training guide in `COMPCARS_SETUP.md`
3. Model automatically saved to `models/`

**Time:** 2-4 hours on GPU  
**Accuracy:** 70-90% (depending on training)

---

## Option 4: Download Pre-trained (If Available) 🌐

**If you have access to a trained CompCars model**

### From GitHub:
```bash
# Configure URLs in src/download_model.py first
python -m src.download_model --method github
```

### From HuggingFace:
```bash
pip install huggingface_hub
# Configure repo in src/download_model.py first
python -m src.download_model --method huggingface
```

---

## Quick Comparison

| Method | Time | Accuracy | Best For |
|--------|------|----------|----------|
| Sample | 10 seconds | 0% (random) | Testing code |
| Base | 5 minutes | N/A (needs training) | Training base |
| Train | 2-4 hours | 70-90% | Real use |
| Download | 5-10 minutes | 70-90% | Real use |

---

## Recommended Path

**For Testing:**
```bash
python -m src.download_model --method sample
python run.py  # Test the system
```

**For Production:**
1. Train on CompCars (see `COMPCARS_SETUP.md`)
2. Or download pre-trained model if available
3. Deploy to production

---

## Verification

After setup, check:

```bash
# Check files exist
ls models/

# Should see:
# - compcars_resnet50.pth
# - compcars_labels.txt

# Test loading
python run.py

# Check logs for:
# ✓ Loaded CompCars fine-tuned model
# ✓ Loaded N vehicle classes
```

---

## Need Help?

- **Model won't load:** Check `models/` directory has both `.pth` and `.txt` files
- **Poor accuracy:** Use trained model, not sample model
- **Training questions:** See `COMPCARS_SETUP.md`
- **Download issues:** See `DOWNLOAD_MODELS.md`

---

## Without Model

The system still works without a vehicle classification model:
- ✅ Color detection works (no model needed)
- ✅ Plate recognition works
- ❌ Make/Model returns 'unknown'

So you can use the system immediately with just color detection!
