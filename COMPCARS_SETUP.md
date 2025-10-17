# CompCars Vehicle Classification Setup

This guide explains how to set up vehicle make/model classification using the CompCars dataset.

## Overview

The system now supports fine-tuned models for vehicle make/model classification. By default, it returns `unknown` for make/model until you provide a trained model.

## Quick Start

### Option 1: Auto-Download (Easiest)

```bash
# Download pre-trained model automatically
python -m src.download_model

# Or with HuggingFace Hub
python -m src.download_model --method huggingface
```

**Note:** You'll need to configure the model URLs in `src/download_model.py` to point to your hosted model or use HuggingFace Hub.

### Option 2: Create Sample Model (Testing Only)

```bash
# Create a sample model structure for testing code
python -m src.create_sample_model

# ⚠️ WARNING: This creates random weights - NOT for real classification!
# Use only to test that the code works
```

### Option 3: Download Manually

1. Download a pre-trained CompCars model:
   - Search for "CompCars ResNet50 PyTorch pretrained" on GitHub/HuggingFace
   - Or use models from: https://github.com/topics/compcars
   - Check model repositories like: https://github.com/XiangZhang1997/MLIC

2. Place the model files:
   ```
   models/
   ├── compcars_resnet50.pth    # Model weights
   └── compcars_labels.txt       # Class labels (one per line)
   ```

3. Restart the application - it will automatically detect and load the model

### Option 2: Train Your Own Model

#### Download CompCars Dataset

1. Visit: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
2. Download the "Surveillance data" or "Web data" subset
3. The dataset contains ~200 car models with make/model/year labels

#### Training Script

Create `train_compcars.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import pandas as pd

class CompCarsDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        """
        Args:
            data_dir: Directory with vehicle images
            labels_file: CSV with columns: image_path, make, model, year
            transform: Torchvision transforms
        """
        self.data_dir = Path(data_dir)
        self.labels = pd.read_csv(labels_file)
        self.transform = transform
        
        # Create class mapping
        self.labels['class_name'] = (
            self.labels['make'] + '_' + 
            self.labels['model'] + '_' + 
            self.labels['year'].astype(str)
        )
        self.classes = sorted(self.labels['class_name'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = self.data_dir / row['image_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        class_name = row['class_name']
        label = self.class_to_idx[class_name]
        
        return image, label

def train_compcars_model():
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = CompCarsDataset(
        'data/compcars/train',
        'data/compcars/train_labels.csv',
        transform=train_transform
    )
    
    val_dataset = CompCarsDataset(
        'data/compcars/val',
        'data/compcars/val_labels.csv',
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    num_classes = len(train_dataset.classes)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'models/compcars_resnet50.pth')
            print(f'Saved best model with val_acc: {val_acc:.2f}%')
        
        scheduler.step()
    
    # Save class labels
    with open('models/compcars_labels.txt', 'w', encoding='utf-8') as f:
        for class_name in train_dataset.classes:
            f.write(f'{class_name}\n')
    
    print(f'Training complete! Best validation accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    train_compcars_model()
```

#### Run Training

```bash
python train_compcars.py
```

## Model Format

The system expects:

1. **Model file** (`models/compcars_resnet50.pth`):
   - PyTorch checkpoint with ResNet50 architecture
   - Modified final layer for CompCars classes
   - Supported formats:
     - Dictionary with `model_state_dict` key
     - Dictionary with `state_dict` key
     - Raw state_dict

2. **Labels file** (`models/compcars_labels.txt`):
   - One class name per line
   - Format: `Make_Model_Year` or `Make Model`
   - Examples:
     ```
     Audi_A4_Sedan_2012
     BMW_3_Series_2015
     Toyota_Camry_2018
     Ford_F150_Pickup_2020
     ```

## Verification

After setting up, check the logs:

```
✓ Loaded CompCars fine-tuned model from models/compcars_resnet50.pth
✓ Loaded 431 vehicle classes from CompCars
✓ CompCars model loaded successfully with 431 classes
```

If the model is not found:
```
⚠ CompCars model not found. Make/model detection will return 'unknown'
  To enable make/model detection:
    1. Download or train a CompCars fine-tuned model
    2. Place it at: models/compcars_resnet50.pth
    3. Place class labels at: models/compcars_labels.txt
```

## Expected Performance

- **Training**: ~2-4 hours on GPU (depending on dataset size)
- **Accuracy**: 70-90% top-1 accuracy (depending on model and training)
- **Inference**: ~50-100ms per vehicle on GPU, ~200-500ms on CPU

## Alternative: Use Pre-trained from Model Zoo

You can also use models from:

1. **MMClassification**: https://github.com/open-mmlab/mmclassification
2. **Timm Models**: https://github.com/huggingface/pytorch-image-models
3. **HuggingFace Hub**: https://huggingface.co/models?pipeline_tag=image-classification&filter=car

Convert them to the expected format or modify the loading code accordingly.

## Troubleshooting

### Model not loading
- Check file paths: `models/compcars_resnet50.pth` and `models/compcars_labels.txt`
- Verify PyTorch version compatibility
- Check logs for specific error messages

### Poor accuracy
- Ensure training data quality
- Try data augmentation
- Increase training epochs
- Use a larger model (ResNet101, EfficientNet)

### Slow inference
- Use GPU if available
- Reduce image resolution
- Use a smaller model (MobileNet, EfficientNet-B0)

## Fallback Behavior

If no model is loaded:
- Make: `unknown`
- Model: `unknown`
- Color: Still detected using HSV analysis (works independently)

This ensures the system remains functional even without make/model classification.
