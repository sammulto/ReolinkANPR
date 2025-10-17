# Upgrading Vehicle Make/Model Detection

This guide explains how to improve the vehicle make/model detection beyond the basic implementation.

## Current Implementation

The current `vehicle_recognizer.py` uses ResNet50 trained on ImageNet, which is **not** vehicle-specific. This is intentional - it provides a working framework without requiring large model downloads or training.

## Option 1: Use Stanford Cars Pre-trained Model (Recommended)

### Step 1: Install Additional Dependencies
```bash
pip install timm  # PyTorch Image Models
```

### Step 2: Update `vehicle_recognizer.py`

Replace the model initialization section with:

```python
import timm

# In __init__:
try:
    logger.info("Loading vehicle classification model...")
    # Use EfficientNet trained on Stanford Cars (if you have weights)
    self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=196)
    # Load your fine-tuned weights:
    # self.model.load_state_dict(torch.load('path/to/stanford_cars_weights.pth'))
    self.model = self.model.to(self.device)
    self.model.eval()
    logger.info(f"Vehicle classification model loaded on {self.device}")
except Exception as e:
    logger.error(f"Failed to load vehicle model: {e}")
    self.enabled = False
```

### Step 3: Create Class Mappings

Download Stanford Cars dataset and create a mapping file:

```python
# stanford_cars_classes.py
MAKE_MODEL_CLASSES = [
    "Acura Integra 2001",
    "Acura RL Sedan 2012",
    "Audi A4 Sedan 2012",
    # ... 196 total classes
]

def parse_class(class_name):
    """Parse class name into make and model"""
    parts = class_name.split()
    make = parts[0]
    model = ' '.join(parts[1:-1])  # Everything except make and year
    return make.lower(), model.lower()
```

### Step 4: Update Classification Method

```python
def _classify_vehicle(self, image: np.ndarray) -> tuple[str, str]:
    try:
        from .stanford_cars_classes import MAKE_MODEL_CLASSES, parse_class
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get class name
        class_idx = predicted_idx.item()
        if class_idx < len(MAKE_MODEL_CLASSES):
            class_name = MAKE_MODEL_CLASSES[class_idx]
            make, model = parse_class(class_name)
            
            logger.info(f"Detected vehicle: {make} {model} (confidence: {confidence.item():.3f})")
            return make, model
        else:
            return 'unknown', 'unknown'
            
    except Exception as e:
        logger.error(f"Error classifying vehicle: {e}")
        return 'unknown', 'unknown'
```

## Option 2: Use Commercial API

### Google Cloud Vision API

```python
# Add to requirements.txt:
# google-cloud-vision

def _classify_vehicle_with_google(self, image: np.ndarray) -> tuple[str, str]:
    """Use Google Cloud Vision API for vehicle classification"""
    try:
        from google.cloud import vision
        
        client = vision.ImageAnnotatorClient()
        
        # Convert image to bytes
        success, encoded_image = cv2.imencode('.jpg', image)
        content = encoded_image.tobytes()
        
        image = vision.Image(content=content)
        
        # Detect objects and logos
        response = client.object_localization(image=image)
        objects = response.localized_object_annotations
        
        # Look for car-related objects
        for obj in objects:
            if 'car' in obj.name.lower() or 'vehicle' in obj.name.lower():
                # Parse make/model from object name
                name_parts = obj.name.split()
                if len(name_parts) >= 2:
                    return name_parts[0].lower(), ' '.join(name_parts[1:]).lower()
        
        # Try label detection as fallback
        response = client.label_detection(image=image)
        labels = response.label_annotations
        
        for label in labels:
            if 'car' in label.description.lower():
                return 'unknown', label.description.lower()
        
        return 'unknown', 'unknown'
        
    except Exception as e:
        logger.error(f"Google Vision API error: {e}")
        return 'unknown', 'unknown'
```

### AWS Rekognition

```python
# Add to requirements.txt:
# boto3

def _classify_vehicle_with_aws(self, image: np.ndarray) -> tuple[str, str]:
    """Use AWS Rekognition for vehicle classification"""
    try:
        import boto3
        
        client = boto3.client('rekognition')
        
        # Convert image to bytes
        success, encoded_image = cv2.imencode('.jpg', image)
        image_bytes = encoded_image.tobytes()
        
        # Detect labels
        response = client.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=10,
            MinConfidence=70
        )
        
        # Parse vehicle attributes from labels
        make = 'unknown'
        model = 'unknown'
        
        for label in response['Labels']:
            name = label['Name'].lower()
            if 'car' in name or 'vehicle' in name:
                # AWS Rekognition returns generic labels
                # You may need custom logic or their Custom Labels feature
                model = name
                break
        
        return make, model
        
    except Exception as e:
        logger.error(f"AWS Rekognition error: {e}")
        return 'unknown', 'unknown'
```

## Option 3: Train Your Own Model

### Step 1: Download Dataset

**Stanford Cars Dataset**:
- 16,185 images of 196 classes
- Download: http://ai.stanford.edu/~jkrause/cars/car_dataset.html

**CompCars Dataset**:
- 100,000+ images
- More comprehensive
- Download: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

### Step 2: Training Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms, datasets

def train_vehicle_classifier():
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    num_classes = 196  # Stanford Cars
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder('path/to/stanford_cars/train', transform=transform_train)
    val_dataset = datasets.ImageFolder('path/to/stanford_cars/test', transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_vehicle_model.pth')
        
        scheduler.step()
    
    print(f'Best validation accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    train_vehicle_classifier()
```

## Option 4: Use Pre-trained ONNX Models

Many vehicle classification models are available in ONNX format:

```python
import onnxruntime as ort

def _classify_vehicle_onnx(self, image: np.ndarray) -> tuple[str, str]:
    """Use ONNX model for vehicle classification"""
    try:
        # Load ONNX model (download from Model Zoo)
        session = ort.InferenceSession('vehicle_classifier.onnx')
        
        # Preprocess
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (224, 224))
        normalized = resized.astype('float32') / 255.0
        normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        
        # Get predictions
        probabilities = outputs[0][0]
        predicted_idx = np.argmax(probabilities)
        
        # Map to make/model
        make, model = self._map_class_to_make_model(predicted_idx)
        
        return make, model
        
    except Exception as e:
        logger.error(f"ONNX classification error: {e}")
        return 'unknown', 'unknown'
```

## Recommended Approach

For **production deployment**, I recommend:

1. **Start with Google Cloud Vision API** for quick results
2. **Collect your own data** of vehicles in your specific use case
3. **Fine-tune a model** on your data using transfer learning
4. **Deploy locally** for privacy and cost efficiency

## Performance Comparison

| Method | Accuracy | Speed | Cost | Setup Difficulty |
|--------|----------|-------|------|------------------|
| Current (ImageNet) | Low | Fast | Free | Easy |
| Stanford Cars Fine-tuned | High | Medium | Free | Medium |
| Google Cloud Vision | High | Medium | $$$$ | Easy |
| AWS Rekognition | Medium | Fast | $$$ | Easy |
| Custom Trained | Very High | Medium | Free* | Hard |

*Initial training costs (GPU time)

## Resources

- Stanford Cars Dataset: http://ai.stanford.edu/~jkrause/cars/car_dataset.html
- CompCars Dataset: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
- PyTorch Image Models: https://github.com/rwightman/pytorch-image-models
- Google Cloud Vision: https://cloud.google.com/vision/docs
- AWS Rekognition: https://aws.amazon.com/rekognition/

## Questions?

If you need help implementing any of these options, please refer to the respective documentation or open an issue in the repository.
