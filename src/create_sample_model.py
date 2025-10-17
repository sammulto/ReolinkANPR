"""Create a sample CompCars model for testing purposes."""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


def create_sample_model():
    """
    Create a sample CompCars model structure for testing.
    
    This creates a randomly initialized model - NOT for production use!
    For actual vehicle classification, you need to train on CompCars dataset.
    """
    print("Creating sample CompCars model structure...")
    
    # Common CompCars classes (subset for demo)
    sample_classes = [
        "Acura_TL_Sedan_2012",
        "Audi_A4_Sedan_2012",
        "Audi_A6_Sedan_2012",
        "BMW_3_Series_Sedan_2012",
        "BMW_5_Series_Sedan_2010",
        "Chevrolet_Camaro_Convertible_2012",
        "Chevrolet_Corvette_Convertible_2012",
        "Chevrolet_Silverado_1500_Hybrid_Crew_Cab_2012",
        "Dodge_Charger_Sedan_2012",
        "Dodge_Durango_SUV_2012",
        "Ford_F-150_Regular_Cab_2012",
        "Ford_F-450_Super_Duty_Crew_Cab_2012",
        "Ford_Focus_Sedan_2012",
        "Ford_Mustang_Convertible_2007",
        "GMC_Acadia_SUV_2012",
        "Honda_Accord_Sedan_2012",
        "Honda_Civic_Sedan_2012",
        "Hyundai_Elantra_Sedan_2007",
        "Hyundai_Genesis_Sedan_2012",
        "Hyundai_Sonata_Sedan_2012",
        "Infiniti_G_Coupe_IPL_2012",
        "Jeep_Grand_Cherokee_SUV_2012",
        "Jeep_Wrangler_SUV_2012",
        "Lexus_IS_250_Sedan_2012",
        "Lexus_RX_SUV_2012",
        "Mazda_3_Sedan_2012",
        "Mazda_CX-9_SUV_2012",
        "Mercedes-Benz_C-Class_Sedan_2012",
        "Mercedes-Benz_E-Class_Sedan_2012",
        "Mercedes-Benz_S-Class_Sedan_2012",
        "Nissan_Altima_Sedan_2012",
        "Nissan_Maxima_Sedan_2012",
        "Nissan_Titan_Crew_Cab_2012",
        "Ram_C-V_Cargo_Van_Minivan_2012",
        "Subaru_Impreza_Sedan_2012",
        "Subaru_Outback_Wagon_2012",
        "Tesla_Model_S_Sedan_2012",
        "Toyota_4Runner_SUV_2012",
        "Toyota_Camry_Sedan_2012",
        "Toyota_Corolla_Sedan_2012",
        "Toyota_Prius_Sedan_2012",
        "Toyota_RAV4_SUV_2012",
        "Toyota_Sequoia_SUV_2008",
        "Toyota_Tacoma_Double_Cab_2012",
        "Volkswagen_Beetle_Hatchback_2012",
        "Volkswagen_Golf_Hatchback_2012",
        "Volkswagen_Jetta_Sedan_2012",
        "Volvo_C30_Hatchback_2012",
        "Volvo_XC60_SUV_2012",
    ]
    
    num_classes = len(sample_classes)
    
    # Create ResNet50 with correct output size
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Initialize with random weights (for structure only)
    # In production, these would be trained weights
    print(f"Model structure created with {num_classes} classes")
    print("⚠️  WARNING: This is a randomly initialized model!")
    print("⚠️  For actual classification, train on CompCars dataset")
    
    return model, sample_classes


def save_sample_model(output_dir: str = 'models'):
    """
    Save a sample model structure.
    
    Args:
        output_dir: Directory to save model files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model, classes = create_sample_model()
    
    # Save model weights
    model_file = output_path / 'compcars_resnet50.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': len(classes),
        'note': 'Sample model for testing structure only - NOT trained!'
    }, model_file)
    print(f"✓ Saved model structure to: {model_file}")
    
    # Save class labels
    labels_file = output_path / 'compcars_labels.txt'
    with open(labels_file, 'w', encoding='utf-8') as f:
        for class_name in classes:
            f.write(f'{class_name}\n')
    print(f"✓ Saved class labels to: {labels_file}")
    
    print("\n" + "="*60)
    print("✅ Sample model structure created!")
    print("="*60)
    print("\n⚠️  IMPORTANT NOTES:")
    print("   - This model has RANDOM weights (not trained)")
    print("   - Predictions will be meaningless")
    print("   - Use this only to test the code structure")
    print("   - For real classification, train on CompCars dataset")
    print("   - See COMPCARS_SETUP.md for training instructions")
    print("\n" + "="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sample CompCars model structure')
    parser.add_argument('--output-dir', default='models', help='Output directory')
    
    args = parser.parse_args()
    
    print("CompCars Sample Model Generator")
    print("="*60)
    print("This creates a MODEL STRUCTURE ONLY (not trained)")
    print("="*60 + "\n")
    
    save_sample_model(args.output_dir)
