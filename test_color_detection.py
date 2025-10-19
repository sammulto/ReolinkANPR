"""Test vehicle color detection on sample images."""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.vehicle_recognizer import VehicleRecognizer


def test_color_detection(image_path: str):
    """Test color detection on a specific image."""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return
    
    print(f"\n[INFO] Testing image: {image_path}")
    print(f"[INFO] Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize recognizer
    recognizer = VehicleRecognizer()
    
    # Get vehicle crops (or use full image if detection fails)
    crops = recognizer.get_all_vehicle_crops(image)
    
    if not crops:
        print("[WARNING] No vehicles detected, using full image")
        crops = [(image, (0, 0, image.shape[1], image.shape[0]), 1.0)]
    
    # Test each crop
    for i, (crop, bbox, conf) in enumerate(crops):
        print(f"\n[INFO] Vehicle {i+1}:")
        print(f"  Crop size: {crop.shape[1]}x{crop.shape[0]}")
        print(f"  Confidence: {conf:.2f}")
        
        # Detect color with detailed debug info
        color = detect_color_debug(crop)
        print(f"  Detected color: {color}")
        
        # Also get make/model
        result = recognizer.recognize_vehicle(crop)
        print(f"  Make: {result['make']}")
        print(f"  Model: {result['model']}")


def detect_color_debug(image: np.ndarray) -> str:
    """
    Detect color with detailed debug output.
    """
    try:
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get image dimensions
        height, width = hsv.shape[:2]
        
        # Sample regions
        regions = []
        
        # Top region (hood area)
        top_region = hsv[int(height * 0.2):int(height * 0.4), int(width * 0.3):int(width * 0.7)]
        regions.append(top_region)
        
        # Middle region (door area)
        mid_region = hsv[int(height * 0.4):int(height * 0.7), int(width * 0.3):int(width * 0.7)]
        regions.append(mid_region)
        
        # Combine all regions
        combined_region = np.vstack(regions)
        
        # Split HSV channels
        h, s, v = cv2.split(combined_region)
        
        # Filter out shadows and reflections
        mask = (v > 30) & (v < 220)
        
        valid_pixels = np.sum(mask)
        print(f"  Valid pixels (after filtering): {valid_pixels}")
        
        if valid_pixels < 100:
            print("  [WARNING] Not enough valid pixels")
            return 'unknown'
        
        filtered_h = h[mask]
        filtered_s = s[mask]
        filtered_v = v[mask]
        
        # Calculate statistics
        median_saturation = np.median(filtered_s)
        median_value = np.median(filtered_v)
        median_hue = np.median(filtered_h)
        
        print(f"  HSV medians: H={median_hue:.1f}, S={median_saturation:.1f}, V={median_value:.1f}")
        
        # Detect achromatic colors first
        if median_saturation < 40:
            print(f"  Low saturation ({median_saturation:.1f}) - achromatic color")
            if median_value < 50:
                return 'black'
            elif median_value > 180:
                return 'white'
            elif median_value > 130:
                return 'silver'
            else:
                return 'gray'
        
        # For chromatic colors
        chromatic_mask = (s > 40) & (v > 30) & (v < 220)
        chromatic_pixels = np.sum(chromatic_mask)
        print(f"  Chromatic pixels: {chromatic_pixels}")
        
        if chromatic_pixels < 50:
            print("  Not enough chromatic pixels, using achromatic fallback")
            if median_value < 80:
                return 'black'
            elif median_value > 160:
                return 'white'
            else:
                return 'gray'
        
        chromatic_hue = h[chromatic_mask]
        
        # Calculate hue histogram
        hist = cv2.calcHist([chromatic_hue], [0], None, [180], [0, 180])
        hist = hist.flatten()
        
        # Smooth histogram
        from scipy.ndimage import gaussian_filter1d
        hist_smooth = gaussian_filter1d(hist, sigma=3)
        
        # Find dominant hue
        dominant_hue = np.argmax(hist_smooth)
        print(f"  Dominant hue: {dominant_hue}")
        
        # Map to color
        if dominant_hue < 10 or dominant_hue >= 160:
            if median_saturation < 100 and median_value < 120:
                return 'brown'
            return 'red'
        elif dominant_hue < 20:
            if median_value < 100:
                return 'brown'
            return 'orange'
        elif dominant_hue < 35:
            if median_saturation < 80:
                return 'gold'
            return 'yellow'
        elif dominant_hue < 80:
            return 'green'
        elif dominant_hue < 95:
            return 'cyan'
        elif dominant_hue < 135:
            return 'blue'
        elif dominant_hue < 160:
            if median_value > 150:
                return 'pink'
            return 'purple'
        else:
            return 'red'
            
    except Exception as e:
        print(f"  [ERROR] Color detection failed: {e}")
        import traceback
        traceback.print_exc()
        return 'unknown'


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_color_detection.py <image_path>")
        print("\nExample:")
        print("  python test_color_detection.py data/images/detection_20250117_120000.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_color_detection(image_path)


if __name__ == '__main__':
    main()
