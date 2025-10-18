"""Vehicle recognition for color, make, and model detection."""

import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
from .logger import logger

try:
    import torch
    import torchvision
    from torchvision import transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Vehicle recognition will be disabled.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available. Vehicle detection will use fallback method.")


class VehicleRecognizer:
    """Recognize vehicle attributes (color, make, model) using deep learning."""
    
    # Common vehicle colors
    COLORS = [
        'black', 'white', 'silver', 'gray', 'red', 'blue', 
        'green', 'yellow', 'orange', 'brown', 'gold', 'beige',
        'cyan', 'purple', 'pink'
    ]
    
    # Common vehicle makes (expandable)
    MAKES = [
        'toyota', 'ford', 'honda', 'chevrolet', 'nissan', 'jeep',
        'ram', 'gmc', 'hyundai', 'subaru', 'kia', 'mazda',
        'dodge', 'volkswagen', 'bmw', 'mercedes', 'audi', 'lexus',
        'buick', 'cadillac', 'volvo', 'tesla', 'acura', 'infiniti'
    ]
    
    def __init__(self, config=None):
        """Initialize vehicle recognizer with models.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available! Install with: pip install torch torchvision")
            self.enabled = False
            return
        
        self.enabled = True
        self.yolo_model = None
        
        # Load YOLO for vehicle detection
        if YOLO_AVAILABLE:
            try:
                logger.info("Loading YOLOv9 vehicle detection model...")
                # Use YOLOv9t (tiny) for fast vehicle detection, matching ALPR's YOLOv9 architecture
                # YOLO can detect: car, truck, bus, motorcycle
                self.yolo_model = YOLO('yolov9t.pt')
                logger.info("YOLOv9 vehicle detection loaded")
            except Exception as e:
                logger.warning(f"Failed to load YOLOv9: {e}. Will use fallback detection.")
                self.yolo_model = None
        else:
            logger.warning("YOLOv9 (ultralytics) not available. Install with: pip install ultralytics")
            logger.info("Will use fallback vehicle detection (center-crop estimation)")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize classification model
        self.classifier_model = None
        self.class_labels = None
        
        try:
            # Try to load fine-tuned CompCars model
            model_path = 'models/compcars_resnet50.pth'
            if self._load_compcars_model(model_path):
                logger.info(f"Loaded CompCars fine-tuned model from {model_path}")
            else:
                logger.warning("CompCars model not found. Make/model detection will return 'unknown'")
                
                # Offer to download automatically
                if self._should_auto_download():
                    logger.info("Attempting to download pre-trained CompCars model...")
                    if self._auto_download_model():
                        logger.info("Model downloaded successfully! Reloading...")
                        if self._load_compcars_model(model_path):
                            logger.info("✓ CompCars model loaded after download")
                        else:
                            logger.warning("Failed to load downloaded model")
                    else:
                        self._show_manual_instructions()
                else:
                    self._show_manual_instructions()
        except Exception as e:
            logger.error(f"Failed to load vehicle classification model: {e}")
            self.classifier_model = None
    
    def _should_auto_download(self) -> bool:
        """
        Check if we should attempt auto-download of model.
        
        Returns:
            True if auto-download should be attempted
        """
        # Check config setting if available
        if self.config and hasattr(self.config, '_config'):
            auto_download = self.config._config.get('alpr', {}).get('auto_download_vehicle_model', True)
            return auto_download
        
        # Default: allow auto-download
        return True
    
    def _show_manual_instructions(self):
        """Show instructions for manual model setup."""
        logger.info("To enable make/model detection:")
        logger.info("  1. Run: python -m src.download_model")
        logger.info("  2. Or manually place model at: models/compcars_resnet50.pth")
        logger.info("  3. Place class labels at: models/compcars_labels.txt")
        logger.info("  See COMPCARS_SETUP.md for details")
    
    def _auto_download_model(self) -> bool:
        """
        Automatically download pre-trained CompCars model.
        
        Returns:
            True if download successful, False otherwise
        """
        try:
            from pathlib import Path
            import urllib.request
            import json
            
            model_dir = Path('models')
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to download from a pre-configured source
            # Option 1: GitHub Release
            github_config = {
                'owner': 'your-username',  # Replace with actual repo
                'repo': 'compcars-models',
                'release': 'latest',
                'weights_file': 'compcars_resnet50.pth',
                'labels_file': 'compcars_labels.txt'
            }
            
            # For now, we'll use a placeholder approach
            # In production, replace with actual model hosting
            logger.warning("Auto-download not configured yet")
            logger.info("Please use one of these methods:")
            logger.info("  1. Run: python -m src.download_model --method github")
            logger.info("  2. Download manually from your model repository")
            logger.info("  3. Train your own model using COMPCARS_SETUP.md")
            
            return False
            
        except Exception as e:
            logger.error(f"Auto-download failed: {e}")
            return False

    def _load_compcars_model(self, model_path: str) -> bool:
        """
        Load a fine-tuned CompCars classification model.
        Automatically detects and prefers CompCars Surveillance model if available.
        
        Args:
            model_path: Path to the saved model weights
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from pathlib import Path
            
            # Priority order: Surveillance model > Standard model
            models_to_try = [
                {
                    'model': Path('models/compcars_surveillance.pth'),
                    'labels': Path('models/compcars_surveillance_labels.txt'),
                    'name': 'CompCars Surveillance (1,716 classes)'
                },
                {
                    'model': Path(model_path),
                    'labels': Path('models/compcars_labels.txt'),
                    'name': 'CompCars Standard (196 classes)'
                }
            ]
            
            model_file = None
            labels_file = None
            model_name = None
            
            # Try each model in priority order
            for model_config in models_to_try:
                if model_config['model'].exists() and model_config['labels'].exists():
                    model_file = model_config['model']
                    labels_file = model_config['labels']
                    model_name = model_config['name']
                    logger.info(f"Found {model_name}")
                    break
            
            if model_file is None or labels_file is None:
                logger.debug(f"No CompCars model found")
                return False
            
            # Load class labels
            with open(labels_file, 'r', encoding='utf-8') as f:
                self.class_labels = [line.strip() for line in f.readlines()]
            
            num_classes = len(self.class_labels)
            logger.info(f"Loading {num_classes} vehicle classes from {model_name}")
            
            # Create model with correct number of output classes
            self.classifier_model = torchvision.models.resnet50(weights=None)
            
            # Modify the final layer to match CompCars classes
            num_features = self.classifier_model.fc.in_features
            self.classifier_model.fc = torch.nn.Linear(num_features, num_classes)
            
            # Load trained weights
            checkpoint = torch.load(model_file, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.classifier_model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.classifier_model.load_state_dict(checkpoint)
            else:
                self.classifier_model.load_state_dict(checkpoint)
            
            self.classifier_model = self.classifier_model.to(self.device)
            self.classifier_model.eval()
            
            logger.info(f"CompCars model loaded successfully with {num_classes} classes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CompCars model: {e}")
            self.classifier_model = None
            self.class_labels = None
            return False

    def detect_vehicles(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect all vehicles in the image and return bounding boxes.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of tuples (x1, y1, x2, y2, confidence) for each detected vehicle
        """
        vehicles = []
        
        # Use YOLOv9 for vehicle detection
        if self.yolo_model is not None:
            try:
                # Run YOLO detection
                results = self.yolo_model(image, verbose=False)
                
                # Vehicle class IDs in COCO dataset:
                # 2: car, 3: motorcycle, 5: bus, 7: truck
                vehicle_classes = {2, 3, 5, 7}
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls in vehicle_classes:
                            conf = float(box.conf[0])
                            # Use configurable confidence threshold
                            min_conf = self.config.vehicle_min_confidence if self.config else 0.3
                            if conf > min_conf:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                vehicles.append((x1, y1, x2, y2, conf))
                
                logger.debug(f"YOLOv9 detected {len(vehicles)} vehicles")
                return vehicles
                
            except Exception as e:
                logger.error(f"Error in YOLOv9 detection: {e}")
        
        # Fallback: Use simple center-crop estimation
        logger.debug("YOLOv9 not available, using fallback vehicle detection")
        return self._detect_vehicle_fallback(image)
    
    def _detect_vehicle_fallback(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Fallback vehicle detection using simple CV when YOLO is not available.
        Assumes vehicle occupies center portion of frame.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List with single bounding box for estimated vehicle region
        """
        height, width = image.shape[:2]
        
        # Assume vehicle is in the center 60% of the frame
        # This is reasonable for camera-triggered events
        margin_x = int(width * 0.2)
        margin_y = int(height * 0.2)
        
        x1 = margin_x
        y1 = margin_y
        x2 = width - margin_x
        y2 = height - margin_y
        
        logger.debug("Using fallback vehicle detection (center crop)")
        return [(x1, y1, x2, y2, 0.5)]
    
    def get_primary_vehicle_crop(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Get the crop of the primary (largest/most confident) vehicle.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Tuple of (cropped_image, bbox) where bbox is (x1, y1, x2, y2)
            Returns (None, None) if no vehicle detected
        """
        vehicles = self.detect_vehicles(image)
        
        if not vehicles:
            logger.warning("No vehicles detected in image")
            return None, None
        
        # Select the primary vehicle (largest area * confidence)
        primary_vehicle = max(vehicles, key=lambda v: (v[2] - v[0]) * (v[3] - v[1]) * v[4])
        
        x1, y1, x2, y2, conf = primary_vehicle
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Crop the vehicle
        vehicle_crop = image[y1:y2, x1:x2].copy()
        
        logger.debug(f"Primary vehicle crop: {x2-x1}x{y2-y1} (conf: {conf:.2f})")
        
        return vehicle_crop, (x1, y1, x2, y2)
    
    def get_all_vehicle_crops(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Get crops of all detected vehicles.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of tuples (cropped_image, bbox, confidence) where bbox is (x1, y1, x2, y2)
        """
        vehicles = self.detect_vehicles(image)
        
        if not vehicles:
            logger.warning("No vehicles detected in image")
            return []
        
        vehicle_crops = []
        height, width = image.shape[:2]
        
        for x1, y1, x2, y2, conf in vehicles:
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Crop the vehicle
            vehicle_crop = image[y1:y2, x1:x2].copy()
            
            vehicle_crops.append((vehicle_crop, (x1, y1, x2, y2), conf))
            logger.debug(f"Vehicle crop {len(vehicle_crops)}: {x2-x1}x{y2-y1} (conf: {conf:.2f})")
        
        return vehicle_crops
    
    def recognize_vehicle(self, image: np.ndarray) -> Dict[str, str]:
        """
        Recognize vehicle attributes from image with enhanced confidence handling.
        
        Args:
            image: OpenCV image (BGR format) of the vehicle
            
        Returns:
            Dict with color, make, model, and confidence information
        """
        if not self.enabled:
            return {
                'color': 'unknown',
                'make': 'unknown',
                'model': 'unknown',
                'confidence': 0.0
            }
        
        try:
            # Detect color using enhanced HSV analysis with lighting normalization
            color = self._detect_color_enhanced(image)
            
            # Use deep learning for make/model recognition with confidence
            make, model, confidence = self._classify_vehicle_with_confidence(image)
            
            # Only return make/model if confidence is above threshold
            min_confidence = getattr(self.config, 'vehicle_min_confidence', 0.4)
            if confidence < min_confidence:
                logger.debug(f"Vehicle classification confidence {confidence:.3f} below threshold {min_confidence}")
                make, model = 'unknown', 'unknown'
                confidence = 0.0
            
            return {
                'color': color,
                'make': make,
                'model': model,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error recognizing vehicle: {e}")
            return {
                'color': 'unknown',
                'make': 'unknown',
                'model': 'unknown',
                'confidence': 0.0
            }
    
    def _detect_color(self, image: np.ndarray) -> str:
        """
        Detect dominant vehicle color using improved HSV analysis with histogram-based approach.
        
        Args:
            image: OpenCV image (BGR format) - should be cropped to vehicle
            
        Returns:
            Color name as string
        """
        # Use enhanced color detection if available, fallback to basic
        try:
            return self._detect_color_enhanced(image)
        except Exception as e:
            logger.warning(f"Enhanced color detection failed, using basic: {e}")
            return self._detect_color_basic(image)
    
    def _detect_color_enhanced(self, image: np.ndarray) -> str:
        """
        Enhanced color detection with lighting normalization and better sampling.
        
        Args:
            image: OpenCV image (BGR format) - should be cropped to vehicle
            
        Returns:
            Color name as string
        """
        try:
            # Normalize lighting using LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_normalized = clahe.apply(l)
            
            # Merge back and convert to BGR
            normalized_lab = cv2.merge([l_normalized, a, b])
            normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
            
            # Convert normalized image to HSV
            hsv = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2HSV)
            
            # Get image dimensions
            height, width = hsv.shape[:2]
            
            # Sample from multiple body regions for better accuracy
            regions = [
                # Center body (primary)
                hsv[int(height * 0.30):int(height * 0.70), int(width * 0.25):int(width * 0.75)],
                # Upper body (hood area)
                hsv[int(height * 0.20):int(height * 0.45), int(width * 0.30):int(width * 0.70)],
                # Side panels
                hsv[int(height * 0.35):int(height * 0.65), int(width * 0.15):int(width * 0.50)]
            ]
            
            # Collect color votes from each region
            color_votes = []
            for region in regions:
                if region.size > 0:
                    color = self._detect_color_from_region(region)
                    if color != 'unknown':
                        color_votes.append(color)
            
            # Return most common color (voting system)
            if color_votes:
                from collections import Counter
                color_counts = Counter(color_votes)
                most_common = color_counts.most_common(1)[0]
                # Require at least 2/3 agreement for confidence
                if most_common[1] >= len(color_votes) * 0.5:
                    return most_common[0]
            
            # Fallback to center region only
            body_region = hsv[
                int(height * 0.30):int(height * 0.70),
                int(width * 0.25):int(width * 0.75)
            ]
            return self._detect_color_from_region(body_region)
            
        except Exception as e:
            logger.error(f"Error in enhanced color detection: {e}")
            return 'unknown'
    
    def _detect_color_basic(self, image: np.ndarray) -> str:
        """
        Basic color detection (original implementation as fallback).
        
        Args:
            image: OpenCV image (BGR format) - should be cropped to vehicle
            
        Returns:
            Color name as string
        """
        try:
            # Convert to HSV and LAB color spaces for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Get image dimensions
            height, width = hsv.shape[:2]
            
            # Sample only from car body area (avoiding windows, wheels, bumpers)
            # Focus on the center body panels where paint color is most visible
            # This is the "sweet spot" for most vehicles:
            # - Vertical: 30-70% (avoids roof/windows and lower wheels/bumpers)
            # - Horizontal: 25-75% (center body, avoiding edges)
            
            body_region = hsv[
                int(height * 0.30):int(height * 0.70),  # Middle 40% vertically
                int(width * 0.25):int(width * 0.75)     # Center 50% horizontally
            ]
            
            # Use only the body region for color detection
            combined_region = body_region
            
            # Split HSV channels
            h, s, v = cv2.split(combined_region)
            
            # Filter out very dark (shadows) and very bright (reflections) pixels
            # Relaxed thresholds for better compatibility with various lighting
            mask = (v > 20) & (v < 235)
            
            # Calculate statistics on filtered pixels
            valid_pixels = np.sum(mask)
            if valid_pixels < 50:  # Reduced threshold for smaller crops
                logger.warning(f"Insufficient valid pixels for color detection ({valid_pixels} pixels)")
                # Try without filtering if we have too few pixels
                if len(h) > 100:
                    mask = np.ones_like(v, dtype=bool)
                    logger.debug("Using all pixels (no filtering)")
                else:
                    return 'unknown'
            
            filtered_h = h[mask]
            filtered_s = s[mask]
            filtered_v = v[mask]
            
            # Calculate median values (more robust than mean)
            median_saturation = np.median(filtered_s)
            median_value = np.median(filtered_v)
            median_hue = np.median(filtered_h)
            
            logger.debug(f"Color detection - H:{median_hue:.1f} S:{median_saturation:.1f} V:{median_value:.1f} (pixels:{valid_pixels})")
            
            # Detect achromatic colors (black, white, gray, silver) first
            # Adjusted thresholds for better accuracy
            if median_saturation < 45:  # Low saturation = achromatic
                if median_value < 60:
                    return 'black'
                elif median_value > 200:
                    return 'white'
                elif median_value > 140:
                    return 'silver'
                else:
                    return 'gray'
            
            # For chromatic colors, use histogram-based approach
            # Create hue histogram (ignore low saturation pixels)
            chromatic_mask = (s > 40) & (v > 30) & (v < 220)
            if np.sum(chromatic_mask) < 50:
                # Not enough chromatic pixels, fall back to achromatic
                if median_value < 80:
                    return 'black'
                elif median_value > 160:
                    return 'white'
                else:
                    return 'gray'
            
            chromatic_hue = h[chromatic_mask]
            
            # Calculate hue histogram with 180 bins (0-180 in OpenCV HSV)
            hist = cv2.calcHist([chromatic_hue], [0], None, [180], [0, 180])
            hist = hist.flatten()
            
            # Smooth histogram to reduce noise
            from scipy.ndimage import gaussian_filter1d
            hist_smooth = gaussian_filter1d(hist, sigma=3)
            
            # Find dominant hue (peak in histogram)
            dominant_hue = np.argmax(hist_smooth)
            
            # Map hue to color names with improved ranges
            # Red: 0-10 and 160-180 (wraps around)
            # Orange: 10-20
            # Yellow: 20-35
            # Green: 35-80
            # Cyan: 80-95
            # Blue: 95-135
            # Purple/Magenta: 135-160
            
            if dominant_hue < 10 or dominant_hue >= 160:
                # Check if it's more brown or red based on saturation and value
                if median_saturation < 100 and median_value < 120:
                    return 'brown'
                return 'red'
            elif dominant_hue < 20:
                # Orange or brown
                if median_value < 100:
                    return 'brown'
                return 'orange'
            elif dominant_hue < 35:
                # Yellow or gold
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
                # Purple, magenta, or pink
                if median_value > 150:
                    return 'pink'
                return 'purple'
            else:
                return 'red'
                
        except Exception as e:
            logger.error(f"Error detecting color: {e}")
            return 'unknown'
    
    def _detect_color_from_region(self, region: np.ndarray) -> str:
        """
        Detect color from a specific HSV region.
        
        Args:
            region: HSV image region
            
        Returns:
            Color name as string
        """
        try:
            if region.size == 0:
                return 'unknown'
            
            # Split HSV channels
            h, s, v = cv2.split(region)
            
            # Filter out very dark and very bright pixels
            mask = (v > 20) & (v < 235)
            
            if np.sum(mask) < 50:
                return 'unknown'
            
            filtered_h = h[mask]
            filtered_s = s[mask]
            filtered_v = v[mask]
            
            # Calculate median values
            median_saturation = np.median(filtered_s)
            median_value = np.median(filtered_v)
            median_hue = np.median(filtered_h)
            
            # Detect achromatic colors
            if median_saturation < 45:
                if median_value < 60:
                    return 'black'
                elif median_value > 200:
                    return 'white'
                elif median_value > 140:
                    return 'silver'
                else:
                    return 'gray'
            
            # For chromatic colors
            chromatic_mask = (s > 40) & (v > 30) & (v < 220)
            if np.sum(chromatic_mask) < 50:
                return 'unknown'
            
            chromatic_h = h[chromatic_mask]
            
            # Determine dominant hue range
            median_hue = np.median(chromatic_h)
            
            # Hue ranges (0-180 in OpenCV)
            if median_hue < 10 or median_hue > 170:
                return 'red'
            elif median_hue < 25:
                return 'orange'
            elif median_hue < 35:
                return 'yellow'
            elif median_hue < 85:
                return 'green'
            elif median_hue < 130:
                return 'blue'
            else:
                return 'purple'
                
        except Exception as e:
            logger.error(f"Error in region color detection: {e}")
            return 'unknown'
    
    def _classify_vehicle(self, image: np.ndarray) -> tuple[str, str]:
        """
        Classify vehicle make and model using CompCars fine-tuned model.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Tuple of (make, model)
        """
        # Check if CompCars model is loaded
        if self.classifier_model is None or self.class_labels is None:
            return 'unknown', 'unknown'
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Preprocess
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.classifier_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get predicted class
            class_idx = predicted_idx.item()
            confidence_score = confidence.item()
            
            if class_idx < len(self.class_labels):
                predicted_class = self.class_labels[class_idx]
                
                # Parse class name (CompCars format: "make_model_year" or "make model")
                # Example: "Audi_A4_Sedan_2012" or "Toyota Camry"
                parts = predicted_class.replace('_', ' ').split()
                
                if len(parts) >= 2:
                    make = parts[0].lower()
                    model = ' '.join(parts[1:]).lower()
                    # Remove year if present (typically 4 digits at the end)
                    if model.split()[-1].isdigit() and len(model.split()[-1]) == 4:
                        model = ' '.join(model.split()[:-1])
                    
                    logger.debug(f"Classified as {make} {model} (confidence: {confidence_score:.2f})")
                    return make, model
                else:
                    logger.debug(f"Could not parse class name: {predicted_class}")
                    return 'unknown', 'unknown'
            else:
                logger.error(f"Predicted index {class_idx} out of range")
                return 'unknown', 'unknown'
            
        except Exception as e:
            logger.error(f"Error classifying vehicle: {e}")
            return 'unknown', 'unknown'
    
    def _classify_vehicle_with_confidence(self, image: np.ndarray) -> tuple[str, str, float]:
        """
        Classify vehicle make and model with confidence score.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Tuple of (make, model, confidence)
        """
        # Check if CompCars model is loaded
        if self.classifier_model is None or self.class_labels is None:
            return 'unknown', 'unknown', 0.0
        
        try:
            # Enhance image quality for better classification
            enhanced_image = self._enhance_vehicle_crop(image)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Preprocess
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.classifier_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top-k predictions for better confidence calibration
                top_k = min(5, len(self.class_labels))
                top_probs, top_indices = torch.topk(probabilities[0], top_k)
                
                # Use the top prediction
                confidence = top_probs[0].item()
                predicted_idx = top_indices[0].item()
            
            # Get predicted class
            if predicted_idx < len(self.class_labels):
                predicted_class = self.class_labels[predicted_idx]
                
                # Parse class name (CompCars format: "make_model_year" or "make model")
                # Example: "Audi_A4_Sedan_2012" or "Toyota Camry"
                parts = predicted_class.replace('_', ' ').split()
                
                if len(parts) >= 2:
                    make = parts[0].lower()
                    model = ' '.join(parts[1:]).lower()
                    # Remove year if present (typically 4 digits at the end)
                    if model.split()[-1].isdigit() and len(model.split()[-1]) == 4:
                        model = ' '.join(model.split()[:-1])
                    
                    # Log top-k predictions for debugging
                    if top_k > 1:
                        top_predictions = []
                        for i in range(min(3, top_k)):
                            idx = top_indices[i].item()
                            prob = top_probs[i].item()
                            if idx < len(self.class_labels):
                                class_name = self.class_labels[idx]
                                top_predictions.append(f"{class_name}({prob:.2f})")
                        logger.debug(f"Top predictions: {', '.join(top_predictions)}")
                    
                    logger.info(f"Classified as {make} {model} (confidence: {confidence:.3f})")
                    return make, model, confidence
                else:
                    logger.debug(f"Could not parse class name: {predicted_class}")
                    return 'unknown', 'unknown', 0.0
            else:
                logger.error(f"Predicted index {predicted_idx} out of range")
                return 'unknown', 'unknown', 0.0
            
        except Exception as e:
            logger.error(f"Error classifying vehicle with confidence: {e}")
            return 'unknown', 'unknown', 0.0
    
    def _enhance_vehicle_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance vehicle crop image for better classification.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge and convert back to BGR
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]]) / 9.0
            sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)
            
            # Blend original and sharpened (50/50)
            result = cv2.addWeighted(enhanced_bgr, 0.5, sharpened, 0.5, 0)
            
            return result
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def recognize_from_bytes(self, image_bytes: bytes) -> Dict[str, str]:
        """
        Recognize vehicle attributes from image bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dict with vehicle attributes
        """
        try:
            # Decode image
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.warning("Failed to decode image for vehicle recognition")
                return {
                    'color': 'unknown',
                    'make': 'unknown',
                    'model': 'unknown',
                    'confidence': 0.0
                }
            
            return self.recognize_vehicle(image)
            
        except Exception as e:
            logger.error(f"Error recognizing vehicle from bytes: {e}")
            return {
                'color': 'unknown',
                'make': 'unknown',
                'model': 'unknown',
                'confidence': 0.0
            }
