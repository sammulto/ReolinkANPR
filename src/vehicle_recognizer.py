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
        
        try:
            # Use ResNet50 as feature extractor
            logger.info("Loading vehicle recognition model...")
            self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Vehicle recognition model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load vehicle recognition model: {e}")
            self.enabled = False
    
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
                            if conf > 0.3:  # Confidence threshold
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
        Recognize vehicle attributes from image.
        
        Args:
            image: OpenCV image (BGR format) of the vehicle
            
        Returns:
            Dict with color, make, and model information
        """
        if not self.enabled:
            return {
                'color': 'unknown',
                'make': 'unknown',
                'model': 'unknown',
                'confidence': 0.0
            }
        
        try:
            # Detect color using traditional CV (more reliable than deep learning for color)
            color = self._detect_color(image)
            
            # Use deep learning for make/model recognition
            # Note: This is a simplified implementation. For production, you would use
            # a model specifically trained on vehicle make/model classification
            make, model = self._classify_vehicle(image)
            
            return {
                'color': color,
                'make': make,
                'model': model,
                'confidence': 0.8  # Placeholder - would come from actual model
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
        try:
            # Convert to HSV and LAB color spaces for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Get image dimensions
            height, width = hsv.shape[:2]
            
            # Sample multiple regions to avoid shadows/reflections
            # Top center (hood), middle center (door/body), lower center (body)
            regions = []
            
            # Top region (hood area) - 20-40% height, 30-70% width
            top_region = hsv[int(height * 0.2):int(height * 0.4), int(width * 0.3):int(width * 0.7)]
            regions.append(top_region)
            
            # Middle region (door area) - 40-70% height, 30-70% width
            mid_region = hsv[int(height * 0.4):int(height * 0.7), int(width * 0.3):int(width * 0.7)]
            regions.append(mid_region)
            
            # Combine all regions
            combined_region = np.vstack(regions)
            
            # Split HSV channels
            h, s, v = cv2.split(combined_region)
            
            # Filter out very dark (shadows) and very bright (reflections) pixels
            mask = (v > 30) & (v < 220)
            
            # Calculate statistics on filtered pixels
            if np.sum(mask) < 100:  # Not enough valid pixels
                logger.warning("Insufficient valid pixels for color detection")
                return 'unknown'
            
            filtered_h = h[mask]
            filtered_s = s[mask]
            filtered_v = v[mask]
            
            # Calculate median values (more robust than mean)
            median_saturation = np.median(filtered_s)
            median_value = np.median(filtered_v)
            median_hue = np.median(filtered_h)
            
            # Detect achromatic colors (black, white, gray, silver) first
            if median_saturation < 40:  # Low saturation = achromatic
                if median_value < 50:
                    return 'black'
                elif median_value > 180:
                    return 'white'
                elif median_value > 130:
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
    
    def _classify_vehicle(self, image: np.ndarray) -> tuple[str, str]:
        """
        Classify vehicle make and model using deep learning.
        
        Note: This is a simplified implementation using ImageNet features.
        For production, use a model specifically trained on vehicle classification
        like Stanford Cars dataset or CompCars dataset.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Tuple of (make, model)
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Preprocess
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features (we're not actually classifying make/model here,
            # as ResNet is trained on ImageNet, not vehicles specifically)
            with torch.no_grad():
                features = self.model(input_tensor)
            
            # For now, return placeholder values
            # In production, you would:
            # 1. Fine-tune a model on vehicle datasets (Stanford Cars, CompCars, etc.)
            # 2. Use a pre-trained vehicle classification model
            # 3. Or use an API service like Google Vision API, AWS Rekognition
            
            # Simplified heuristic based on image features
            # This is just a placeholder - not actual make/model detection
            feature_sum = features.sum().item()
            
            # Pseudo-random selection based on feature sum (for demonstration)
            # In production, replace with actual classification
            make_idx = int(abs(feature_sum) % len(self.MAKES))
            make = self.MAKES[make_idx]
            
            # Model detection would require specific training data
            model = 'sedan'  # Placeholder
            
            # TODO: Integrate actual vehicle classification model
            # Recommended approaches:
            # 1. Use MMClassification with vehicle-specific models
            # 2. Fine-tune EfficientNet on Stanford Cars dataset
            # 3. Use commercial APIs (Google Vision, AWS Rekognition)
            
            logger.debug(f"Classified vehicle as {make} {model} (placeholder)")
            return make, model
            
        except Exception as e:
            logger.error(f"Error classifying vehicle: {e}")
            return 'unknown', 'unknown'
    
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
