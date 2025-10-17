"""ALPR processing using FastALPR."""

import cv2
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .logger import logger
from .vehicle_recognizer import VehicleRecognizer

try:
    from fast_alpr import ALPR
    FAST_ALPR_AVAILABLE = True
except ImportError:
    FAST_ALPR_AVAILABLE = False
    logger.error("fast-alpr not available! Install with: pip install fast-alpr")


class ALPRProcessor:
    """Process images for license plate recognition using FastALPR."""

    def __init__(self, config):
        self.config = config

        if not FAST_ALPR_AVAILABLE:
            raise ImportError("fast-alpr library not available")

        # Initialize FastALPR with configured models
        self.alpr = ALPR(
            detector_model=config.detector_model,
            ocr_model=config.ocr_model,
            detector_conf_thresh=0.1  # Detection confidence (lower than recognition threshold)
        )

        # Initialize vehicle recognizer
        try:
            if config.vehicle_recognition_enabled:
                self.vehicle_recognizer = VehicleRecognizer(config)
                if self.vehicle_recognizer.enabled:
                    logger.info("Vehicle recognition enabled")
                else:
                    logger.warning("Vehicle recognition disabled - missing dependencies")
            else:
                logger.info("Vehicle recognition disabled in config")
                self.vehicle_recognizer = None
        except Exception as e:
            logger.warning(f"Could not initialize vehicle recognizer: {e}")
            self.vehicle_recognizer = None

        logger.info(f"FastALPR initialized with {config.detector_model}")

    def process_frames(
        self,
        frame_bytes_list: List[bytes],
        save_dir: Path
    ) -> Optional[Dict]:
        """
        Process multiple frames and return the best result.

        Args:
            frame_bytes_list: List of frame bytes to process
            save_dir: Directory to save images

        Returns:
            Dict with plate info and paths, or None if no plate found
        """
        if not frame_bytes_list:
            logger.warning("No frames provided for processing")
            return None

        logger.info(f"Processing {len(frame_bytes_list)} frames...")

        best_result = None
        best_confidence = 0.0
        best_image = None
        best_plate_crop = None

        for i, frame_bytes in enumerate(frame_bytes_list):
            try:
                # Decode image
                image_array = np.frombuffer(frame_bytes, np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if img is None:
                    logger.warning(f"Failed to decode frame {i+1}")
                    continue

                # Run detection-only first (lighter weight)
                detections = self.alpr.detector.predict(img)

                if not detections:
                    if (i+1) % 20 == 0:  # Log every 20th frame to reduce noise
                        logger.debug(f"Frame {i+1}: No plates detected")
                    continue

                # Run full ALPR (detection + OCR) on frames with detections
                results = self.alpr.predict(img)

                if not results:
                    logger.debug(f"Frame {i+1}: Detected {len(detections)} plates but OCR failed")
                    continue

                # Process each detection
                for result in results:
                    plate_text = result.ocr.text.upper().replace(' ', '')
                    ocr_confidence = result.ocr.confidence
                    detection_confidence = result.detection.confidence

                    # Check confidence threshold
                    if ocr_confidence < self.config.min_confidence:
                        logger.debug(f"Frame {i+1}: Low confidence {ocr_confidence:.3f} for {plate_text}")
                        continue

                    logger.info(f"Frame {i+1}: {plate_text} (confidence: {ocr_confidence:.3f})")

                    # Track best result - OCR priority, detection as tiebreaker
                    is_better = False
                    if best_result is None:
                        is_better = True
                    elif ocr_confidence > best_result.ocr.confidence:
                        # Higher OCR confidence always wins
                        is_better = True
                    elif ocr_confidence == best_result.ocr.confidence:
                        # Same OCR confidence - use detection as tiebreaker
                        if detection_confidence > best_result.detection.confidence:
                            is_better = True
                    
                    if is_better:
                        best_confidence = ocr_confidence
                        best_result = result
                        best_image = img.copy()

                        # Crop plate region
                        bbox = result.detection.bounding_box
                        x1, y1 = int(bbox.x1), int(bbox.y1)
                        x2, y2 = int(bbox.x2), int(bbox.y2)
                        best_plate_crop = img[y1:y2, x1:x2].copy()

            except Exception as e:
                logger.error(f"Error processing frame {i+1}: {e}")
                continue

        # If no plate detected, still try vehicle recognition
        if not best_result:
            logger.info("No valid plates detected")
            
            # Still perform vehicle recognition on the best quality frame
            if self.vehicle_recognizer and self.vehicle_recognizer.enabled and self.config.vehicle_only_detection_enabled:
                return self._process_vehicle_only(frame_bytes_list, save_dir)
            else:
                # Save debug frames if vehicle recognition is disabled
                logger.info("Saving debug frames")
                self._save_debug_frames(frame_bytes_list, save_dir)
                return None

        # Return best result if plate found
        if best_result and best_image is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save full image
            image_filename = f"{timestamp}_{best_result.ocr.text.upper().replace(' ', '')}.jpg"
            image_path = save_dir / "images" / image_filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), best_image)

            # Save plate crop
            crop_filename = f"{timestamp}_{best_result.ocr.text.upper().replace(' ', '')}_crop.jpg"
            crop_path = save_dir / "images" / crop_filename
            if best_plate_crop is not None:
                cv2.imwrite(str(crop_path), best_plate_crop)

            bbox = best_result.detection.bounding_box

            # Recognize vehicle attributes (color, make, model) for ALL detected vehicles
            vehicles_data = []
            
            if self.vehicle_recognizer and self.vehicle_recognizer.enabled:
                try:
                    logger.info("Detecting and recognizing all vehicles...")
                    
                    # Get all vehicle crops
                    vehicle_crops = self.vehicle_recognizer.get_all_vehicle_crops(best_image)
                    
                    if vehicle_crops:
                        for idx, (vehicle_crop, vehicle_bbox, conf) in enumerate(vehicle_crops, 1):
                            # Save vehicle crop
                            vehicle_crop_filename = f"{timestamp}_{best_result.ocr.text.upper().replace(' ', '')}_vehicle_{idx}.jpg"
                            vehicle_crop_path = save_dir / "images" / vehicle_crop_filename
                            cv2.imwrite(str(vehicle_crop_path), vehicle_crop)
                            logger.info(f"Saved vehicle crop {idx}: {vehicle_crop_filename}")
                            
                            # Recognize attributes from the cropped vehicle
                            vehicle_attrs = self.vehicle_recognizer.recognize_vehicle(vehicle_crop)
                            
                            vehicles_data.append({
                                'crop_path': f"images/{vehicle_crop_filename}",
                                'bbox': vehicle_bbox,
                                'detection_confidence': conf,
                                'color': vehicle_attrs['color'],
                                'make': vehicle_attrs['make'],
                                'model': vehicle_attrs['model'],
                                'confidence': vehicle_attrs['confidence']
                            })
                            
                            logger.info(f"Vehicle {idx}: {vehicle_attrs['color']} {vehicle_attrs['make']} {vehicle_attrs['model']}")
                    else:
                        # Fallback: no vehicles detected, use full image
                        logger.warning("No vehicles detected, processing full image")
                        vehicle_attrs = self.vehicle_recognizer.recognize_vehicle(best_image)
                        vehicles_data.append({
                            'crop_path': None,
                            'bbox': None,
                            'detection_confidence': 0.0,
                            'color': vehicle_attrs['color'],
                            'make': vehicle_attrs['make'],
                            'model': vehicle_attrs['model'],
                            'confidence': vehicle_attrs['confidence']
                        })
                    
                except Exception as e:
                    logger.error(f"Error during vehicle recognition: {e}")
                    # Add unknown vehicle data
                    vehicles_data.append({
                        'crop_path': None,
                        'bbox': None,
                        'detection_confidence': 0.0,
                        'color': 'unknown',
                        'make': 'unknown',
                        'model': 'unknown',
                        'confidence': 0.0
                    })

            # Use primary vehicle (first one) for backward compatibility
            primary_vehicle = vehicles_data[0] if vehicles_data else {
                'crop_path': None,
                'bbox': None,
                'detection_confidence': 0.0,
                'color': 'unknown',
                'make': 'unknown',
                'model': 'unknown',
                'confidence': 0.0
            }

            return {
                'plate_number': best_result.ocr.text.upper().replace(' ', ''),
                'confidence': best_result.ocr.confidence,
                'image_path': f"images/{image_filename}",
                'plate_crop_path': f"images/{crop_filename}",
                'vehicle_crop_path': primary_vehicle['crop_path'],
                'box_coordinates': {
                    'xmin': bbox.x1,
                    'ymin': bbox.y1,
                    'xmax': bbox.x2,
                    'ymax': bbox.y2
                },
                'frame_count': len(frame_bytes_list),
                'vehicle_color': primary_vehicle['color'],
                'vehicle_make': primary_vehicle['make'],
                'vehicle_model': primary_vehicle['model'],
                'vehicle_confidence': primary_vehicle['confidence'],
                'vehicles': vehicles_data  # List of all detected vehicles
            }

        logger.info("No valid plates detected in any frame")
        return None

    def _process_vehicle_only(self, frame_bytes_list: List[bytes], save_dir: Path) -> Optional[Dict]:
        """
        Process vehicle recognition when no plate is detected.
        
        Args:
            frame_bytes_list: List of frame bytes to process
            save_dir: Directory to save images
            
        Returns:
            Dict with vehicle info but no plate data, or None if processing fails
        """
        try:
            logger.info("No plate detected - performing vehicle-only recognition...")
            
            # Select the middle frame (usually best quality)
            if not frame_bytes_list:
                return None
            
            mid_idx = len(frame_bytes_list) // 2
            frame_bytes = frame_bytes_list[mid_idx]
            
            # Decode image
            image_array = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.warning("Failed to decode frame for vehicle-only processing")
                return None
            
            # Get all vehicle crops
            vehicle_crops = self.vehicle_recognizer.get_all_vehicle_crops(img)
            
            if not vehicle_crops:
                logger.info("No vehicles detected")
                self._save_debug_frames(frame_bytes_list, save_dir)
                return None
            
            # Process all vehicles
            vehicles_data = []
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for idx, (vehicle_crop, vehicle_bbox, conf) in enumerate(vehicle_crops, 1):
                # Recognize vehicle attributes from cropped image
                vehicle_attrs = self.vehicle_recognizer.recognize_vehicle(vehicle_crop)
                
                if vehicle_attrs['color'] == 'unknown' and vehicle_attrs['make'] == 'unknown':
                    continue  # Skip unknown vehicles
                
                # Save vehicle crop
                vehicle_desc = f"{vehicle_attrs['color']}_{vehicle_attrs['make']}"
                vehicle_crop_filename = f"{timestamp}_NO_PLATE_{vehicle_desc}_vehicle_{idx}.jpg"
                vehicle_crop_path = save_dir / "images" / vehicle_crop_filename
                vehicle_crop_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vehicle_crop_path), vehicle_crop)
                logger.info(f"Saved vehicle crop {idx}: {vehicle_crop_filename}")
                
                vehicles_data.append({
                    'crop_path': f"images/{vehicle_crop_filename}",
                    'bbox': vehicle_bbox,
                    'detection_confidence': conf,
                    'color': vehicle_attrs['color'],
                    'make': vehicle_attrs['make'],
                    'model': vehicle_attrs['model'],
                    'confidence': vehicle_attrs['confidence']
                })
                
                logger.info(f"Vehicle {idx} (no plate): {vehicle_attrs['color']} {vehicle_attrs['make']} {vehicle_attrs['model']}")
            
            if not vehicles_data:
                logger.info("No valid vehicle attributes detected")
                self._save_debug_frames(frame_bytes_list, save_dir)
                return None
            
            # Save full image with vehicle-only naming
            primary_vehicle = vehicles_data[0]
            vehicle_desc = f"{primary_vehicle['color']}_{primary_vehicle['make']}"
            image_filename = f"{timestamp}_NO_PLATE_{vehicle_desc}.jpg"
            image_path = save_dir / "images" / image_filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), img)
            
            return {
                'plate_number': 'NO_PLATE',
                'confidence': 0.0,
                'image_path': f"images/{image_filename}",
                'plate_crop_path': None,
                'vehicle_crop_path': primary_vehicle['crop_path'],
                'box_coordinates': {},
                'frame_count': len(frame_bytes_list),
                'vehicle_color': primary_vehicle['color'],
                'vehicle_make': primary_vehicle['make'],
                'vehicle_model': primary_vehicle['model'],
                'vehicle_confidence': primary_vehicle['confidence'],
                'vehicles': vehicles_data  # List of all detected vehicles
            }
            
        except Exception as e:
            logger.error(f"Error during vehicle-only processing: {e}")
            self._save_debug_frames(frame_bytes_list, save_dir)
            return None

    def _save_debug_frames(self, frame_bytes_list: List[bytes], save_dir: Path):
        """Save first and last frames for debugging when no plates detected."""
        try:
            debug_dir = save_dir / "debug_frames"
            debug_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save first frame
            if len(frame_bytes_list) > 0:
                image_array = np.frombuffer(frame_bytes_list[0], np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if img is not None:
                    first_path = debug_dir / f"{timestamp}_first.jpg"
                    cv2.imwrite(str(first_path), img)
                    logger.info(f"Saved first frame: {first_path}")

            # Save last frame
            if len(frame_bytes_list) > 1:
                image_array = np.frombuffer(frame_bytes_list[-1], np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if img is not None:
                    last_path = debug_dir / f"{timestamp}_last.jpg"
                    cv2.imwrite(str(last_path), img)
                    logger.info(f"Saved last frame: {last_path}")

            # Save middle frame
            if len(frame_bytes_list) > 2:
                mid_idx = len(frame_bytes_list) // 2
                image_array = np.frombuffer(frame_bytes_list[mid_idx], np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if img is not None:
                    mid_path = debug_dir / f"{timestamp}_middle.jpg"
                    cv2.imwrite(str(mid_path), img)
                    logger.info(f"Saved middle frame: {mid_path}")

        except Exception as e:
            logger.error(f"Failed to save debug frames: {e}")
