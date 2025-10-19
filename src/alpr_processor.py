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
    ) -> Optional[List[Dict]]:
        """
        Process multiple frames and return results for all vehicles.
        Each vehicle gets its own plate detection.

        Args:
            frame_bytes_list: List of frame bytes to process
            save_dir: Directory to save images

        Returns:
            List of dicts with plate info and paths for each vehicle, or None if no vehicles/plates found
        """
        if not frame_bytes_list:
            logger.warning("No frames provided for processing")
            return None

        logger.info(f"Processing {len(frame_bytes_list)} frames for multiple vehicles...")

        # Step 1: Track all vehicles across frames
        if not self.vehicle_recognizer or not self.vehicle_recognizer.enabled:
            # Fallback to old logic if vehicle recognition is disabled
            return self._process_frames_legacy(frame_bytes_list, save_dir)
        
        vehicle_tracks = self._track_vehicles_across_frames(frame_bytes_list)
        
        if not vehicle_tracks:
            logger.info("No vehicles detected in any frame")
            if self.config.vehicle_only_detection_enabled:
                return None
            # Fallback to frame-level plate detection
            return self._process_frames_legacy(frame_bytes_list, save_dir)
        
        logger.info(f"Found {len(vehicle_tracks)} unique vehicles across frames")
        
        # Step 2: For each vehicle track, find the best plate detection
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = []
        
        for vehicle_idx, track in enumerate(vehicle_tracks, 1):
            logger.info(f"Processing vehicle {vehicle_idx}/{len(vehicle_tracks)}...")
            
            # Find best plate for this vehicle across all its detections
            best_plate = self._find_best_plate_for_vehicle(track, frame_bytes_list)
            
            # Aggregate vehicle attributes across detections
            aggregated_vehicle = self._aggregate_vehicle_results(track['detections'])
            
            if best_plate:
                # Vehicle with plate detected
                logger.info(f"Vehicle {vehicle_idx}: {best_plate['plate_text']} "
                          f"(plate conf: {best_plate['confidence']:.3f}, "
                          f"vehicle: {aggregated_vehicle['color']} {aggregated_vehicle['type']})")
                
                # Save images
                image_filename = f"{timestamp}_{best_plate['plate_text']}_vehicle_{vehicle_idx}.jpg"
                crop_filename = f"{timestamp}_{best_plate['plate_text']}_crop_{vehicle_idx}.jpg"
                vehicle_crop_filename = f"{timestamp}_{best_plate['plate_text']}_vehicle_crop_{vehicle_idx}.jpg"
                
                image_path = save_dir / "images" / image_filename
                crop_path = save_dir / "images" / crop_filename
                vehicle_crop_path = save_dir / "images" / vehicle_crop_filename
                
                image_path.parent.mkdir(parents=True, exist_ok=True)
                
                cv2.imwrite(str(image_path), best_plate['full_image'])
                cv2.imwrite(str(crop_path), best_plate['plate_crop'])
                cv2.imwrite(str(vehicle_crop_path), track['best_detection']['crop'])
                
                results.append({
                    'plate_number': best_plate['plate_text'],
                    'confidence': best_plate['confidence'],
                    'image_path': f"images/{image_filename}",
                    'plate_crop_path': f"images/{crop_filename}",
                    'vehicle_crop_path': f"images/{vehicle_crop_filename}",
                    'box_coordinates': best_plate['bbox'],
                    'frame_count': len(frame_bytes_list),
                    'vehicle_color': aggregated_vehicle['color'],
                    'vehicle_type': aggregated_vehicle['type'],
                    'vehicle_confidence': aggregated_vehicle['confidence'],
                    'vehicle_bbox': track['best_detection']['bbox']
                })
            else:
                # Vehicle without plate (vehicle-only detection)
                logger.info(f"Vehicle {vehicle_idx}: No plate detected "
                          f"(vehicle: {aggregated_vehicle['color']} {aggregated_vehicle['type']})")
                
                if self.config.vehicle_only_detection_enabled:
                    # Save images for vehicle-only detection
                    vehicle_desc = f"{aggregated_vehicle['color']}_{aggregated_vehicle['type']}"
                    image_filename = f"{timestamp}_NO_PLATE_{vehicle_desc}_vehicle_{vehicle_idx}.jpg"
                    vehicle_crop_filename = f"{timestamp}_NO_PLATE_{vehicle_desc}_crop_{vehicle_idx}.jpg"
                    
                    image_path = save_dir / "images" / image_filename
                    vehicle_crop_path = save_dir / "images" / vehicle_crop_filename
                    
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Use best detection frame for full image
                    best_det = track['best_detection']
                    cv2.imwrite(str(image_path), track['best_frame_img'])
                    cv2.imwrite(str(vehicle_crop_path), best_det['crop'])
                    
                    results.append({
                        'plate_number': 'NO_PLATE',
                        'confidence': 0.0,
                        'image_path': f"images/{image_filename}",
                        'plate_crop_path': None,
                        'vehicle_crop_path': f"images/{vehicle_crop_filename}",
                        'box_coordinates': {},
                        'frame_count': len(frame_bytes_list),
                        'vehicle_color': aggregated_vehicle['color'],
                        'vehicle_type': aggregated_vehicle['type'],
                        'vehicle_confidence': aggregated_vehicle['confidence'],
                        'vehicle_bbox': best_det['bbox']
                    })
        
        return results if results else None
    
    def _process_frames_legacy(
        self,
        frame_bytes_list: List[bytes],
        save_dir: Path
    ) -> Optional[List[Dict]]:
        """
        Legacy frame-level processing (original logic wrapped in list for compatibility).
        Used when vehicle recognition is disabled.
        """
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
            # Sample across multiple frames for better accuracy
            vehicles_data = []
            
            if self.vehicle_recognizer and self.vehicle_recognizer.enabled:
                try:
                    logger.info("Detecting and recognizing all vehicles across multiple frames...")
                    vehicles_data = self._sample_vehicle_recognition_multiframe(
                        frame_bytes_list, 
                        best_image, 
                        timestamp, 
                        best_result.ocr.text.upper().replace(' ', ''),
                        save_dir
                    )
                    
                except Exception as e:
                    logger.error(f"Error during vehicle recognition: {e}")
                    # Add unknown vehicle data
                    vehicles_data.append({
                        'crop_path': None,
                        'bbox': None,
                        'detection_confidence': 0.0,
                        'color': 'unknown',
                        'type': 'unknown',
                        'confidence': 0.0
                    })

            # Use primary vehicle (first one) for backward compatibility
            primary_vehicle = vehicles_data[0] if vehicles_data else {
                'crop_path': None,
                'bbox': None,
                'detection_confidence': 0.0,
                'color': 'unknown',
                'type': 'unknown',
                'confidence': 0.0
            }

            # Wrap in list for consistency with new multi-vehicle API
            return [{
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
                'vehicle_type': primary_vehicle['type'],
                'vehicle_confidence': primary_vehicle['confidence'],
                'vehicles': vehicles_data  # List of all detected vehicles
            }]

        logger.info("No valid plates detected in any frame")
        return None

    def _track_vehicles_across_frames(self, frame_bytes_list: List[bytes]) -> List[Dict]:
        """
        Track all vehicles across multiple frames.
        
        Args:
            frame_bytes_list: All frames to process
            
        Returns:
            List of vehicle tracks, each containing detections list and best detection
        """
        vehicle_tracks = []
        
        for frame_idx, frame_bytes in enumerate(frame_bytes_list):
            try:
                # Decode image
                image_array = np.frombuffer(frame_bytes, np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # Get all vehicle crops in this frame
                vehicle_crops = self.vehicle_recognizer.get_all_vehicle_crops(img)
                
                if not vehicle_crops:
                    continue
                
                # Process each detected vehicle
                for vehicle_crop, vehicle_bbox, det_conf in vehicle_crops:
                    # Recognize vehicle attributes
                    vehicle_attrs = self.vehicle_recognizer.recognize_vehicle(vehicle_crop)
                    
                    # Try to match with existing tracks using IoU
                    matched_track = None
                    best_iou = 0.3  # Minimum IoU threshold
                    
                    for track in vehicle_tracks:
                        last_detection = track['detections'][-1]
                        iou = self._calculate_iou(vehicle_bbox, last_detection['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            matched_track = track
                    
                    detection = {
                        'frame_idx': frame_idx,
                        'frame_img': img.copy(),
                        'bbox': vehicle_bbox,
                        'crop': vehicle_crop.copy(),
                        'det_conf': det_conf,
                        'color': vehicle_attrs.get('color', 'unknown'),
                        'type': vehicle_attrs.get('type', 'unknown'),
                        'confidence': vehicle_attrs.get('confidence', 0.0)
                    }
                    
                    if matched_track:
                        matched_track['detections'].append(detection)
                        if vehicle_attrs['confidence'] > matched_track['best_confidence']:
                            matched_track['best_confidence'] = vehicle_attrs['confidence']
                            matched_track['best_detection'] = detection
                            matched_track['best_frame_img'] = img.copy()
                    else:
                        vehicle_tracks.append({
                            'detections': [detection],
                            'best_confidence': vehicle_attrs['confidence'],
                            'best_detection': detection,
                            'best_frame_img': img.copy()
                        })
                
            except Exception as e:
                logger.debug(f"Error tracking vehicles in frame {frame_idx}: {e}")
                continue
        
        return vehicle_tracks
    
    def _find_best_plate_for_vehicle(
        self, 
        vehicle_track: Dict, 
        frame_bytes_list: List[bytes]
    ) -> Optional[Dict]:
        """
        Find the best plate detection for a specific vehicle across its tracked frames.
        Samples multiple detections and aggregates results for better accuracy.
        
        Args:
            vehicle_track: Vehicle track containing detection list
            frame_bytes_list: Original frame bytes list
            
        Returns:
            Dict with plate info (plate_text, confidence, bbox, full_image, plate_crop) or None
        """
        # Collect all plate detections across all vehicle detections
        plate_candidates = {}  # {plate_text: [detection_dicts]}
        
        logger.debug(f"Scanning {len(vehicle_track['detections'])} vehicle detections for plates...")
        
        for detection in vehicle_track['detections']:
            try:
                # Get the vehicle crop from this detection
                vehicle_crop = detection['crop']
                frame_idx = detection['frame_idx']
                
                # Run ALPR on the vehicle crop
                plate_results = self.alpr.predict(vehicle_crop)
                
                if not plate_results:
                    continue
                
                # Process each plate found in this vehicle crop
                for result in plate_results:
                    plate_text = result.ocr.text.upper().replace(' ', '')
                    ocr_confidence = result.ocr.confidence
                    
                    # Check confidence threshold
                    if ocr_confidence < self.config.min_confidence:
                        logger.debug(f"Frame {frame_idx}: Low confidence {ocr_confidence:.3f} for {plate_text}")
                        continue
                    
                    logger.debug(f"Frame {frame_idx}: Found {plate_text} (conf: {ocr_confidence:.3f})")
                    
                    # Convert plate bbox from vehicle crop coordinates to full image coordinates
                    plate_bbox_crop = result.detection.bounding_box
                    vehicle_bbox = detection['bbox']
                    
                    # Calculate plate bbox in full image coordinates
                    plate_x1 = vehicle_bbox[0] + int(plate_bbox_crop.x1)
                    plate_y1 = vehicle_bbox[1] + int(plate_bbox_crop.y1)
                    plate_x2 = vehicle_bbox[0] + int(plate_bbox_crop.x2)
                    plate_y2 = vehicle_bbox[1] + int(plate_bbox_crop.y2)
                    
                    # Crop plate from vehicle crop image
                    crop_x1, crop_y1 = int(plate_bbox_crop.x1), int(plate_bbox_crop.y1)
                    crop_x2, crop_y2 = int(plate_bbox_crop.x2), int(plate_bbox_crop.y2)
                    plate_crop = vehicle_crop[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    
                    # Store this detection
                    plate_detection = {
                        'plate_text': plate_text,
                        'confidence': ocr_confidence,
                        'bbox': {
                            'xmin': plate_x1,
                            'ymin': plate_y1,
                            'xmax': plate_x2,
                            'ymax': plate_y2
                        },
                        'full_image': detection['frame_img'],
                        'plate_crop': plate_crop,
                        'frame_idx': frame_idx
                    }
                    
                    # Group by plate text
                    if plate_text not in plate_candidates:
                        plate_candidates[plate_text] = []
                    plate_candidates[plate_text].append(plate_detection)
                
            except Exception as e:
                logger.debug(f"Error finding plate in vehicle crop: {e}")
                continue
        
        if not plate_candidates:
            logger.debug("No plates found in any vehicle detection")
            return None
        
        # Aggregate results: select plate text with most detections and highest average confidence
        best_plate_text = None
        best_score = 0.0
        
        for plate_text, detections in plate_candidates.items():
            # Score = detection_count * average_confidence
            avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
            score = len(detections) * avg_confidence
            
            logger.debug(f"Plate {plate_text}: {len(detections)} detections, avg conf={avg_confidence:.3f}, score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_plate_text = plate_text
        
        if best_plate_text:
            # Get the best individual detection for this plate text (highest confidence)
            detections = plate_candidates[best_plate_text]
            best_detection = max(detections, key=lambda d: d['confidence'])
            
            logger.info(f"Selected plate {best_plate_text} from {len(detections)} detections "
                       f"(best conf: {best_detection['confidence']:.3f})")
            
            return best_detection
        
        return None

    def _sample_vehicle_recognition_multiframe(
        self,
        frame_bytes_list: List[bytes],
        best_plate_image: np.ndarray,
        timestamp: str,
        plate_text: str,
        save_dir: Path
    ) -> List[Dict]:
        """
        Sample vehicle recognition across multiple frames for better accuracy.
        
        Args:
            frame_bytes_list: All frames to sample from
            best_plate_image: The image with best plate detection (fallback)
            timestamp: Timestamp for filenames
            plate_text: Plate text for filenames
            save_dir: Directory to save images
            
        Returns:
            List of vehicle data dicts
        """
        vehicle_tracks = []
        
        # Sample a subset of frames (e.g., every 3rd frame for efficiency)
        frame_step = max(1, len(frame_bytes_list) // 10)  # Sample up to 10 frames
        sampled_indices = range(0, len(frame_bytes_list), frame_step)
        
        logger.info(f"Sampling vehicle recognition from {len(sampled_indices)} frames...")
        
        for frame_idx in sampled_indices:
            try:
                # Decode image
                image_array = np.frombuffer(frame_bytes_list[frame_idx], np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # Get all vehicle crops in this frame
                vehicle_crops = self.vehicle_recognizer.get_all_vehicle_crops(img)
                
                if not vehicle_crops:
                    continue
                
                # Process each detected vehicle
                for vehicle_crop, vehicle_bbox, det_conf in vehicle_crops:
                    # Recognize vehicle attributes
                    vehicle_attrs = self.vehicle_recognizer.recognize_vehicle(vehicle_crop)
                    
                    # Try to match with existing tracks using IoU
                    matched_track = None
                    best_iou = 0.3  # Minimum IoU threshold
                    
                    for track in vehicle_tracks:
                        last_detection = track['detections'][-1]
                        iou = self._calculate_iou(vehicle_bbox, last_detection['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            matched_track = track
                    
                    detection = {
                        'frame_idx': frame_idx,
                        'bbox': vehicle_bbox,
                        'crop': vehicle_crop.copy(),
                        'det_conf': det_conf,
                        'color': vehicle_attrs.get('color', 'unknown'),
                        'type': vehicle_attrs.get('type', 'unknown'),
                        'confidence': vehicle_attrs.get('confidence', 0.0)
                    }
                    
                    if matched_track:
                        matched_track['detections'].append(detection)
                        if vehicle_attrs.get('confidence', 0.0) > matched_track['best_confidence']:
                            matched_track['best_confidence'] = vehicle_attrs.get('confidence', 0.0)
                            matched_track['best_detection'] = detection
                    else:
                        vehicle_tracks.append({
                            'detections': [detection],
                            'best_confidence': vehicle_attrs['confidence'],
                            'best_detection': detection
                        })
                
            except Exception as e:
                logger.debug(f"Error sampling frame {frame_idx}: {e}")
                continue
        
        # If no vehicles found in sampled frames, fall back to best plate image
        if not vehicle_tracks:
            logger.info("No vehicles in sampled frames, using best plate frame")
            vehicle_crops = self.vehicle_recognizer.get_all_vehicle_crops(best_plate_image)
            
            if vehicle_crops:
                for vehicle_crop, vehicle_bbox, det_conf in vehicle_crops:
                    vehicle_attrs = self.vehicle_recognizer.recognize_vehicle(vehicle_crop)
                    vehicle_tracks.append({
                        'detections': [{
                            'frame_idx': -1,
                            'bbox': vehicle_bbox,
                            'crop': vehicle_crop.copy(),
                            'det_conf': det_conf,
                            'color': vehicle_attrs.get('color', 'unknown'),
                            'type': vehicle_attrs.get('type', 'unknown'),
                            'confidence': vehicle_attrs.get('confidence', 0.0)
                        }],
                        'best_confidence': vehicle_attrs.get('confidence', 0.0),
                        'best_detection': {
                            'bbox': vehicle_bbox,
                            'crop': vehicle_crop.copy(),
                            'det_conf': det_conf,
                            'color': vehicle_attrs.get('color', 'unknown'),
                            'type': vehicle_attrs.get('type', 'unknown'),
                            'confidence': vehicle_attrs.get('confidence', 0.0)
                        }
                    })
            else:
                # Last resort: process full image
                logger.warning("No vehicles detected, processing full image")
                vehicle_attrs = self.vehicle_recognizer.recognize_vehicle(best_plate_image)
                return [{
                    'crop_path': None,
                    'bbox': None,
                    'detection_confidence': 0.0,
                    'color': vehicle_attrs.get('color', 'unknown'),
                    'type': vehicle_attrs.get('type', 'unknown'),
                    'confidence': vehicle_attrs.get('confidence', 0.0),
                    'sample_count': 1
                }]
        
        # Process vehicle tracks and aggregate results
        vehicles_data = []
        logger.info(f"Found {len(vehicle_tracks)} unique vehicles")
        
        for idx, track in enumerate(vehicle_tracks, 1):
            # Aggregate results from multiple detections
            aggregated = self._aggregate_vehicle_results(track['detections'])
            best_det = track['best_detection']
            
            logger.info(f"Vehicle {idx}: {len(track['detections'])} samples, {aggregated['color']} {aggregated['type']}")
            
            # Save best crop from this track
            vehicle_crop_filename = f"{timestamp}_{plate_text}_vehicle_{idx}.jpg"
            vehicle_crop_path = save_dir / "images" / vehicle_crop_filename
            cv2.imwrite(str(vehicle_crop_path), best_det['crop'])
            
            vehicles_data.append({
                'crop_path': f"images/{vehicle_crop_filename}",
                'bbox': best_det['bbox'],
                'detection_confidence': best_det['det_conf'],
                'color': aggregated['color'],
                'type': aggregated['type'],
                'confidence': aggregated['confidence'],
                'sample_count': len(track['detections'])
            })
        
        return vehicles_data

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1, bbox2: Bounding boxes in format (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0 and 1
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

    def _aggregate_vehicle_results(self, detections: List[Dict]) -> Dict:
        """
        Aggregate vehicle attributes from multiple detections using voting and confidence.
        
        Args:
            detections: List of vehicle detection dicts with color, type, confidence
            
        Returns:
            Dict with aggregated color, type, and average confidence
        """
        if not detections:
            return {
                'color': 'unknown',
                'type': 'unknown',
                'confidence': 0.0
            }
        
        # Count occurrences and track confidences
        colors = {}
        types = {}
        confidences = []
        
        for det in detections:
            color = det.get('color', 'unknown')
            vtype = det.get('type', 'unknown')
            conf = det.get('confidence', 0.0)
            
            if color != 'unknown':
                colors[color] = colors.get(color, 0) + 1
            if vtype != 'unknown':
                types[vtype] = types.get(vtype, 0) + 1
            
            confidences.append(conf)
        
        # Select most common value (voting)
        best_color = max(colors.items(), key=lambda x: x[1])[0] if colors else 'unknown'
        best_type = max(types.items(), key=lambda x: x[1])[0] if types else 'unknown'
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'color': best_color,
            'type': best_type,
            'confidence': avg_confidence
        }

    def _process_vehicle_only(self, frame_bytes_list: List[bytes], save_dir: Path) -> Optional[Dict]:
        """
        Process vehicle recognition when no plate is detected.
        Samples multiple frames for better accuracy.
        
        Args:
            frame_bytes_list: List of frame bytes to process
            save_dir: Directory to save images
            
        Returns:
            Dict with vehicle info but no plate data, or None if processing fails
        """
        try:
            logger.info(f"No plate detected - performing vehicle-only recognition on {len(frame_bytes_list)} frames...")
            
            if not frame_bytes_list:
                return None
            
            # Track vehicles across frames: {vehicle_id: [detections]}
            vehicle_tracks = []
            best_frame_img = None
            
            # Process each frame to detect and recognize vehicles
            for frame_idx, frame_bytes in enumerate(frame_bytes_list):
                try:
                    # Decode image
                    image_array = np.frombuffer(frame_bytes, np.uint8)
                    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        continue
                    
                    # Get all vehicle crops in this frame
                    vehicle_crops = self.vehicle_recognizer.get_all_vehicle_crops(img)
                    
                    if not vehicle_crops:
                        continue
                    
                    # Process each detected vehicle
                    for vehicle_crop, vehicle_bbox, det_conf in vehicle_crops:
                        # Recognize vehicle attributes
                        vehicle_attrs = self.vehicle_recognizer.recognize_vehicle(vehicle_crop)
                        
                        if vehicle_attrs.get('color', 'unknown') == 'unknown' and vehicle_attrs.get('type', 'unknown') == 'unknown':
                            continue
                        
                        # Try to match with existing tracks using IoU
                        matched_track = None
                        best_iou = 0.3  # Minimum IoU threshold
                        
                        for track in vehicle_tracks:
                            # Compare with last bbox in this track
                            last_detection = track['detections'][-1]
                            iou = self._calculate_iou(vehicle_bbox, last_detection['bbox'])
                            if iou > best_iou:
                                best_iou = iou
                                matched_track = track
                        
                        detection = {
                            'frame_idx': frame_idx,
                            'bbox': vehicle_bbox,
                            'crop': vehicle_crop.copy(),
                            'det_conf': det_conf,
                            'color': vehicle_attrs.get('color', 'unknown'),
                            'type': vehicle_attrs.get('type', 'unknown'),
                            'confidence': vehicle_attrs.get('confidence', 0.0)
                        }
                        
                        if matched_track:
                            # Add to existing track
                            matched_track['detections'].append(detection)
                            # Update with highest confidence detection
                            if vehicle_attrs.get('confidence', 0.0) > matched_track['best_confidence']:
                                matched_track['best_confidence'] = vehicle_attrs.get('confidence', 0.0)
                                matched_track['best_detection'] = detection
                                matched_track['best_frame_img'] = img.copy()
                        else:
                            # Create new track
                            vehicle_tracks.append({
                                'detections': [detection],
                                'best_confidence': vehicle_attrs.get('confidence', 0.0),
                                'best_detection': detection,
                                'best_frame_img': img.copy()
                            })
                    
                except Exception as e:
                    logger.debug(f"Error processing frame {frame_idx+1}: {e}")
                    continue
            
            if not vehicle_tracks:
                logger.info("No vehicles detected in any frame")
                self._save_debug_frames(frame_bytes_list, save_dir)
                return None
            
            logger.info(f"Tracked {len(vehicle_tracks)} unique vehicles across {len(frame_bytes_list)} frames")
            
            # Process each vehicle track
            vehicles_data = []
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for idx, track in enumerate(vehicle_tracks, 1):
                # Aggregate results from all detections
                aggregated = self._aggregate_vehicle_results(track['detections'])
                best_det = track['best_detection']
                
                logger.info(f"Vehicle {idx}: {len(track['detections'])} detections, aggregated as {aggregated['color']} {aggregated['type']}")
                
                # Save best crop from this track
                vehicle_desc = f"{aggregated['color']}_{aggregated['type']}"
                vehicle_crop_filename = f"{timestamp}_NO_PLATE_{vehicle_desc}_vehicle_{idx}.jpg"
                vehicle_crop_path = save_dir / "images" / vehicle_crop_filename
                vehicle_crop_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vehicle_crop_path), best_det['crop'])
                logger.info(f"Saved vehicle crop {idx}: {vehicle_crop_filename}")
                
                vehicles_data.append({
                    'crop_path': f"images/{vehicle_crop_filename}",
                    'bbox': best_det['bbox'],
                    'detection_confidence': best_det['det_conf'],
                    'color': aggregated['color'],
                    'type': aggregated['type'],
                    'confidence': aggregated['confidence'],
                    'sample_count': len(track['detections'])
                })
                
                # Use the best frame for full image (from first vehicle)
                if idx == 1:
                    best_frame_img = track['best_frame_img']
            
            if not vehicles_data:
                logger.info("No valid vehicle attributes detected")
                self._save_debug_frames(frame_bytes_list, save_dir)
                return None
            
            # Save full image with vehicle-only naming (from best frame)
            primary_vehicle = vehicles_data[0]
            vehicle_desc = f"{primary_vehicle['color']}_{primary_vehicle['type']}"
            image_filename = f"{timestamp}_NO_PLATE_{vehicle_desc}.jpg"
            image_path = save_dir / "images" / image_filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), best_frame_img)
            
            return {
                'plate_number': 'NO_PLATE',
                'confidence': 0.0,
                'image_path': f"images/{image_filename}",
                'plate_crop_path': None,
                'vehicle_crop_path': primary_vehicle['crop_path'],
                'box_coordinates': {},
                'frame_count': len(frame_bytes_list),
                'vehicle_color': primary_vehicle['color'],
                'vehicle_type': primary_vehicle['type'],
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
