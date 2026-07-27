import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Import tracker wrapper
try:
    from .tracker_wrapper import Tracker
except ImportError:
    # Fallback if relative import fails
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tracker_wrapper import Tracker


class LitterDetectionPipeline:
    """Unified litter detection pipeline with tracking."""
    
    def __init__(self, api_key: str, model_id: str):
        """
        Initialize detection pipeline.
        
        Args:
            api_key: Roboflow API key
            model_id: Roboflow model ID
        """
        
        # Setup Roboflow inference client (key from caller / env — never hardcode)
        if not api_key:
            raise ValueError(
                "Roboflow API key is empty. Set ROBOFLOW_API_KEY in .env "
                "(see .env.example)."
            )
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        self.model_id = model_id
        
        # Setup tracker
        try:
            self.tracker = Tracker()
            logger.info("✅ Tracker initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize tracker: {e}")
            raise
    
    def detect_and_track(
        self,
        frame: np.ndarray
    ) -> Tuple[List, List, List]:
        """
        Detect litter and people, track them.
        
        Args:
            frame: Input video frame (H x W x 3)
        
        Returns:
            Tuple of (litter_detections, person_tracks, class_labels)
            where:
            - litter_detections: List of [normalized_x, normalized_y, normalized_z]
            - person_tracks: List of [track_id, center_x, center_y]
            - class_labels: List of class names for each detection
        """
        
        frame_h, frame_w = frame.shape[:2]
        
        # Run inference on frame
        try:
            result = self.client.infer(frame, model_id=self.model_id)
            print(f"🔍 Inference result keys: {result.keys()}")
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return [], [], []
        
        detections = []
        class_labels = []
        raw_litter_data = []
        person_detections_with_indices = []  # Track person detections with their indices
        
        # Parse predictions from Roboflow
        if 'predictions' not in result:
            logger.warning("No predictions in inference result")
            return [], [], []
        
        print(f"📊 Total predictions: {len(result['predictions'])}")
        
        for idx, pred in enumerate(result['predictions']):
            try:
                # Convert from center coordinates to tlbr
                x_center = pred['x']
                y_center = pred['y']
                width = pred['width']
                height = pred['height']
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                confidence = pred['confidence']
                class_name = pred['class']
                
                print(f"  [{idx}] {class_name}: ({x_center:.0f}, {y_center:.0f}) conf={confidence:.2f}")
                
                # Add to detections (tlbr format with confidence)
                detections.append([x1, y1, x2, y2, confidence])
                class_labels.append(class_name)
                
                # Track person detections for later matching
                if class_name.lower() == 'person':
                    person_detections_with_indices.append({
                        'index': len(detections) - 1,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence
                    })
                
                # Process litter-specific data (normalized coordinates)
                if class_name.lower() == 'litter':
                    # Normalize to 0-1
                    nx = x_center / frame_w
                    ny = y_center / frame_h
                    
                    # Calculate area and depth estimate
                    area = (width * height) / (frame_w * frame_h)
                    nz = 1.0 - area  # Rough depth estimate
                    
                    raw_litter_data.append([
                        round(nx, 4),
                        round(ny, 4),
                        round(nz, 4)
                    ])
                    
            except KeyError as e:
                logger.warning(f"Missing key in prediction: {e}")
                print(f"  ❌ Missing key: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing prediction: {e}")
                continue
        
        print(f"✅ Parsed {len(detections)} detections, {len(person_detections_with_indices)} persons")
        
        # Update tracker with detections
        if detections:
            try:
                self.tracker.update(frame, detections, class_labels)
                print(f"✅ Tracker updated with {len(detections)} detections")
            except Exception as e:
                logger.error(f"Tracker update failed: {e}")
                print(f"❌ Tracker update failed: {e}")
        
        # Extract tracked persons - FIXED VERSION
        tracked_persons = []
        if self.tracker.tracks:
            print(f"📊 Tracker has {len(self.tracker.tracks)} tracks")
            
            for track in self.tracker.tracks:
                print(f"  Track {track.track_id}: label='{track.label}'")
                
                # Check if this track is a person (case-insensitive)
                if track.label.lower() == 'person':
                    # Calculate center of bbox
                    bbox = track.tlbr
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    
                    print(f"    ✅ Person track {track.track_id} at ({cx:.0f}, {cy:.0f})")
                    
                    tracked_persons.append([
                        track.track_id,
                        cx,
                        cy
                    ])
                else:
                    print(f"    ❌ Not a person (label: {track.label})")
        else:
            print("⚠️ No tracks in tracker")
        
        logger.info(f"Detected: {len(raw_litter_data)} litter, {len(tracked_persons)} persons")
        print(f"📊 Result: {len(raw_litter_data)} litter, {len(tracked_persons)} persons\n")
        
        return raw_litter_data, tracked_persons, class_labels
    
    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """
        Process entire video file.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
        """
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Run detection and tracking
            litter_data, persons, labels = self.detect_and_track(frame)
            
            logger.info(f"Frame {frame_count}: {len(litter_data)} litter, {len(persons)} persons")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Processed {frame_count} frames")