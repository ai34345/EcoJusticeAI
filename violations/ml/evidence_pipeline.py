# violations/ml/evidence_pipeline_FIXED.py
# FIXED: Saves complete frame + face metadata (bbox, coordinates, quality scores)

import numpy as np
import cv2
import base64
import logging
import requests
from datetime import datetime
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class FaceQualityEvaluator:
    """Extract best face with COMPLETE metadata."""
    
    def __init__(self):
        logger.info("✅ FaceQualityEvaluator initialized")

    def extract_best_face(self, frame, face_detections, frame_number=0, timestamp=None):
        """
        Extract best face from frame with ALL metadata.
        
        Args:
            frame: Video frame
            face_detections: List of face detections from Roboflow
            frame_number: Which frame number this is
            timestamp: When this frame was captured
        
        Returns:
            Complete face dict with image + metadata
        """
        if not face_detections:
            return None
        
        best_face = None
        best_score = -1
        
        for face_data in face_detections:
            # Extract coordinates
            x_center = face_data.get('x', 0)
            y_center = face_data.get('y', 0)
            width = face_data.get('width', face_data.get('w', 0))
            height = face_data.get('height', face_data.get('h', 0))
            conf = face_data.get('confidence', 0.0)

            # Convert to bounding box [x1, y1, x2, y2]
            x1 = int(max(0, x_center - width/2))
            y1 = int(max(0, y_center - height/2))
            x2 = int(min(frame.shape[1], x_center + width/2))
            y2 = int(min(frame.shape[0], y_center + height/2))
            
            # Extract face crop
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Calculate quality scores
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness_score = np.mean(gray)
            
            # Normalize scores
            sharpness_normalized = min(1.0, sharpness_score / 100.0)
            brightness_normalized = 1.0 if 80 <= brightness_score <= 180 else 0.5
            
            # Weighted overall quality score
            overall_score = (conf * 0.4) + (sharpness_normalized * 0.4) + (brightness_normalized * 0.2)

            if overall_score > best_score:
                best_score = overall_score
                best_face = {
                    # IMAGE
                    'face_image': face_crop,
                    
                    # QUALITY SCORES
                    'quality_score': overall_score,
                    'sharpness_score': sharpness_normalized,
                    'brightness_score': brightness_normalized,
                    'detection_confidence': conf,
                    
                    # COORDINATES ✅ CRITICAL
                    'bbox': [x1, y1, x2, y2],
                    'center': (int(x_center), int(y_center)),
                    'width': int(width),
                    'height': int(height),
                    
                    # SOURCE INFORMATION ✅ CRITICAL
                    'source_frame_number': frame_number,
                    'source_timestamp': timestamp or datetime.now().isoformat(),
                    
                    # METADATA
                    'metadata': {
                        'sharpness_variance': sharpness_score,
                        'brightness_mean': brightness_score,
                        'face_area_pixels': (x2 - x1) * (y2 - y1)
                    }
                }
        
        return best_face


class EvidenceCollector:
    """Collect violation frames with COMPLETE metadata."""
    
    def __init__(self):
        self.violation_frames = []
        self.face_evaluator = FaceQualityEvaluator()
        self.best_face_overall = None
        self.frame_counter = 0

    def add_frame(self, frame, is_violation, confidence, face_detections):
        """
        Add frame to buffer with COMPLETE metadata.
        
        Stores:
        - Frame image
        - LSTM confidence
        - All face detections (coordinates, bbox, confidence)
        - Timestamp
        - Frame number
        """
        self.frame_counter += 1
        timestamp = datetime.now().isoformat()
        
        # Extract best face from THIS frame (with full metadata)
        current_face = self.face_evaluator.extract_best_face(
            frame, 
            face_detections,
            frame_number=self.frame_counter,
            timestamp=timestamp
        )
        
        # Update overall best face
        if current_face:
            if (self.best_face_overall is None or 
                current_face['quality_score'] > self.best_face_overall['quality_score']):
                self.best_face_overall = current_face
                logger.info(f"✅ New best face found in frame {self.frame_counter} "
                           f"(quality: {current_face['quality_score']:.3f})")

        # Store COMPLETE frame data ✅ FIXED
        if is_violation:
            frame_data = {
                'frame': frame.copy(),
                'confidence': confidence,
                'timestamp': timestamp,
                'frame_number': self.frame_counter,
                
                # ✅ NEW: Store all face detections with full metadata
                'face_detections': [
                    {
                        'x': f.get('x', 0),
                        'y': f.get('y', 0),
                        'width': f.get('width', f.get('w', 0)),
                        'height': f.get('height', f.get('h', 0)),
                        'confidence': f.get('confidence', 0.0),
                        'bbox': [
                            int(max(0, f.get('x', 0) - f.get('width', f.get('w', 0))/2)),
                            int(max(0, f.get('y', 0) - f.get('height', f.get('h', 0))/2)),
                            int(min(frame.shape[1], f.get('x', 0) + f.get('width', f.get('w', 0))/2)),
                            int(min(frame.shape[0], f.get('y', 0) + f.get('height', f.get('h', 0))/2))
                        ]
                    }
                    for f in face_detections
                ] if face_detections else []
            }
            
            self.violation_frames.append(frame_data)
            logger.debug(f"Frame {self.frame_counter} added: confidence={confidence:.3f}, "
                        f"faces={len(frame_data['face_detections'])}")

    def get_evidence_package(self, submitter):
        """
        Get complete evidence package for submission.
        
        Returns all frames + best face + all metadata
        """
        if not self.violation_frames:
            logger.warning("No violation frames to submit")
            return [], None, 0.0

        # Encode all violation frames to Base64
        all_b64 = [submitter.encode_image(f['frame']) for f in self.violation_frames]
        
        # Get max confidence
        max_conf = max(f['confidence'] for f in self.violation_frames)
        
        if self.best_face_overall:
            quality_value = self.best_face_overall['quality_score']
            logger.info(f"✅ Evidence package ready: {len(all_b64)} frames, best face quality: {quality_value:.3f}")
        else:
            logger.info(f"✅ Evidence package ready: {len(all_b64)} frames, best face quality: N/A")
        return all_b64, self.best_face_overall, max_conf

    def clear(self):
        """Clear buffer after submission."""
        self.violation_frames = []
        self.best_face_overall = None
        self.frame_counter = 0
        logger.info("✅ Evidence buffer cleared")


class BackendSubmitter:
    """Submit evidence to backend with COMPLETE metadata."""
    
   
    def __init__(self, backend_url, api_key):
        self.backend_url = backend_url
        self.endpoint = f"{backend_url}/api/violations/ingest/"
        self.api_key = api_key
        logger.info(f"✅ BackendSubmitter initialized: {self.endpoint}")

    def encode_image(self, image):
        """Encode image to base64 JPEG."""
        try:
            success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                return base64.b64encode(buffer).decode('utf-8')
            else:
                logger.warning("Image encoding failed")
                return None
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None

    def submit_violation(self, track_id, all_frames_b64, best_face, confidence, location):
        """
        Submit violation to backend with COMPLETE metadata.
        
        Args:
            track_id: Person track ID
            all_frames_b64: List of base64 encoded frames
            best_face: Best face dict with full metadata
            confidence: LSTM confidence
            location: Where violation occurred
        """
        print("\n" + "="*70)
        print("📤 SUBMITTING VIOLATION WITH COMPLETE EVIDENCE")
        print("="*70)
        
        # Encode best face
        face_b64 = None
        face_quality = 0.0
        face_bbox = None
        face_metadata = {}
        
        if isinstance(best_face, dict) and 'face_image' in best_face:
            face_b64 = self.encode_image(best_face['face_image'])
            face_quality = best_face.get('quality_score', 0.0)
            face_bbox = best_face.get('bbox')  # ✅ CRITICAL
            face_metadata = {
                'sharpness_score': best_face.get('sharpness_score', 0.0),
                'brightness_score': best_face.get('brightness_score', 0.0),
                'detection_confidence': best_face.get('detection_confidence', 0.0),
                'center': best_face.get('center'),
                'dimensions': {
                    'width': best_face.get('width', 0),
                    'height': best_face.get('height', 0)
                },
                'source_frame': best_face.get('source_frame_number', 0),
                'source_timestamp': best_face.get('source_timestamp')
            }
        
        # Build payload ✅ COMPLETE
        payload = {
            "track_id": int(track_id),
            "location": str(location),
            "confidence": float(confidence),
            "face_image_b64": face_b64,
            "face_quality": face_quality,
            "face_bbox": face_bbox,  # ✅ NEW: Include bbox
            "face_metadata": face_metadata,  # ✅ NEW: Include all metadata
            "frames": all_frames_b64,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                'detection_source': 'ecojustice_ml_pipeline',
                'total_frames': len(all_frames_b64),
                'face_quality_score': face_quality
            }
        }
        
        print(f"\n📊 Payload Summary:")
        print(f"   Track ID: {track_id}")
        print(f"   Location: {location}")
        print(f"   LSTM Confidence: {confidence:.3f}")
        print(f"   Frames: {len(all_frames_b64)}")
        print(f"   Face Quality: {face_quality:.3f}")
        print(f"   Face BBox: {face_bbox}")
        print(f"   Face Metadata Keys: {list(face_metadata.keys())}")
        
        # Submit to backend
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        try:
            print(f"\n📡 Sending to {self.endpoint}...")
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 201:
                result = response.json()
                print(f"✅ Submission successful!")
                print(f"   Violation ID: {result.get('violation_id')}")
                return {
                    'status': 'success',
                    'violation_id': result.get('violation_id'),
                    'response': result
                }
            else:
                logger.error(f"Backend error: {response.status_code}")
                print(f"❌ Backend error: {response.status_code}")
                print(f"   {response.text}")
                return {
                    'status': 'error',
                    'code': response.status_code,
                    'message': response.text
                }
        
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            print("❌ Request timeout")
            return {'status': 'error', 'message': 'Backend timeout'}
        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            print(f"❌ Cannot connect to backend")
            return {'status': 'error', 'message': f'Cannot connect to {self.backend_url}'}
        
        except Exception as e:
            logger.error(f"Submission error: {e}")
            print(f"❌ Error: {e}")
            return {'status': 'error', 'message': str(e)}