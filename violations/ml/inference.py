# violations/ml/inference.py
# FIXED - Makes predictions WHILE person is visible using sliding window

import logging
import numpy as np
from typing import List, Dict
from violations.ml.config import CONFIG
from violations.ml.detection import LitterDetectionPipeline
from violations.ml.features import FeatureExtractor
from violations.ml.models import build_model
import tensorflow as tf

logger = logging.getLogger(__name__)

class InferenceEngine:
    """End-to-end inference with continuous predictions (sliding window)."""
    
    def __init__(self, model_path: str, api_key: str, model_id: str):
        """Initialize inference engine."""
        
        print("\n" + "="*60)
        print("🔧 Initializing InferenceEngine (Sliding Window)")
        print("="*60)
        
        try:
            print(f"📦 Loading model: {model_path}")
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print(f"✅ Model loaded: input shape {self.model.input_shape}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        try:
            print(f"🔍 Initializing detection pipeline...")
            self.detector = LitterDetectionPipeline(api_key=api_key, model_id=model_id)
            print(f"✅ Detection pipeline initialized")
        except Exception as e:
            print(f"❌ Failed to initialize detector: {e}")
            raise
        
        try:
            print(f"📊 Initializing feature extractor...")
            self.extractor = FeatureExtractor()
            print(f"✅ Feature extractor initialized")
        except Exception as e:
            print(f"❌ Failed to initialize extractor: {e}")
            raise
        
        # Buffers
        self.feature_buffer = []  # Keep last 60 frames
        self.frame_count = 0
        self.last_valid_features = np.zeros(19, dtype=np.float32)
        
        # Prediction tracking
        self.prediction_started = False
        self.start_prediction_at_frame = 30  # Start predictions at 30 frames (1 second)
        
        print("="*60)
        print("✅ InferenceEngine ready!\n")
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process single frame with CONTINUOUS SLIDING WINDOW predictions.
        """
        
        self.frame_count += 1
        
        results = {
            'frame_num': self.frame_count,
            'detections': [],
            'persons': 0,
            'features': np.zeros(19, dtype=np.float32),
            'prediction': None,
            'buffer_size': len(self.feature_buffer)
        }
        
        try:
            # ========== STEP 1: RUN DETECTION ==========
            litter_data, persons, labels = self.detector.detect_and_track(frame)
            
            results['detections'] = litter_data
            results['persons'] = len(persons) if isinstance(persons, list) else 0
            
            # ========== STEP 2: EXTRACT LITTER DATA ==========
            
            current_litter = None
            if isinstance(litter_data, list) and len(litter_data) > 0:
                litter_raw = litter_data[0]
                
                if isinstance(litter_raw, (list, tuple)) and len(litter_raw) >= 3:
                    current_litter = {
                        'x': float(litter_raw[0]) if litter_raw[0] else 0.0,
                        'y': float(litter_raw[1]) if litter_raw[1] else 0.0,
                        'z': float(litter_raw[2]) if litter_raw[2] else 0.0,
                    }
                elif isinstance(litter_raw, dict):
                    current_litter = litter_raw
            
            # ========== STEP 3: CHECK IF PERSON DETECTED ==========
            
            person_in_labels = any(
                label.lower() == 'person' 
                for label in labels if isinstance(label, str)
            )
            person_in_tracks = isinstance(persons, list) and len(persons) > 0
            person_detected = person_in_labels or person_in_tracks
            
            # ========== STEP 4: GET PERSON POSE ==========
            
            person_pose = None
            if person_detected:
                try:
                    person_pose = self.extractor.pose_estimator.estimate_pose(frame)
                except Exception as e:
                    logger.warning(f"Pose estimation failed: {e}")
                    person_pose = None
            
            # ========== STEP 5: EXTRACT 19D FEATURES ==========
            
            if person_detected and current_litter:
                # Extract fresh features
                try:
                    features = self.extractor.extract(
                        frame=frame,
                        litter_data=current_litter,
                        person_pose=person_pose,
                        timestamp=self.frame_count / 30.0
                    )
                    
                    if features is not None and len(features) == 19:
                        feat_vec = features.copy() if isinstance(features, np.ndarray) else np.array(features, dtype=np.float32)
                        self.last_valid_features = feat_vec.copy()
                    else:
                        feat_vec = self.last_valid_features.copy()
                
                except Exception as e:
                    logger.error(f"Feature extraction error: {e}")
                    feat_vec = self.last_valid_features.copy()
            
            else:
                # Use last valid features
                feat_vec = self.last_valid_features.copy()
                feat_vec[18] = 1.0 if person_detected else 0.0
            
            results['features'] = feat_vec
            
            # ========== STEP 6: BUFFER AND PREDICT (SLIDING WINDOW) ==========
            
            # Add to buffer
            self.feature_buffer.append(feat_vec)
            
            # Keep only last 60 frames
            if len(self.feature_buffer) > 60:
                self.feature_buffer = self.feature_buffer[-60:]
            
            results['buffer_size'] = len(self.feature_buffer)
            
            # ========== KEY FIX: CONTINUOUS PREDICTIONS ==========
            
            # Make prediction as soon as buffer has enough frames
            if len(self.feature_buffer) >= self.start_prediction_at_frame:
                
                # Mark that predictions have started
                if not self.prediction_started:
                    self.prediction_started = True
                    logger.info(f"Starting predictions at frame {self.frame_count}")
                
                try:
                    # Get last 60 frames, pad if necessary
                    window = np.array(self.feature_buffer[-60:])
                    
                    # If buffer is not yet 60 frames, pad at the beginning
                    if len(window) < 60:
                        pad_width = ((60 - len(window), 0), (0, 0))
                        window = np.pad(window, pad_width, mode='constant', constant_values=0)
                    
                    # Prepare input: use only first 18 features (exclude person_detected)
                    X = window[:, :18].reshape(1, 60, 18)
                    
                    # Run LSTM prediction
                    pred_prob = float(self.model.predict(X, verbose=0)[0][0])
                    
                    # Store result
                    is_suspicious = pred_prob > 0.5
                    results['prediction'] = {
                        'confidence': pred_prob,
                        'is_suspicious': is_suspicious,
                        'threshold': 0.5
                    }
                    
                    if is_suspicious:
                        logger.info(f"Frame {self.frame_count}: ⚠️ LITTERING DETECTED (confidence={pred_prob:.3f})")
                    else:
                        logger.debug(f"Frame {self.frame_count}: Prediction={pred_prob:.3f}")
                
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    import traceback
                    traceback.print_exc()
            
            else:
                # Still filling buffer
                results['prediction'] = None
        
        except Exception as e:
            logger.error(f"Process frame error: {e}")
            import traceback
            traceback.print_exc()
        
        return results


def print_prediction_result(results: dict):
    """Pretty print the results."""
    
    print(f"\n📊 Frame {results['frame_num']} Results:")
    print(f"   Litter detected: {len(results['detections'])} items")
    print(f"   Persons detected: {results['persons']}")
    print(f"   Features extracted: {np.sum(results['features'] != 0)}/19")
    print(f"   Buffer size: {results['buffer_size']}/60")
    
    if results['prediction']:
        conf = results['prediction']['confidence']
        is_suspicious = results['prediction']['is_suspicious']
        
        if is_suspicious:
            print(f"   ⚠️  ALERT! Littering detected (confidence: {conf:.3f})")
        else:
            print(f"   ✅ No littering (confidence: {conf:.3f})")
    else:
        print(f"   ⏳ Waiting for {30} frames... ({results['buffer_size']}/30)")


if __name__ == "__main__":
    print("Test inference.py directly")