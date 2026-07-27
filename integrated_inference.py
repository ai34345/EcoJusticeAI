# violations/ml/integrated_inference_FIXED.py
# FIXED: Proper face extraction from Roboflow predictions
import mediapipe as mp
import numpy as np
import cv2
import time
import logging
from violations.ml.inference import InferenceEngine
from violations.ml.evidence_pipeline import EvidenceCollector, BackendSubmitter, FaceQualityEvaluator
from violations.ml.config import CONFIG

logger = logging.getLogger(__name__)

class IntegratedInferenceEngine:
    """
    Complete pipeline with PROPER face metadata:
    1. Run LSTM inference (littering detection)
    2. Collect violation evidence frames
    3. Track face detections (coordinates, bbox, quality)
    4. Extract best face from ALL frames
    5. Submit complete evidence to Django backend
    """
    
    def __init__(self, model_path, roboflow_api_key, roboflow_model_id, 
                 backend_url, backend_api_key,
                 peak_threshold=0.7, sustained_frames=8):
        """
        Initialize complete pipeline.
        
        Args:
            model_path: Path to trained LSTM model (.h5)
            roboflow_api_key: Roboflow API key
            roboflow_model_id: Roboflow model ID
            backend_url: Django backend URL
            backend_api_key: Backend API key
            peak_threshold: Peak confidence for violation
            sustained_frames: Min sustained frames for violation
        """
        
        print("\n" + "="*70)
        print("🚀 INITIALIZING INTEGRATED INFERENCE ENGINE")
        print("="*70)
        
        # Thresholds
        self.peak_threshold = peak_threshold
        self.sustained_frames = sustained_frames
        
        print(f"\nThresholds:")
        print(f"  Peak: >{peak_threshold}")
        print(f"  Sustained: {sustained_frames}+ frames")
        
        # 1. Inference engine
        print("\n[1/3] Initializing LSTM Inference Engine...")
        self.inference_engine = InferenceEngine(
            model_path=model_path,
            api_key=roboflow_api_key,
            model_id=roboflow_model_id
        )
        print("     ✅ Ready")
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 for more accuracy, 0 for speed
            min_detection_confidence=0.5
        )
        # 2. Evidence collector
        print("\n[2/3] Initializing Evidence Collector...")
        self.evidence_collector = EvidenceCollector()
        print("     ✅ Ready")
        
        # 3. Backend submitter
        print("\n[3/3] Initializing Backend Submitter...")
        self.backend_submitter = BackendSubmitter(
            backend_url=backend_url,
            api_key=backend_api_key
        )
        print("     ✅ Ready")
        
        # Tracking for peak + sustained logic
        self.peak_confidence = 0.0
        self.sustained_count = 0
        self.confidence_history = []
        
        print("\n" + "="*70)
        print("✅ PIPELINE INITIALIZED!")
        print("="*70 + "\n")
    
    def _extract_faces_with_mediapipe(self, frame):
        """Use MediaPipe instead of Roboflow for faces"""
        results = self.mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                faces.append({
                    'x': int(bbox.xmin * w + bbox.width * w / 2),
                    'y': int(bbox.ymin * h + bbox.height * h / 2),
                    'width': int(bbox.width * w),
                    'height': int(bbox.height * h),
                    'confidence': detection.score[0]
                })
        
        return faces    
    def process_frame(self, frame, face_detections=None):
        """
        Process single frame: inference + evidence collection.
        
        Args:
            frame: Video frame (H × W × 3)
            face_detections: Optional pre-extracted face detections
        
        Returns:
            {
                'is_violation': True if littering detected,
                'confidence': LSTM confidence,
                'peak': peak confidence so far,
                'sustained': sustained high frame count,
                'should_submit': True if both conditions met
            }
        """
        
        # Step 1: Run LSTM inference
        results = self.inference_engine.process_frame(frame)
        face_detections = self._extract_faces_with_mediapipe(frame)
        
        is_violation = (
            results['prediction'] and 
            results['prediction']['is_suspicious']
        )
        
        confidence = results['prediction']['confidence'] if results['prediction'] else 0.0
        self.confidence_history.append(confidence)
        
        # Update peak
        if confidence > self.peak_confidence:
            self.peak_confidence = confidence
        
        # Track sustained high confidence
        if confidence > 0.6:  # "High" threshold
            self.sustained_count += 1
        else:
            self.sustained_count = 0
        
        
            # Step 3: Collect evidence
        # Pass faces for best face extraction
        self.evidence_collector.add_frame(
            frame=frame,
            is_violation=is_violation, 
            confidence=confidence,
            face_detections=face_detections  # ✅ CRITICAL: Pass the faces
        )
        
        # Step 4: Check if should submit
        should_submit = (self.peak_confidence > self.peak_threshold and 
                        self.sustained_count >= self.sustained_frames)
        
        return {
            'is_violation': is_violation,
            'confidence': confidence,
            'peak': self.peak_confidence,
            'sustained': self.sustained_count,
            'should_submit': should_submit,
            'buffered_frames': len(self.evidence_collector.violation_frames),
            'best_face_found': self.evidence_collector.best_face_overall is not None,
            'faces_in_frame': len(face_detections)
        }
    
    def submit_violation(self, location="Unknown", track_id=1):
        """
        Submit violation with COMPLETE evidence (frames + face metadata).
        
        Args:
            location: Where violation occurred
            track_id: Person track ID
        
        Returns:
            Backend submission response
        """
        
        print("\n" + "="*70)
        print("📸 EXTRACTING & SUBMITTING EVIDENCE")
        print("="*70)
        
        # Get evidence package
        print("\n[1/3] Preparing evidence package...")
        frames_list, best_face, max_conf = self.evidence_collector.get_evidence_package(
            self.backend_submitter
        )
        
        if not frames_list:
            print("⚠️  No frames to submit")
            return {'status': 'error', 'message': 'No violation frames'}
        
        print(f"      ✅ {len(frames_list)} frames ready")
        
        if best_face:
            print(f"      ✅ Best face ready")
            print(f"         Quality: {best_face['quality_score']:.3f}")
            print(f"         BBox: {best_face['bbox']}")
            print(f"         Center: {best_face['center']}")
            print(f"         From frame: {best_face['source_frame_number']}")
        else:
            print(f"      ⚠️  No face detected in frames")
        
        # Submit to backend
        print("\n[2/3] Submitting to backend...")
        submission = self.backend_submitter.submit_violation(
            track_id=track_id,
            all_frames_b64=frames_list,
            best_face=best_face,
            confidence=max(self.peak_confidence, max_conf),
            location=location
        )
        
        # Clear buffer on success
        if submission['status'] == 'success':
            print("\n[3/3] Clearing buffer...")
            self.evidence_collector.clear()
            self.peak_confidence = 0.0
            self.sustained_count = 0
            self.confidence_history = []
            print("      ✅ Ready for next violation")
        
        return submission
    
    def get_buffer_status(self):
        """Get current buffer and detection status."""
        return {
            'buffered_frames': len(self.evidence_collector.violation_frames),
            'peak_confidence': self.peak_confidence,
            'sustained_frames': self.sustained_count,
            'best_face_quality': (self.evidence_collector.best_face_overall['quality_score']
                                 if self.evidence_collector.best_face_overall else None),
            'should_submit': (self.peak_confidence > self.peak_threshold and 
                            self.sustained_count >= self.sustained_frames),
            'frames_detail': [
                {
                    'frame_num': f.get('frame_number', 0),
                    'confidence': f['confidence'],
                    'faces': len(f.get('face_detections', []))
                }
                for f in self.evidence_collector.violation_frames
            ]
        }


# ========== USAGE EXAMPLE ==========

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("EXAMPLE: Using Integrated Inference Engine")
    print("="*70)
    
    # Initialize
    CONFIG.require_secrets()
    engine = IntegratedInferenceEngine(
        model_path=CONFIG.model_path,
        roboflow_api_key=CONFIG.api_key,
        roboflow_model_id=CONFIG.model_id,
        backend_url=CONFIG.backend_url,
        backend_api_key=CONFIG.backend_api_key,
        peak_threshold=0.7,
        sustained_frames=8
    )
    
    # Open video
    cap = cv2.VideoCapture('1_littering.mp4')
    frame_count = 0
    
    print("\n🎬 Processing video...")
    print("="*70)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame - face extraction happens INSIDE
        results = engine.process_frame(frame)
        
        # Print status every 10 frames
        if frame_count % 10 == 0:
            status = engine.get_buffer_status()
            print(f"\nFrame {frame_count}:")
            print(f"  Peak: {status['peak_confidence']:.3f}")
            print(f"  Sustained: {status['sustained_frames']}")
            print(f"  Should submit: {status['should_submit']}")
            if status['best_face_quality']:
                print(f"  Best face quality: {status['best_face_quality']:.3f}")
        
        # Print when violation detected
        if results['is_violation']:
            print(f"\n  Frame {frame_count}: LITTERING ALERT! "
                  f"(confidence: {results['confidence']:.3f}, "
                  f"faces in frame: {results['faces_in_frame']})")
        
        # Submit when conditions met
        if results['should_submit']:
            print(f"\n✅ Conditions met at frame {frame_count}! Submitting...")
            submission = engine.submit_violation(
                location="Gate-A",
                track_id=int(time.time())
            )
            
            if submission['status'] == 'success':
                print(f"   Violation ID: {submission.get('violation_id')}")
                # Reset for next violation
                engine.peak_confidence = 0.0
                engine.sustained_count = 0
            else:
                print(f"   Error: {submission.get('message')}")
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("✅ Processing complete")
    print("="*70)