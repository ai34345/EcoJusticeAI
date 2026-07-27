# violations/ml/pose_estimator.py
# Copy this entire file to violations/ml/ folder

import mediapipe as mp
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class PoseEstimator:
    """Extract pose keypoints using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe Pose."""
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=light, 1=full, 2=heavy
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("✅ MediaPipe Pose initialized")
            print("✅ MediaPipe Pose initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaPipe: {e}")
            print(f"❌ Failed to initialize MediaPipe: {e}")
            raise
    
    def estimate_pose(self, frame):
        """
        Detect pose keypoints in frame.
        
        Args:
            frame: Input frame (H × W × 3) BGR format from OpenCV
        
        Returns:
            List of 33 keypoints: [(x, y, z, confidence), ...]
            OR None if no person detected
        """
        
        try:
            # Check frame validity
            if frame is None or frame.size == 0:
                logger.warning("Empty frame received")
                return None
            
            # Convert BGR to RGB (MediaPipe needs RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run pose detection
            results = self.pose.process(rgb_frame)
            
            # No person detected
            if results.pose_landmarks is None:
                logger.debug("No person detected in frame")
                return None
            
            # Extract keypoints
            keypoints = []
            
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x  # 0-1 normalized (left-right)
                y = landmark.y  # 0-1 normalized (top-bottom)
                z = landmark.z  # depth (0-1)
                confidence = landmark.visibility  # how visible (0-1)
                
                keypoints.append((x, y, z, confidence))
            
            logger.debug(f"Detected {len(keypoints)} keypoints")
            return keypoints
        
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            print(f"❌ Pose estimation error: {e}")
            return None
    
    def get_keypoint_names(self):
        """Get names of all 33 MediaPipe keypoints."""
        return [
            # Head (0-9)
            "nose",                    # 0
            "left_eye_inner",          # 1
            "left_eye",                # 2
            "left_eye_outer",          # 3
            "right_eye_inner",         # 4
            "right_eye",               # 5
            "right_eye_outer",         # 6
            "left_ear",                # 7
            "right_ear",               # 8
            "mouth_left",              # 9
            "mouth_right",             # 10
            
            # Torso & Arms (11-22)
            "left_shoulder",           # 11
            "right_shoulder",          # 12
            "left_elbow",              # 13
            "right_elbow",             # 14
            "left_wrist",              # 15
            "right_wrist",             # 16
            "left_pinky",              # 17
            "right_pinky",             # 18
            "left_index",              # 19
            "right_index",             # 20
            "left_thumb",              # 21
            "right_thumb",             # 22
            
            # Legs (23-32)
            "left_hip",                # 23
            "right_hip",               # 24
            "left_knee",               # 25
            "right_knee",              # 26
            "left_ankle",              # 27
            "right_ankle",             # 28
            "left_heel",               # 29
            "right_heel",              # 30
            "left_foot_index",         # 31
            "right_foot_index"         # 32
        ]


# Test the pose estimator
if __name__ == "__main__":
    import cv2
    import os
    import numpy as np
    
    print("\n" + "="*60)
    print("Testing PoseEstimator")
    print("="*60)
    
    estimator = PoseEstimator()
    
    # Prefer images relative to project / cwd
    test_images = [
        "A1.png",
        "test_image.jpg",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "A1.png"),
    ]
    
    frame = None
    for img_path in test_images:
        if os.path.exists(img_path):
            frame = cv2.imread(img_path)
            if frame is not None:
                print(f"✅ Success! Loaded: {img_path}")
                break
        else:
            print(f"❓ Not found: {img_path}")
    
    if frame is None:
        print("⚠️ Still no image found. Creating dummy frame...")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Now run the actual MediaPipe detection on the loaded frame
    print("\nRunning pose estimation...")
    keypoints = estimator.estimate_pose(frame)
    
    if keypoints:
        print(f"✅ Found {len(keypoints)} keypoints!")
        print(keypoints)
        # This is where your l_angle and r_angle calculation will happen next!
    else:
        print("❌ MediaPipe ran but found no person in the image.")