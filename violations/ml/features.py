# violations/ml/features.py
# SIMPLE & FOCUSED - Only 19D features as defined

import numpy as np
import logging
from violations.ml.pose_estimator import PoseEstimator

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract EXACTLY 19D features for littering detection."""
    
    def __init__(self):
        """Initialize with MediaPipe pose estimator."""
        self.pose_estimator = PoseEstimator()
        
        # Track previous values for velocity/acceleration
        self.prev_litter_pos = None
        self.prev_litter_velocity = 0.0
        self.prev_timestamp = None
        
        logger.info("✅ FeatureExtractor initialized - 19D only")
    
    def extract(self, frame, litter_data, person_pose, timestamp=0.0):
        """
        Extract EXACTLY 19D feature vector.
        
        Args:
            frame: Video frame (H × W × 3)
            litter_data: Detected litter {x, y, w, h, conf}
            person_pose: Person keypoints from MediaPipe (33 points)
            timestamp: Frame timestamp
        
        Returns:
            19D numpy array with EXACT structure:
            [0] litter_velocity
            [1] litter_acceleration
            [2] distance to left hand
            [3] distance to right hand
            [4] left arm angle (shoulder-elbow-wrist)
            [5] right arm angle (shoulder-elbow-wrist)
            [6-8] litter position (x, y, z)
            [9-11] left hand position (x, y, z)
            [12-14] right hand position (x, y, z)
            [15] dt (time delta)
            [16] relative_y (litter y vs hand y)
            [17] relative_x (litter x vs hand x)
            [18] person_detected (0 or 1)
        """
        
        features = np.zeros(19, dtype=np.float32)
        
        # Get keypoints from MediaPipe
        if person_pose is None:
            keypoints = self.pose_estimator.estimate_pose(frame)
        else:
            keypoints = person_pose
        
        person_detected = keypoints is not None
        
        if keypoints is None:
            keypoints = [(0.0, 0.0, 0.0, 0.0)] * 33
            person_detected = False
        if keypoints is None or (isinstance(keypoints, list) and len(keypoints) < 33):
            keypoints = self.pose_estimator.estimate_pose(frame)
        
        # 2. DOUBLE FIX: If MediaPipe STILL fails, fill with a "Safe Zero List"
        if keypoints is None or (isinstance(keypoints, list) and len(keypoints) < 33):
            keypoints = [(0.0, 0.0, 0.0, 0.0)] * 33
            person_detected = False
        else:
            person_detected = True
        
        # Define keypoint indices
        LEFT_SHOULDER = 11
        LEFT_ELBOW = 13
        LEFT_WRIST = 15
        RIGHT_SHOULDER = 12
        RIGHT_ELBOW = 14
        RIGHT_WRIST = 16
        
        # Extract hand positions (x, y, z)
        left_hand = keypoints[LEFT_WRIST] if LEFT_WRIST < len(keypoints) else (0.0, 0.0, 0.0, 0.0)
        right_hand = keypoints[RIGHT_WRIST] if RIGHT_WRIST < len(keypoints) else (0.0, 0.0, 0.0, 0.0)
        
        # Extract litter position (if detected)
        litter_pos = None
        if litter_data:
            # Normalize litter position to 0-1
            frame_h, frame_w = frame.shape[:2]
            litter_x = litter_data.get('x', 0) / frame_w if frame_w > 0 else 0
            litter_y = litter_data.get('y', 0) / frame_h if frame_h > 0 else 0
            litter_z = 1.0 - (litter_data.get('w', 0) * litter_data.get('h', 0)) / (frame_w * frame_h) if (frame_w * frame_h) > 0 else 1.0
            litter_pos = (litter_x, litter_y, litter_z)
        
        # ========== FEATURE [0]: LITTER VELOCITY ==========
        litter_velocity = self._compute_litter_velocity(litter_pos, timestamp)
        features[0] = litter_velocity
        
        # ========== FEATURE [1]: LITTER ACCELERATION ==========
        litter_accel = self._compute_litter_acceleration(litter_velocity, timestamp)
        features[1] = litter_accel
        
        # ========== FEATURES [2-3]: DISTANCE TO HANDS ==========
        
        # Distance from litter to left hand
        if litter_pos:
            dist_l_hand = np.linalg.norm(
                np.array(litter_pos[:2]) - np.array(left_hand[:2])
            )
        else:
            dist_l_hand = 0.0
        features[2] = dist_l_hand
        
        # Distance from litter to right hand
        if litter_pos:
            dist_r_hand = np.linalg.norm(
                np.array(litter_pos[:2]) - np.array(right_hand[:2])
            )
        else:
            dist_r_hand = 0.0
        features[3] = dist_r_hand
        
        # ========== FEATURES [4-5]: ARM ANGLES (3-POINT) ==========
        
        # Left arm angle: shoulder (11) → elbow (13) → wrist (15)
        left_angle = self._compute_3point_angle(
            keypoints[LEFT_SHOULDER],
            keypoints[LEFT_ELBOW],
            keypoints[LEFT_WRIST]
        )
        features[4] = left_angle
        
        # Right arm angle: shoulder (12) → elbow (14) → wrist (16)
        right_angle = self._compute_3point_angle(
            keypoints[RIGHT_SHOULDER],
            keypoints[RIGHT_ELBOW],
            keypoints[RIGHT_WRIST]
        )
        features[5] = right_angle
        
        # ========== FEATURES [6-8]: LITTER POSITION (x, y, z) ==========
        
        if litter_pos:
            features[6] = litter_pos[0]  # x (normalized 0-1)
            features[7] = litter_pos[1]  # y (normalized 0-1)
            features[8] = litter_pos[2]  # z (depth estimate 0-1)
        else:
            features[6] = 0.0
            features[7] = 0.0
            features[8] = 0.0
        
        # ========== FEATURES [9-11]: LEFT HAND POSITION (x, y, z) ==========
        
        features[9] = left_hand[0] if len(left_hand) > 0 else 0.0   # x
        features[10] = left_hand[1] if len(left_hand) > 1 else 0.0  # y
        features[11] = left_hand[2] if len(left_hand) > 2 else 0.0  # z
        
        # ========== FEATURES [12-14]: RIGHT HAND POSITION (x, y, z) ==========
        
        features[12] = right_hand[0] if len(right_hand) > 0 else 0.0   # x
        features[13] = right_hand[1] if len(right_hand) > 1 else 0.0   # y
        features[14] = right_hand[2] if len(right_hand) > 2 else 0.0   # z
        
        # ========== FEATURE [15]: TIME DELTA (dt) ==========
        
        # Time between frames (assuming 30 FPS = 0.033 seconds)
        dt = 1.0 / 30.0
        features[15] = dt
        
        # ========== FEATURE [16]: RELATIVE Y (litter y vs hand y) ==========
        
        if litter_pos and (left_hand or right_hand):
            # Average hand height
            hand_y = (left_hand[1] + right_hand[1]) / 2 if (left_hand and right_hand) else 0.0
            relative_y = litter_pos[1] - hand_y
            features[16] = relative_y
        else:
            features[16] = 0.0
        
        # ========== FEATURE [17]: RELATIVE X (litter x vs hand x) ==========
        
        if litter_pos and (left_hand or right_hand):
            # Average hand x position
            hand_x = (left_hand[0] + right_hand[0]) / 2 if (left_hand and right_hand) else 0.0
            relative_x = litter_pos[0] - hand_x
            features[17] = relative_x
        else:
            features[17] = 0.0
        
        # ========== FEATURE [18]: PERSON DETECTED (state) ==========
        
        # 1.0 = person detected, 0.0 = no person
        features[18] = 1.0 if person_detected else 0.0
        
        logger.debug(f"Extracted 19D features: {features}")
        
        return features
    
    # ========== HELPER METHODS ==========
    
    def _compute_litter_velocity(self, litter_pos, timestamp):
        """
        Compute velocity of litter from position change.
        
        Args:
            litter_pos: Current (x, y, z) normalized
            timestamp: Current frame timestamp
        
        Returns:
            Velocity magnitude (0-1 typically)
        """
        
        if litter_pos is None:
            self.prev_litter_pos = None
            self.prev_timestamp = timestamp
            return 0.0
        
        if self.prev_litter_pos is None or self.prev_timestamp is None:
            self.prev_litter_pos = litter_pos
            self.prev_timestamp = timestamp
            return 0.0
        
        dt = timestamp - self.prev_timestamp
        if dt == 0:
            return 0.0
        
        # Distance moved
        distance = np.linalg.norm(
            np.array(litter_pos[:2]) - np.array(self.prev_litter_pos[:2])
        )
        
        # Velocity
        velocity = distance / dt
        
        # Update for next frame
        self.prev_litter_pos = litter_pos
        self.prev_timestamp = timestamp
        
        return float(velocity)
    
    def _compute_litter_acceleration(self, velocity, timestamp):
        """
        Compute acceleration from velocity change.
        
        Args:
            velocity: Current velocity
            timestamp: Current frame timestamp
        
        Returns:
            Acceleration (change in velocity)
        """
        
        if self.prev_timestamp is None:
            return 0.0
        
        dt = timestamp - self.prev_timestamp
        if dt == 0:
            return 0.0
        
        accel = (velocity - self.prev_litter_velocity) / dt
        
        self.prev_litter_velocity = velocity
        
        return float(accel)
    
    def _compute_3point_angle(self, point_a, point_b, point_c):
        """
        Compute angle at point B (vertex) formed by A-B-C.
        
        This calculates the angle at the JOINT:
        - For arm: angle at ELBOW (shoulder → elbow → wrist)
        - For leg: angle at KNEE (hip → knee → ankle)
        
        Args:
            point_a: Start point (x, y, z, confidence)
            point_b: Vertex point (x, y, z, confidence)
            point_c: End point (x, y, z, confidence)
        
        Returns:
            Angle in radians (0 to π)
            0°   = straight down/aligned
            90°  = right angle bend
            180° = fully extended opposite direction
        """
        
        try:
            # Extract (x, y) coordinates only
            a = np.array(point_a[:2])
            b = np.array(point_b[:2])
            c = np.array(point_c[:2])
            
            # Vectors from vertex to other points
            ba = a - b
            bc = c - b
            
            # Avoid division by zero
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            
            if norm_ba == 0 or norm_bc == 0:
                return 0.0
            
            # Compute angle using dot product
            cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
            
            # Clamp to valid range
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Get angle in radians
            angle = np.arccos(cos_angle)
            
            return float(angle)
        
        except Exception as e:
            logger.error(f"Error computing 3-point angle: {e}")
            return 0.0


# Test the feature extractor
if __name__ == "__main__":
    import cv2
    
    print("\n" + "="*60)
    print("Testing FeatureExtractor - 19D ONLY")
    print("="*60)
    
    # Initialize
    extractor = FeatureExtractor()
    
    # Try to load a test image
    frame = cv2.imread("A1.png")
    if frame is None:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Get pose
    keypoints = extractor.pose_estimator.estimate_pose(frame)
    
    # Mock litter data
    litter_data = {
        'x': 320,
        'y': 240,
        'w': 50,
        'h': 60,
        'conf': 0.87
    }
    
    # Extract 19D features
    features = extractor.extract(frame, litter_data, keypoints, timestamp=0.033)
    
    # Print
    print("\n19D FEATURE VECTOR (EXACTLY AS DEFINED):")
    print("="*60)
    
    feature_names = [
        "[0] litter_velocity",
        "[1] litter_acceleration",
        "[2] distance_left_hand",
        "[3] distance_right_hand",
        "[4] left_arm_angle",
        "[5] right_arm_angle",
        "[6] litter_x",
        "[7] litter_y",
        "[8] litter_z",
        "[9] left_hand_x",
        "[10] left_hand_y",
        "[11] left_hand_z",
        "[12] right_hand_x",
        "[13] right_hand_y",
        "[14] right_hand_z",
        "[15] dt (1/30)",
        "[16] relative_y",
        "[17] relative_x",
        "[18] person_detected",
    ]
    
    for name, value in zip(feature_names, features):
        print(f"{name:30s}: {value:10.6f}")
    
    print("="*60)
    print("✅ 19D features extracted successfully!")
    print("="*60 + "\n")