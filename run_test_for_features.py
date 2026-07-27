import numpy as np
import cv2
# This is the line you are likely missing:
from violations.ml.features import FeatureExtractor 

# ... the rest of your test code follows
if __name__ == "__main__":
    # This block only runs if you run features.py directly.
    # It won't run when you import FeatureExtractor in other files.
    import numpy as np

    # Create dummy data so the test actually works
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    extractor = FeatureExtractor()

    try:
        # FRAME 1: The object is at (0.5, 0.5) at time 0.0
        extractor.extract(
            frame=dummy_frame,
            litter_data=[[0.5, 0.5, 0.3]],
            pose_data={'left': {'coords': (0.3, 0.3, 0.1), 'angle': 90}},
            timestamp=0.0
        )
        
        # FRAME 2: The object moves to (0.6, 0.7) at time 0.033
        features = extractor.extract(
            frame=dummy_frame,
            litter_data=[[0.6, 0.7, 0.3]], # Coordinates changed!
            pose_data={'left': {'coords': (0.32, 0.32, 0.1), 'angle': 95}},
            timestamp=0.033
        )
        
        print("✅ Unit Test Passed: 19D Vector generated successfully.")
        print(f"Vector: {features.feature_vector}")
        
        # Checking specific indices
        print(f"Calculated Velocity: {features.feature_vector[0]}")
        print(f"Calculated Acceleration: {features.feature_vector[1]}")

    except Exception as e:
        print(f"❌ Unit Test Failed: {e}")