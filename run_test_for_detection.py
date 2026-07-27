import sys
import os
import cv2

# Set up paths properly
base_path = os.path.dirname(os.path.abspath(__file__))
ml_folder_path = os.path.join(base_path, "violations", "ml")
sys.path.insert(0, ml_folder_path)

# Import directly to see errors
from violations.ml.detection import LitterDetectionPipeline
from violations.ml.config import CONFIG

print("✅ Pipeline and Config loaded successfully.")

def run_screenshot_test():
    # 1. Image path relative to project root (or pass your own)
    image_path = os.path.join(base_path, "A1.png")
    if not os.path.exists(image_path):
        print(f"❌ Still can't find A1.png at: {image_path}")
        folder_to_check = os.path.dirname(image_path)
        if os.path.exists(folder_to_check):
            print(f"📂 Files actually in that folder: {os.listdir(folder_to_check)[:5]}")
        else:
            print(f"❌ Even the folder {folder_to_check} doesn't seem to exist.")
    else:
        print(f"✅ Found it! Loading A1.png now...")

    # 2. Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Error: OpenCV failed to open the image. Is it a valid .jpg?")
        return

    # 3. Initialize the Pipeline (This is where the magic happens)
    print("🚀 Initializing Pipeline (DeepSORT & Roboflow API)...")
    try:
        pipeline = LitterDetectionPipeline(
            api_key=CONFIG.api_key, 
            model_id=CONFIG.model_id
        )
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        return

    # 4. Run the "detect_and_track" function
    print(f"🧠 Processing {image_path}...")
    litter_points, tracked_persons, labels = pipeline.detect_and_track(frame)

    # 5. Output results to the console
    print("\n" + "="*40)
    print(f"📊 TEST RESULTS FOR A1.jpg")
    print("="*40)
    
    print(f"🗑️ Litter Detected: {len(litter_points)}")
    for i, pt in enumerate(litter_points):
        print(f"   - Item {i}: [X:{pt[0]}, Y:{pt[1]}, Z:{pt[2]}]")

    print(f"👤 Persons Tracked: {len(tracked_persons)}")
    for p in tracked_persons:
        print(f"   - ID {p[0]} at center ({p[1]}, {p[2]})")

    # 6. Final visual check
    h, w = frame.shape[:2]
    for pt in litter_points:
        # Draw a red circle where the AI thinks the trash is
        cv2.circle(frame, (int(pt[0]*w), int(pt[1]*h)), 10, (0, 0, 255), -1)

    try:
        cv2.imshow("EcoJustice AI - Detection Test", frame)
        print("Press any key on the image window to close it...")
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    except cv2.error:
        print("⚠️ Could not open window. Saving result to 'result.jpg' instead.")
        cv2.imwrite("result.jpg", frame)

if __name__ == "__main__":
    run_screenshot_test()