# run_evaluation.py

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from violations.ml.model_evaluation import ModelEvaluator
from violations.ml.inference import InferenceEngine
from violations.ml.config import CONFIG

# ========== CONFIGURATION ==========

MODEL_PATH = CONFIG.model_path
ROBOFLOW_API_KEY = CONFIG.api_key
ROBOFLOW_MODEL_ID = CONFIG.model_id
TEST_VIDEOS_DIR = 'test_videos'  # Directory with littering/ and normal/ folders
BACKEND_URL = CONFIG.backend_url
BACKEND_API_KEY = CONFIG.backend_api_key

# ========== MAIN EVALUATION ==========

def main():
    print("\n" + "="*70)
    print("🚀 STARTING MODEL EVALUATION")
    print("="*70)

    try:
        CONFIG.require_secrets()
    except EnvironmentError as e:
        print(f"❌ {e}")
        return
    
    # Step 1: Initialize inference engine
    print("\n[1/4] Initializing Inference Engine...")
    try:
        inference_engine = InferenceEngine(
            model_path=MODEL_PATH,
            api_key=ROBOFLOW_API_KEY,
            model_id=ROBOFLOW_MODEL_ID
        )
        print("✅ Inference engine ready")
    except Exception as e:
        print(f"❌ Error initializing inference engine: {e}")
        return
    
    # Step 2: Load model
    print("\n[2/4] Loading trained model...")
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Step 3: Initialize evaluator
    print("\n[3/4] Initializing Model Evaluator...")
    try:
        evaluator = ModelEvaluator(model, inference_engine)
        print("✅ Evaluator ready")
    except Exception as e:
        print(f"❌ Error initializing evaluator: {e}")
        return
    
    # Step 4: Run evaluation
    print("\n[4/4] Running evaluation on test set...")
    print("This may take a few minutes...\n")
    
    try:
        results = evaluator.evaluate_on_videos(
            test_videos_dir=TEST_VIDEOS_DIR
        )
        
        if not results:
            print("❌ Evaluation failed")
            return
        
        # Print summary
        print("\n" + "="*70)
        evaluator.print_summary()
        
        # Generate visualizations
        print("\n[BONUS] Generating visualizations for thesis...")
        evaluator.generate_visualizations(
            output_dir='./evaluation_results'
        )
        
        # Generate JSON report
        print("\n[BONUS] Generating JSON report...")
        evaluator.generate_report(
            output_file='evaluation_report.json'
        )
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE!")
        print("="*70)
        print("\nOutputs saved in:")
        print("  - ./evaluation_results/ (charts)")
        print("  - ./evaluation_report.json (detailed metrics)")
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()