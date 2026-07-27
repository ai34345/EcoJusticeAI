# run_test_for_inference.py
# FIXED v4 - Check PEAK confidence and SUSTAINED high frames (correct littering detection)

import cv2
import sys
import numpy as np
from pathlib import Path
from violations.ml.inference import InferenceEngine, print_prediction_result
from violations.ml.config import CONFIG

def main():
    """Run inference on video and detect littering based on PEAK + SUSTAINED confidence."""
    
    # ========== CONFIGURATION ==========
    
    CONFIG.require_secrets()
    model_path = CONFIG.model_path
    api_key = CONFIG.api_key
    model_id = CONFIG.model_id
    video_path = "1_littering (3).mp4"
    
    # Threshold settings
    PEAK_THRESHOLD = 0.8  # Peak confidence must be > this
    SUSTAINED_FRAMES = 8  # Must have 5+ consecutive high frames
    HIGH_CONFIDENCE_THRESHOLD = 0.6  # What counts as "high"
    
    print("\n" + "="*70)
    print("🚀 EcoJustice AI - Littering Detection (PEAK + SUSTAINED)")
    print("="*70)
    print(f"\nSettings:")
    print(f"  - Peak threshold: >{PEAK_THRESHOLD}")
    print(f"  - Sustained high frames: {SUSTAINED_FRAMES}+")
    print(f"  - High confidence: >{HIGH_CONFIDENCE_THRESHOLD}")
    
    # ========== INITIALIZE ENGINE ==========
    
    try:
        print("\n[STEP 1] Initializing Engine...")
        engine = InferenceEngine(
            model_path=model_path,
            api_key=api_key,
            model_id=model_id
        )
        print("✅ Engine initialized successfully!")
    
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== OPEN VIDEO ==========
    
    try:
        print(f"\n[STEP 2] Loading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            sys.exit(1)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 30.0
        print(f"✅ Video loaded: {frame_count} frames at {fps:.1f} FPS ({width}x{height})")

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"annotated_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"❌ Failed to create output video: {output_path}")
            cap.release()
            sys.exit(1)
        print(f"✅ Saving annotated video to: {output_path}")
    
    except Exception as e:
        print(f"❌ Failed to load video: {e}")
        sys.exit(1)
    
    # ========== PROCESS VIDEO ==========
    
    print(f"\n[STEP 3] Processing video frames...")
    print("="*70)
    
    frame_num = 0
    alert_count = 0
    all_confidences = []  # All frame confidences
    high_confidence_streak = 0  # Consecutive high confidence frames
    peak_confidence = 0.0  # Maximum confidence seen
    peak_frame = 0  # Frame with peak confidence
    sustained_high_frames = 0  # How many sustained high frames
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_num += 1
            
            # Process frame with inference engine
            results = engine.process_frame(frame)
            
            # Print results
            print_prediction_result(results)
            
            # Track confidence
            if results['prediction']:
                confidence = results['prediction']['confidence']
                all_confidences.append(confidence)
                
                # Update peak
                if confidence > peak_confidence:
                    peak_confidence = confidence
                    peak_frame = frame_num
                
                # Track sustained high confidence
                if confidence > HIGH_CONFIDENCE_THRESHOLD:
                    high_confidence_streak += 1
                    sustained_high_frames = max(sustained_high_frames, high_confidence_streak)
                else:
                    high_confidence_streak = 0
            
            # Count alerts
            if results['prediction'] and results['prediction']['is_suspicious']:
                alert_count += 1
                
                # Draw alert on frame
                cv2.putText(
                    frame,
                    f"ALERT! ({results['prediction']['confidence']:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),  # Red
                    2
                )
                cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), (0, 0, 255), 3)
            else:
                # Show "ok" on frame
                cv2.putText(
                    frame,
                    f"OK ({results['prediction']['confidence']:.2f})" if results['prediction'] else "OK",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),  # Green
                    2
                )
            
            # Save annotated frame (same frame shown in imshow)
            writer.write(frame)

            # Show frame
            cv2.imshow("EcoJustice AI - Littering Detection", frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️  Stopped by user")
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"\n💾 Annotated video saved to: {output_path}")
    
    # ========== SUMMARY ==========
    
    print("\n" + "="*70)
    print("📊 ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total frames processed: {frame_num}")
    print(f"Total alerts: {alert_count}")
    
    # Calculate statistics
    alert_percentage = (alert_count / frame_num) * 100 if frame_num > 0 else 0
    all_avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
    
    print(f"\n📈 Confidence Statistics:")
    print(f"   Alert rate: {alert_percentage:.1f}%")
    print(f"   Average confidence (all frames): {all_avg_confidence:.3f}")
    print(f"   Peak confidence: {peak_confidence:.3f} (at frame {peak_frame})")
    print(f"   Sustained high frames (>{HIGH_CONFIDENCE_THRESHOLD}): {sustained_high_frames}")
    
    # Decision logic based on PEAK + SUSTAINED
    print(f"\n🎯 LITTERING DETECTION LOGIC:")
    print(f"   Requirement 1: Peak > {PEAK_THRESHOLD}? {peak_confidence > PEAK_THRESHOLD} ({peak_confidence:.3f})")
    print(f"   Requirement 2: Sustained {SUSTAINED_FRAMES}+ high frames? {sustained_high_frames >= SUSTAINED_FRAMES} ({sustained_high_frames})")
    
    if peak_confidence > PEAK_THRESHOLD and sustained_high_frames >= SUSTAINED_FRAMES:
        print(f"\n⚠️  VIOLATION DETECTED - Littering behavior identified!")
        print(f"   Peak confidence: {peak_confidence:.3f} at frame {peak_frame}")
        print(f"   Sustained high confidence: {sustained_high_frames} consecutive frames")
        print(f"   Alert rate: {alert_percentage:.1f}%")
        violation_detected = True
    
    elif peak_confidence > PEAK_THRESHOLD:
        print(f"\n⚠️  POSSIBLE VIOLATION - High confidence but not sustained")
        print(f"   Peak confidence: {peak_confidence:.3f}")
        print(f"   Sustained high frames: {sustained_high_frames} (need {SUSTAINED_FRAMES})")
        print(f"   Recommendation: Manual review")
        violation_detected = False
    
    elif sustained_high_frames >= SUSTAINED_FRAMES:
        print(f"\n⚠️  POSSIBLE VIOLATION - Sustained but not high peak")
        print(f"   Peak confidence: {peak_confidence:.3f} (need >{PEAK_THRESHOLD})")
        print(f"   Sustained frames: {sustained_high_frames}")
        print(f"   Recommendation: Manual review")
        violation_detected = False
    
    else:
        print(f"\n✅ NO VIOLATION - Normal behavior")
        print(f"   Peak confidence: {peak_confidence:.3f}")
        print(f"   Sustained high frames: {sustained_high_frames}")
        violation_detected = False
    
    print("\n" + "="*70)
    print("✅ Analysis finished.\n")
    
    return violation_detected


if __name__ == "__main__":
    main()