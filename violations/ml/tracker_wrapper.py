import sys
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Get the path to 'violations/ml'
ml_dir = os.path.dirname(os.path.abspath(__file__))

# Add deep_sort folder to path
deep_sort_path = os.path.join(ml_dir, "deep_sort")
sys.path.insert(0, deep_sort_path)

print(f"📁 ML Dir: {ml_dir}")
print(f"📁 DeepSORT Path: {deep_sort_path}")
print(f"📁 Checking if deep_sort exists: {os.path.exists(deep_sort_path)}")

# Try importing DeepSORT components
try:
    from deep_sort import nn_matching
    from deep_sort.tracker import Tracker as DeepSortTracker
    from deep_sort.detection import Detection
    from tools import generate_detections as gdet
    logger.info("✅ DeepSORT imported successfully!")
    DEEPSORT_LOADED = True
except ImportError as e:
    logger.error(f"❌ DeepSORT import error: {e}")
    DEEPSORT_LOADED = False
    DeepSortTracker = None
    nn_matching = None
    Detection = None
    gdet = None


class Tracker:
    """Wrapper around DeepSORT tracker with label matching."""
    
    def __init__(self):
        """Initialize tracker with DeepSORT."""
        
        if not DEEPSORT_LOADED:
            raise RuntimeError(
                "❌ DeepSORT not loaded. Check your deep_sort folder structure.\n"
                "Expected: violations/ml/deep_sort/\n"
                "With: tracker.py, nn_matching.py, detection.py, tools/generate_detections.py"
            )
        
        try:
            max_cosine_distance = 0.4
            nn_budget = None
            
            # Get path to model
            base_dir = os.path.dirname(os.path.abspath(__file__))
            encoder_model_filename = os.path.join(base_dir, 'model_data', 'mars-small128.pb')
            
            print(f"🔍 Looking for model at: {encoder_model_filename}")
            print(f"✅ Model exists: {os.path.exists(encoder_model_filename)}")
            
            if not os.path.exists(encoder_model_filename):
                logger.warning(f"⚠️ Model file not found at {encoder_model_filename}")
                logger.warning("⚠️ Create violations/ml/model_data/ folder")
                logger.warning("⚠️ Download mars-small128.pb from deep_sort repo")
                raise FileNotFoundError(f"Model not found: {encoder_model_filename}")

            # Create metric and tracker
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            self.tracker = DeepSortTracker(metric)
            self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
            self.tracks = []
            
            logger.info("✅ Tracker initialized successfully!")
        
        except Exception as e:
            logger.error(f"❌ Tracker init failed: {e}")
            raise

    def update(self, frame, detections, class_labels=None):
        """
        Update tracker with new detections.
        
        Args:
            frame: Video frame (H x W x 3)
            detections: List of [x1, y1, x2, y2, confidence]
            class_labels: List of class names matching each detection
        """

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        # If no class labels provided, default to "unknown"
        if class_labels is None:
            class_labels = ["unknown"] * len(detections)

        # Convert detections to format expected by DeepSORT
        # Input format: [x1, y1, x2, y2, confidence]
        # DeepSORT expects: [x1, y1, width, height]
        
        bboxes = np.asarray([d[:-1] for d in detections])  # [x1, y1, x2, y2]
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]    # Convert to [x1, y1, w, h]
        scores = np.array([d[-1] for d in detections])

        # Get features from encoder
        features = self.encoder(frame, bboxes)

        # Create Detection objects
        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        # Update tracker
        self.tracker.predict()
        self.tracker.update(dets)

        # Update tracks with labels
        self.update_tracks(bboxes, class_labels)

    def update_tracks(self, bboxes=None, class_labels=None):
        """
        Extract confirmed tracks and match them with class labels.
        
        Args:
            bboxes: Array of detection bboxes in tlwh format
            class_labels: List of class names
        """
        tracks = []
        
        for track in self.tracker.tracks:
            # Only keep confirmed tracks
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlbr()  # Convert to tlbr format [x1, y1, x2, y2]
            track_id = track.track_id

            # Match this track to closest detection using IoU
            label = "unknown"
            if bboxes is not None and class_labels is not None:
                label = self._match_label(bbox, bboxes, class_labels)

            tracks.append(Track(track_id, bbox, label))

        self.tracks = tracks

    def _match_label(self, track_bbox, bboxes, class_labels):
        """
        Find the class label of the detection that best matches this track.
        
        Args:
            track_bbox: Track bbox in tlbr format [x1, y1, x2, y2]
            bboxes: Array of detection bboxes in tlwh format
            class_labels: List of class names
            
        Returns:
            Best matching class label
        """
        best_iou = 0
        best_label = "unknown"

        for bbox, label in zip(bboxes, class_labels):
            # Convert bbox from tlwh to tlbr for IoU computation
            box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            iou = self._compute_iou(track_bbox, box)
            
            if iou > best_iou:
                best_iou = iou
                best_label = label

        return best_label

    @staticmethod
    def _compute_iou(boxA, boxB):
        """
        Compute Intersection over Union (IoU) between two boxes.
        
        Args:
            boxA: Box in tlbr format [x1, y1, x2, y2]
            boxB: Box in tlbr format [x1, y1, x2, y2]
            
        Returns:
            IoU score (0-1)
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        
        return max(0.0, min(1.0, iou))  # Clamp to [0, 1]


class Track:
    """Represents a single tracked object."""
    
    def __init__(self, track_id, bbox, label="unknown"):
        """
        Initialize a track.
        
        Args:
            track_id: Unique identifier for this track
            bbox: Bounding box in tlbr format [x1, y1, x2, y2]
            label: Class label (e.g., "person", "litter")
        """
        self.track_id = track_id
        self.bbox = bbox  # tlbr format [x1, y1, x2, y2]
        self.label = label
    
    @property
    def tlbr(self):
        """Get bbox in top-left-bottom-right format."""
        return self.bbox
    
    @property
    def tlwh(self):
        """Get bbox in top-left-width-height format."""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]
    
    def __repr__(self):
        return f"Track(id={self.track_id}, label={self.label}, bbox={self.bbox})"


# Debug info
if __name__ == "__main__":
    print("=" * 60)
    print("DeepSORT Tracker Wrapper")
    print("=" * 60)
    print(f"ML Directory: {ml_dir}")
    print(f"DeepSORT Path: {deep_sort_path}")
    print(f"DeepSORT Loaded: {DEEPSORT_LOADED}")
    
    if DEEPSORT_LOADED:
        try:
            tracker = Tracker()
            print("✅ Tracker initialized successfully!")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ DeepSORT not loaded!")