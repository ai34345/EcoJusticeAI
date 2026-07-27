"""
Configuration for litter detection ML pipeline.
Secrets come from environment variables (.env); never hardcode API keys here.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _require_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    return value


@dataclass
class DetectionConfig:
    """Detection model configuration."""
    api_url: str = field(default_factory=lambda: _require_env("ROBOFLOW_API_URL", "https://serverless.roboflow.com"))
    api_key: str = field(default_factory=lambda: _require_env("ROBOFLOW_API_KEY"))
    model_id: str = field(default_factory=lambda: _require_env("ROBOFLOW_MODEL_ID", "my-first-project-xaurp/7"))
    confidence_threshold: float = 0.5
    max_detections: int = 10


@dataclass
class PipelineConfig:
    """Overall pipeline configuration."""
    api_url: str = field(default_factory=lambda: _require_env("ROBOFLOW_API_URL", "https://serverless.roboflow.com"))
    api_key: str = field(default_factory=lambda: _require_env("ROBOFLOW_API_KEY"))
    model_id: str = field(default_factory=lambda: _require_env("ROBOFLOW_MODEL_ID", "my-first-project-xaurp/7"))
    backend_url: str = field(default_factory=lambda: _require_env("BACKEND_URL", "http://localhost:8000"))
    backend_api_key: str = field(default_factory=lambda: _require_env("BACKEND_API_KEY"))
    model_path: str = field(default_factory=lambda: _require_env("MODEL_PATH", "weights/best_ecojustice_model.h5"))
    confidence_threshold: float = 0.5
    max_detections: int = 10
    # Feature engineering
    window_size_frames: int = 60
    feature_dimension: int = 19
    feature_fps: int = 30
    
    # Model architecture
    lstm_hidden_units: int = 64
    lstm_layers: int = 2
    dropout_rate: float = 0.4
    batch_size: int = 4
    
    # Training
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    
    # Inference
    min_detection_conf: float = 0.5
    
    # Behavior verification
    away_distance_threshold: float = 0.1
    confirmation_frames: int = 30
    min_evidence_frames: int = 20
    
    # Data paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    dataset_path: str = field(default="")
    
    def __post_init__(self):
        if not self.dataset_path:
            self.dataset_path = os.path.join(self.BASE_DIR, "dataset")
    
    def require_secrets(self) -> None:
        """Raise if required API credentials are missing."""
        missing = []
        if not self.api_key:
            missing.append("ROBOFLOW_API_KEY")
        if missing:
            raise EnvironmentError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill in values."
            )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Load from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__


# Usage
CONFIG = PipelineConfig(
    window_size_frames=60,
    lstm_hidden_units=64,
    dropout_rate=0.4,
)

print(f"Window size: {CONFIG.window_size_frames} frames")
print(f"LSTM units: {CONFIG.lstm_hidden_units}")
