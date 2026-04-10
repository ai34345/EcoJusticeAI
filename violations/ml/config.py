"""Configuration for ML pipeline."""
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Feature engineering
    window_size: int = 60
    feature_dim: int = 19
    fps: int = 30
    
    # Model
    lstm_units: int = 64
    lstm_layers: int = 2
    dropout: float = 0.4
    batch_size: int = 4
    
    # Training
    epochs: int = 100
    learning_rate: float = 0.001
    
    # Inference
    confidence_threshold: float = 0.7
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    
    dataset_path: str = os.path.join(BASE_DIR, "dataset")
    model_path: str = os.path.join(BASE_DIR, "weights", "best_litter_model.h5")
    
    # Add your Ngrok and API keys here too!
    NGROK_URL: str = "https://preyouthful-kymberly-nonlocally.ngrok-free.dev"

CONFIG = Config()