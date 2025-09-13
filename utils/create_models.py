# create_models.py
import numpy as np
from ml_detector import MLDetector
import joblib
import os

def create_and_save_models():
    """Create and save pre-trained ML models"""
    detector = MLDetector()
    print("Training models...")
    results = detector.train_models(5000)  # Train with 5000 samples
    
    # Save the feature count for consistency
    joblib.dump(detector.num_features, "models/num_features.pkl")
    print("Models trained and saved successfully!")
    
    for name, result in results.items():
        print(f"{name}: Accuracy = {result['accuracy']:.4f}")

if __name__ == "__main__":
    create_and_save_models()