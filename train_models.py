import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

from data.data_loader import DataLoader
from utils.ml_detector import MLDetector

def train_models():
    """Train ML models with Kaggle dataset"""
    print("Starting model training...")
    
    # Load data
    data_loader = DataLoader()
    
    # Load Kaggle dataset (replace with your actual file path)
    kaggle_file = "data/kaggle_dataset.csv"  # Update this path
    training_data = data_loader.load_kaggle_data(kaggle_file)
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Deadlock rate: {training_data['deadlock_occurred'].mean():.2%}")
    
    # Train models
    ml_detector = MLDetector()
    results = ml_detector.train_models(training_data)
    
    if results:
        print("\nTraining completed successfully!")
        for method, result in results.items():
            print(f"   {method}: Accuracy = {result['accuracy']:.4f}, F1 = {result['f1_score']:.4f}")
    else:
        print("Training failed!")

if __name__ == "__main__":
    train_models()