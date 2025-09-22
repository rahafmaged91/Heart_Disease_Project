"""
Heart Disease Prediction - Deployment Script
Generated on: 2025-09-22 15:39:16

Model: Logistic Regression
Test Accuracy: 0.883
Test F1-Score: 0.882
"""

import joblib
import pandas as pd
import numpy as np
import os

class HeartDiseasePredictor:
    def __init__(self, model_path=None):
        """Initialize the predictor with the trained pipeline."""
        if model_path is None:
            model_path = 'models/deployment/heart_disease_pipeline_latest.joblib'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.pipeline = joblib.load(model_path)
        self.feature_names = ['thal', 'ca', 'cp', 'exang', 'oldpeak', 'thalach', 'age', 'slope', 'trestbps', 'sex', 'chol', 'restecg']
        
    def predict(self, data):
        """Make predictions on new data."""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
            
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        return self.pipeline.predict(data)
    
    def predict_proba(self, data):
        """Get prediction probabilities."""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
            
        return self.pipeline.predict_proba(data)
    
    def get_feature_names(self):
        """Get the list of required feature names."""
        return self.feature_names.copy()

# Example usage
if __name__ == "__main__":
    try:
        # Initialize predictor
        predictor = HeartDiseasePredictor()
        
        # Example with sample data (replace with real values)
        # Create a dictionary with keys from predictor.get_feature_names()
        sample_data = {k: [0] for k in predictor.get_feature_names()}
        
        # Make prediction
        prediction = predictor.predict(sample_data)
        probabilities = predictor.predict_proba(sample_data)
        
        result_text = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
        print(f"Prediction: {prediction[0]} ({result_text})")
        print(f"Probability of No Disease: {probabilities[0][0]:.3f}")
        print(f"Probability of Disease: {probabilities[0][1]:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
