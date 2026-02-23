"""
Predictor for AMQSS AI Layer
Purpose: Load trained model, make predictions on new AMQSS features.
Role: Market Environment Classifier (not signal generator).
"""

import pandas as pd
import numpy as np
import pickle
import os
import json


class AMQSSPredictor:
    """Load and use trained RandomForest model for market environment classification."""
    
    def __init__(self, models_dir):
        """
        Load model, scaler, and metadata.
        
        Args:
            models_dir: Directory containing model.pkl, scaler.pkl, metadata.json
        """
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.metadata = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, scaler, and metadata from disk."""
        model_path = os.path.join(self.models_dir, 'model.pkl')
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        meta_path = os.path.join(self.models_dir, 'metadata.json')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
            raise FileNotFoundError(f"Missing model artifacts in {self.models_dir}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_cols = self.metadata['feature_columns']
        print(f"[Predictor] Model loaded from {self.models_dir}")
        print(f"  Features: {self.feature_cols}")
        print(f"  Trained: {self.metadata.get('trained_at', 'unknown')}")
    
    def predict(self, amqss_features):
        """
        Predict probability that market is high quality.
        
        Args:
            amqss_features: Dict with keys matching feature_cols
                            (volatility_regime, true_liquidity, etc.)
        
        Returns:
            float in [0, 1] (probability of high-quality market)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Extract features in correct order
        X = []
        for col in self.feature_cols:
            if col not in amqss_features:
                raise ValueError(f"Missing feature: {col}")
            X.append(amqss_features[col])
        
        X = np.array([X], dtype=float)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict probability
        proba = self.model.predict_proba(X_scaled)[0][1]
        
        return float(proba)
    
    def predict_batch(self, amqss_features_list):
        """
        Predict on multiple samples.
        
        Args:
            amqss_features_list: List of dicts
        
        Returns:
            np.array of probabilities
        """
        X = []
        for features in amqss_features_list:
            row = [features.get(col, 0) for col in self.feature_cols]
            X.append(row)
        
        X = np.array(X, dtype=float)
        X_scaled = self.scaler.transform(X)
        probas = self.model.predict_proba(X_scaled)[:, 1]
        
        return probas


def load_predictor(models_dir):
    """Convenience function to load predictor."""
    return AMQSSPredictor(models_dir)


if __name__ == '__main__':
    # Example usage
    models_dir = r'C:\Users\bkiyo\Desktop\Project.X\models'
    
    try:
        predictor = load_predictor(models_dir)
        
        # Example features
        example_features = {
            'volatility_regime': 5.0,
            'true_liquidity': 7.0,
            'structure': 8.0,
            'momentum': 6.0,
            'volatility_level': 4.0,
            'current_atr': 0.001
        }
        
        prob = predictor.predict(example_features)
        print(f"\nExample prediction: {prob:.4f}")
        print(f"Market environment quality: {'HIGH' if prob > 0.5 else 'LOW'}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
