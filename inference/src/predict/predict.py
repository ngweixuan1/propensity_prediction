import joblib
import os
import pandas as pd
import numpy as np

class ModelPredictor:
    def __init__(self, model_name: str, model_folder: str, feature_cols: list):
        """
        Load saved model and threshold from the given folder.
        
        Args:
            model_name (str): Name of the model (e.g., "random_forest", "xgboost").
            model_folder (str): Folder path where model and threshold are saved.
            feature_cols (list): List of feature column names expected by the model.
        """
        self.model_name = model_name
        self.feature_cols = feature_cols
        
        model_path = os.path.join(model_folder, f"{model_name}_model.joblib")
        threshold_path = os.path.join(model_folder, f"{model_name}_threshold.txt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(threshold_path):
            raise FileNotFoundError(f"Threshold file not found: {threshold_path}")
        
        self.model = joblib.load(model_path)
        
        with open(threshold_path, "r") as f:
            self.best_threshold = float(f.read().strip())
        
        print(f"Loaded model from {model_path}")
        print(f"Loaded threshold: {self.best_threshold:.3f}")

        if hasattr(self.model, "feature_names_in_"):
            print(f"Model trained with features: {self.model.feature_names_in_.tolist()}")

    def validate_features(self, X: pd.DataFrame):
        """
        Validate if prediction dataset matches expected features.
        Raises clear errors if mismatches found.
        """
        missing = [col for col in self.feature_cols if col not in X.columns]
        extra = [col for col in X.columns if col not in self.feature_cols]

        if missing:
            raise ValueError(f"Missing columns for prediction: {missing}")
        if extra:
            print(f"Warning: Extra columns present in prediction data (ignored during filtering): {extra}")

    def predict(self, X: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Predict binary labels and probabilities for input features.
        
        Args:
            X (pd.DataFrame): DataFrame containing feature columns.
            
        Returns:
            preds (np.ndarray): Binary predictions based on threshold.
            probs (np.ndarray): Predicted probabilities (positive class).
        """
        self.validate_features(X)
        
        X_filtered = X[self.feature_cols]
        
        print(f"Using columns for prediction: {X_filtered.columns.tolist()}")

        probs = self.model.predict_proba(X_filtered)[:, 1]
        preds = (probs >= self.best_threshold).astype(int)
        
        return preds, probs