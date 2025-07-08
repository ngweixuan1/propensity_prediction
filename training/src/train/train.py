import numpy as np
import pandas as pd
import yaml
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV
import wandb
import warnings
import os
import joblib
import itertools
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class ModelHyperparams(BaseModel):
    """Configuration for model hyperparameter grids."""
    random_forest: dict = Field(default_factory=dict)
    xgboost: dict = Field(default_factory=dict)


class WandbConfig(BaseModel):
    """W&B tracking configuration."""
    project_name: str
    run_name: str = None
    hyperparameters: ModelHyperparams


def load_wandb_config(yaml_path: str) -> WandbConfig:
    """
    Load W&B config from a YAML file.

    Args:
        yaml_path (str): Path to YAML config file.

    Returns:
        WandbConfig: Parsed W&B config object.
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return WandbConfig.parse_obj(cfg)


class ClassifierTrainer:
    """
    Class to handle model initialization, training, evaluation, and saving.

    Supports RandomForest and XGBoost models.
    """
    def __init__(
        self,
        model_name: str,
        config: ModelHyperparams,
        feature_cols: List[str],
        label_col: str,
        wandb_config: WandbConfig,
    ):
        """
        Initialize the trainer.

        Args:
            model_name (str): Name of model ('random_forest' or 'xgboost').
            config (ModelHyperparams): Hyperparameter grid.
            feature_cols (List[str]): List of feature column names.
            label_col (str): Name of label column.
            wandb_config (WandbConfig): Weights & Biases tracking config.
        """
        self.model_name = model_name
        self.config = config
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.best_threshold = 0.5
        self.model = self._init_model()

        self.wandb_run = wandb.init(
            project=wandb_config.project_name,
            name=wandb_config.run_name,
            config=config.dict(),
        )
        wandb.config.update({"model_name": model_name})

    def _init_model(self):
        """Initialize the model based on name."""
        if self.model_name == "random_forest":
            return RandomForestClassifier(random_state=42)
        elif self.model_name == "xgboost":
            return XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_estimators=80,
                learning_rate=0.001,
            )
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

    def top_15_metrics(self, probs: np.ndarray, y_true: Union[pd.Series, np.ndarray]) -> tuple:
        """
        Calculate precision, recall, and lift for top 15% highest scores.

        Args:
            probs (np.ndarray): Predicted probabilities.
            y_true (array-like): Ground truth labels.

        Returns:
            tuple: (top15_precision, top15_recall, top15_lift)
        """
        top_k_percent = 0.15
        n_top = int(len(probs) * top_k_percent)
        sorted_indices = probs.argsort()[::-1][:n_top]
        top_y = y_true.iloc[sorted_indices].values if isinstance(y_true, pd.Series) else y_true[sorted_indices]
        total_positives = y_true.sum()

        positives_in_top = top_y.sum()
        top15_precision = positives_in_top / n_top if n_top > 0 else 0.0
        top15_recall = positives_in_top / total_positives if total_positives > 0 else 0.0
        baseline_positive_rate = total_positives / len(y_true) if len(y_true) > 0 else 0.0
        lift = (top15_precision / baseline_positive_rate) if baseline_positive_rate > 0 else 0.0

        return top15_precision, top15_recall, lift

    def expected_calibration_error(self, y_true, probs, n_bins=10):
        """
        Compute Expected Calibration Error (ECE).

        Args:
            y_true (array-like): Ground truth.
            probs (np.ndarray): Predicted probabilities.
            n_bins (int): Number of bins.

        Returns:
            float: ECE score.
        """
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy='uniform')
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bins=bin_edges, right=True) - 1
        unique_bins = np.unique(bin_indices)
        bin_counts = np.array([np.sum(bin_indices == b) for b in unique_bins])
        weights = bin_counts / bin_counts.sum() if bin_counts.sum() > 0 else np.ones_like(bin_counts) / len(bin_counts)
        return float(np.sum(weights * np.abs(prob_true - prob_pred)))

    def compute_all_metrics(self, y_true: Union[pd.Series, np.ndarray], probs: np.ndarray) -> Dict[str, float]:
        """
        Compute full set of evaluation metrics.

        Args:
            y_true (array-like): Ground truth labels.
            probs (np.ndarray): Predicted probabilities.

        Returns:
            dict: Dictionary of metrics.
        """
        preds = (probs >= self.best_threshold).astype(int)

        f1 = f1_score(y_true, preds)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds)
        auc = roc_auc_score(y_true, probs)
        brier = brier_score_loss(y_true, probs)
        ece = self.expected_calibration_error(y_true, probs)
        top15_precision, top15_recall, top15_lift = self.top_15_metrics(probs, y_true)

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "brier": brier,
            "ece": ece,
            "top15_precision": top15_precision,
            "top15_recall": top15_recall,
            "top15_lift": top15_lift,
        }

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        """
        Tune hyperparameters using validation F1 score.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.DataFrame): Validation labels.
        """
        X_train_filtered = X_train[self.feature_cols]
        X_val_filtered = X_val[self.feature_cols]
        y_train = y_train[self.label_col]
        y_val = y_val[self.label_col]

        param_grid = getattr(self.config, self.model_name, {})
        if not param_grid:
            raise ValueError(f"No hyperparameter grid provided for {self.model_name}.")

        best_f1 = -1
        best_params = None
        best_model = None
        best_threshold = 0.5

        keys, values = zip(*param_grid.items())
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            model = self.model.__class__(**params)
            model.fit(X_train_filtered, y_train)

            val_probs = model.predict_proba(X_val_filtered)[:, 1]
            threshold = self._tune_threshold(np.array(y_val), val_probs)
            val_preds = (val_probs >= threshold).astype(int)

            f1 = f1_score(y_val, val_preds)
            if f1 > best_f1:
                best_f1 = f1
                best_params = params
                best_model = model
                best_threshold = threshold

        # Save best model and threshold
        self.model = best_model
        self.best_threshold = best_threshold

        print(f"\nBest hyperparameters for {self.model_name}: {best_params}")
        print(f"Best threshold tuned on validation set: {best_threshold:.3f}")
        print(f"Best validation F1: {best_f1:.4f}\n")

        # Compute metrics on train and val
        train_probs = best_model.predict_proba(X_train_filtered)[:, 1]
        val_probs = best_model.predict_proba(X_val_filtered)[:, 1]

        train_metrics = self.compute_all_metrics(y_train, train_probs)
        val_metrics = self.compute_all_metrics(y_val, val_probs)

        print("Metrics using Best Model BEFORE Retraining:")
        print(f"Train F1: {train_metrics['f1']:.4f} | Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f} | ROC AUC: {train_metrics['auc']:.4f}")
        print(f"Train Brier: {train_metrics['brier']:.4f} | ECE: {train_metrics['ece']:.4f}")
        print(f"Train Top15% Precision: {train_metrics['top15_precision']:.4f} | Recall: {train_metrics['top15_recall']:.4f} | Lift: {train_metrics['top15_lift']:.4f}")

        print(f"Val F1: {val_metrics['f1']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | ROC AUC: {val_metrics['auc']:.4f}")
        print(f"Val Brier: {val_metrics['brier']:.4f} | ECE: {val_metrics['ece']:.4f}")
        print(f"Val Top15% Precision: {val_metrics['top15_precision']:.4f} | Recall: {val_metrics['top15_recall']:.4f} | Lift: {val_metrics['top15_lift']:.4f}\n")

        wandb.log({
            "best_hyperparameters": best_params,
            "best_threshold": self.best_threshold,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # Retrain on combined train + val set with best params
        X_combined = pd.concat([X_train_filtered, X_val_filtered], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)

        final_model = self.model.__class__(**best_params)
        final_model.fit(X_combined, y_combined)
        self.model = final_model

    def _tune_threshold(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """
        Tune probability threshold to maximize F1.

        Args:
            y_true (np.ndarray): True labels.
            y_probs (np.ndarray): Predicted probabilities.

        Returns:
            float: Best threshold.
        """
        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = [f1_score(y_true, (y_probs >= t).astype(int)) for t in thresholds]
        return thresholds[np.argmax(f1_scores)]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
        """
        Evaluate model on test data.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.DataFrame): Test labels.

        Returns:
            dict: Evaluation metrics.
        """

        X_test_filtered = X_test[self.feature_cols]
        y_test = y_test[self.label_col]

        probs = self.model.predict_proba(X_test_filtered)[:, 1]
        preds = (probs >= self.best_threshold).astype(int)

        # Original threshold-based metrics
        metrics = {
            "f1": f1_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds),
            "auc": roc_auc_score(y_test, probs),
            "threshold": self.best_threshold,
        }

        # Top 15% targeting logic
        top_k_percent = 0.15
        n_top = int(len(probs) * top_k_percent)
        sorted_indices = probs.argsort()[::-1]

        top_indices = sorted_indices[:n_top]
        top_y = y_test.iloc[top_indices].values

        positives_in_top = top_y.sum()
        total_positives = y_test.sum()

        top15_precision = positives_in_top / n_top if n_top > 0 else 0
        top15_recall = positives_in_top / total_positives if total_positives > 0 else 0
        baseline_positive_rate = total_positives / len(y_test) if len(y_test) > 0 else 0
        lift = (top15_precision / baseline_positive_rate) if baseline_positive_rate > 0 else 0

        metrics.update({
            "top15_precision": top15_precision,
            "top15_recall": top15_recall,
            "top15_lift": lift,
        })

        # Calibration metrics
        brier = brier_score_loss(y_test, probs)
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy='uniform')
   
        bin_edges = np.linspace(0, 1, 11) 
        bin_indices = np.digitize(probs, bins=bin_edges, right=True) - 1

        unique_bins = np.unique(bin_indices)
        bin_counts = np.array([np.sum(bin_indices == b) for b in unique_bins])

        weights = bin_counts / bin_counts.sum() if bin_counts.sum() > 0 else np.ones_like(bin_counts) / len(bin_counts)

        # Expected Calibration Error (ECE)
        ece = np.sum(weights * np.abs(prob_true - prob_pred))
        metrics.update({
            "brier_score": brier,
            "expected_calibration_error": ece,
        })

        # ROC Curve plotting
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        auc_score = metrics["auc"]

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid()

        # Save plot to file if needed
        plt.savefig("roc_curve.png", bbox_inches="tight")
        plt.show()

        # Optionally log to wandb
        wandb.log({"roc_curve": wandb.Image("roc_curve.png")})
        wandb.log(metrics)

        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        return metrics       

    def predict(self, X: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Predict class labels and probabilities.

        Args:
            X (pd.DataFrame): Feature input.

        Returns:
            tuple: (predicted labels, predicted probabilities)
        """
        X_filtered = X[self.feature_cols]
        probs = self.model.predict_proba(X_filtered)[:, 1]
        preds = (probs >= self.best_threshold).astype(int)
        return preds, probs

    def save_model(self, output_folder: str):
        """
        Save trained model and threshold to disk.

        Args:
            output_folder (str): Output directory.
        """
        os.makedirs(output_folder, exist_ok=True)
        joblib.dump(self.model, os.path.join(output_folder, f"{self.model_name}_model.joblib"))
        with open(os.path.join(output_folder, f"{self.model_name}_threshold.txt"), "w") as f:
            f.write(str(self.best_threshold))

    def close(self):
        """Close the wandb run."""
        wandb.finish()