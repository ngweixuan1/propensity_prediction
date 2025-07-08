import pytest
import pandas as pd
from train.train import ClassifierTrainer, ModelHyperparams, WandbConfig
from data_processing.process import DataPreprocessor, PreprocessingConfig

@pytest.fixture(scope="module")
def data_and_config():
    config = PreprocessingConfig(file_path="data/DataScientist_CaseStudy_Dataset.xlsx")
    dp = DataPreprocessor(config)
    dp.load_and_merge()
    df = dp.preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = dp.split_data()
    return X_train, X_val, X_test, y_train, y_val, y_test, df

@pytest.fixture
def model_config():
    return ModelHyperparams(
        random_forest={'n_estimators': [10], 'max_depth': [3]}
    )

@pytest.fixture
def wandb_config(model_config):
    return WandbConfig(
        project_name="unit_test_project",
        run_name="rf_test_run",
        hyperparameters=model_config,
    )

@pytest.fixture(autouse=True)
def mock_wandb(monkeypatch):
    monkeypatch.setattr("wandb.init", lambda *a, **k: type("WandbMock", (), {"log": lambda *a, **k: None, "finish": lambda: None}))
    monkeypatch.setattr("wandb.Image", lambda *a, **k: None)

def test_trainer_pipeline(data_and_config, model_config, wandb_config):
    X_train, X_val, X_test, y_train, y_val, y_test, df = data_and_config

    trainer = ClassifierTrainer(
        model_name="random_forest",
        config=model_config,
        feature_cols=[col for col in X_train.columns if col != "Client"],
        label_col="Sale_MF",
        wandb_config=wandb_config,
    )

    trainer.tune_hyperparameters(X_train, y_train, X_val, y_val)
    metrics = trainer.evaluate(X_test, y_test)

    assert "f1" in metrics
    assert 0.0 <= metrics["f1"] <= 1.0
    assert "top15_precision" in metrics
    assert "expected_calibration_error" in metrics