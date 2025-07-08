
import os
import yaml
import pandas as pd

from data_processing.process import PreprocessingConfig, DataPreprocessor
from train.train import ClassifierTrainer, ModelHyperparams, load_wandb_config


def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(
    preprocessing_config_path: str = "configs/config.yaml",
    wandb_config_path: str = "configs/config.yaml",
    model_name: str = "xgboost",
    label_col: str = "Sale_CC",
    output_dir: str = "models/"
):
    # Load configs
    preprocess_cfg_dict = load_config(preprocessing_config_path)
    wandb_cfg = load_wandb_config(wandb_config_path)
    preprocessing_config = PreprocessingConfig(**preprocess_cfg_dict)

    # Data Preprocessing
    preprocessor = DataPreprocessor(preprocessing_config)
    preprocessor.load_and_merge()
    df_processed = preprocessor.preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()

    feature_cols = [col for col in X_train.columns]

    # Model Training
    trainer = ClassifierTrainer(
        model_name=model_name,
        config=wandb_cfg.hyperparameters,
        feature_cols=feature_cols,
        label_col=label_col,
        wandb_config=wandb_cfg,
    )

    trainer.tune_hyperparameters(X_train, y_train, X_val, y_val)
    trainer.evaluate(X_test, y_test)
    trainer.save_model(output_dir)
    trainer.close()


if __name__ == "__main__":
    run_pipeline()