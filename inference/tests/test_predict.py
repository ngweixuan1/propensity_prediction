import pytest
import os
import pandas as pd
from predict.predict import ModelPredictor

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

feature_cols = ['ActBal_CA', 'ActBal_SA', 'ActBal_MF', 'ActBal_OVD', 'ActBal_CC',
                'ActBal_CL', 'VolumeCred', 'VolumeCred_CA', 'TransactionsCred',
                'TransactionsCred_CA', 'VolumeDeb', 'VolumeDeb_CA',
                'VolumeDebCash_Card', 'VolumeDebCashless_Card',
                'VolumeDeb_PaymentOrder', 'TransactionsDeb', 'TransactionsDeb_CA',
                'TransactionsDebCash_Card', 'TransactionsDebCashless_Card',
                'TransactionsDeb_PaymentOrder', 'Age', 'Tenure', 'Sex_M']

@pytest.fixture
def predictor():
    model_name = "random_forest"
    return ModelPredictor(model_name=model_name, model_folder=MODEL_DIR, feature_cols=feature_cols)

def test_load_model_and_threshold(predictor):
    assert predictor.model is not None
    assert isinstance(predictor.best_threshold, float)

def test_validate_features_success(predictor):
    df = pd.read_csv(os.path.join(DATA_DIR, 'test_predict.csv'))
    predictor.validate_features(df)

def test_validate_features_missing_column(predictor):
    df = pd.read_csv(os.path.join(DATA_DIR, 'test_predict.csv'))
    df = df.drop(columns=[feature_cols[0]])
    with pytest.raises(ValueError):
        predictor.validate_features(df)

def test_predict_output_shapes(predictor):
    df = pd.read_csv(os.path.join(DATA_DIR, 'test_predict.csv'))
    preds, probs = predictor.predict(df)
    assert len(preds) == len(df)
    assert len(probs) == len(df)
    assert set(preds).issubset({0, 1})
    assert (probs >= 0).all() and (probs <= 1).all()