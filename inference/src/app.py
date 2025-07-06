from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml
import os
import numpy as np

# XGBoost imports
import xgboost as xgb

# Sklearn imports for RandomForest
import joblib

app = FastAPI()

# Load config on startup
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

model_type = config.get('model_type', '').lower()
model_path = config.get('model_path', '')

if not model_type or not model_path:
    raise RuntimeError("Config file must specify 'model_type' and 'model_path'")

# Load the appropriate model
if model_type == 'xgboost':
    model = xgb.Booster()
    model.load_model(model_path)
elif model_type == 'randomforest':
    model = joblib.load(model_path)
else:
    raise RuntimeError(f"Unsupported model_type '{model_type}' in config")

# Input data model
class ModelInput(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: ModelInput):
    features = np.array([data.features])

    if model_type == 'xgboost':
        dmatrix = xgb.DMatrix(features)
        preds = model.predict(dmatrix)
        prediction = preds[0].item()
    elif model_type == 'randomforest':
        preds = model.predict(features)
        prediction = preds[0].item()
    else:
        raise HTTPException(status_code=500, detail="Model type unsupported")

    return {"prediction": prediction}