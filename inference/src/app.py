# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import yaml

from predict.predict import ModelPredictor
from optimization.optimize import optimization_selection

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_info = config["model"]["folders"] 
FEATURE_COLS = config["model"]["feature_cols"]

# Load all models into a dict keyed by model_name
predictors = {}
for key, info in model_info.items():
    folder = info["folder_path"]
    name = info["model_name"]
    predictors[name] = ModelPredictor(
        model_name=name,
        model_folder=folder,
        feature_cols=FEATURE_COLS
    )

app = FastAPI()

class PredictRequest(BaseModel):
    data: List[Dict]

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        df = pd.DataFrame(request.data)
        all_probs = {}
        for model_name, predictor in predictors.items():
            _, probs = predictor.predict(df)
            all_probs[model_name] = probs.tolist()
        return {"predicted_probabilities": all_probs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
def optimize(request: PredictRequest):
    try:
        df = pd.DataFrame(request.data)
        offers_df = optimization_selection(df)
        return offers_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
