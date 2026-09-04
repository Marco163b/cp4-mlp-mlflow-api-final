from pathlib import Path
from typing import Union
import json
import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
BEST_MODEL_INFO_PATH = BASE_DIR / "best_model_info.json"

SAMPLE_REQUEST = {
    "features": {
        "alcohol": 13.2,
        "malic_acid": 1.78,
        "ash": 2.14,
        "alcalinity_of_ash": 11.2,
        "magnesium": 100.0,
        "total_phenols": 2.65,
        "flavanoids": 2.76,
        "nonflavanoid_phenols": 0.26,
        "proanthocyanins": 1.28,
        "color_intensity": 4.38,
        "hue": 1.05,
        "od280/od315_of_diluted_wines": 3.4,
        "proline": 1050.0,
    }
}


class PredictionRequest(BaseModel):
    features: Union[list[float], dict[str, float]] = Field(
        ...,
        description="Lista com 13 valores na ordem das features ou objeto com os nomes das features.",
    )

    model_config = {
        "json_schema_extra": {
            "example": SAMPLE_REQUEST,
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    model_run: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 0,
                "prediction_label": "class_0",
                "model_run": "MLP-01",
            }
        }
    }


def load_best_model_info():
    if not BEST_MODEL_INFO_PATH.exists():
        raise RuntimeError(
            "Arquivo best_model_info.json nao encontrado. Execute primeiro: python train.py"
        )

    return json.loads(BEST_MODEL_INFO_PATH.read_text(encoding="utf-8"))


def build_dataframe(features, feature_names):
    if isinstance(features, list):
        if len(features) != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Foram recebidas {len(features)} features, mas o modelo espera {len(feature_names)}.",
            )

        return pd.DataFrame([features], columns=feature_names)

    missing_features = [name for name in feature_names if name not in features]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Features ausentes: {missing_features}",
        )

    row = {name: features[name] for name in feature_names}
    return pd.DataFrame([row])


mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}")
best_model_info = load_best_model_info()
model = mlflow.pyfunc.load_model(best_model_info["model_uri"])

FEATURE_NAMES = best_model_info["feature_names"]
TARGET_NAMES = best_model_info["target_names"]

app = FastAPI(title="CP4 - API de Predicao MLP", version="1.0.0")


@app.get("/")
def home():
    return {
        "message": "API local de predicao do projeto CP4",
        "dataset": best_model_info["dataset"]["name"],
        "best_run": best_model_info["best_run_name"],
        "selection_metric": best_model_info["selection_metric"],
        "docs": "/docs",
    }


@app.get("/features")
def get_features():
    return {
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "sample_request": SAMPLE_REQUEST,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_data = build_dataframe(request.features, FEATURE_NAMES)
    prediction = model.predict(input_data)

    prediction_id = int(prediction[0])
    prediction_label = TARGET_NAMES[prediction_id]

    return {
        "prediction": prediction_id,
        "prediction_label": prediction_label,
        "model_run": best_model_info["best_run_name"],
    }
