from __future__ import annotations

import os
import json
import joblib
import pandas as pd
import numpy as np
import uvicorn
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Body

# Artifacts paths
ARTIFACT_DIR = "artifacts"
PIPELINE_PATH = os.path.join(ARTIFACT_DIR, "student_dropout_pipeline_v1.joblib")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler_v1.joblib")
PCA_PATH = os.path.join(ARTIFACT_DIR, "pca_v1.joblib")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgboost_model_v1.joblib")
LABEL_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "target_label_encoder_v1.joblib")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")

app = FastAPI(title="Student Dropout Prediction API")

# ---- CORS Middleware Configuration ----
origins = [
    "*",  # Allows all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# ---- module-level globals (initialized at startup) ----
ARTIFACTS: dict = {}
# ... rest of your code
PIPELINE = None
ESTIMATOR_ONLY = None
LABEL_ENCODER = None
METADATA: dict = {}
FEATURE_NAMES: Optional[List[str]] = None


# -----------------------
# Utilities
# -----------------------
def load_json(path: str) -> Optional[dict]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_artifacts() -> dict:
    artifacts: dict = {}
    # Prefer pipeline if present
    try:
        if os.path.exists(PIPELINE_PATH):
            artifacts["pipeline"] = joblib.load(PIPELINE_PATH)
        else:
            scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
            pca = joblib.load(PCA_PATH) if os.path.exists(PCA_PATH) else None
            model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

            if scaler is not None and pca is not None and model is not None:
                from sklearn.pipeline import Pipeline as SkPipeline
                artifacts["pipeline"] = SkPipeline([("scaler", scaler), ("pca", pca), ("estimator", model)])
            elif model is not None:
                artifacts["estimator_only"] = model
            else:
                raise FileNotFoundError("No usable model artifacts found in 'artifacts' directory.")
    except Exception as exc:
        # re-raise with context to make startup errors clear
        raise RuntimeError(f"Failed to load model artifacts: {exc}") from exc

    # label encoder & metadata (optional)
    artifacts["label_encoder"] = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None
    artifacts["metadata"] = load_json(METADATA_PATH) or {}
    return artifacts


def ensure_dataframe_from_input(data: Any, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Accept dict / list[dict] / DataFrame and return DataFrame aligned to feature_names.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported input type. Provide dict, list of dicts, or pandas.DataFrame.")

    if feature_names is not None:
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        # Reorder and keep only required columns
        df = df[feature_names]

    # Attempt to coerce object columns that look numeric
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except Exception:
                # keep original if cannot convert; model will raise meaningful error later
                pass

    return df


def predict_from_pipeline(pipeline, df: pd.DataFrame, label_encoder=None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        probs = pipeline.predict_proba(df)
    except Exception:
        probs = None

    preds = pipeline.predict(df)
    for i, pred in enumerate(preds):
        r: Dict[str, Any] = {"prediction_index": int(pred)}
        if probs is not None:
            r["raw_probs"] = probs[i].tolist()
            r["probability"] = float(probs[i].max())
        else:
            r["raw_probs"] = None
            r["probability"] = None

        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            try:
                r["prediction_label"] = str(label_encoder.inverse_transform([pred])[0])
            except Exception:
                classes = list(label_encoder.classes_)
                r["prediction_label"] = classes[int(pred)] if 0 <= int(pred) < len(classes) else None
        results.append(r)
    return results


def predict_from_estimator_only(estimator, df_transformed: np.ndarray, label_encoder=None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        probs = estimator.predict_proba(df_transformed)
    except Exception:
        probs = None
    preds = estimator.predict(df_transformed)
    for i, pred in enumerate(preds):
        r: Dict[str, Any] = {"prediction_index": int(pred)}
        if probs is not None:
            r["raw_probs"] = probs[i].tolist()
            r["probability"] = float(probs[i].max())
        else:
            r["raw_probs"] = None
            r["probability"] = None
        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            try:
                r["prediction_label"] = str(label_encoder.inverse_transform([pred])[0])
            except Exception:
                classes = list(label_encoder.classes_)
                r["prediction_label"] = classes[int(pred)] if 0 <= int(pred) < len(classes) else None
        results.append(r)
    return results


# -----------------------
# App startup: load artifacts once
# -----------------------
@app.on_event("startup")
def startup_load():
    global ARTIFACTS, PIPELINE, ESTIMATOR_ONLY, LABEL_ENCODER, METADATA, FEATURE_NAMES
    ARTIFACTS = {}
    try:
        ARTIFACTS = load_artifacts()
        PIPELINE = ARTIFACTS.get("pipeline", None)
        ESTIMATOR_ONLY = ARTIFACTS.get("estimator_only", None)
        LABEL_ENCODER = ARTIFACTS.get("label_encoder", None)
        METADATA = ARTIFACTS.get("metadata", {}) or {}
        FEATURE_NAMES = METADATA.get("feature_names", None)

        if FEATURE_NAMES is None:
            FEATURE_NAMES = [
                "Application order", "Inflation rate", "Application mode", "GDP", "Unemployment rate", "Course",
                "Curricular units 1st sem (evaluations)", "Curricular units 2nd sem (evaluations)",
                "Age at enrollment", "Admission grade",
                "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
                "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (approved)"
            ]
    except Exception as exc:
        # Fail fast with an informative error so the server doesn't silently run without model
        raise RuntimeError(f"Failed to load artifacts during startup: {exc}") from exc


# -----------------------
# Request/Response models
# -----------------------
class PredictResponse(BaseModel):
    prediction_index: int
    raw_probs: Optional[List[float]] = None
    probability: Optional[float] = None
    prediction_label: Optional[str] = None


# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "details": {
            "pipeline_loaded": PIPELINE is not None,
            "estimator_only": ESTIMATOR_ONLY is not None,
            "feature_names_count": len(FEATURE_NAMES) if FEATURE_NAMES is not None else None,
        },
    }

@app.get("/")
def read_root():
    return {"message": "FastAPI running on Vercel"}

@app.post("/predict", response_model=List[PredictResponse])
def predict(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]] = Body(
        ...,
        example={
            "Application order": 1,
            "Inflation rate": 2.5,
            "Application mode": 5,
            "GDP": 1.8,
            "Unemployment rate": 7.4,
            "Course": 9500,
            "Curricular units 1st sem (evaluations)": 6,
            "Curricular units 2nd sem (evaluations)": 5,
            "Age at enrollment": 20,
            "Admission grade": 145,
            "Curricular units 1st sem (approved)": 5,
            "Curricular units 1st sem (grade)": 13.5,
            "Curricular units 2nd sem (grade)": 14.0,
            "Curricular units 2nd sem (approved)": 4,
        },
    )
) -> List[PredictResponse]:
    # Normalize input to list[dict]
    if isinstance(payload, dict):
        instances = [payload]
    elif isinstance(payload, list):
        instances = payload
    else:
        raise HTTPException(status_code=400, detail="Payload must be a dict or list of dicts.")

    # Validate expected features present
    if FEATURE_NAMES:
        missing_any = []
        for idx, inst in enumerate(instances):
            missing = [c for c in FEATURE_NAMES if c not in inst]
            if missing:
                missing_any.append({"index": idx, "missing": missing})
        if missing_any:
            raise HTTPException(status_code=422, detail={"error": "Missing required features", "details": missing_any})

    # Build DataFrame
    try:
        df_in = ensure_dataframe_from_input(instances, feature_names=FEATURE_NAMES)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Prediction path
    try:
        if PIPELINE is not None:
            out = predict_from_pipeline(PIPELINE, df_in, label_encoder=LABEL_ENCODER)
            return out
        elif ESTIMATOR_ONLY is not None:
            # need scaler+pca
            if os.path.exists(SCALER_PATH) and os.path.exists(PCA_PATH):
                scaler = joblib.load(SCALER_PATH)
                pca = joblib.load(PCA_PATH)
                X_scaled = scaler.transform(df_in)
                X_pca = pca.transform(X_scaled)
                out = predict_from_estimator_only(ESTIMATOR_ONLY, X_pca, label_encoder=LABEL_ENCODER)
                return out
            else:
                raise HTTPException(status_code=500, detail="Estimator-only present but scaler/pca missing.")
        else:
            raise HTTPException(status_code=500, detail="No model pipeline or estimator available.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


# Run with: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000,  reload=True)

