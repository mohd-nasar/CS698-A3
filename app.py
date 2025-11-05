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
from fastapi.middleware.cors import CORSMiddleware
import shap  # explainability

# Artifacts paths
ARTIFACT_DIR = "artifacts"
PIPELINE_PATH = os.path.join(ARTIFACT_DIR, "student_dropout_pipeline_v1.joblib")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler_v1.joblib")
PCA_PATH = os.path.join(ARTIFACT_DIR, "pca_v1.joblib")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgboost_model_v1.joblib")
LABEL_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "target_label_encoder_v1.joblib")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")

app = FastAPI(title="Student Dropout Prediction API with Explainability")

# ---- CORS Middleware Configuration ----
origins = [
    "*",  # Allows all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ---- module-level globals (initialized at startup) ----
ARTIFACTS: dict = {}
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
    return {"message": "FastAPI running with explainability endpoints"}


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


# ---- GLOBAL EXPLANATION ----
@app.get("/global_explanation")
def global_explanation():
    """
    Returns global feature importances for the model.
    Works when the underlying model exposes `feature_importances_` (e.g., XGBoost).
    """
    # find underlying estimator
    model = None
    if PIPELINE is not None:
        # pipeline may have 'estimator' or 'model' step
        model = PIPELINE.named_steps.get("estimator", None) or PIPELINE.named_steps.get("model")
    if model is None:
        model = ESTIMATOR_ONLY

    if model is None:
        raise HTTPException(status_code=500, detail="No model found to compute global explanation.")

    # Attempt to get feature importances
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise AttributeError("Model does not expose 'feature_importances_'")
        # ensure feature names length matches
        if FEATURE_NAMES is None or len(FEATURE_NAMES) != len(importances):
            # If lengths mismatch, include limited mapping but warn user.
            feature_importance = [
                {"feature": f"feature_{i}", "importance": float(v)}
                for i, v in enumerate(importances)
            ]
        else:
            feature_importance = sorted(
                [
                    {"feature": f, "importance": float(i)}
                    for f, i in zip(FEATURE_NAMES, importances)
                ],
                key=lambda x: x["importance"], reverse=True
            )
        top_features = feature_importance[:5]
        return {
            "model_type": type(model).__name__,
            "goal": "Predict student graduation/dropout",
            "top_features": top_features,
            "explanation_type": "Global",
            "how_it_helps": "Shows which features influence predictions most across all students.",
            "feature_importances": feature_importance,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Global explanation failed: {e}")


# ---- LOCAL EXPLANATION ----
@app.post("/local_explanation")
def local_explanation(payload: Dict[str, Any] = Body(...)):
    """
    Produces a local (per-instance) explanation using SHAP.
    If a full pipeline is present, explanations are computed w.r.t. original input features.
    If only an estimator is present and scaler/pca exist, local explanations are not supported
    because shap values would be relative to transformed components.
    """
    # Accept a single instance (dict) or list-of-one
    if isinstance(payload, list):
        if len(payload) == 0:
            raise HTTPException(status_code=422, detail="Payload list must contain at least one instance.")
        payload_instance = payload[0]
    elif isinstance(payload, dict):
        payload_instance = payload
    else:
        raise HTTPException(status_code=400, detail="Payload must be a dict or a single-item list of dicts.")

    # Validate required features
    if FEATURE_NAMES:
        missing = [c for c in FEATURE_NAMES if c not in payload_instance]
        if missing:
            raise HTTPException(status_code=422, detail={"error": "Missing required features", "missing": missing})

    try:
        df_in = ensure_dataframe_from_input(payload_instance, feature_names=FEATURE_NAMES)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Determine explainability path
    if PIPELINE is not None:
        # Explain the pipeline's prediction function w.r.t original input features
        try:
            explainer = shap.Explainer(PIPELINE.predict, df_in)
            shap_values = explainer(df_in)
            instance_values = shap_values[0].values  # shape: (n_features,)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Local explanation (pipeline) failed: {e}")
    else:
        # If only estimator exists, we can only explain in transformed space (PCA), which isn't helpful for original features.
        if ESTIMATOR_ONLY is not None and os.path.exists(SCALER_PATH) and os.path.exists(PCA_PATH):
            raise HTTPException(
                status_code=422,
                detail="Local explanations require the full pipeline. Estimator-only present; scaler/pca available but shap would explain PCA components, not original features. Provide a pipeline artifact or the pipeline joblib file."
            )
        else:
            raise HTTPException(status_code=500, detail="No pipeline available for local explanation.")

    # Map shap values to feature names and build sorted impacts
    try:
        if len(instance_values) != len(FEATURE_NAMES):
            # fallback: use numbered features if mismatch
            feature_list = [f"feature_{i}" for i in range(len(instance_values))]
        else:
            feature_list = FEATURE_NAMES

        explanation = [
            {"feature": f, "impact": float(v)}
            for f, v in zip(feature_list, instance_values)
        ]

        # Top positive (largest positive impacts)
        pos_sorted = sorted([e for e in explanation], key=lambda x: x["impact"], reverse=True)
        top_positive = [p for p in pos_sorted if p["impact"] > 0][:3]
        # If none positive, return the top-3 by absolute impact
        if not top_positive:
            top_positive = pos_sorted[:3]

        # Top negative (most negative impacts)
        neg_sorted = sorted([e for e in explanation], key=lambda x: x["impact"])
        top_negative = [n for n in neg_sorted if n["impact"] < 0][:3]
        if not top_negative:
            top_negative = neg_sorted[:3]

        return {
            "explanation_type": "Local",
            "instance_input": payload_instance,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "all_impacts": explanation,
            "how_it_helps": "Shows which features most increased or decreased the dropout probability for this student."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare local explanation: {e}")


# Run with: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
