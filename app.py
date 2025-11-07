# app.py
from __future__ import annotations

import os
import json
import joblib
import pandas as pd
import numpy as np
import uvicorn
import traceback
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import shap  # explainability

# --------------------
# ✅ FIXED: CORS SETUP
# --------------------
app = FastAPI(title="Student Dropout Prediction API with Explainability")

# Allow your local and deployed frontend
origins = [
    "http://localhost:3000",
    "https://cs-698-frontend-asgn-5.vercel.app",
]

# ⚠️ Must come immediately after FastAPI() and before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # only these domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Artifacts paths ----
ARTIFACT_DIR = "artifacts"
PIPELINE_PATH = os.path.join(ARTIFACT_DIR, "student_dropout_pipeline_v1.joblib")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler_v1.joblib")
PCA_PATH = os.path.join(ARTIFACT_DIR, "pca_v1.joblib")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgboost_model_v1.joblib")
LABEL_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "target_label_encoder_v1.joblib")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")
BACKGROUND_DATA_PATH = os.path.join(ARTIFACT_DIR, "background_data_sample.csv")


# ---- module-level globals (initialized at startup) ----
ARTIFACTS: dict = {}
PIPELINE = None
ESTIMATOR_ONLY = None
LABEL_ENCODER = None
METADATA: dict = {}
FEATURE_NAMES: Optional[List[str]] = None
BACKGROUND_DATA: Optional[pd.DataFrame] = None
SHAP_EXPLAINER: Optional[shap.Explainer] = None
CLASS_NAMES: List[str] = ["Class 0", "Class 1"]  # Default, will be overwritten if label encoder exists


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
                # don't raise here; return empty artifacts and let endpoints handle absence gracefully
                artifacts["pipeline"] = None
                artifacts["estimator_only"] = None
    except Exception as exc:
        # log and continue — avoid crashing startup so health endpoints work
        print(f"Warning: Failed to load some model artifacts: {exc}")
        traceback.print_exc()

    # label encoder & metadata (optional)
    try:
        artifacts["label_encoder"] = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None
    except Exception as exc:
        print(f"Warning: could not load label encoder: {exc}")
        artifacts["label_encoder"] = None

    artifacts["metadata"] = load_json(METADATA_PATH) or {}

    # --- NEW: Load SHAP background data ---
    if os.path.exists(BACKGROUND_DATA_PATH):
        try:
            artifacts["background_data"] = pd.read_csv(BACKGROUND_DATA_PATH)
        except Exception as exc:
            print(f"Warning: failed to read background data CSV: {exc}")
            artifacts["background_data"] = None
    else:
        artifacts["background_data"] = None

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
            if label_encoder is not None and hasattr(label_encoder, "classes_"):
                r["class_probabilities"] = {str(label_encoder.classes_[j]): float(prob) for j, prob in enumerate(probs[i])}
            else:
                r["class_probabilities"] = {f"Class {j}": float(prob) for j, prob in enumerate(probs[i])}
        else:
            r["raw_probs"] = None
            r["probability"] = None
            r["class_probabilities"] = None

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
            if label_encoder is not None and hasattr(label_encoder, "classes_"):
                r["class_probabilities"] = {str(label_encoder.classes_[j]): float(prob) for j, prob in enumerate(probs[i])}
            else:
                r["class_probabilities"] = {f"Class {j}": float(prob) for j, prob in enumerate(probs[i])}
        else:
            r["raw_probs"] = None
            r["probability"] = None
            r["class_probabilities"] = None
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
    global BACKGROUND_DATA, SHAP_EXPLAINER, CLASS_NAMES

    ARTIFACTS = {}
    try:
        ARTIFACTS = load_artifacts()
        PIPELINE = ARTIFACTS.get("pipeline", None)
        ESTIMATOR_ONLY = ARTIFACTS.get("estimator_only", None)
        LABEL_ENCODER = ARTIFACTS.get("label_encoder", None)
        METADATA = ARTIFACTS.get("metadata", {}) or {}
        FEATURE_NAMES = METADATA.get("feature_names", None)
        BACKGROUND_DATA = ARTIFACTS.get("background_data", None)

        if LABEL_ENCODER is not None and hasattr(LABEL_ENCODER, "classes_"):
            CLASS_NAMES = list(LABEL_ENCODER.classes_)

        if FEATURE_NAMES is None:
            if BACKGROUND_DATA is not None:
                FEATURE_NAMES = list(BACKGROUND_DATA.columns)
            else:
                FEATURE_NAMES = [
                    "Application order", "Inflation rate", "Application mode", "GDP", "Unemployment rate", "Course",
                    "Curricular units 1st sem (evaluations)", "Curricular units 2nd sem (evaluations)",
                    "Age at enrollment", "Admission grade",
                    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
                    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (approved)"
                ]

        # Initialize SHAP Explainer only if pipeline & background provided
        if PIPELINE is not None and BACKGROUND_DATA is not None:
            try:
                print("Initializing SHAP Explainer...")
                background_df_aligned = BACKGROUND_DATA[FEATURE_NAMES]
                masker = shap.maskers.Tabular(background_df_aligned, hclustering=False)
                SHAP_EXPLAINER = shap.Explainer(PIPELINE.predict_proba, masker)
                print("SHAP Explainer initialized successfully.")
            except Exception as exc:
                print(f"Warning: Failed to initialize SHAP explainer: {exc}")
                traceback.print_exc()
                SHAP_EXPLAINER = None
        else:
            print("Warning: SHAP Explainer not initialized. Missing pipeline or background data.")
    except Exception as exc:
        # Don't crash the server on startup; log for debugging and continue
        print(f"Warning: Exception during startup artifact load: {exc}")
        traceback.print_exc()


# -----------------------
# Request/Response models
# -----------------------
class PredictResponse(BaseModel):
    prediction_index: int
    raw_probs: Optional[List[float]] = None
    probability: Optional[float] = None
    prediction_label: Optional[str] = None
    class_probabilities: Optional[Dict[str, float]] = None


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
            "shap_explainer_ready": SHAP_EXPLAINER is not None,
            "background_data_rows": len(BACKGROUND_DATA) if BACKGROUND_DATA is not None else 0
        },
    }


@app.get("/")
def read_root():
    return {"message": "Student Dropout Prediction API with full explainability suite."}


# --- NEW: GLOBAL EXPLANATION (REPORT) ----
@app.get("/model_info")
def get_model_info():
    """
    Safe endpoint returning metadata; provides fallback values if metadata.json missing
    to avoid 5xx/502 responses that break CORS in clients.
    """
    global METADATA
    try:
        if not METADATA:
            # fallback metadata so endpoint always responds
            METADATA = {
                "data_input_description": "Student academic and macroeconomic features (fallback).",
                "model_output_description": "Predicted probability of graduation or dropout (fallback).",
                "model_performance": {
                    "test_accuracy": None,
                    "f1_score_weighted": None,
                    "notes": "Fallback placeholder: update artifacts/metadata.json for real metrics."
                },
                "model_how_it_works": "XGBoost classifier ensemble (fallback description).",
                "feature_names": FEATURE_NAMES or []
            }
        return {
            "global_explanation_report": {
                "INPUT": METADATA.get("data_input_description", "No description provided."),
                "OUTPUT": METADATA.get("model_output_description", "No description provided."),
                "PERFORMANCE": METADATA.get("model_performance", "No performance metrics provided."),
                "HOW": METADATA.get("model_how_it_works", "No high-level description provided."),
            },
            "metadata": METADATA
        }
    except Exception as exc:
        # Return a 500 only if something truly unexpected happens, but include readable message
        raise HTTPException(status_code=500, detail=f"Failed to return model info: {exc}")


@app.post("/predict", response_model=List[PredictResponse])
def predict(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]] = Body(...),
) -> List[PredictResponse]:
    if isinstance(payload, dict):
        instances = [payload]
    elif isinstance(payload, list):
        instances = payload
    else:
        raise HTTPException(status_code=400, detail="Payload must be a dict or list of dicts.")

    if FEATURE_NAMES:
        missing_any = []
        for idx, inst in enumerate(instances):
            missing = [c for c in FEATURE_NAMES if c not in inst]
            if missing:
                missing_any.append({"index": idx, "missing": missing})
        if missing_any:
            raise HTTPException(status_code=422, detail={"error": "Missing required features", "details": missing_any})

    try:
        df_in = ensure_dataframe_from_input(instances, feature_names=FEATURE_NAMES)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        if PIPELINE is not None:
            out = predict_from_pipeline(PIPELINE, df_in, label_encoder=LABEL_ENCODER)
            return out
        elif ESTIMATOR_ONLY is not None:
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


# ---- GLOBAL EXPLANATION (FEATURE IMPORTANCE) ----
@app.get("/global_explanation")
def global_explanation():
    """
    Returns global feature importances for the model.
    """
    model = None
    if PIPELINE is not None:
        model = PIPELINE.named_steps.get("estimator", None) or PIPELINE.named_steps.get("model")
    if model is None:
        model = ESTIMATOR_ONLY

    if model is None:
        raise HTTPException(status_code=500, detail="No model found to compute global explanation.")

    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise AttributeError("Model does not expose 'feature_importances_'")

        # Handle PCA component case
        if PIPELINE is not None and "pca" in PIPELINE.named_steps:
            pca = PIPELINE.named_steps["pca"]
            n_components = getattr(pca, "n_components_", None)
            if n_components and len(importances) == n_components:
                component_importance = sorted(
                    [{"feature": f"PCA_Component_{i}", "importance": float(imp)} for i, imp in enumerate(importances)],
                    key=lambda x: x["importance"], reverse=True
                )
                return {
                    "model_type": type(model).__name__,
                    "explanation_type": "Global (PCA Component Importance)",
                    "note": "Model uses PCA; importances are for components.",
                    "top_features": component_importance[:5],
                    "feature_importances": component_importance,
                }

        if FEATURE_NAMES is None or len(FEATURE_NAMES) != len(importances):
            feature_importance = [{"feature": f"feature_{i}", "importance": float(v)} for i, v in enumerate(importances)]
        else:
            feature_importance = sorted(
                [{"feature": f, "importance": float(i)} for f, i in zip(FEATURE_NAMES, importances)],
                key=lambda x: x["importance"], reverse=True
            )

        top_features = feature_importance[:5]
        return {
            "model_type": type(model).__name__,
            "goal": "Predict student graduation/dropout",
            "top_features": top_features,
            "explanation_type": "Global (Feature Importance)",
            "how_it_helps": "Shows which features influence predictions most across all students.",
            "feature_importances": feature_importance,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Global explanation failed: {e}")


# ---- LOCAL EXPLANATION (WHY / WHY NOT) ----
@app.post("/local_explanation")
def local_explanation(payload: Dict[str, Any] = Body(...)):
    if SHAP_EXPLAINER is None:
        raise HTTPException(status_code=503, detail="SHAP Explainer is not available. Check server startup logs.")

    if isinstance(payload, list):
        payload_instance = payload[0] if len(payload) > 0 else {}
    elif isinstance(payload, dict):
        payload_instance = payload
    else:
        raise HTTPException(status_code=400, detail="Payload must be a dict or a single-item list of dicts.")

    if not payload_instance:
        raise HTTPException(status_code=422, detail="Payload is empty.")

    if FEATURE_NAMES:
        missing = [c for c in FEATURE_NAMES if c not in payload_instance]
        if missing:
            raise HTTPException(status_code=422, detail={"error": "Missing required features", "missing": missing})

    try:
        df_in = ensure_dataframe_from_input(payload_instance, feature_names=FEATURE_NAMES)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        shap_values = SHAP_EXPLAINER(df_in)
        instance_values_by_class = shap_values.values[0]
        instance_base_values = shap_values.base_values[0]

        explanations_by_class = []
        for i, class_name in enumerate(CLASS_NAMES):
            impacts = instance_values_by_class[:, i]
            base_value = instance_base_values[i]

            feature_impacts = [{"feature": f, "impact": float(v)} for f, v in zip(FEATURE_NAMES, impacts)]
            sorted_impacts = sorted(feature_impacts, key=lambda x: abs(x["impact"]), reverse=True)

            explanations_by_class.append({
                "class_name": class_name,
                "base_value": float(base_value),
                "top_5_features": sorted_impacts[:5],
                "all_impacts": feature_impacts
            })

        return {
            "explanation_type": "Local (SHAP values per class)",
            "instance_input": payload_instance,
            "class_names": CLASS_NAMES,
            "explanations": explanations_by_class,
            "how_it_helps": "Shows the impact of each feature on the probability of each outcome."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare local explanation: {e}")


# ---- ACTIONABLE EXPLANATIONS ----
@app.post("/actionable_explanations")
def actionable_explanations(
    payload: Dict[str, Any] = Body(...),
    target_class: str = "Graduate",
    undesirable_class: str = "Dropout"
):
    if SHAP_EXPLAINER is None:
        raise HTTPException(status_code=503, detail="SHAP Explainer is not available. Check server startup logs.")

    try:
        df_in = ensure_dataframe_from_input(payload, feature_names=FEATURE_NAMES)
        prediction_result = predict_from_pipeline(PIPELINE, df_in, LABEL_ENCODER)[0]
        pred_label = prediction_result.get("prediction_label")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction for actionable advice: {e}")

    try:
        shap_values = SHAP_EXPLAINER(df_in)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get SHAP values for actionable advice: {e}")

    try:
        if pred_label == undesirable_class:
            try:
                target_class_index = CLASS_NAMES.index(target_class)
            except ValueError:
                return {"recommendation_type": "Error", "message": f"Target class '{target_class}' not found in model classes: {CLASS_NAMES}"}

            impacts = shap_values.values[0, :, target_class_index]
            feature_impacts = [{"feature": f, "impact": float(v)} for f, v in zip(FEATURE_NAMES, impacts)]
            sorted_negative_impacts = sorted([f for f in feature_impacts if f["impact"] < 0], key=lambda x: x["impact"])
            suggestions = [f"Improve '{f['feature']}' (it strongly decreased your '{target_class}' score)" for f in sorted_negative_impacts[:3]]
            if not suggestions:
                suggestions = ["Your profile is complex. No simple changes are apparent."]
            return {
                "recommendation_type": "How to Improve (Counterfactual)",
                "current_prediction": pred_label,
                "target_prediction": target_class,
                "message": f"To improve your chances of '{target_class}', focus on these areas:",
                "suggestions": suggestions
            }

        elif pred_label == target_class:
            try:
                target_class_index = CLASS_NAMES.index(target_class)
            except ValueError:
                return {"recommendation_type": "Error", "message": f"Target class '{target_class}' not found in model classes: {CLASS_NAMES}"}

            impacts = shap_values.values[0, :, target_class_index]
            feature_impacts = [{"feature": f, "impact": float(v)} for f, v in zip(FEATURE_NAMES, impacts)]
            sorted_weakest_links = sorted(feature_impacts, key=lambda x: x["impact"])
            suggestions = [f"Maintain your performance in '{f['feature']}' (it's your weakest positive factor or a negative factor)" for f in sorted_weakest_links[:3]]
            if not suggestions:
                suggestions = ["Your profile is strong across the board."]
            return {
                "recommendation_type": "How to Maintain (Robustness)",
                "current_prediction": pred_label,
                "message": f"Your '{target_class}' prediction is strong. To maintain it, be mindful of these areas:",
                "suggestions": suggestions
            }

        else:
            return {
                "recommendation_type": "General",
                "current_prediction": pred_label,
                "message": f"No specific actions defined for the '{pred_label}' status."
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate actionable advice: {e}")


# Run with: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
