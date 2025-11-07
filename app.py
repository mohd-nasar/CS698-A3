# app.py
from __future__ import annotations

import os
import json
import joblib
import pandas as pd
import numpy as np
import uvicorn
import traceback
import logging
import math
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.encoders import jsonable_encoder
import shap

# Basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("student-dropout-api")

# FastAPI app
app = FastAPI(title="Student Dropout Prediction API with Explainability")

# --- CORS: allow all for debugging; restrict in production ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- change to specific origins for production
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

# ---- Globals filled at startup
ARTIFACTS: dict = {}
PIPELINE = None
ESTIMATOR_ONLY = None
LABEL_ENCODER = None
METADATA: dict = {}
FEATURE_NAMES: Optional[List[str]] = None
BACKGROUND_DATA: Optional[pd.DataFrame] = None
SHAP_EXPLAINER: Optional[Any] = None   # shap.Explainer or TreeExplainer
CLASS_NAMES: List[str] = ["Class 0", "Class 1"]


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
    try:
        if os.path.exists(PIPELINE_PATH):
            logger.info("Loading pipeline from %s", PIPELINE_PATH)
            artifacts["pipeline"] = joblib.load(PIPELINE_PATH)
        else:
            scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
            pca = joblib.load(PCA_PATH) if os.path.exists(PCA_PATH) else None
            model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

            if scaler is not None and pca is not None and model is not None:
                from sklearn.pipeline import Pipeline as SkPipeline
                logger.info("Building pipeline from scaler+pca+model")
                artifacts["pipeline"] = SkPipeline([("scaler", scaler), ("pca", pca), ("estimator", model)])
            elif model is not None:
                logger.info("Loaded estimator-only model from %s", MODEL_PATH)
                artifacts["estimator_only"] = model
            else:
                artifacts["pipeline"] = None
                artifacts["estimator_only"] = None
    except Exception as exc:
        logger.exception("Failed to load model artifacts: %s", exc)

    try:
        artifacts["label_encoder"] = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None
        if artifacts["label_encoder"] is not None:
            logger.info("Loaded label encoder")
    except Exception as exc:
        logger.exception("Could not load label encoder: %s", exc)
        artifacts["label_encoder"] = None

    artifacts["metadata"] = load_json(METADATA_PATH) or {}

    if os.path.exists(BACKGROUND_DATA_PATH):
        try:
            artifacts["background_data"] = pd.read_csv(BACKGROUND_DATA_PATH)
            logger.info("Loaded background data with shape %s", artifacts["background_data"].shape)
        except Exception as exc:
            logger.exception("Failed to read background data CSV: %s", exc)
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
        df = df[feature_names]

    # Try coerce object columns that look numeric
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except Exception:
                pass

    return df


def predict_from_pipeline(pipeline, df: pd.DataFrame, label_encoder=None) -> List[Dict[str, Any]]:
    """
    Return list of prediction dicts (index, probs, label, etc.). Converts numpy types to python.
    """
    results: List[Dict[str, Any]] = []
    probs = None
    try:
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(df)
    except Exception:
        # ignore, probs remain None
        logger.exception("predict_proba on pipeline failed or unsupported.")
        probs = None

    try:
        preds = pipeline.predict(df)
    except Exception as exc:
        logger.exception("predict on pipeline failed: %s", exc)
        raise

    for i, pred in enumerate(preds):
        r: Dict[str, Any] = {"prediction_index": int(pred)}
        if probs is not None:
            row_probs = probs[i]
            # safe conversions
            r["raw_probs"] = [None if (np.isnan(x) or np.isinf(x)) else float(x) for x in row_probs.tolist()]
            maxprob = max([x for x in r["raw_probs"] if x is not None], default=None)
            r["probability"] = maxprob
            if label_encoder is not None and hasattr(label_encoder, "classes_"):
                r["class_probabilities"] = {
                    str(label_encoder.classes_[j]): r["raw_probs"][j] for j in range(len(r["raw_probs"]))
                }
            else:
                r["class_probabilities"] = {f"Class {j}": r["raw_probs"][j] for j in range(len(r["raw_probs"]))}
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
    probs = None
    try:
        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(df_transformed)
    except Exception:
        logger.exception("predict_proba on estimator failed.")
        probs = None
    try:
        preds = estimator.predict(df_transformed)
    except Exception:
        logger.exception("predict on estimator failed.")
        raise

    for i, pred in enumerate(preds):
        r: Dict[str, Any] = {"prediction_index": int(pred)}
        if probs is not None:
            row_probs = probs[i]
            r["raw_probs"] = [None if (np.isnan(x) or np.isinf(x)) else float(x) for x in row_probs.tolist()]
            maxprob = max([x for x in r["raw_probs"] if x is not None], default=None)
            r["probability"] = maxprob
            if label_encoder is not None and hasattr(label_encoder, "classes_"):
                r["class_probabilities"] = {
                    str(label_encoder.classes_[j]): r["raw_probs"][j] for j in range(len(r["raw_probs"]))
                }
            else:
                r["class_probabilities"] = {f"Class {j}": r["raw_probs"][j] for j in range(len(r["raw_probs"]))}
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
# Sanitizer for JSON-safe responses
# -----------------------
def sanitize_for_json(obj: Any) -> Any:
    """Convert numpy/pandas types to Python primitives and replace NaN/Inf with None."""
    # primitives
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj

    # arrays and series
    if isinstance(obj, (np.ndarray, pd.Series)):
        return [sanitize_for_json(v) for v in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return [sanitize_for_json(row) for _, row in obj.to_dict(orient="records")]

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]

    # fallback to jsonable_encoder then sanitize again
    try:
        encoded = jsonable_encoder(obj)
        if encoded is obj:
            return obj
        return sanitize_for_json(encoded)
    except Exception:
        return str(obj)


# -----------------------
# SHAP explainer init helper (fallback-friendly)
# -----------------------
def init_shap_explainer():
    """
    Try initialization in order:
      1) pipeline + background -> shap.Explainer(pipeline.predict_proba, masker)
      2) TreeExplainer(estimator) if model looks tree-based
      3) shap.Explainer(estimator.predict_proba) if available
    Sets global SHAP_EXPLAINER.
    """
    global SHAP_EXPLAINER
    SHAP_EXPLAINER = None

    # 1) pipeline + background
    try:
        if PIPELINE is not None and BACKGROUND_DATA is not None and FEATURE_NAMES is not None:
            logger.info("Attempting SHAP explainer with pipeline + background data")
            background_df_aligned = BACKGROUND_DATA[FEATURE_NAMES]
            masker = shap.maskers.Tabular(background_df_aligned, hclustering=False)
            SHAP_EXPLAINER = shap.Explainer(PIPELINE.predict_proba, masker)
            logger.info("Initialized SHAP Explainer using pipeline + masker.")
            return
    except Exception:
        logger.exception("Failed to init SHAP Explainer with pipeline + background.")

    # get estimator
    estimator = None
    if PIPELINE is not None:
        estimator = PIPELINE.named_steps.get("estimator", None) or PIPELINE.named_steps.get("model")
    if estimator is None:
        estimator = ESTIMATOR_ONLY

    if estimator is None:
        logger.info("No estimator available for SHAP.")
        SHAP_EXPLAINER = None
        return

    # 2) TreeExplainer for tree models
    try:
        est_name = type(estimator).__name__.lower()
        if any(k in est_name for k in ("xgb", "xgboost", "forest", "randomforest", "tree", "lgbm", "lightgbm")):
            logger.info("Attempting SHAP TreeExplainer for estimator: %s", type(estimator).__name__)
            SHAP_EXPLAINER = shap.TreeExplainer(estimator)
            logger.info("Initialized SHAP TreeExplainer.")
            return
    except Exception:
        logger.exception("Failed to init TreeExplainer fallback.")

    # 3) Generic shap.Explainer with predict_proba
    try:
        if hasattr(estimator, "predict_proba"):
            logger.info("Attempting generic SHAP Explainer using estimator.predict_proba")
            SHAP_EXPLAINER = shap.Explainer(estimator.predict_proba)
            logger.info("Initialized generic SHAP Explainer using predict_proba.")
            return
    except Exception:
        logger.exception("Failed to init generic SHAP Explainer from estimator.predict_proba")

    logger.info("Could not initialize SHAP Explainer; SHAP endpoints will return 503 until fixed.")
    SHAP_EXPLAINER = None


# -----------------------
# App startup: load artifacts once
# -----------------------
@app.on_event("startup")
def startup_load():
    global ARTIFACTS, PIPELINE, ESTIMATOR_ONLY, LABEL_ENCODER, METADATA, FEATURE_NAMES, BACKGROUND_DATA, CLASS_NAMES

    ARTIFACTS = {}
    try:
        ARTIFACTS = load_artifacts()
        PIPELINE = ARTIFACTS.get("pipeline", None)
        ESTIMATOR_ONLY = ARTIFACTS.get("estimator_only", None)
        LABEL_ENCODER = ARTIFACTS.get("label_encoder", None)
        METADATA = ARTIFACTS.get("metadata", {}) or {}
        BACKGROUND_DATA = ARTIFACTS.get("background_data", None)

        if LABEL_ENCODER is not None and hasattr(LABEL_ENCODER, "classes_"):
            CLASS_NAMES = list(LABEL_ENCODER.classes_)
            logger.info("CLASS_NAMES set to %s", CLASS_NAMES)

        FEATURE_NAMES = METADATA.get("feature_names", None) or (list(BACKGROUND_DATA.columns) if BACKGROUND_DATA is not None else None)

        if FEATURE_NAMES is None:
            # fallback default feature names (keep consistent with training)
            FEATURE_NAMES = [
                "Application order", "Inflation rate", "Application mode", "GDP", "Unemployment rate", "Course",
                "Curricular units 1st sem (evaluations)", "Curricular units 2nd sem (evaluations)",
                "Age at enrollment", "Admission grade",
                "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
                "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (approved)"
            ]
            logger.info("Using fallback FEATURE_NAMES with length %s", len(FEATURE_NAMES))
        else:
            logger.info("FEATURE_NAMES loaded with length %s", len(FEATURE_NAMES))

        # attempt to init SHAP explainer with best-effort
        init_shap_explainer()

    except Exception as exc:
        logger.exception("Exception during startup artifact load: %s", exc)


# middleware to log exceptions
@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception("Unhandled exception for %s %s: %s", request.method, request.url, e)
        raise


# -----------------------
# Routes
# -----------------------

@app.get("/health")
def health() -> JSONResponse:
    payload = {
        "status": "ok",
        "details": {
            "pipeline_loaded": PIPELINE is not None,
            "estimator_only": ESTIMATOR_ONLY is not None,
            "feature_names_count": len(FEATURE_NAMES) if FEATURE_NAMES is not None else None,
            "shap_explainer_ready": SHAP_EXPLAINER is not None,
            "background_data_rows": len(BACKGROUND_DATA) if BACKGROUND_DATA is not None else 0
        },
    }
    return JSONResponse(content=sanitize_for_json(payload))


@app.get("/")
def read_root() -> JSONResponse:
    return JSONResponse(content=sanitize_for_json({"message": "Student Dropout Prediction API with Explainability."}))


@app.get("/model_info")
def get_model_info() -> JSONResponse:
    global METADATA
    try:
        if not METADATA:
            METADATA = {
                "data_input_description": "Student academic and macroeconomic features (fallback).",
                "model_output_description": "Predicted probability of graduation or dropout (fallback).",
                "model_performance": {"test_accuracy": None, "f1_score_weighted": None},
                "model_how_it_works": "XGBoost classifier (fallback).",
                "feature_names": FEATURE_NAMES or []
            }
        resp = {
            "global_explanation_report": {
                "INPUT": METADATA.get("data_input_description", ""),
                "OUTPUT": METADATA.get("model_output_description", ""),
                "PERFORMANCE": METADATA.get("model_performance", ""),
                "HOW": METADATA.get("model_how_it_works", ""),
            },
            "metadata": METADATA
        }
        return JSONResponse(content=sanitize_for_json(resp))
    except Exception as exc:
        logger.exception("Failed to return model info: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to return model info: {exc}")


@app.post("/predict")
def predict(payload: Union[Dict[str, Any], List[Dict[str, Any]]] = Body(...)) -> JSONResponse:
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
        elif ESTIMATOR_ONLY is not None:
            if os.path.exists(SCALER_PATH) and os.path.exists(PCA_PATH):
                scaler = joblib.load(SCALER_PATH)
                pca = joblib.load(PCA_PATH)
                X_scaled = scaler.transform(df_in)
                X_pca = pca.transform(X_scaled)
                out = predict_from_estimator_only(ESTIMATOR_ONLY, X_pca, label_encoder=LABEL_ENCODER)
            else:
                raise HTTPException(status_code=500, detail="Estimator-only present but scaler/pca missing.")
        else:
            raise HTTPException(status_code=500, detail="No model pipeline or estimator available.")
        return JSONResponse(content=sanitize_for_json(out))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.get("/global_explanation")
def global_explanation() -> JSONResponse:
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
            # fallback: if shap provides global importance via SHAP values using background
            if SHAP_EXPLAINER is not None and BACKGROUND_DATA is not None:
                # compute mean absolute SHAP values on background sample (expensive)
                try:
                    logger.info("Computing global importances using SHAP values on background sample (may be slow).")
                    shap_vals = SHAP_EXPLAINER(BACKGROUND_DATA[FEATURE_NAMES])
                    # shap_vals.values shape: (n_samples, n_features, n_classes?) -> handle both binary and multiclass
                    vals = shap_vals.values
                    # if multiclass, take mean over absolute values across samples and classes
                    if vals.ndim == 3:
                        mean_abs = np.mean(np.abs(vals), axis=(0, 2))
                    else:
                        mean_abs = np.mean(np.abs(vals), axis=0)
                    importances = mean_abs
                except Exception:
                    logger.exception("Failed to compute SHAP-based global importances.")
                    importances = None

        if importances is None:
            raise HTTPException(status_code=500, detail="Model does not expose 'feature_importances_' and SHAP fallback failed.")

        # convert to list and map to feature names
        importances_list = [None if (math.isnan(float(v)) or math.isinf(float(v))) else float(v) for v in np.asarray(importances).tolist()]

        if len(importances_list) != len(FEATURE_NAMES):
            # label features generically
            feature_importance = [{"feature": f"feature_{i}", "importance": importances_list[i]} for i in range(len(importances_list))]
        else:
            feature_importance = [{"feature": fname, "importance": importances_list[i]} for i, fname in enumerate(FEATURE_NAMES)]

        feature_importance_sorted = sorted(feature_importance, key=lambda x: (x["importance"] if x["importance"] is not None else -float("inf")), reverse=True)
        resp = {
            "model_type": type(model).__name__,
            "explanation_type": "Global (feature importance)",
            "top_features": feature_importance_sorted[:10],
            "feature_importances": feature_importance_sorted
        }
        return JSONResponse(content=sanitize_for_json(resp))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Global explanation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Global explanation failed: {e}")


@app.post("/local_explanation")
def local_explanation(payload: Union[Dict[str, Any], List[Dict[str, Any]]] = Body(...)) -> JSONResponse:
    # attempt on-demand SHAP initialization
    if SHAP_EXPLAINER is None:
        logger.info("SHAP_EXPLAINER is None at request time; attempting on-demand init.")
        init_shap_explainer()
    if SHAP_EXPLAINER is None:
        raise HTTPException(status_code=503, detail="SHAP Explainer is not available. Check server startup logs.")

    if isinstance(payload, list):
        payload_instance = payload[0] if len(payload) > 0 else {}
    elif isinstance(payload, dict):
        payload_instance = payload
    else:
        raise HTTPException(status_code=400, detail="Payload must be a dict or list")

    if not payload_instance:
        raise HTTPException(status_code=422, detail="Payload is empty")

    if FEATURE_NAMES:
        missing = [c for c in FEATURE_NAMES if c not in payload_instance]
        if missing:
            raise HTTPException(status_code=422, detail={"error": "Missing required features", "missing": missing})

    try:
        df_in = ensure_dataframe_from_input(payload_instance, feature_names=FEATURE_NAMES)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        # SHAP explain -> returns shap values object
        shap_values = SHAP_EXPLAINER(df_in)

        # shap_values.values shape: (n_samples, n_features) or (n_samples, n_features, n_classes)
        vals = shap_values.values
        base_vals = shap_values.base_values

        explanations_by_class = []

        # If multiclass => vals ndim == 3: (1, n_features, n_classes)
        if vals.ndim == 3:
            # iterate classes
            for class_idx, class_name in enumerate(CLASS_NAMES):
                impacts = vals[0, :, class_idx]
                base_value = base_vals[class_idx] if hasattr(base_vals, "__len__") and len(base_vals) > class_idx else base_vals
                feature_impacts = [{"feature": f, "impact": (float(v) if not (math.isnan(float(v)) or math.isinf(float(v))) else None)} for f, v in zip(FEATURE_NAMES, impacts)]
                sorted_impacts = sorted(feature_impacts, key=lambda x: abs(x["impact"]) if x["impact"] is not None else 0, reverse=True)
                explanations_by_class.append({
                    "class_name": class_name,
                    "base_value": (float(base_value) if not (math.isnan(float(base_value)) or math.isinf(float(base_value))) else None),
                    "top_5_features": sorted_impacts[:5],
                    "all_impacts": feature_impacts
                })
        else:
            # binary or regression case: vals shape (1, n_features)
            impacts = vals[0, :]
            base_value = base_vals if not (hasattr(base_vals, "__len__") and len(base_vals) > 1) else base_vals[0]
            feature_impacts = [{"feature": f, "impact": (float(v) if not (math.isnan(float(v)) or math.isinf(float(v))) else None)} for f, v in zip(FEATURE_NAMES, impacts)]
            sorted_impacts = sorted(feature_impacts, key=lambda x: abs(x["impact"]) if x["impact"] is not None else 0, reverse=True)
            explanations_by_class.append({
                "class_name": CLASS_NAMES[0] if CLASS_NAMES else "class_0",
                "base_value": (float(base_value) if not (math.isnan(float(base_value)) or math.isinf(float(base_value))) else None),
                "top_5_features": sorted_impacts[:5],
                "all_impacts": feature_impacts
            })

        resp = {
            "explanation_type": "Local (SHAP)",
            "instance_input": payload_instance,
            "class_names": CLASS_NAMES,
            "explanations": explanations_by_class,
            "how_it_helps": "Shows per-feature contribution to the prediction (positive increases score, negative decreases)."
        }
        return JSONResponse(content=sanitize_for_json(resp))

    except Exception as e:
        logger.exception("Failed to prepare local explanation: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to prepare local explanation: {e}")


@app.post("/actionable_explanations")
def actionable_explanations(
    payload: Dict[str, Any] = Body(...),
    target_class: str = "Graduate",
    undesirable_class: str = "Dropout"
) -> JSONResponse:
    if SHAP_EXPLAINER is None:
        logger.info("SHAP_EXPLAINER is None at actionable_explanations; attempting init.")
        init_shap_explainer()
    if SHAP_EXPLAINER is None:
        raise HTTPException(status_code=503, detail="SHAP Explainer is not available. Check server startup logs.")

    try:
        df_in = ensure_dataframe_from_input(payload, feature_names=FEATURE_NAMES)
        prediction_result = predict_from_pipeline(PIPELINE, df_in, LABEL_ENCODER)[0]
        pred_label = prediction_result.get("prediction_label")
    except Exception as e:
        logger.exception("Failed to get prediction for actionable advice: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get prediction for actionable advice: {e}")

    try:
        shap_values = SHAP_EXPLAINER(df_in)
    except Exception as e:
        logger.exception("Failed to get SHAP values for actionable advice: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to get SHAP values for actionable advice: {e}")

    try:
        # simple heuristic: if predicted undesirable_class, show top negative contributors to target class
        if pred_label == undesirable_class:
            # try to find target_class index (if present)
            try:
                target_idx = CLASS_NAMES.index(target_class)
            except ValueError:
                return JSONResponse(content=sanitize_for_json({"recommendation_type": "Error", "message": f"Target class '{target_class}' not in {CLASS_NAMES}"}))

            vals = shap_values.values
            if vals.ndim == 3:
                impacts = vals[0, :, target_idx]
            else:
                impacts = vals[0, :]

            feature_impacts = [{"feature": f, "impact": (float(v) if not (math.isnan(float(v)) or math.isinf(float(v))) else None)} for f, v in zip(FEATURE_NAMES, impacts)]
            # negative impacts that reduce target probability
            negatives = [f for f in feature_impacts if f["impact"] is not None and f["impact"] < 0]
            negatives_sorted = sorted(negatives, key=lambda x: x["impact"])
            suggestions = [f"Improve '{f['feature']}' (it decreases chance of '{target_class}')" for f in negatives_sorted[:3]]
            if not suggestions:
                suggestions = ["No clear negative contributors found; consider improving overall profile features."]
            return JSONResponse(content=sanitize_for_json({
                "recommendation_type": "How to Improve",
                "current_prediction": pred_label,
                "target_prediction": target_class,
                "suggestions": suggestions
            }))

        elif pred_label == target_class:
            # suggest areas to maintain (weak positive factors)
            vals = shap_values.values
            if vals.ndim == 3:
                target_idx = CLASS_NAMES.index(target_class) if target_class in CLASS_NAMES else 0
                impacts = vals[0, :, target_idx]
            else:
                impacts = vals[0, :]
            feature_impacts = [{"feature": f, "impact": (float(v) if not (math.isnan(float(v)) or math.isinf(float(v))) else None)} for f, v in zip(FEATURE_NAMES, impacts)]
            weakest = sorted(feature_impacts, key=lambda x: (x["impact"] if x["impact"] is not None else float("inf")))[:3]
            suggestions = [f"Maintain or improve '{f['feature']}' (weak positive/negative contributor)" for f in weakest]
            return JSONResponse(content=sanitize_for_json({
                "recommendation_type": "How to Maintain",
                "current_prediction": pred_label,
                "suggestions": suggestions
            }))

        else:
            return JSONResponse(content=sanitize_for_json({
                "recommendation_type": "General",
                "current_prediction": pred_label,
                "message": "No specific actionable advice for this label."
            }))

    except Exception as e:
        logger.exception("Failed to generate actionable advice: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate actionable advice: {e}")


@app.get("/explainability_report")
def explainability_report() -> JSONResponse:
    """
    Short report summarizing the explainability methods used and why.
    Useful to include in deliverable / grading.
    """
    report = {
        "title": "Explainability Report (Student Dropout Prediction)",
        "local_explanation": {
            "method": "SHAP (local values)",
            "why": "SHAP gives per-feature contributions for individual predictions (additive feature attribution). This answers 'Why was this student predicted dropout/graduate?' and 'Which features drove this decision?'",
            "how_users_use_it": [
                "Understand personal risk factors",
                "Get targeted recommendations (actionable_explanations)",
                "Inspect which features increase or decrease predicted probability"
            ]
        },
        "global_explanation": {
            "method": "Feature importances (model feature_importances_ or mean(|SHAP|) on background sample)",
            "why": "Provides overall importance ranking across dataset; answers 'Which features generally matter most?' and 'What drives the model globally?'",
            "how_users_use_it": [
                "Prioritize interventions across student population",
                "Validate model alignment with domain knowledge"
            ]
        },
        "notes": "Local explanations are computed with SHAP. Global explanations prefer built-in model importances and fall back to mean absolute SHAP values computed on a background sample (if available)."
    }
    return JSONResponse(content=sanitize_for_json(report))


# Run server
if __name__ == "__main__":
    # Development server; in production render/gunicorn will run app
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
