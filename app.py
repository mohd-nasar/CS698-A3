# app.py
"""
Complete FastAPI application that:
- loads the xgboost_student_dropout_model.joblib from repo root
- creates a small synthetic background for SHAP (no external data file)
- attempts to initialize SHAP safely (tries shap.Explainer then TreeExplainer)
- logs detailed startup errors (stdout -> host logs)
- exposes:
    GET  /        -> health + explainer readiness
    POST /local_explanation -> returns shap values or a lightweight fallback
Notes:
- This file assumes `xgboost_student_dropout_model.joblib` is present in repo root.
- Deployers: use `uvicorn app:app --host 0.0.0.0 --port $PORT` or run this file directly.
"""
import os
import traceback
import logging
import joblib
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Logging configured to INFO so hosted logs capture tracebacks/messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="Local Explanation Service (safe SHAP init)")

# CORS (allow all for convenience; lock down in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path (repo root)
MODEL_PATH = "xgboost_student_dropout_model.joblib"

# Globals
model = None
explainer = None
explainer_ready = False
background_data = None

# Request schema
class ExplainRequest(BaseModel):
    # Accept either a single sample: [x1,x2,...] or a batch: [[x11,x12,...], [...]]
    input: List[float] | List[List[float]]

def safe_load_model(path: str):
    logger.info("Attempting to load model from: %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    m = joblib.load(path)
    logger.info("Model loaded type: %s", type(m))
    return m

def make_synthetic_background(m, nrows: int = 20):
    """
    Create a tiny synthetic background dataset suitable for SHAP initialization.
    - Uses model.n_features_in_ if available
    - otherwise infers from feature_importances_, else falls back to 5 features
    """
    try:
        if hasattr(m, "n_features_in_"):
            nfeat = int(m.n_features_in_)
            logger.info("Model reports n_features_in_ = %d", nfeat)
        elif hasattr(m, "feature_importances_"):
            nfeat = len(getattr(m, "feature_importances_"))
            logger.info("Inferred n_features from feature_importances_: %d", nfeat)
        else:
            nfeat = 5
            logger.info("Falling back to default n_features = %d", nfeat)
    except Exception as e:
        logger.warning("Error inferring feature count: %s - defaulting to 5", e)
        nfeat = 5

    rng = np.random.default_rng(12345)
    base = np.zeros((nrows, nfeat), dtype=float)
    for i in range(nrows):
        base[i] = (i / float(max(1, nrows - 1))) * 0.2 + rng.normal(0, 0.01, size=(nfeat,))
    logger.info("Synthetic background shape: %s", base.shape)
    return base

def model_predict(m, X: np.ndarray):
    """
    Unified predict wrapper for different model types.
    - xgboost.Booster -> uses DMatrix
    - scikit-learn style -> predict_proba (if available) else predict
    """
    # xgboost Booster case
    try:
        import xgboost as xgb  # optional
        if isinstance(m, xgb.Booster):
            d = xgb.DMatrix(X)
            return m.predict(d)
    except Exception:
        # xgboost not installed or model not Booster; ignore
        pass

    if hasattr(m, "predict_proba"):
        try:
            return m.predict_proba(X)
        except Exception:
            logger.warning("predict_proba failed; falling back to predict")
    if hasattr(m, "predict"):
        return m.predict(X)
    # last resort: try calling
    try:
        return m(X)
    except Exception as e:
        raise RuntimeError("Model prediction failed: " + str(e))

# Initialize model + SHAP explainer (best-effort, logs tracebacks)
try:
    model = safe_load_model(MODEL_PATH)
    background_data = make_synthetic_background(model, nrows=20)

    try:
        import shap  # type: ignore
        logger.info("Detected shap version: %s", getattr(shap, "__version__", "unknown"))

        # Try generic Explainer with a light background (safe for many models)
        try:
            predictor = lambda x: model_predict(model, x)
            # shap.Explainer can accept a function + background
            explainer = shap.Explainer(predictor, background_data)
            explainer_ready = True
            logger.info("Initialized shap.Explainer successfully.")
        except Exception as e_gen:
            logger.warning("shap.Explainer failed: %s", e_gen)
            # fallback to TreeExplainer for tree-based models
            try:
                explainer = shap.TreeExplainer(model)
                explainer_ready = True
                logger.info("Initialized shap.TreeExplainer successfully.")
            except Exception as e_tree:
                logger.error("shap.TreeExplainer also failed: %s", e_tree)
                traceback.print_exc()
                explainer_ready = False

    except Exception as shap_e:
        logger.error("SHAP import/initialization error: %s", shap_e)
        traceback.print_exc()
        explainer_ready = False

except Exception as init_e:
    logger.error("Model or explainer startup failed: %s", init_e)
    traceback.print_exc()
    explainer_ready = False

def fallback_explain(X: np.ndarray):
    """
    Lightweight fallback explanation:
    - If model.feature_importances_ exists -> normalized importance vector
    - Otherwise -> uniform importance across features
    """
    try:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        nfeat = X.shape[1]
        result = {"method": None, "importance": None}
        if model is not None and hasattr(model, "feature_importances_"):
            try:
                importances = np.array(model.feature_importances_, dtype=float)
                s = float(np.sum(importances)) if np.sum(importances) != 0 else 1.0
                importances = (importances / s).tolist()
                result["method"] = "model_feature_importances"
                result["importance"] = importances
            except Exception as e:
                logger.warning("Reading feature_importances_ failed: %s", e)
                result["method"] = "uniform_fallback"
                result["importance"] = [1.0 / nfeat] * nfeat
        else:
            result["method"] = "uniform_fallback"
            result["importance"] = [1.0 / nfeat] * nfeat
        return {"ok": True, "explanation": result}
    except Exception as e:
        logger.error("fallback_explain error: %s", e)
        return {"ok": False, "error": str(e)}

@app.get("/")
def root():
    return {"status": "ok", "explainer_ready": explainer_ready}

@app.post("/local_explanation")
def local_explanation(req: ExplainRequest):
    """
    Accepts:
      { "input": [x1,x2,...] } or { "input": [[x11,x12,...], [...]] }
    Returns:
      - If SHAP available: { ok: True, method: "shap", shap_values: [...], base_values: ... }
      - Else: fallback explanation with normalized importances
    """
    try:
        X = np.array(req.input, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.size == 0:
            raise HTTPException(status_code=400, detail="Empty input provided")

        if explainer_ready and explainer is not None:
            try:
                shap_out = explainer(X)
                # shap_out.values may be an ndarray or list depending on model
                try:
                    values = np.array(shap_out.values).tolist()
                except Exception:
                    values = None
                base_values = None
                if hasattr(shap_out, "base_values"):
                    try:
                        base_values = np.array(shap_out.base_values).tolist()
                    except Exception:
                        base_values = None
                return {
                    "ok": True,
                    "method": "shap",
                    "shap_values": values,
                    "base_values": base_values,
                }
            except Exception as e_sh:
                logger.error("Runtime SHAP explain failed: %s", e_sh)
                traceback.print_exc()
                return fallback_explain(X)
        else:
            return fallback_explain(X)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled exception in /local_explanation: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# If run directly, start uvicorn (useful for local dev).
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    # --host 0.0.0.0 so container binding works on hosts like Render
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
