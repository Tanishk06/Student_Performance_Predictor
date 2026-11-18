"""FastAPI service for the Student Performance Predictor website.

Run from the project root after training:
    pip install fastapi uvicorn pandas joblib
    uvicorn backend_main:app --reload

The model bundle is created at outputs/student_performance_model.joblib.
"""
from pathlib import Path
import math
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "student_performance_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run the training script first."
    )

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
category_options = bundle.get("category_options", {})
numeric_ranges = bundle.get("numeric_ranges", {})

app = FastAPI(
    title="Student Performance Prediction API",
    description="Predicts an estimated Exam_Score from the dataset's 19 input features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "model": bundle["model_name"]}


@app.get("/schema")
def get_schema():
    """Returns exact feature names, categorical options and numeric ranges."""
    return {
        "target": bundle["target"],
        "features": features,
        "categorical_features": category_options,
        "numeric_features": numeric_ranges,
        "model_name": bundle["model_name"],
    }


@app.post("/predict")
def predict(payload: dict):
    missing = [name for name in features if name not in payload]
    extra = [name for name in payload if name not in features]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    if extra:
        raise HTTPException(status_code=422, detail={"unexpected_features": extra})

    row = {}
    for name in features:
        value = payload[name]
        if value is None or (isinstance(value, str) and not value.strip()):
            raise HTTPException(status_code=422, detail=f"{name} cannot be empty.")

        if name in category_options:
            value = str(value)
            if value not in category_options[name]:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "feature": name,
                        "provided": value,
                        "allowed_values": category_options[name],
                    },
                )
        else:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail=f"{name} must be numeric.")
            if not math.isfinite(value):
                raise HTTPException(status_code=422, detail=f"{name} must be finite.")

        row[name] = value

    input_frame = pd.DataFrame([row], columns=features)
    prediction = float(model.predict(input_frame)[0])

    return {
        "predicted_exam_score": round(prediction, 2),
        "target": bundle["target"],
        "model": bundle["model_name"],
        "notice": "This is an estimate from a statistical model, not a guaranteed exam result.",
    }
