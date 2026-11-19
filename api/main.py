#!/usr/bin/env python3
"""
DHS Stunting Risk — FastAPI Prediction Service
================================================
DSA 8401 Applied Machine Learning Capstone

Endpoints
---------
GET  /             → API info & health check
GET  /health       → readiness probe
GET  /schema       → feature schema (accepted inputs)
POST /predict      → single-child stunting risk prediction
POST /predict/batch → batch predictions (list of children)

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add the project root to sys.path so we can import src.predict
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import StuntingPredictor  # noqa: E402

# ── Initialise the predictor at startup ──────────────────────────────────
predictor = StuntingPredictor(model_dir=PROJECT_ROOT / "models")

# ── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(
    title="DHS Stunting Risk Prediction API",
    description=(
        "Predicts childhood stunting risk using Kenya 2022 DHS data. "
        "The model is a calibrated Gradient Boosting classifier trained on "
        "17,327 child records with cluster-aware spatial leakage prevention."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ────────────────────────────────────────────────────
class ChildRecord(BaseModel):
    """Input schema for a single child record.

    All fields are optional — missing values will be median/mode imputed
    by the trained preprocessing pipeline.
    """
    # Numeric features
    child_age_months: Optional[float] = Field(None, ge=0, le=59, description="Child's age in months (0–59)")
    birth_order: Optional[float] = Field(None, ge=1, description="Birth order of the child")
    birth_weight_kg: Optional[float] = Field(None, ge=0.5, le=6.0, description="Birth weight in kilograms")
    bw_measured: Optional[float] = Field(None, description="1 if birth weight was measured, 0 otherwise")
    mother_age: Optional[float] = Field(None, ge=15, le=49, description="Mother's age at birth")
    mother_education_years: Optional[float] = Field(None, ge=0, description="Mother's years of schooling")
    mother_height_cm: Optional[float] = Field(None, ge=100, le=200, description="Mother's height in cm")
    mother_bmi: Optional[float] = Field(None, ge=12, le=60, description="Mother's BMI")
    bmi_measured: Optional[float] = Field(None, description="1 if BMI was measured, 0 otherwise")
    time_to_water_min: Optional[float] = Field(None, ge=0, description="Time to water source in minutes")
    altitude_m: Optional[float] = Field(None, description="Altitude of the household in metres")
    improved_water: Optional[float] = Field(None, description="1 if improved water source, 0 otherwise")
    improved_sanitation: Optional[float] = Field(None, description="1 if improved sanitation, 0 otherwise")
    has_electricity: Optional[float] = Field(None, description="1 if household has electricity, 0 otherwise")
    currently_breastfed: Optional[float] = Field(None, description="1 if currently breastfed, 0 otherwise")
    months_breastfed: Optional[float] = Field(None, ge=0, description="Total months of breastfeeding")
    preceding_birth_interval_m: Optional[float] = Field(None, ge=0, description="Months since previous birth")
    short_birth_interval: Optional[float] = Field(None, description="1 if interval < 24 months, 0 otherwise")
    had_diarrhoea_recent: Optional[float] = Field(None, description="1 if diarrhoea in last 2 weeks, 0 otherwise")

    # Categorical features
    child_sex: Optional[str] = Field(None, description="Child's sex: 'male' or 'female'")
    mother_education_level: Optional[str] = Field(None, description="'none', 'primary', 'secondary', or 'higher'")
    wealth_quintile: Optional[str] = Field(None, description="'poorest', 'poorer', 'middle', 'richer', 'richest'")
    urban_rural: Optional[str] = Field(None, description="'urban' or 'rural'")
    county: Optional[str] = Field(None, description="Kenyan county name (47 counties)")
    birth_size: Optional[float] = Field(None, description="Mother's recall of birth size (1–5)")
    caesarean: Optional[float] = Field(None, description="1 if delivered by caesarean, 0 otherwise")

    model_config = {"json_schema_extra": {
        "examples": [{
            "child_age_months": 24,
            "birth_order": 3,
            "birth_weight_kg": 2.8,
            "mother_age": 28,
            "mother_education_years": 8,
            "mother_height_cm": 155.0,
            "mother_bmi": 21.5,
            "child_sex": "male",
            "mother_education_level": "primary",
            "wealth_quintile": "poorer",
            "urban_rural": "rural",
            "county": "Turkana",
            "improved_water": 0,
            "improved_sanitation": 0,
            "has_electricity": 0,
        }]
    }}


class PredictionResponse(BaseModel):
    """Output schema for a single prediction."""
    stunting_risk: float = Field(..., description="Calibrated probability of stunting (0–1)")
    risk_label: str = Field(..., description="'High Risk', 'Moderate Risk', or 'Low Risk'")
    prediction: int = Field(..., description="Binary prediction: 1=stunted, 0=not stunted")
    model_name: str = Field(..., description="Name of the underlying ML model")


class BatchRequest(BaseModel):
    """Input for batch predictions."""
    records: list[ChildRecord]


class BatchResponse(BaseModel):
    """Output for batch predictions."""
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str


# ── Endpoints ────────────────────────────────────────────────────────────
STATIC_DIR = PROJECT_ROOT / "static"


@app.get("/", tags=["Info"], include_in_schema=False)
async def root():
    """Serve the frontend web application."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {
        "project": "DHS Stunting Risk Prediction",
        "version": "1.0.0",
        "description": "Childhood stunting risk screening tool based on Kenya 2022 DHS data",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    """Readiness probe for container orchestrators."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model is not None,
        model_name=predictor.schema.get("ml_model_name", "unknown"),
    )


@app.get("/schema", tags=["Info"])
async def feature_schema():
    """Return the feature schema — useful for frontend form generation."""
    return predictor.get_feature_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(record: ChildRecord):
    """Predict stunting risk for a single child record.

    All fields are optional. Missing values are imputed using the
    training-set median (numeric) or mode (categorical).
    """
    try:
        data = record.model_dump(exclude_none=False)
        # Convert None back for imputation
        clean_data = {k: v for k, v in data.items() if v is not None}
        result = predictor.predict(clean_data)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(batch: BatchRequest):
    """Predict stunting risk for a batch of child records."""
    if len(batch.records) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 records per batch")
    try:
        results = []
        for record in batch.records:
            data = {k: v for k, v in record.model_dump().items() if v is not None}
            results.append(predictor.predict(data))
        return BatchResponse(
            predictions=[PredictionResponse(**r) for r in results],
            count=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/stats", tags=["Public Health Data"])
async def get_dhs_stats():
    """Return Kenya DHS 2022 national overview and 47-county stunting metrics."""
    stats_file = PROJECT_ROOT / "models" / "dhs_stats.json"
    if stats_file.exists():
        import json
        with open(stats_file) as f:
            return json.load(f)
    return {"overview": {}, "counties": []}


@app.get("/api/sample-batch", tags=["Public Health Data"])
async def get_sample_batch():
    """Return pre-packaged realistic child records across 10 Kenyan counties for batch demo."""
    batch_file = PROJECT_ROOT / "models" / "sample_batch.json"
    if batch_file.exists():
        import json
        with open(batch_file) as f:
            return json.load(f)
    return []


# ── Mount static files (must be AFTER route definitions) ─────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
