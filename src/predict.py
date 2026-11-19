#!/usr/bin/env python3
"""
DHS Stunting Risk — Prediction / Inference Module
==================================================
Loads the serialised preprocessor + model and provides a predict() function
for both the FastAPI service and any CLI consumer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"


class StuntingPredictor:
    """Wraps the trained preprocessor + classifier for inference."""

    def __init__(self, model_dir: Path | str = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._load()

    def _load(self):
        """Load serialised artifacts from disk."""
        self.preprocessor = joblib.load(self.model_dir / "preprocessor.joblib")
        self.model = joblib.load(self.model_dir / "ml_model.joblib")
        with open(self.model_dir / "feature_schema.json") as f:
            self.schema = json.load(f)

        self.numeric_features = self.schema["numeric_features"]
        self.categorical_features = self.schema["categorical_features"]
        self.all_features = self.numeric_features + self.categorical_features
        self.categories = self.schema.get("categories", {})
        self.medians = self.schema.get("numeric_medians", {})

    def _validate_and_fill(self, data: dict[str, Any]) -> pd.DataFrame:
        """Validate input, fill missing numerics with training medians."""
        row = {}

        for feat in self.numeric_features:
            val = data.get(feat)
            if val is None:
                row[feat] = np.nan  # will be median-imputed by preprocessor
            else:
                row[feat] = float(val)

        for feat in self.categorical_features:
            val = data.get(feat)
            if val is None:
                row[feat] = np.nan  # will be mode-imputed by preprocessor
            else:
                row[feat] = val

        return pd.DataFrame([row], columns=self.all_features)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Predict stunting risk for a single child record.

        Parameters
        ----------
        data : dict
            Keys should correspond to feature names from feature_schema.json.
            Missing keys will be imputed (median for numeric, mode for categorical).

        Returns
        -------
        dict with keys:
            - stunting_risk : float   (calibrated probability 0–1)
            - risk_label    : str     ("High Risk" | "Moderate Risk" | "Low Risk")
            - prediction    : int     (1 = stunted, 0 = not stunted at 0.5 threshold)
            - model_name    : str     (name of the underlying model)
        """
        X = self._validate_and_fill(data)
        X_processed = self.preprocessor.transform(X)

        proba = float(self.model.predict_proba(X_processed)[0, 1])
        pred = int(proba >= 0.5)

        if proba >= 0.7:
            label = "High Risk"
        elif proba >= 0.4:
            label = "Moderate Risk"
        else:
            label = "Low Risk"

        return {
            "stunting_risk": round(proba, 4),
            "risk_label": label,
            "prediction": pred,
            "model_name": self.schema.get("ml_model_name", "unknown"),
        }

    def predict_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict for a list of child records."""
        return [self.predict(r) for r in records]

    def get_feature_info(self) -> dict:
        """Return schema information for API documentation."""
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "categories": self.categories,
            "numeric_medians": self.medians,
        }
