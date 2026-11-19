#!/usr/bin/env python3
"""
DHS Stunting Risk — Reproducible Training Pipeline
===================================================
DSA 8401 Applied Machine Learning Capstone · MSc Data Science & Analytics

This script reproduces the full ML pipeline from the analysis notebook:
  1.  Load the DHS analysis-ready dataset
  2.  Engineer indicator flags (bw_measured, bmi_measured)
  3.  Define numeric / categorical feature sets
  4.  Cluster-aware train / test split (GroupShuffleSplit)
  5.  Build a ColumnTransformer (median impute → scale  |  mode impute → OHE)
  6.  Train & compare: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
  7.  Isotonic calibration of the best model
  8.  Serialize preprocessor, model, and feature schema to  models/

Usage
-----
    python src/train.py                            # defaults
    python src/train.py --data path/to/data.csv    # custom path
    python src/train.py --seed 123                 # reproducibility knob
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Project root (one level up from src/) ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "Kenya_2022_DHS_Stunting_AnalysisReady.csv"
MODEL_DIR = PROJECT_ROOT / "models"

# ── Feature definitions (mirrors the notebook's Cell 58) ─────────────────
TARGET = "stunted"

NUMERIC_FEATURES = [
    "child_age_months", "birth_order", "birth_weight_kg", "bw_measured",
    "mother_age", "mother_education_years", "mother_height_cm",
    "mother_bmi", "bmi_measured",
    "time_to_water_min", "altitude_m",
    "improved_water", "improved_sanitation", "has_electricity",
    "currently_breastfed", "months_breastfed",
    "preceding_birth_interval_m", "short_birth_interval",
    "had_diarrhoea_recent",
]

CATEGORICAL_FEATURES = [
    "child_sex", "mother_education_level", "wealth_quintile",
    "urban_rural", "county", "birth_size", "caesarean",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ═══════════════════════════════════════════════════════════════════════════
#  1.  DATA INGESTION & CLEANING
# ═══════════════════════════════════════════════════════════════════════════
def load_and_clean(data_path: Path) -> pd.DataFrame:
    """Load the analysis-ready CSV and engineer missingness-indicator flags."""
    print(f"\n📂  Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Rows: {len(df):,}   Columns: {df.shape[1]}")

    # Missingness-indicator flags must be created BEFORE imputation so the
    # ColumnTransformer does not erase the original NaN pattern.
    df["bw_measured"] = df["birth_weight_kg"].notna().astype(int)
    df["bmi_measured"] = df["mother_bmi"].notna().astype(int)

    missing = df[ALL_FEATURES].isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"   Columns with NaN (will be pipeline-imputed):")
        for col, n in missing.sort_values(ascending=False).items():
            print(f"     {col}: {n:,} ({100*n/len(df):.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  2.  CLUSTER-AWARE TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════
def split_data(df: pd.DataFrame, seed: int):
    """GroupShuffleSplit on DHS cluster_id to prevent spatial leakage."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    groups = df["cluster_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    w_train = df["sample_weight"].iloc[train_idx]
    w_test = df["sample_weight"].iloc[test_idx]
    cl_train = groups.iloc[train_idx]
    cl_test = groups.iloc[test_idx]

    # Assert zero cluster overlap
    assert set(cl_train) & set(cl_test) == set(), "Cluster leakage detected!"

    print(f"\n🔀  Cluster-aware split (seed={seed})")
    print(f"   Train: {len(X_train):,} children  ·  {cl_train.nunique():,} clusters  ·  stunted: {y_train.mean():.2%}")
    print(f"   Test:  {len(X_test):,} children  ·  {cl_test.nunique():,} clusters  ·  stunted: {y_test.mean():.2%}")
    print(f"   Cluster overlap: 0 (verified)")

    return X_train, X_test, y_train, y_test, w_train, w_test


# ═══════════════════════════════════════════════════════════════════════════
#  3.  PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def build_preprocessor():
    """ColumnTransformer: median-impute+scale for numerics, mode-impute+OHE for categoricals."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


# ═══════════════════════════════════════════════════════════════════════════
#  4.  MODEL TRAINING & COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
def train_models(X_train_p, y_train, w_train, seed: int) -> dict:
    """Train and compare multiple classifiers; return name→model dict."""

    # ── Random Forest with RandomizedSearchCV ────────────────────────────
    print("\n🌲  Tuning Random Forest via RandomizedSearchCV …")
    rf_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [2, 5, 10],
    }
    rf_base = RandomForestClassifier(
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    rf_search = RandomizedSearchCV(
        rf_base, rf_grid, n_iter=5, scoring="roc_auc",
        cv=3, random_state=seed, n_jobs=-1,
    )
    rf_search.fit(X_train_p, y_train)
    print(f"   Best RF params: {rf_search.best_params_}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=seed,
        ),
        "Random Forest (Tuned)": rf_search.best_estimator_,
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=seed,
        ),
    }

    # ── XGBoost (optional dependency) ────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        xgb_grid = {
            "n_estimators": [200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
        }
        xgb_base = XGBClassifier(
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        )
        xgb_search = RandomizedSearchCV(
            xgb_base, xgb_grid, n_iter=5, scoring="roc_auc",
            cv=3, random_state=seed, n_jobs=-1,
        )
        xgb_search.fit(X_train_p, y_train)
        print(f"   Best XGBoost params: {xgb_search.best_params_}")
        models["XGBoost (Tuned)"] = xgb_search.best_estimator_
    except ImportError:
        print("   ⚠  xgboost not installed — skipping")

    # ── Fit non-tuned models ─────────────────────────────────────────────
    for name, m in models.items():
        if "Tuned" not in name:
            m.fit(X_train_p, y_train, sample_weight=w_train.values)
        print(f"   ✓ {name}")

    return models


# ═══════════════════════════════════════════════════════════════════════════
#  5.  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_models(models, X_test_p, y_test, w_test):
    """Print classification metrics and return best model name + AUC."""
    print("\n📊  Test-set evaluation")
    results = []
    w_arr = w_test.values

    for name, model in models.items():
        proba = model.predict_proba(X_test_p)[:, 1]
        auc = roc_auc_score(y_test, proba, sample_weight=w_arr)
        brier = brier_score_loss(y_test, proba, sample_weight=w_arr)
        results.append({"Model": name, "AUC": auc, "Brier": brier})
        print(f"   {name:<28s}  AUC={auc:.4f}  Brier={brier:.4f}")

    results_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
    best_name = results_df.iloc[0]["Model"]
    print(f"\n   🏆 Best model: {best_name}  (AUC={results_df.iloc[0]['AUC']:.4f})")
    return best_name, results_df


# ═══════════════════════════════════════════════════════════════════════════
#  6.  CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════
def calibrate_model(models, best_name, X_train_p, y_train, w_train, X_test_p, y_test, w_test):
    """Apply isotonic calibration to the best model."""
    print(f"\n🎯  Isotonic calibration for: {best_name}")
    best_model = models[best_name]

    calibrator = CalibratedClassifierCV(
        estimator=best_model, method="isotonic", cv=5, n_jobs=-1,
    )
    calibrator.fit(X_train_p, y_train, sample_weight=w_train.values)

    proba_raw = best_model.predict_proba(X_test_p)[:, 1]
    proba_cal = calibrator.predict_proba(X_test_p)[:, 1]
    w_arr = w_test.values

    print(f"   AUC   raw={roc_auc_score(y_test, proba_raw, sample_weight=w_arr):.4f}"
          f"   calibrated={roc_auc_score(y_test, proba_cal, sample_weight=w_arr):.4f}")
    print(f"   Brier raw={brier_score_loss(y_test, proba_raw, sample_weight=w_arr):.4f}"
          f"   calibrated={brier_score_loss(y_test, proba_cal, sample_weight=w_arr):.4f}")

    # Also print the classification report for the calibrated model
    preds_cal = (proba_cal >= 0.5).astype(int)
    print("\n   Classification report (calibrated, threshold=0.5):")
    report = classification_report(y_test, preds_cal, target_names=["Not Stunted", "Stunted"])
    for line in report.split("\n"):
        print(f"   {line}")

    return calibrator


# ═══════════════════════════════════════════════════════════════════════════
#  7.  SERIALIZE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════
def save_artifacts(preprocessor, best_model, df, model_name: str):
    """Persist preprocessor, model, and feature schema to models/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(best_model, MODEL_DIR / "ml_model.joblib")

    schema = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "ml_model_name": model_name,
        "categories": {
            c: sorted(df[c].dropna().unique().tolist())
            for c in CATEGORICAL_FEATURES
        },
        "numeric_medians": {
            c: float(df[c].median()) for c in NUMERIC_FEATURES
        },
    }
    with open(MODEL_DIR / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2, default=str)

    print("\n💾  Saved artifacts to models/")
    for p in MODEL_DIR.iterdir():
        print(f"   {p.name}  ({p.stat().st_size / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="DHS Stunting Risk — Reproducible Training Pipeline",
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="Path to the analysis-ready CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"❌  Data file not found: {args.data}")
        sys.exit(1)

    # 1. Load & clean
    df = load_and_clean(args.data)

    # 2. Split
    X_train, X_test, y_train, y_test, w_train, w_test = split_data(df, args.seed)

    # 3. Preprocess
    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    print(f"\n⚙️   Preprocessed: {X_train_p.shape[1]} features after OHE")

    # 4. Train
    models = train_models(X_train_p, y_train, w_train, args.seed)

    # 5. Evaluate
    best_name, results_df = evaluate_models(models, X_test_p, y_test, w_test)

    # 6. Calibrate
    calibrator = calibrate_model(
        models, best_name, X_train_p, y_train, w_train, X_test_p, y_test, w_test,
    )

    # 7. Save — we save the calibrated model as the deployment artifact
    save_artifacts(preprocessor, calibrator, df, best_name)

    print("\n✅  Training pipeline complete.\n")


if __name__ == "__main__":
    main()
