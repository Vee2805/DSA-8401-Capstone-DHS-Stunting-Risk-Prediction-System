# DHS Stunting Risk Prediction System

**DSA 8401 — Applied Machine Learning Capstone Project**
*MSc Data Science & Analytics · iLabAfrica · Strathmore University*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render)](https://stunting-risk-api.onrender.com/)
[![Swagger Docs](https://img.shields.io/badge/API%20Docs-Swagger%20UI-85EA2D?style=for-the-badge&logo=swagger)](https://stunting-risk-api.onrender.com/docs)
[![Health Probe](https://img.shields.io/badge/Service-Healthy-brightgreen?style=for-the-badge)](https://stunting-risk-api.onrender.com/health)

> **Live Web Application:** [https://stunting-risk-api.onrender.com/](https://stunting-risk-api.onrender.com/)  
> **Interactive Swagger UI:** [https://stunting-risk-api.onrender.com/docs](https://stunting-risk-api.onrender.com/docs)  
> **Health Check:** [https://stunting-risk-api.onrender.com/health](https://stunting-risk-api.onrender.com/health)

---

## Problem Statement

Childhood stunting (height-for-age Z-score < −2 SD) affects **18%** of Kenyan children under five (DHS 2022), with prevalence reaching **31%** in counties like West Pokot and Turkana. Stunting reflects chronic malnutrition and is linked to impaired cognitive development, reduced economic productivity, and increased disease susceptibility.

This project builds an **end-to-end machine learning system** that:
1. Ingests raw DHS survey data (17,327 child records across 47 counties)
2. Predicts individual stunting risk using a calibrated Gradient Boosting model
3. Provides SHAP-based interpretability for actionable policy targeting
4. Serves predictions via a containerised REST API

The system is designed as a **community health screening tool** — a CHW can input a child's vitals and household indicators to get a calibrated risk score.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DHS Stunting Risk System                  │
├──────────────┬──────────────┬───────────────┬───────────────┤
│  Data Layer  │ Training     │  Inference    │  Deployment   │
│              │ Pipeline     │  Module       │               │
│  DHS CSV     │  src/        │  src/         │  api/         │
│  (17,327     │  train.py    │  predict.py   │  main.py      │
│   records)   │              │               │  (FastAPI)    │
│              │  ┌────────┐  │  ┌─────────┐  │               │
│  ┌────────┐  │  │Preproc.│  │  │Preproc. │  │  ┌─────────┐  │
│  │Raw CSV │──│─▶│Pipeline│──│─▶│.joblib  │──│─▶│/predict │  │
│  └────────┘  │  │  ↓     │  │  │  ↓      │  │  │  ↓      │  │
│              │  │Models  │  │  │ML Model │  │  │ JSON    │  │
│  ┌────────┐  │  │.joblib │  │  │.joblib  │  │  │Response │  │
│  │Data    │  │  └────────┘  │  └─────────┘  │  └─────────┘  │
│  │Dict.   │  │              │               │               │
│  └────────┘  │  GroupShuffle│  Calibrated   │  Docker +     │
│              │  Split (no   │  Probabilities│  Compose      │
│              │  spatial     │               │               │
│              │  leakage)    │               │               │
└──────────────┴──────────────┴───────────────┴───────────────┘
```

---

## Repository Structure

```
STUNTING GROWTH/
├── README.md                          ← This file
├── AI_USE.md                          ← AI assistance disclosure
├── requirements.txt                   ← Python dependencies
├── Dockerfile                         ← Container image definition
├── docker-compose.yml                 ← Container orchestration
│
├── Kenya_2022_DHS_Stunting_AnalysisReady.csv   ← Source data (17,327 records)
├── Kenya_2022_DHS_Stunting_DataDictionary.csv  ← Variable definitions (50 vars)
│
├── DHS_Stunting_Cleaning_and_EDA.ipynb ← Full EDA & modelling notebook
│
├── src/                               ← Source code (scripted pipeline)
│   ├── __init__.py
│   ├── train.py                       ← Reproducible training pipeline
│   └── predict.py                     ← Inference module
│
├── api/                               ← FastAPI prediction service
│   ├── __init__.py
│   └── main.py                        ← API endpoints
│
├── models/                            ← Serialized artifacts
│   ├── preprocessor.joblib            ← Fitted ColumnTransformer
│   ├── ml_model.joblib                ← Calibrated Gradient Boosting
│   ├── logit_model.joblib             ← Econometric logit (statsmodels)
│   └── feature_schema.json            ← Feature names, types, categories
│
└── figures/                           ← Publication-quality visualizations
    ├── haz_distribution.png
    ├── stunting_by_age.png
    ├── stunting_by_county.png
    ├── stunting_by_strata.png
    ├── stunting_by_wash.png
    ├── correlation_heatmap.png
    ├── cluster_map_static.png
    ├── stunting_map.html              ← Interactive Folium/Leaflet map
    ├── roc_pr_curves.png
    ├── confusion_matrix.png
    ├── calibration_plot.png
    ├── calibration_before_after.png
    ├── shap_summary.png
    ├── shap_force_top_risk.png
    ├── pdp_top6.png
    ├── cost_benefit_comparison.png
    └── model_comparison.csv
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)

### 1. Install Dependencies

```bash
cd "STUNTING GROWTH"
pip install -r requirements.txt
```

### 2. Train the Model (from scratch)

```bash
python src/train.py
# or with custom options:
python src/train.py --data path/to/data.csv --seed 42
```

This will:
- Load and preprocess the DHS dataset
- Perform cluster-aware train/test split (no spatial leakage)
- Train & tune Random Forest, Gradient Boosting, XGBoost via RandomizedSearchCV
- Calibrate the best model (isotonic calibration)
- Save artifacts to `models/`

### 3. Run the API Locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit **http://localhost:8000/docs** for interactive Swagger documentation.

### 4. Deploy with Docker

```bash
docker-compose up --build
```

The API will be available at **http://localhost:8000**.

### 5. Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "has_electricity": 0
  }'
```

**Response:**
```json
{
  "stunting_risk": 0.6842,
  "risk_label": "Moderate Risk",
  "prediction": 1,
  "model_name": "Gradient Boosting"
}
```

---

## Methodology

### Data Source
**Kenya 2022 Demographic and Health Survey (DHS)** — 17,327 children under five linked to ~1,700 DHS survey clusters across all 47 Kenyan counties. The dataset captures child anthropometry, maternal characteristics, household WASH indicators, and GPS cluster coordinates.

### Feature Engineering
- **26 features** (19 numeric + 7 categorical) mapped to the UNICEF conceptual framework for stunting
- **Missingness-indicator flags** (`bw_measured`, `bmi_measured`) preserve the informative missingness pattern before imputation
- **Imputation** happens inside the `ColumnTransformer` to prevent train→test leakage

### Spatial Leakage Prevention
`GroupShuffleSplit(groups=cluster_id)` ensures all children within the same DHS survey cluster (PSU) are in either the train or test set — never both. This prevents the model from memorizing cluster-level geographic correlates.

### Models Compared

| Model | Class Balancing | Tuning |
|-------|----------------|--------|
| Logistic Regression | `class_weight='balanced'` | — |
| Random Forest | `class_weight='balanced'` | RandomizedSearchCV (5 iter) |
| Gradient Boosting | — | Fixed: 200 trees, lr=0.05, depth=4 |
| XGBoost | `scale_pos_weight` | RandomizedSearchCV (5 iter) |

### Calibration
The best model is wrapped in `CalibratedClassifierCV(method='isotonic', cv=5)` so that predicted probabilities are trustworthy as real-world risk scores.

### Interpretability
- **SHAP TreeExplainer** — global feature importance + individual force plots
- **Partial Dependence Plots** — marginal effect of top 6 features
- **Cost-Benefit Decision Threshold** — optimal screening cut-off under asymmetric misclassification costs

---

## Technique Justification

### Why These Methods

| Method | Justification |
|--------|---------------|
| **Gradient Boosting** | Handles mixed feature types well; outperformed RF and XGBoost on AUC for this tabular survey dataset; interpretable via SHAP |
| **Random Forest** | Ensemble baseline resistant to overfitting; class balancing handles the 18% stunting prevalence |
| **XGBoost** | Industry-standard gradient boosting; `scale_pos_weight` for class imbalance; used as a comparison point |
| **Logistic Regression** | Linear baseline for calibration comparison and coefficient interpretation |
| **Isotonic Calibration** | Class-balancing distorts probability scale; isotonic calibration restores meaningful probabilities for screening decisions |
| **SHAP** | Model-agnostic interpretability critical for health policy stakeholders who need to understand *why* a child is flagged |

### Why Not Other Methods

| Method | Reason for Exclusion |
|--------|---------------------|
| **Deep Learning (MLP/CNN/RNN)** | Tabular data with 17K rows and 26 features — gradient boosting consistently outperforms deep learning on structured data of this scale (Grinsztajn et al., 2022). No sequential or image modality to exploit. |
| **SVM** | Scales poorly (O(n²) kernel computation) and offers no probability calibration natively. Gradient boosting dominates SVMs on tabular data with mixed types. |
| **LightGBM** | Functionally equivalent to XGBoost for this dataset size. XGBoost was chosen for wider deployment ecosystem support. |
| **K-Means / DBSCAN** | The problem is supervised classification, not clustering. Unsupervised methods would not directly predict stunting. |
| **LLMs / Transformers** | No text data in this project. The features are structured survey responses, not natural language. |

---

## Key Results

- **Best model AUC**: ~0.72 (Gradient Boosting, calibrated)
- **Brier score improvement**: Isotonic calibration reduces Brier score vs. raw probabilities
- **Top SHAP features**: `child_age_months`, `mother_height_cm`, `mother_bmi`, `birth_weight_kg`, `wealth_quintile`
- **Policy threshold**: Cost-benefit analysis identifies optimal screening cut-off that minimizes total expected cost under asymmetric FP/FN penalties

---

## Limitations

1. **Cross-sectional design**: DHS captures a single time point — cannot model stunting trajectories
2. **Survey weight approximation**: ML models use `sample_weight` at training but SHAP values are unweighted
3. **Geographic generalization**: Model is trained on Kenya 2022 data and may not transfer to other countries without retraining
4. **Missing dietary data**: Individual dietary diversity (IDDS) is not captured in the analysis-ready dataset
5. **AUC ceiling ~0.72**: Stunting is multifactorial with many unmeasured causes (genetics, micronutrient status, infections); this AUC is consistent with published DHS-based stunting models

---

## License

This project was developed for academic purposes as part of the DSA 8401 course at Strathmore University.
