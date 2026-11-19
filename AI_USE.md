# AI Assistance Disclosure

**DSA 8401 — Applied Machine Learning Capstone**
*Per course policy, this document discloses all AI tool usage during the project.*

---

## Tools Used

| Tool | Purpose | Scope of Use |
|------|---------|--------------|
| **Google Antigravity (Gemini-based IDE agent)** | Code assistance, debugging, documentation | Described below |

## How AI Was Used

### 1. Debugging & Code Fixes
- Identified and fixed a `statsmodels` GLM compatibility issue where `.prsquared` attribute was unavailable; the AI suggested computing McFadden's Pseudo R² via deviance ratios.
- Resolved a SHAP `expected_value` array shape error for binary classification (scalar extraction from 1D array).
- Fixed minor Python string syntax issues (f-string formatting) in notebook cells.

### 2. Code Scaffolding
- The FastAPI service (`api/main.py`), Dockerfile, and `docker-compose.yml` were scaffolded with AI assistance, then reviewed and customised to match the project's specific model schema and deployment requirements.
- The standalone training script (`src/train.py`) was extracted from the notebook with AI help to ensure the logic was faithfully replicated.

### 3. Documentation
- The README, this AI_USE.md file, and parts of the technique justification were drafted with AI assistance and then reviewed for accuracy.

## What Was NOT AI-Generated

- **Problem selection and framing**: The choice to model stunting using DHS Kenya 2022 data was an independent decision.
- **Modelling decisions**: Feature selection (UNICEF conceptual framework mapping), choice of GroupShuffleSplit for spatial leakage prevention, isotonic calibration strategy, and cost-benefit threshold analysis were designed by the student.
- **Data analysis and interpretation**: All EDA insights, SHAP interpretations, and policy recommendations reflect the student's domain understanding.
- **Notebook analysis flow**: The overall structure of `DHS_Stunting_Cleaning_and_EDA.ipynb` (data ingestion → cleaning → EDA → econometric modelling → ML comparison → calibration → SHAP → geospatial mapping → policy cost-benefit) was designed by the student.

---

*Last updated: November 2026*
