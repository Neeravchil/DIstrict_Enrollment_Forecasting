# CPS District Enrollment Forecasting — Session Context

Load this at the start of every session. Do NOT re-derive what is here from the notebooks.

---

## Problem Statement

CPS needs to forecast **how many students will enroll in each school, each grade, for the next school year**. Currently done manually in SAS by **Iliana Vargas** (Director, Planning & Data Management). Goal: replace with a Python + ML pipeline on **Microsoft Fabric**.

Forecasts drive: teacher hiring, budgeting, lunch planning, space/capacity, over/under-enrollment decisions. A bad Kindergarten forecast cascades into every grade above it.

---

## Primary Table: `fact_enrollment_annualized`

| Dimension | Value |
|---|---|
| Raw rows | 19,751,688 |
| Unique students | 9,875,844 (each appears 2x — ETL duplication bug) |
| School years | 2001–2027 |
| Schools | 1,502 |

### Data Quality Issues (all fixed in Notebook 2):
| Problem | Action |
|---|---|
| Every row duplicated (ETL 2x load) | `dropDuplicates()` |
| `SYS_DELETE_STATUS=1` (soft-deleted rows) | Filter out |
| `SCHOOL_YEAR` stored as double (e.g. 2024.0) | Cast to IntegerType |
| Sentinel exit date `9999-12-31` = still enrolled | Replace with NULL |
| Entry date > exit date (5 rows) | Drop |
| `ENTRY_REASON` 97.33% NULL, `EXIT_REASON` 89.24% NULL | Drop both columns |
| Future school year records (SY ≥ 2027) | Flagged as `IS_FUTURE_SCHOOL_YEAR` |
| `ENROLLMENT_DAYS = 0` (69.65% of rows) | Flagged as `IS_ZERO_DAY_ENROLLMENT` |

### Clean Table: `fact_enrollment_annualized_clean`
- ~9,875,703 rows, 32 columns
- Added flag columns: `IS_FUTURE_SCHOOL_YEAR`, `IS_ENTRY_GRADE`, `IS_ZERO_DAY_ENROLLMENT`, `IS_CURRENTLY_ENROLLED`, `ENROLLMENT_DURATION_DAYS`

### Standard ML Training Filter:
```python
df_train_ready = df.filter(
    (F.col("IS_FUTURE_SCHOOL_YEAR") == False) &
    (F.col("IS_ZERO_DAY_ENROLLMENT") == False)
)
```

---

## The Pipeline — Notebook Versions

| Notebook | Input | Output | Status |
|---|---|---|---|
| `CPS_Enrollment_EDA.ipynb` | `fact_enrollment_annualized` (raw) | EDA findings | Done |
| `CPS_Enrollment_Cleaning.ipynb` | raw table | `fact_enrollment_annualized_clean` | Done |
| `CPS_Enrollment_Feature_Engineering_Basic.ipynb` (V1) | clean table | `ml_enrollment_feature_table` | Superseded |
| `CPS_Enrollment_Feature_Engineering_V2.py` | clean table | `ml_enrollment_feature_table_v2` | Superseded |
| `CPS_Enrollment_Model_Training_V2.py` | `ml_enrollment_feature_table_v2` | metrics + forecast | Superseded |
| `CPS_Enrollment_Feature_Engineering_V3.py` | clean table + dim_school | `ml_enrollment_feature_table_v3` | **Ready to run in Fabric** |
| `CPS_Enrollment_Model_Training_V3.py` | `ml_enrollment_feature_table_v3` | metrics + `enrollment_forecast_sy2027_v3` | **Ready to run in Fabric** |

---

## Current Model (V1) — Baseline Results

- **Algorithm:** GBT (Gradient Boosted Trees), PySpark MLlib
- **Split:** Train ≤ 2025 | Validate = 2026 | Forecast = 2027
- **Target grades:** 1–8, 10–12 (K and 9 excluded — separate models needed)
- **V1 Validation results (SY2026):** RMSE=77.8, MAE=13.2, R²=0.48

### V1 Features (4 features):
1. `SAME_GRADE_LAST_YEAR`
2. `SAME_GRADE_2YR_AGO`
3. `FEEDER_GRADE_LAST_YEAR`
4. `SCHOOL_TOTAL_ENROLLMENT`

---

## V2 Model — What Changed

### Feature Engineering V2 (`CPS_Enrollment_Feature_Engineering_V2.py`):
All V1 features retained, plus:

| New Feature | What It Captures |
|---|---|
| `COHORT_SURVIVAL_RATE` | Enrollment this year / last year for same grade — Iliana's core SAS metric as a feature |
| `AVG_SURVIVAL_RATE_3YR` | 3-year rolling average survival rate — smooths migrant spike |
| `DISTRICT_GRADE_ENROLLMENT_LAST_YEAR` | Total CPS-wide enrollment for this grade, prior year — captures district decline trend |
| `HAS_FEEDER_GRADE` | Binary flag: does this school actually have a feeder grade? Replaces misleading NULL→0 fill |
| `IS_MIGRANT_ANOMALY_YEAR` | 1 for SY2022–2024 (migrant spike), 0 otherwise |

**Bug fixes vs V1:**
- `IS_ZERO_DAY_ENROLLMENT` filter now applied BEFORE headcount aggregation (V1 missed this — was inflating counts)
- NULL feeder grades filled with `SAME_GRADE_LAST_YEAR` as fallback (not 0)

### Model Training V2 (`CPS_Enrollment_Model_Training_V2.py`):
- `SCHOOL_KEY` kept as numeric feature (not `SCHOOL_NAME` — fragile if school is renamed)
- NULL imputation uses **column median from training data** (not 0 — zero was misleading the model)
- **Sample weights:** migrant anomaly years (2022–2024) down-weighted to 0.3 vs 1.0
- **Size-stratified evaluation:** RMSE/MAE reported separately for small (<50), medium (50–199), large (200+) schools
- **CSR baseline comparison:** V2 prints side-by-side: ML model vs Iliana's method (predict = last year's count). If ML RMSE isn't lower, the model adds no value.
- **Feature importance:** printed after training — share with Iliana to show model is using sensible signals

**GBT Hyperparameters (V2):**
```
maxDepth=5, maxIter=100, stepSize=0.1, subsamplingRate=0.8, maxBins=256, seed=42
```

---

## Known Data Anomalies

1. **2022–2024 Migrant Spike:** Unusual surge followed by sharp reversal in 2025–26 post Trump policies. Flagged via `IS_MIGRANT_ANOMALY_YEAR`, down-weighted in training.
2. **Small Schools:** Volatile. A school going 15→22 students is 47% swing but statistically meaningless. V2 evaluates these separately.
3. **Entry Grades (K and 9):** Excluded from current ML targets. Depend on external signals — need separate models.

---

## What's Still Needed for the Next Model Version

### GoCPS Data (Lisa Lynn's team) — Grade 9 separate model
**Status:** Meeting held June 2026. Data access being set up via Connor.

**What the data looks like:**
- Grain: one row per student per application to a program (not school)
- Key fields: `CPS_STUDENT_ID`, `SCHOOL_PROGRAM_CODE`, `OFFER_STATUS`, `ACCEPTANCE_STATUS`, `ROUND`
- Tables named by year: `Merge` (2018), then `1920`, `2021`, etc.
- Goes back to 2018 only (schema differs before that)
- Must limit to **Round 1 only** — Round 2+ allows repeated applications, inflating counts

**What you need to build:**
- Aggregate to school level using program → school crosswalk (Lisa will provide)
- Features: `APPLICATIONS_COUNT`, `OFFERS_COUNT`, `ACCEPTANCES_COUNT` per school per year
- Deduplicate students across programs before counting (one student can apply to multiple programs at same school)

**Key insight from Lisa:** Predictive power varies by school type:
- Selective enrollment (e.g. Payton): ~90% show-up rate — near-perfect predictor
- Neighborhood high schools: weaker signal (students can walk in without applying)

### Other Data Still Needed:
| Data | Source | Status |
|---|---|---|
| `dim_school` (school type, community area) | Fabric Lakehouse | Needed — school type is high-signal missing feature |
| Zoned school assignments | 7001 server, GIS_CPS database | Iliana will share once DB access set up |
| Grade 9 application / GoCPS data | Lisa Lynn's team | Access being set up via Connor |
| Neighborhood birth data (for K model) | Census Table S1301 (fertility by census tract) | Not started |

---

## Planned Future Improvements (Priority Order)

1. **Run V3 in Fabric** — V3 scripts are written and ready. Copy-paste into Fabric notebooks and execute.
2. **Grade 9 separate model** — Use GoCPS acceptance data as primary feature. School type moderates predictive power. (Waiting on Connor for data access.)
3. **Kindergarten separate model** — Use neighborhood birth data (Census S1301) + GoCPS K applications (partial signal).
4. **Longer lag features** — 3-year and 5-year rolling averages for enrollment count (not just survival rate).
5. **School-level bias correction** — Post-hoc adjustment for schools that consistently over/under-predict.
6. **Hyperparameter tuning** — Once V3 baseline is established. Cross-validate maxDepth, maxIter, stepSize.

## V3 Model — What Changed from V2

### New dim_school features added (`CPS_Enrollment_Feature_Engineering_V3.py`):

| Feature | Encoding | What It Captures |
|---|---|---|
| `GOVERNANCE_ENCODED` | Charter=1, CPS=0, Contract=2 | Charter vs district schools have fundamentally different enrollment dynamics |
| `IS_SELECTIVE` | 0/1 | Selective enrollment schools (Payton, Jones, etc.) — demand-constrained, very stable enrollment |
| `IS_ATTENDANCE_AREA` | 0/1 | Zoned/neighborhood school vs citywide — zoned more predictable from demographics |
| `IS_SMALL_SCHOOL` | 0/1 | CPS-designated small school — more volatile enrollment |
| `IS_HIGH_SCHOOL` | 0/1 | Elementary vs high school structure |
| `REGION_ENCODED` | 0–14 | Geographic region — enrollment decline is spatially uneven (South/West declining faster) |

### Other V3 improvements:
- Forecast output filtered to `IS_SCHOOL_OPEN=1` — no point forecasting closed schools
- Validation metrics broken down by governance type (Charter vs CPS vs Contract) and selective vs non-selective
- Feature importance report highlights how much school-type features contribute

---

## Key People

| Person | Role |
|---|---|
| Rohit Jindal | Data scientist, THG/RAN — building the ML system |
| Iliana Vargas | CPS Director, Planning & Data Management — owns current SAS system |
| Lisa Lynn | CPS — manages GoCPS application/admissions data |
| Connor | CPS technical contact — provisions data access |

---

## Development Environment

- **Platform:** Microsoft Fabric (Synapse / Lakehouse)
- **Language:** Python + PySpark
- **Tables:** `MDMF_COPY_RN.dbo.*` schema on Fabric
- **Code cannot run locally** — must be copied to Fabric notebooks and executed there
- **Evaluation metrics:** RMSE, MAE, R²
