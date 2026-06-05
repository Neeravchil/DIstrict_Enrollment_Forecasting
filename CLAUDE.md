# CPS District Enrollment Forecasting — Project Context

This file is the authoritative init document for this project folder. Load this at the start of every conversation.

---

## What This Project Is

Chicago Public Schools (CPS) wants to replace their manual SAS-based enrollment forecasting system with a Python + ML solution on **Microsoft Fabric**. The goal is to forecast how many students will enroll in each school, each grade, for the next school year.

This work is being done under the THG / RAN engagement. The key CPS stakeholder is **Iliana Vargas** (Director, Planning & Data Management). She currently maintains the existing forecasting system largely by herself.

---

## Why This Matters

CPS uses enrollment forecasts for:
- Staffing (how many teachers to hire)
- Budgeting
- Lunch planning
- School space / capacity planning
- Over/under-enrollment decisions

A bad forecast compounds — if you get Kindergarten wrong, every grade downstream is wrong too.

---

## The Current CPS Methodology (Iliana's SAS System)

The existing system is called **Cohort Survival Rate (CSR)** forecasting. It works like this:

```
Survival Rate = Current Year Grade 7 Enrollment / Previous Year Grade 6 Enrollment
```

- Calculated per school, per grade transition
- Uses ~10 years of historical data
- Automatically selects averaging windows to minimize historical error
- Heavily manual, hard to maintain, implemented in SAS

**Key weakness:** The district enrollment has been declining for years, so historical averages consistently *overestimate* future enrollment. Entry grades (K and Grade 9) have the highest forecast errors because they depend on external signals (births, school choice) that historical cohort patterns don't capture.

---

## The Data — `fact_enrollment_annualized`

The primary table is `fact_enrollment_annualized` on **Microsoft Fabric Lakehouse**.

### Raw table facts (from actual EDA):
- 19,751,688 raw rows
- 29 columns
- 9,875,844 unique students (each appears EXACTLY 2x — ETL duplication bug)
- School years: 2001 to 2027
- 1,502 unique schools

### Severe data quality issues found:
| Problem | Scale | Action |
|---|---|---|
| Every row duplicated (ETL 2x load) | 9,875,844 rows (50%) | `dropDuplicates()` |
| Soft-deleted rows (`SYS_DELETE_STATUS=1`) | 136 rows | Filter out |
| `SCHOOL_YEAR` stored as `double` (e.g. 2024.0) | All rows | Cast to `IntegerType` |
| Sentinel exit dates `9999-12-31` = "still enrolled" | 1,359,429 rows (13.77%) | Replace with NULL |
| Entry date > exit date (logically impossible) | 5 rows | Drop |
| `ENTRY_REASON` 97.33% NULL, `EXIT_REASON` 89.24% NULL | All rows | Drop both columns |
| Future school year records (SY ≥ 2027) | 331,776 rows | Flag, don't delete |
| `ENROLLMENT_DAYS = 0` (69.65% of rows) | 6,878,469 rows | Flag, don't delete |

### After cleaning: `fact_enrollment_annualized_clean`
- 9,875,703 rows, 32 columns
- Added flag columns: `IS_FUTURE_SCHOOL_YEAR`, `IS_ENTRY_GRADE`, `IS_ZERO_DAY_ENROLLMENT`, `IS_CURRENTLY_ENROLLED`, `ENROLLMENT_DURATION_DAYS`

### Key columns you will use:
- `STUDENT_KEY` — student identifier
- `SCHOOL_KEY` — school identifier (join to `dim_school` for `SCHOOL_NAME`)
- `SCHOOL_YEAR` — integer year (e.g. 2025)
- `ENTRY_GRADE_LEVEL` — grade string: "K", "1"–"8", "9"–"12", "PK", "PE"
- `ENROLLMENT_DAYS` — days the student was enrolled in this window
- `IS_FUTURE_SCHOOL_YEAR` — True if year > 2026; exclude from ML training
- `IS_ZERO_DAY_ENROLLMENT` — True if ENROLLMENT_DAYS = 0; exclude for 20th-day headcount

---

## How to Filter for ML Training

```python
# Standard ML training filter (exclude future + zero-day)
df_train_ready = df.filter(
    (F.col("IS_FUTURE_SCHOOL_YEAR") == False) &
    (F.col("IS_ZERO_DAY_ENROLLMENT") == False)
)
```

---

## The ML Pipeline (4 Notebooks, in order)

All development is done in Python + PySpark on **Microsoft Fabric**. Code cannot be run locally — it must be copied and executed in Fabric.

### Notebook 1: `CPS_Enrollment_EDA.ipynb`
- Explores raw `fact_enrollment_annualized`
- Documents all data quality issues
- Produces the evidence base for cleaning decisions

### Notebook 2: `CPS_Enrollment_Cleaning.ipynb`
- Input: `fact_enrollment_annualized` (raw)
- Output: `fact_enrollment_annualized_clean`
- All cleaning steps documented with before/after counts
- Passes 6 hard quality assertions before writing

### Notebook 3: `CPS_Enrollment_Feature_Engineering_Basic.ipynb`
- Input: `fact_enrollment_annualized_clean`
- Output: `ml_enrollment_feature_table`
- Grain: one row per school × grade × year
- 85,338 rows, grades 1–8 and 10–12, years 2005–2026
- 4 features:
  1. `SAME_GRADE_LAST_YEAR` — same school+grade, year-1
  2. `SAME_GRADE_2YR_AGO` — same school+grade, year-2
  3. `FEEDER_GRADE_LAST_YEAR` — same school, grade G-1, year-1 (the advancing cohort)
  4. `SCHOOL_TOTAL_ENROLLMENT` — total school size that year

### Notebook 4: `CPS_Enrollment_Model_Training.ipynb`
- Input: `ml_enrollment_feature_table` + `dim_school` (for school names)
- Model: GBT (Gradient Boosted Trees) via PySpark MLlib
- Time-based split: train ≤ 2025, validate = 2026, forecast = 2027
- Current validation results (SY2026): RMSE=77.8, MAE=13.2, R²=0.48
- Forecast output: predicted enrollment per school × grade for SY2027

---

## Known Data Anomalies to Handle

1. **2022–2024 Migrant Enrollment Spike:** CPS saw an unusual spike in enrollment due to migrant students, followed by a reversal in 2025–26. Any model trained on 2001–2024 data will be influenced by this anomaly. It should be explicitly handled (e.g., flagged, or years weighted down).

2. **Small Schools:** Many CPS schools are small, making enrollment numbers volatile. A school going from 15 to 22 students in one grade is a 47% swing but means nothing statistically. Model evaluation should be size-stratified.

3. **Entry Grades (K and 9):** These are excluded from the current ML targets because they depend on external signals. They are kept as feeder data. K and 9 need separate models eventually.

---

## Data Still Needed (Not clear so ask if you want to know )

From email thread (June 2026):
- `dim_school` table for school names, school type (selective/zoned/magnet/charter), community area
- Zoned school assignments (on 7001 server, GIS_CPS database) — Iliana will share once DB access is set up
- Grade 9 application / GoCPS admissions data — Lisa Lynn's team will set up a call
- Neighborhood-level birth data — IDPH has city-level only; Census Table S1301 (fertility by census tract) is a potential source

---

## Domain Facts

- Use your knowledge of Chicago Public School , whatever you know
- They are trying to implement ML solutions, leadership is highly non tech
- Grade structure: PK → K → 1–8 → 9–12
- Students don't always attend their geographically assigned (zoned) school — school choice is a major enrollment driver
- Entry grades K and 9 are "hardest to forecast" — highest error in the current SAS system
- Primary forecast need is **annual, school-level, grade-level** for the **start of next school year** (not monthly)

---

## Development Environment

- **Platform:** Microsoft Fabric (Synapse / Lakehouse)
- **Language:** Python + PySpark
- **Tables live in:** `MDMF_COPY_RN.dbo.*` schema on Fabric
- **Code cannot be run in VS Code** — must be copied to Fabric notebooks and executed there
- **Evaluation metrics used:** RMSE, MAE, R²

---

## Key People

| Person | Role |
|---|---|
| Rohit (you) | Data scientist, THG/RAN, building the ML system |
| Iliana Vargas | CPS Director, Planning & Data Management; owns the current SAS system |
| Lisa Lynn | CPS; manages GoCPS application/admissions data |
