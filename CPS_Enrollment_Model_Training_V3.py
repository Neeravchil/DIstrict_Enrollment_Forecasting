# =============================================================================
# CPS Enrollment — Model Training V3
# =============================================================================
# Input  : ml_enrollment_feature_table_v3  (Fabric Lakehouse)
# Output : Printed metrics + forecast DataFrame saved to Lakehouse
#
# What changed from V2:
#   1. NEW  — School type features added to the model:
#             GOVERNANCE_ENCODED, IS_SELECTIVE, IS_ATTENDANCE_AREA,
#             IS_SMALL_SCHOOL, IS_HIGH_SCHOOL, REGION_ENCODED
#   2. NEW  — Forecast output filtered to IS_SCHOOL_OPEN=1 only.
#             Forecasting closed schools is meaningless — V2 included them.
#   3. NEW  — Size-stratified metrics broken down by school TYPE as well.
#             Lets you see: are Charter schools harder to forecast than CPS schools?
#   4. NEW  — School name joined for readable forecast output.
#   5. KEEP — All V2 improvements: median imputation, sample weights,
#             CSR baseline comparison, feature importance, size stratification.
# =============================================================================

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd
import numpy as np

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

FEATURE_TABLE = "ml_enrollment_feature_table_v3"
LABEL_COL     = "ENROLLMENT"
TIME_COL      = "SCHOOL_YEAR"

# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = spark.read.table(FEATURE_TABLE)

year_stats = df.select(
    F.min(TIME_COL).alias("min_year"),
    F.max(TIME_COL).alias("max_year")
).collect()[0]

min_year      = year_stats["min_year"]
max_year      = year_stats["max_year"]
val_year      = max_year
train_cutoff  = max_year - 1
forecast_year = max_year + 1

print(f"Data range    : {min_year} – {max_year}")
print(f"Train through : {train_cutoff}")
print(f"Validate on   : {val_year}")
print(f"Forecast year : {forecast_year}")

# =============================================================================
# 2. FEATURE COLUMNS
# All numeric — no StringIndexer needed for school type features because they
# are already encoded as integers in V3 feature engineering.
# =============================================================================

FEATURE_COLS = [
    # V2 features
    "SAME_GRADE_LAST_YEAR",
    "SAME_GRADE_2YR_AGO",
    "FEEDER_GRADE_LAST_YEAR",
    "HAS_FEEDER_GRADE",
    "SCHOOL_TOTAL_ENROLLMENT",
    "COHORT_SURVIVAL_RATE",
    "AVG_SURVIVAL_RATE_3YR",
    "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR",
    "IS_MIGRANT_ANOMALY_YEAR",
    "GRADE_NUMERIC",
    "SCHOOL_KEY",
    # V3 new — school type
    "GOVERNANCE_ENCODED",
    "IS_SELECTIVE",
    "IS_ATTENDANCE_AREA",
    "IS_SMALL_SCHOOL",
    "IS_HIGH_SCHOOL",
    "REGION_ENCODED",
]

CATEGORICAL_COLS = ["GRADE"]

print(f"\nFeatures used ({len(FEATURE_COLS)} numeric + {len(CATEGORICAL_COLS)} categorical):")
for f in FEATURE_COLS:
    print(f"  - {f}")
print(f"  - GRADE (categorical → indexed)")

# =============================================================================
# 3. SPLIT: train / validate (time-based — never random)
# =============================================================================

train_df = df.filter(F.col(TIME_COL) <= train_cutoff)
val_df   = df.filter(F.col(TIME_COL) == val_year)

print(f"\nTrain rows : {train_df.count():,}")
print(f"Val rows   : {val_df.count():,}")

# =============================================================================
# 4. NULL IMPUTATION — column median from training data only
# =============================================================================

null_fill_cols = [
    "SAME_GRADE_2YR_AGO",
    "FEEDER_GRADE_LAST_YEAR",
    "COHORT_SURVIVAL_RATE",
    "AVG_SURVIVAL_RATE_3YR",
    "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR",
]

print("\nComputing medians from TRAINING data for NULL imputation:")
median_fills = {}
for col_name in null_fill_cols:
    median_val = train_df.approxQuantile(col_name, [0.5], 0.01)[0]
    median_fills[col_name] = median_val
    print(f"  {col_name:45s}: {median_val:.2f}")

train_df = train_df.fillna(median_fills)
val_df   = val_df.fillna(median_fills)

# =============================================================================
# 5. SAMPLE WEIGHTS — down-weight migrant anomaly years
# =============================================================================

WEIGHT_COL = "sample_weight"

train_df = train_df.withColumn(
    WEIGHT_COL,
    F.when(F.col("IS_MIGRANT_ANOMALY_YEAR") == 1, F.lit(0.3))
     .otherwise(F.lit(1.0))
)
val_df = val_df.withColumn(WEIGHT_COL, F.lit(1.0))

migrant_rows = train_df.filter(F.col("IS_MIGRANT_ANOMALY_YEAR") == 1).count()
normal_rows  = train_df.filter(F.col("IS_MIGRANT_ANOMALY_YEAR") == 0).count()
print(f"\nSample weights: normal={normal_rows:,} (w=1.0), anomaly={migrant_rows:,} (w=0.3)")

# =============================================================================
# 6. BUILD ML PIPELINE
# =============================================================================

grade_indexer = StringIndexer(
    inputCol="GRADE",
    outputCol="GRADE_idx",
    handleInvalid="keep"
)

all_feature_cols = FEATURE_COLS + ["GRADE_idx"]

assembler = VectorAssembler(
    inputCols=all_feature_cols,
    outputCol="features",
    handleInvalid="keep"
)

gbt = GBTRegressor(
    labelCol=LABEL_COL,
    featuresCol="features",
    weightCol=WEIGHT_COL,
    maxDepth=5,
    maxIter=100,
    stepSize=0.1,
    subsamplingRate=0.8,
    seed=42,
    maxBins=256
)

pipeline = Pipeline(stages=[grade_indexer, assembler, gbt])

# =============================================================================
# 7. TRAIN
# =============================================================================

print("\nTraining GBT model (V3)...")
model = pipeline.fit(train_df)
print("Training complete.")

# =============================================================================
# 8. EVALUATE ON VALIDATION YEAR
# =============================================================================

val_pred = model.transform(val_df)

def compute_metrics(pred_df, label_col, pred_col):
    evaluators = {
        "RMSE": RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="rmse"),
        "MAE":  RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="mae"),
        "R2":   RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="r2"),
    }
    return {name: ev.evaluate(pred_df) for name, ev in evaluators.items()}

gbt_metrics = compute_metrics(val_pred, LABEL_COL, "prediction")

print(f"\n{'='*60}")
print(f"  GBT MODEL V3 — Validation Metrics (SY{val_year})")
print(f"{'='*60}")
print(f"  RMSE : {gbt_metrics['RMSE']:.2f}")
print(f"  MAE  : {gbt_metrics['MAE']:.2f}")
print(f"  R²   : {gbt_metrics['R2']:.4f}")

# --- CSR Baseline ---
csr_metrics = compute_metrics(
    val_pred.withColumnRenamed("SAME_GRADE_LAST_YEAR", "csr_prediction"),
    LABEL_COL, "csr_prediction"
)
print(f"\n  ILIANA'S CSR BASELINE (predict = last year's count)")
print(f"  RMSE : {csr_metrics['RMSE']:.2f}")
print(f"  MAE  : {csr_metrics['MAE']:.2f}")
print(f"  R²   : {csr_metrics['R2']:.4f}")

rmse_improvement = (csr_metrics['RMSE'] - gbt_metrics['RMSE']) / csr_metrics['RMSE'] * 100
print(f"\n  ML improvement over CSR baseline: {rmse_improvement:.1f}% RMSE reduction")
if rmse_improvement > 0:
    print("  ✔ ML model is BETTER than the CSR baseline.")
else:
    print("  ✘ ML model is NOT better than CSR yet — more features or tuning needed.")
print(f"{'='*60}")

# --- Size-stratified metrics ---
print(f"\n  METRICS BY SCHOOL SIZE (grade-level enrollment)")
print(f"  {'Size Group':<15} {'N':>8} {'RMSE':>8} {'MAE':>8}")
print(f"  {'-'*43}")

for label, (lo, hi) in [("Small  (<50)", (0, 49)), ("Medium (50-199)", (50, 199)), ("Large  (200+)", (200, 9999))]:
    subset = val_pred.filter((F.col(LABEL_COL) >= lo) & (F.col(LABEL_COL) <= hi))
    n = subset.count()
    if n == 0:
        print(f"  {label:<15} {'0':>8}")
        continue
    m = compute_metrics(subset, LABEL_COL, "prediction")
    print(f"  {label:<15} {n:>8,} {m['RMSE']:>8.2f} {m['MAE']:>8.2f}")

# --- NEW V3: Metrics by school governance type ---
print(f"\n  METRICS BY GOVERNANCE TYPE")
print(f"  {'Type':<15} {'N':>8} {'RMSE':>8} {'MAE':>8}")
print(f"  {'-'*43}")

for label, gov_code in [("CPS District", 0), ("Charter", 1), ("Contract", 2)]:
    subset = val_pred.filter(F.col("GOVERNANCE_ENCODED") == gov_code)
    n = subset.count()
    if n == 0:
        print(f"  {label:<15} {'0':>8}")
        continue
    m = compute_metrics(subset, LABEL_COL, "prediction")
    print(f"  {label:<15} {n:>8,} {m['RMSE']:>8.2f} {m['MAE']:>8.2f}")

# --- NEW V3: Metrics by selective vs non-selective ---
print(f"\n  METRICS BY SELECTIVE ENROLLMENT")
print(f"  {'Type':<20} {'N':>8} {'RMSE':>8} {'MAE':>8}")
print(f"  {'-'*48}")

for label, sel_code in [("Selective", 1), ("Non-Selective", 0)]:
    subset = val_pred.filter(F.col("IS_SELECTIVE") == sel_code)
    n = subset.count()
    if n == 0:
        continue
    m = compute_metrics(subset, LABEL_COL, "prediction")
    print(f"  {label:<20} {n:>8,} {m['RMSE']:>8.2f} {m['MAE']:>8.2f}")

# --- Sample predictions (worst errors) ---
print(f"\n  Sample predictions — highest error cases (SY{val_year}):")
display(
    val_pred
    .select("SCHOOL_KEY", "GOVERNANCE", "GRADE", TIME_COL, LABEL_COL,
            "prediction", "SAME_GRADE_LAST_YEAR", "COHORT_SURVIVAL_RATE", "IS_SELECTIVE")
    .withColumn("prediction", F.round("prediction", 0))
    .withColumn("error", F.round(F.col("prediction") - F.col(LABEL_COL), 0))
    .orderBy(F.abs("error").desc())
    .limit(30)
)

# =============================================================================
# 9. FEATURE IMPORTANCE
# =============================================================================

gbt_model     = model.stages[-1]
feature_names = all_feature_cols
importances   = gbt_model.featureImportances.toArray()

fi_df = pd.DataFrame({
    "feature":    feature_names,
    "importance": importances
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nFeature Importance (V3 — what the model relies on most):")
for _, row in fi_df.iterrows():
    bar = "█" * int(row["importance"] * 40)
    print(f"  {row['feature']:45s} {row['importance']:.4f}  {bar}")

# If school type features rank highly, it confirms they add signal.
# If they rank near zero, the model found them uninformative.
new_v3_features = ["GOVERNANCE_ENCODED", "IS_SELECTIVE", "IS_ATTENDANCE_AREA",
                   "IS_SMALL_SCHOOL", "IS_HIGH_SCHOOL", "REGION_ENCODED"]
v3_importance = fi_df[fi_df["feature"].isin(new_v3_features)]["importance"].sum()
print(f"\n  V3 school-type features combined importance: {v3_importance:.4f}")
print(f"  (If > 0.05, school type is meaningfully contributing to predictions)")

# =============================================================================
# 10. FORECAST — predict SY{forecast_year}
# Only for OPEN schools — no point forecasting schools that are closed.
# =============================================================================

w_latest = Window.partitionBy("SCHOOL_KEY", "GRADE").orderBy(F.col(TIME_COL).desc())

latest_per_group = (
    df
    .filter(F.col("IS_SCHOOL_OPEN") == 1)   # V3: exclude closed schools
    .fillna(median_fills)
    .withColumn("_rn", F.row_number().over(w_latest))
    .filter(F.col("_rn") == 1)
    .drop("_rn")
    .withColumn(TIME_COL, F.lit(forecast_year))
    .withColumn(WEIGHT_COL, F.lit(1.0))
    .withColumn("IS_MIGRANT_ANOMALY_YEAR", F.lit(0))
)

forecast_pred = model.transform(latest_per_group)

forecast_out = (
    forecast_pred
    .select(
        "SCHOOL_KEY",
        "GOVERNANCE",
        "ANNUAL_REGIONAL_ANALYSIS_REGION",
        "COMMUNITY",
        "GRADE",
        F.col(TIME_COL).alias("FORECAST_YEAR"),
        F.round("prediction", 0).alias("PREDICTED_ENROLLMENT"),
        "IS_SELECTIVE",
        "IS_ATTENDANCE_AREA",
    )
    .orderBy("SCHOOL_KEY", "GRADE")
)

print(f"\nForecast for SY{forecast_year} (open schools only):")
print(f"  School-grade combinations: {forecast_out.count():,}")
display(forecast_out.limit(30))

# Save forecast
FORECAST_TABLE = f"enrollment_forecast_sy{forecast_year}_v3"
forecast_out.write.mode("overwrite").saveAsTable(FORECAST_TABLE)
print(f"\n✔ Forecast saved to: '{FORECAST_TABLE}'")

# =============================================================================
# 11. SUMMARY
# =============================================================================

print(f"""
{'='*60}
  V3 MODEL SUMMARY
{'='*60}
  Training years   : {min_year} – {train_cutoff}
  Validation year  : {val_year}
  Forecast year    : {forecast_year}

  V3 GBT Results:
    RMSE : {gbt_metrics['RMSE']:.2f}
    MAE  : {gbt_metrics['MAE']:.2f}
    R²   : {gbt_metrics['R2']:.4f}

  CSR Baseline:
    RMSE : {csr_metrics['RMSE']:.2f}
    R²   : {csr_metrics['R2']:.4f}

  ML vs CSR improvement: {rmse_improvement:.1f}% RMSE reduction

  New V3 features added:
    - GOVERNANCE_ENCODED  (Charter=1, CPS=0, Contract=2)
    - IS_SELECTIVE        (selective enrollment school flag)
    - IS_ATTENDANCE_AREA  (zoned/neighborhood school flag)
    - IS_SMALL_SCHOOL     (CPS-designated small school flag)
    - IS_HIGH_SCHOOL      (elementary vs high school)
    - REGION_ENCODED      (geographic region)

  Next steps:
    - Grade 9 separate model (waiting on GoCPS data from Connor)
    - Kindergarten separate model (needs birth data)
    - Hyperparameter tuning once V3 baseline is established
{'='*60}
""")
