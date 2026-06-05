# =============================================================================
# CPS Enrollment — Model Training V2
# =============================================================================
# Input  : ml_enrollment_feature_table_v2  (Fabric Lakehouse)
# Output : Printed metrics + forecast DataFrame (save to table as needed)
#
# What changed from V1:
#   1. FIX  — Keep SCHOOL_KEY (stable integer) as school identifier instead of
#             SCHOOL_NAME (fragile string). SCHOOL_NAME breaks if name changes;
#             a new school gets no prediction; StringIndexer indices are arbitrary.
#   2. FIX  — NULL features filled with column MEDIAN, not 0.
#             Filling with 0 makes the model think "zero enrollment" for missing
#             values. Median is a neutral, data-grounded fill.
#   3. NEW  — Sample weights: down-weight migrant anomaly years (2022-2024).
#             Those years had an external shock that won't repeat. We don't want
#             the model to "learn" that spike as a normal pattern.
#   4. NEW  — Size-stratified evaluation: report RMSE and MAE separately for
#             small schools (<50 students/grade), medium (50-200), large (200+).
#             Overall RMSE hides that small schools are much harder to forecast.
#   5. NEW  — CSR baseline comparison: compute what Iliana's method would have
#             predicted (just use last year's count) and compare to GBT.
#             This is the key question: is ML actually better?
#   6. NEW  — Feature importance: which features drove the predictions?
#             Explains the model to Iliana in plain language.
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
# 0. CONFIGURATION — change these values each year if needed
# =============================================================================

FEATURE_TABLE  = "ml_enrollment_feature_table_v2"
LABEL_COL      = "ENROLLMENT"
TIME_COL       = "SCHOOL_YEAR"

# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = spark.read.table(FEATURE_TABLE)

year_stats = df.select(
    F.min(TIME_COL).alias("min_year"),
    F.max(TIME_COL).alias("max_year")
).collect()[0]

min_year     = year_stats["min_year"]
max_year     = year_stats["max_year"]
val_year     = max_year          # most recent year used for validation
train_cutoff = max_year - 1      # train on everything before this
forecast_year = max_year + 1     # the year we want to predict

print(f"Data range    : {min_year} – {max_year}")
print(f"Train through : {train_cutoff}")
print(f"Validate on   : {val_year}")
print(f"Forecast year : {forecast_year}")

# =============================================================================
# 2. FEATURE COLUMNS
# These are the columns the model will learn from.
# SCHOOL_KEY is kept as a numeric feature (not StringIndexer) — it's already
# an integer and more stable than school name.
# GRADE_NUMERIC maps grade string to integer order (K=2, 1=3, ... 12=14).
# =============================================================================

FEATURE_COLS = [
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
]

CATEGORICAL_COLS = ["GRADE"]  # only Grade remains as a string category

print(f"\nFeatures used ({len(FEATURE_COLS)} numeric + {len(CATEGORICAL_COLS)} categorical):")
for f in FEATURE_COLS:
    print(f"  - {f}")
for f in CATEGORICAL_COLS:
    print(f"  - {f} (categorical → indexed)")

# =============================================================================
# 3. SPLIT: train / validate
# IMPORTANT: always split by time, never randomly.
# A random split would let the model "see" future years during training,
# making the validation score misleadingly optimistic.
# =============================================================================

train_df = df.filter(F.col(TIME_COL) <= train_cutoff)
val_df   = df.filter(F.col(TIME_COL) == val_year)

print(f"\nTrain rows : {train_df.count():,}")
print(f"Val rows   : {val_df.count():,}")

# =============================================================================
# 4. FIX: fill NULLs with COLUMN MEDIAN, not zero
#
# Why median and not zero:
#   - Zero tells the model "no students were in this group" — factually wrong
#     when the NULL means "we don't have data for this combination"
#   - Median is data-grounded and won't pull predictions toward zero
#
# We compute medians from training data ONLY (not from validation data).
# Using validation data to compute fill values would be data leakage.
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

# Apply fills to both train and validation
train_df = train_df.fillna(median_fills)
val_df   = val_df.fillna(median_fills)

# =============================================================================
# 5. SAMPLE WEIGHTS for migrant anomaly years
#
# School years 2022-2024 saw a migrant enrollment surge that reversed in 2025-26.
# If the model learns this spike as "normal", it will over-predict for 2027.
# We give those rows less influence (weight 0.3 vs 1.0 for normal years).
# The model still sees those rows but doesn't let them dominate.
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
print(f"\nSample weights applied:")
print(f"  Normal years (weight=1.0)  : {normal_rows:,} rows")
print(f"  Anomaly years (weight=0.3) : {migrant_rows:,} rows (SY2022-2024)")

# =============================================================================
# 6. BUILD ML PIPELINE
#
# Steps in the pipeline:
#   a. StringIndexer — converts GRADE string ("1","2",...,"12") to numeric index
#   b. VectorAssembler — combines all feature columns into one "features" vector
#   c. GBTRegressor — trains the Gradient Boosted Tree model
#
# maxBins=256 is enough because SCHOOL_KEY and GRADE_NUMERIC are numeric,
# not high-cardinality categoricals. This is faster than the 2048 in V1.
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

print("\nTraining GBT model...")
model = pipeline.fit(train_df)
print("Training complete.")

# =============================================================================
# 8. EVALUATE ON VALIDATION YEAR
# =============================================================================

val_pred = model.transform(val_df)

# --- 8a. Overall metrics ---
def compute_metrics(pred_df, label_col, pred_col):
    evaluators = {
        "RMSE": RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="rmse"),
        "MAE":  RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="mae"),
        "R2":   RegressionEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="r2"),
    }
    return {name: ev.evaluate(pred_df) for name, ev in evaluators.items()}

gbt_metrics = compute_metrics(val_pred, LABEL_COL, "prediction")

print(f"\n{'='*55}")
print(f"  GBT MODEL — Validation Metrics (SY{val_year})")
print(f"{'='*55}")
print(f"  RMSE : {gbt_metrics['RMSE']:.2f}  (avg error in student count)")
print(f"  MAE  : {gbt_metrics['MAE']:.2f}  (avg absolute error)")
print(f"  R²   : {gbt_metrics['R2']:.4f}  (1.0 = perfect, 0.0 = predicting mean)")

# --- 8b. CSR BASELINE comparison ---
# Iliana's method = just predict last year's number (SAME_GRADE_LAST_YEAR).
# If our model isn't better than this, it adds no value.
csr_metrics = compute_metrics(
    val_pred.withColumnRenamed("SAME_GRADE_LAST_YEAR", "csr_prediction"),
    LABEL_COL,
    "csr_prediction"
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
print(f"{'='*55}")

# --- 8c. Size-stratified metrics ---
# Small schools (few students per grade) are much harder to forecast.
# Reporting a single RMSE hides this. We break it down by school size.
print(f"\n  METRICS BY SCHOOL SIZE (grade-level enrollment in validation year)")
print(f"  {'Size Group':<15} {'Schools':>8} {'RMSE':>8} {'MAE':>8}")
print(f"  {'-'*43}")

size_buckets = [
    ("Small  (<50)",    (0,   49)),
    ("Medium (50-199)", (50, 199)),
    ("Large  (200+)",  (200, 9999)),
]

for label, (lo, hi) in size_buckets:
    subset = val_pred.filter(
        (F.col(LABEL_COL) >= lo) & (F.col(LABEL_COL) <= hi)
    )
    n = subset.count()
    if n == 0:
        print(f"  {label:<15} {'0':>8}")
        continue
    m = compute_metrics(subset, LABEL_COL, "prediction")
    print(f"  {label:<15} {n:>8,} {m['RMSE']:>8.2f} {m['MAE']:>8.2f}")

# --- 8d. Sample predictions ---
print(f"\n  Sample predictions (school × grade, SY{val_year}):")
display(
    val_pred
    .select("SCHOOL_KEY", "GRADE", TIME_COL, LABEL_COL, "prediction",
            "SAME_GRADE_LAST_YEAR", "COHORT_SURVIVAL_RATE")
    .withColumn("prediction", F.round("prediction", 0))
    .withColumn("error", F.round(F.col("prediction") - F.col(LABEL_COL), 0))
    .orderBy(F.abs("error").desc())
    .limit(30)
)

# =============================================================================
# 9. FEATURE IMPORTANCE
# Shows which features the GBT model relied on most.
# Share this with Iliana — it validates that the model is using sensible signals.
# =============================================================================

gbt_model     = model.stages[-1]
feature_names = all_feature_cols
importances   = gbt_model.featureImportances.toArray()

fi_df = pd.DataFrame({
    "feature":    feature_names,
    "importance": importances
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nFeature Importance (what the model relies on most):")
for _, row in fi_df.iterrows():
    bar = "█" * int(row["importance"] * 40)
    print(f"  {row['feature']:45s} {row['importance']:.4f}  {bar}")

# =============================================================================
# 10. FORECAST — predict SY{forecast_year}
#
# Strategy: for each school × grade, take their most recent available row
# and use it as the "last known state" to predict the next year.
#
# This is correct for 1-year-ahead forecasting because:
#   - SAME_GRADE_LAST_YEAR from the 2026 row = actual 2026 enrollment = correct
#   - FEEDER_GRADE_LAST_YEAR from the 2026 row = grade G-1 in 2026 = correct
#   - DISTRICT_GRADE_ENROLLMENT_LAST_YEAR from 2026 = district 2026 = correct
# =============================================================================

w_latest = Window.partitionBy("SCHOOL_KEY", "GRADE").orderBy(F.col(TIME_COL).desc())

latest_per_group = (
    df.fillna(median_fills)
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
        "GRADE",
        F.col(TIME_COL).alias("FORECAST_YEAR"),
        F.round("prediction", 0).alias("PREDICTED_ENROLLMENT")
    )
    .orderBy("SCHOOL_KEY", "GRADE")
)

print(f"\nForecast for SY{forecast_year}:")
display(forecast_out.limit(30))

# Optional: save forecast to Lakehouse
# forecast_out.write.mode("overwrite").saveAsTable(f"enrollment_forecast_sy{forecast_year}")
# print(f"✔ Forecast saved to: enrollment_forecast_sy{forecast_year}")
