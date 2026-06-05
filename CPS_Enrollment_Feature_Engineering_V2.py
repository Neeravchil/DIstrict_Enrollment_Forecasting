# =============================================================================
# CPS Enrollment — Feature Engineering V2
# =============================================================================
# Input  : fact_enrollment_annualized_clean  (Fabric Lakehouse)
# Output : ml_enrollment_feature_table_v2   (Fabric Lakehouse)
#
# What changed from V1:
#   1. FIX  — Filter out IS_ZERO_DAY_ENROLLMENT before counting headcount.
#             V1 counted students with 0 attendance days. This makes the
#             enrollment count match the CPS 20th-day headcount definition.
#   2. NEW  — COHORT_SURVIVAL_RATE: how well this school retained this grade
#             last year. This is the core of Iliana's SAS method, now as a feature.
#   3. NEW  — AVG_SURVIVAL_RATE_3YR: average survival rate over last 3 years.
#             Smooths out one-year spikes (e.g. migrant surge).
#   4. NEW  — DISTRICT_GRADE_ENROLLMENT_LAST_YEAR: total CPS-wide enrollment
#             for this grade last year. Tells the model about the district trend.
#   5. NEW  — HAS_FEEDER_GRADE flag: instead of filling NULL feeder with 0
#             (which told the model "zero students came up"), we now tell the
#             model "this school simply has no feeder grade."
#   6. FIX  — NULL feeder values filled with school's own SAME_GRADE_LAST_YEAR
#             as a fallback (not zero). Still imperfect but far less wrong.
# =============================================================================

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

# =============================================================================
# 1. LOAD & FILTER
# =============================================================================

CLEAN_TABLE = "fact_enrollment_annualized_clean"
df_raw = spark.read.table(CLEAN_TABLE)

# GRADE ordering for numeric sort (PE/PK excluded from targets)
GRADE_ORDER = {
    "PE": 0, "PK": 1, "K": 2,
    "1": 3, "2": 4, "3": 5, "4": 6, "5": 7, "6": 8,
    "7": 9, "8": 10, "9": 11, "10": 12, "11": 13, "12": 14
}
grade_map_expr = F.create_map([F.lit(x) for pair in GRADE_ORDER.items() for x in pair])

# FIX #1: Apply the IS_ZERO_DAY_ENROLLMENT filter BEFORE counting headcount.
# Without this, students who are pre-enrolled but haven't attended are counted,
# inflating enrollment numbers vs the actual 20th-day count CPS uses.
df = (
    df_raw
    .filter(F.col("IS_FUTURE_SCHOOL_YEAR") == False)
    .filter(F.col("IS_ZERO_DAY_ENROLLMENT") == False)      # <- V1 was missing this
    .withColumn("GRADE", F.col("ENTRY_GRADE_LEVEL").cast("string"))
    .withColumn("GRADE_NUMERIC", grade_map_expr[F.col("GRADE")])
    .filter(F.col("GRADE_NUMERIC").isNotNull())             # drop PE/PK
)

print(f"Rows after filtering (no future, no zero-day, no PE/PK): {df.count():,}")

# =============================================================================
# 2. AGGREGATE — one row per school × grade × year
# ALL grades kept (K and 9 needed as feeders for Grade 1 and Grade 10)
# =============================================================================

df_agg = (
    df
    .groupBy("SCHOOL_KEY", "SCHOOL_YEAR", "GRADE", "GRADE_NUMERIC")
    .agg(F.countDistinct("STUDENT_KEY").alias("ENROLLMENT"))
)

total_agg = df_agg.count()
print(f"Aggregated rows (school × grade × year): {total_agg:,}")

# =============================================================================
# 3. FEATURE: SAME_GRADE_LAST_YEAR and SAME_GRADE_2YR_AGO
# How many students were in this same grade at this same school last year?
# =============================================================================

w_school_grade = Window.partitionBy("SCHOOL_KEY", "GRADE").orderBy("SCHOOL_YEAR")

df_feat = (
    df_agg
    .withColumn("SAME_GRADE_LAST_YEAR", F.lag("ENROLLMENT", 1).over(w_school_grade))
    .withColumn("SAME_GRADE_2YR_AGO",   F.lag("ENROLLMENT", 2).over(w_school_grade))
)

print("Added: SAME_GRADE_LAST_YEAR, SAME_GRADE_2YR_AGO")

# =============================================================================
# 4. FEATURE: COHORT_SURVIVAL_RATE and AVG_SURVIVAL_RATE_3YR
#
# Survival Rate = this year's enrollment / last year's enrollment for same grade.
# A rate of 0.95 means 5% of that grade's students left vs last year.
# This IS Iliana's core SAS metric — now given to the ML model directly.
#
# AVG_SURVIVAL_RATE_3YR smooths over spikes (like the 2022-2024 migrant surge).
# =============================================================================

df_feat = df_feat.withColumn(
    "COHORT_SURVIVAL_RATE",
    F.when(
        F.col("SAME_GRADE_LAST_YEAR") > 0,
        F.round(F.col("ENROLLMENT") / F.col("SAME_GRADE_LAST_YEAR"), 4)
    ).otherwise(F.lit(None))
)

# Lag the survival rate to get prior years' rates
df_feat = (
    df_feat
    .withColumn("_SR_LAG1", F.lag("COHORT_SURVIVAL_RATE", 1).over(w_school_grade))
    .withColumn("_SR_LAG2", F.lag("COHORT_SURVIVAL_RATE", 2).over(w_school_grade))
    .withColumn(
        "AVG_SURVIVAL_RATE_3YR",
        # Average of up to 3 years; handles NULLs gracefully
        (
            F.coalesce(F.col("COHORT_SURVIVAL_RATE"), F.lit(0)) +
            F.coalesce(F.col("_SR_LAG1"), F.lit(0)) +
            F.coalesce(F.col("_SR_LAG2"), F.lit(0))
        ) / (
            F.when(F.col("COHORT_SURVIVAL_RATE").isNotNull(), 1).otherwise(0) +
            F.when(F.col("_SR_LAG1").isNotNull(), 1).otherwise(0) +
            F.when(F.col("_SR_LAG2").isNotNull(), 1).otherwise(0)
        )
    )
    .drop("_SR_LAG1", "_SR_LAG2")
)

print("Added: COHORT_SURVIVAL_RATE, AVG_SURVIVAL_RATE_3YR")

# =============================================================================
# 5. FEATURE: FEEDER_GRADE_LAST_YEAR + HAS_FEEDER_GRADE flag
#
# How many students were in grade G-1 at this school last year?
# Those students are advancing into grade G this year.
# Example: Grade 8 ← Grade 7 last year.  Grade 10 ← Grade 9 last year.
#
# FIX #2: Add HAS_FEEDER_GRADE so model knows when feeder is truly absent
# (e.g. a high school that starts at grade 9 has no feeder for grade 10
# in its own building — those come from feeder middle schools elsewhere).
# =============================================================================

feeder_lookup = (
    df_agg.select(
        F.col("SCHOOL_KEY").alias("_fk_school"),
        F.col("GRADE_NUMERIC").alias("_fk_grade_num"),
        F.col("SCHOOL_YEAR").alias("_fk_year"),
        F.col("ENROLLMENT").alias("FEEDER_GRADE_LAST_YEAR")
    )
)

df_feat = (
    df_feat.join(
        feeder_lookup,
        on=[
            df_feat["SCHOOL_KEY"]    == feeder_lookup["_fk_school"],
            df_feat["GRADE_NUMERIC"] == feeder_lookup["_fk_grade_num"] + 1,
            df_feat["SCHOOL_YEAR"]   == feeder_lookup["_fk_year"] + 1
        ],
        how="left"
    )
    .drop("_fk_school", "_fk_grade_num", "_fk_year")
)

# FIX #2 cont: Flag whether feeder existed, then fill NULL with same-grade-last-year
# (not zero — zero would actively mislead the model into under-predicting)
df_feat = (
    df_feat
    .withColumn(
        "HAS_FEEDER_GRADE",
        F.when(F.col("FEEDER_GRADE_LAST_YEAR").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    )
    .withColumn(
        "FEEDER_GRADE_LAST_YEAR",
        F.coalesce(F.col("FEEDER_GRADE_LAST_YEAR"), F.col("SAME_GRADE_LAST_YEAR"))
    )
)

print("Added: FEEDER_GRADE_LAST_YEAR, HAS_FEEDER_GRADE")

# =============================================================================
# 6. FEATURE: SCHOOL_TOTAL_ENROLLMENT
# Total students across all grades at this school this year.
# Tells the model how large the school is overall.
# =============================================================================

school_totals = (
    df_agg
    .groupBy("SCHOOL_KEY", "SCHOOL_YEAR")
    .agg(F.sum("ENROLLMENT").alias("SCHOOL_TOTAL_ENROLLMENT"))
)

df_feat = df_feat.join(school_totals, on=["SCHOOL_KEY", "SCHOOL_YEAR"], how="left")

print("Added: SCHOOL_TOTAL_ENROLLMENT")

# =============================================================================
# 7. FEATURE: DISTRICT_GRADE_ENROLLMENT_LAST_YEAR
#
# Total enrollment for this GRADE across ALL schools in the district, last year.
# This is the single biggest missing signal from V1.
# It tells the model: "the whole district is declining" or "this grade is growing
# district-wide" — a trend no school-level feature can capture alone.
# =============================================================================

district_grade_totals = (
    df_agg
    .groupBy("GRADE", "SCHOOL_YEAR")
    .agg(F.sum("ENROLLMENT").alias("_district_grade_total"))
)

w_district_grade = Window.partitionBy("GRADE").orderBy("SCHOOL_YEAR")

district_grade_totals = (
    district_grade_totals
    .withColumn(
        "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR",
        F.lag("_district_grade_total", 1).over(w_district_grade)
    )
    .drop("_district_grade_total")
)

df_feat = df_feat.join(district_grade_totals, on=["GRADE", "SCHOOL_YEAR"], how="left")

print("Added: DISTRICT_GRADE_ENROLLMENT_LAST_YEAR")

# =============================================================================
# 8. FEATURE: IS_MIGRANT_ANOMALY_YEAR
#
# The 2022-2024 school years had an unusual migrant enrollment surge that reversed
# in 2025-26. Any model trained on this data will learn the wrong patterns for
# those years. We flag them so the model can treat them differently.
# This is used as a feature AND for sample weighting in training.
# =============================================================================

df_feat = df_feat.withColumn(
    "IS_MIGRANT_ANOMALY_YEAR",
    F.when(F.col("SCHOOL_YEAR").between(2022, 2024), F.lit(1)).otherwise(F.lit(0))
)

print("Added: IS_MIGRANT_ANOMALY_YEAR  (flags SY2022-2024 migrant spike)")

# =============================================================================
# 9. BUILD FINAL FEATURE TABLE
# Filter to target grades only (exclude K and 9 — they need separate models).
# Require at least SAME_GRADE_LAST_YEAR to be non-null (need at least 1 year history).
# =============================================================================

TARGET_GRADES = ["1", "2", "3", "4", "5", "6", "7", "8", "10", "11", "12"]

final_columns = [
    # Identifiers
    "SCHOOL_KEY",
    "GRADE",
    "GRADE_NUMERIC",
    "SCHOOL_YEAR",

    # TARGET — what we are trying to predict
    "ENROLLMENT",

    # Features — school-level, grade-level history
    "SAME_GRADE_LAST_YEAR",
    "SAME_GRADE_2YR_AGO",
    "FEEDER_GRADE_LAST_YEAR",
    "HAS_FEEDER_GRADE",
    "SCHOOL_TOTAL_ENROLLMENT",

    # Features — survival rates (Iliana's methodology as a signal)
    "COHORT_SURVIVAL_RATE",
    "AVG_SURVIVAL_RATE_3YR",

    # Features — district-level trend
    "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR",

    # Feature — anomaly flag
    "IS_MIGRANT_ANOMALY_YEAR",
]

df_final = (
    df_feat
    .filter(F.col("GRADE").isin(TARGET_GRADES))
    .filter(F.col("SAME_GRADE_LAST_YEAR").isNotNull())
    .select(final_columns)
)

final_count = df_final.count()
print(f"\nFinal feature table: {final_count:,} rows × {len(final_columns)} columns")

# =============================================================================
# 10. VERIFY — check fill rates for all features
# =============================================================================

print("\nFeature fill rates:")
feature_only = [c for c in final_columns if c not in
                ["SCHOOL_KEY", "GRADE", "GRADE_NUMERIC", "SCHOOL_YEAR", "ENROLLMENT"]]

for col_name in feature_only:
    non_null = df_final.filter(F.col(col_name).isNotNull()).count()
    pct = round(non_null / final_count * 100, 1)
    flag = "✔" if pct >= 80 else "⚠"
    print(f"  {flag}  {col_name:45s}: {pct}%")

# =============================================================================
# 11. WRITE TO LAKEHOUSE
# =============================================================================

OUTPUT_TABLE = "ml_enrollment_feature_table_v2"

df_final.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)

verify = spark.read.table(OUTPUT_TABLE)
print(f"\n✔ Written: '{OUTPUT_TABLE}'")
print(f"  Rows   : {verify.count():,}")
print(f"  Columns: {len(verify.columns)}")
print(f"  Grades : {sorted([r[0] for r in verify.select('GRADE').distinct().collect()])}")
print(f"  Years  : {verify.agg(F.min('SCHOOL_YEAR')).collect()[0][0]} – {verify.agg(F.max('SCHOOL_YEAR')).collect()[0][0]}")

print("""
\n DONE. Next notebook: CPS_Enrollment_Model_Training_V2.py
  Read from: ml_enrollment_feature_table_v2
""")
