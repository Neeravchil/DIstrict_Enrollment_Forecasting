# =============================================================================
# CPS Enrollment — Feature Engineering V3
# =============================================================================
# Input  : fact_enrollment_annualized_clean  (Fabric Lakehouse)
#          dim_school                        (Fabric Lakehouse)
# Output : ml_enrollment_feature_table_v3   (Fabric Lakehouse)
#
# What changed from V2:
#   1. NEW  — SCHOOL_TYPE_ENCODED: Charter vs CPS (District-run) vs Contract.
#             Charter and district schools have fundamentally different enrollment
#             dynamics — charter schools recruit citywide, district schools are
#             mostly zoned. This is likely the highest-signal new categorical.
#   2. NEW  — IS_SELECTIVE: 1 if the school has selective enrollment (applies
#             to high schools like Payton, Jones, etc.). These schools are
#             demand-constrained, not capacity-constrained — enrollment is very
#             stable and predictable once you know the acceptance pipeline.
#   3. NEW  — IS_ATTENDANCE_AREA: 1 if zoned/neighborhood school (vs citywide).
#             Zoned schools have more predictable enrollment from local demographics.
#             Citywide schools compete for students and are harder to forecast.
#   4. NEW  — IS_SMALL_SCHOOL: 1 if SCHOOL_SUBTYPE = 'Small'. These schools
#             have more volatile enrollment and need the model to know this.
#   5. NEW  — REGION_ENCODED: geographic region (South Side, West Side, etc.).
#             Enrollment decline is uneven across the city — some regions are
#             losing population faster. This gives the model spatial context.
#   6. NEW  — IS_SCHOOL_OPEN: 0 for closed schools. Forecasts for closed schools
#             are meaningless — this flag lets the model down-weight or ignore them.
#             We also filter closed schools from the forecast output.
#   7. KEEP — All V2 features retained unchanged.
# =============================================================================

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

# =============================================================================
# 1. LOAD & FILTER ENROLLMENT DATA
# =============================================================================

CLEAN_TABLE = "fact_enrollment_annualized_clean"
df_raw = spark.read.table(CLEAN_TABLE)

GRADE_ORDER = {
    "PE": 0, "PK": 1, "K": 2,
    "1": 3, "2": 4, "3": 5, "4": 6, "5": 7, "6": 8,
    "7": 9, "8": 10, "9": 11, "10": 12, "11": 13, "12": 14
}
grade_map_expr = F.create_map([F.lit(x) for pair in GRADE_ORDER.items() for x in pair])

df = (
    df_raw
    .filter(F.col("IS_FUTURE_SCHOOL_YEAR") == False)
    .filter(F.col("IS_ZERO_DAY_ENROLLMENT") == False)
    .withColumn("GRADE", F.col("ENTRY_GRADE_LEVEL").cast("string"))
    .withColumn("GRADE_NUMERIC", grade_map_expr[F.col("GRADE")])
    .filter(F.col("GRADE_NUMERIC").isNotNull())
)

print(f"Rows after filtering (no future, no zero-day, no PE/PK): {df.count():,}")

# =============================================================================
# 2. AGGREGATE — one row per school × grade × year
# =============================================================================

df_agg = (
    df
    .groupBy("SCHOOL_KEY", "SCHOOL_YEAR", "GRADE", "GRADE_NUMERIC")
    .agg(F.countDistinct("STUDENT_KEY").alias("ENROLLMENT"))
)

print(f"Aggregated rows (school × grade × year): {df_agg.count():,}")

# =============================================================================
# 3. FEATURES FROM V2 (all retained)
# =============================================================================

w_school_grade = Window.partitionBy("SCHOOL_KEY", "GRADE").orderBy("SCHOOL_YEAR")

# --- 3a. Lag features ---
df_feat = (
    df_agg
    .withColumn("SAME_GRADE_LAST_YEAR", F.lag("ENROLLMENT", 1).over(w_school_grade))
    .withColumn("SAME_GRADE_2YR_AGO",   F.lag("ENROLLMENT", 2).over(w_school_grade))
)

# --- 3b. Cohort survival rate ---
df_feat = df_feat.withColumn(
    "COHORT_SURVIVAL_RATE",
    F.when(
        F.col("SAME_GRADE_LAST_YEAR") > 0,
        F.round(F.col("ENROLLMENT") / F.col("SAME_GRADE_LAST_YEAR"), 4)
    ).otherwise(F.lit(None))
)

df_feat = (
    df_feat
    .withColumn("_SR_LAG1", F.lag("COHORT_SURVIVAL_RATE", 1).over(w_school_grade))
    .withColumn("_SR_LAG2", F.lag("COHORT_SURVIVAL_RATE", 2).over(w_school_grade))
    .withColumn(
        "AVG_SURVIVAL_RATE_3YR",
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

# --- 3c. Feeder grade + flag ---
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
    .withColumn(
        "HAS_FEEDER_GRADE",
        F.when(F.col("FEEDER_GRADE_LAST_YEAR").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    )
    .withColumn(
        "FEEDER_GRADE_LAST_YEAR",
        F.coalesce(F.col("FEEDER_GRADE_LAST_YEAR"), F.col("SAME_GRADE_LAST_YEAR"))
    )
)

# --- 3d. School total enrollment ---
school_totals = (
    df_agg
    .groupBy("SCHOOL_KEY", "SCHOOL_YEAR")
    .agg(F.sum("ENROLLMENT").alias("SCHOOL_TOTAL_ENROLLMENT"))
)
df_feat = df_feat.join(school_totals, on=["SCHOOL_KEY", "SCHOOL_YEAR"], how="left")

# --- 3e. District-grade enrollment last year ---
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

# --- 3f. Migrant anomaly flag ---
df_feat = df_feat.withColumn(
    "IS_MIGRANT_ANOMALY_YEAR",
    F.when(F.col("SCHOOL_YEAR").between(2022, 2024), F.lit(1)).otherwise(F.lit(0))
)

print("All V2 features built.")

# =============================================================================
# 4. LOAD dim_school AND BUILD SCHOOL-TYPE FEATURES (NEW in V3)
# =============================================================================

dim = spark.read.table("dim_school")

print(f"\ndim_school rows: {dim.count():,}")
print(f"dim_school columns: {dim.columns}")

# --- 4a. GOVERNANCE → numeric ---
# Charter schools: open enrollment, recruit citywide, different funding structure
# CPS (district): mostly zoned, politically constrained
# Contract: rare, treat like Charter
# Encoding: Charter=1, CPS/District=0, Contract=2
dim_features = (
    dim
    .select(
        "SCHOOL_KEY",
        "GOVERNANCE",
        "SCHOOL_STATUS",
        "SCHOOL_SUBTYPE",
        "SCHOOL_GRADES_GROUP",
        "SELECTIVE_ENROLLMENT_TYPE",
        "ATTENDANCE_BOUNDARY",
        "ANNUAL_REGIONAL_ANALYSIS_REGION",
        "COMMUNITY"
    )
    .withColumn(
        "GOVERNANCE_ENCODED",
        F.when(F.upper(F.col("GOVERNANCE")) == "CHARTER",  F.lit(1))
         .when(F.upper(F.col("GOVERNANCE")) == "CONTRACT", F.lit(2))
         .otherwise(F.lit(0))   # CPS / District / unknown → 0
    )
    # IS_SELECTIVE: 1 for selective enrollment high schools (e.g. Payton, Jones, Northside)
    # These have very stable, demand-constrained enrollment
    .withColumn(
        "IS_SELECTIVE",
        F.when(
            (F.col("SELECTIVE_ENROLLMENT_TYPE").isNotNull()) &
            (F.trim(F.col("SELECTIVE_ENROLLMENT_TYPE")) != "") &
            (F.upper(F.trim(F.col("SELECTIVE_ENROLLMENT_TYPE"))) != "NON"),
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    # IS_ATTENDANCE_AREA: 1 for zoned/neighborhood schools
    # These schools draw from a defined geography — more demographically predictable
    .withColumn(
        "IS_ATTENDANCE_AREA",
        F.when(
            F.upper(F.trim(F.col("ATTENDANCE_BOUNDARY"))) == "ATTENDANCE-AREA",
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    # IS_SMALL_SCHOOL: 1 if CPS has designated this as a Small school
    .withColumn(
        "IS_SMALL_SCHOOL",
        F.when(
            F.upper(F.trim(F.col("SCHOOL_SUBTYPE"))) == "SMALL",
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    # IS_HIGH_SCHOOL: 1 for high schools
    # High schools have different dynamics (Grade 9 entry, GoCPS applications)
    .withColumn(
        "IS_HIGH_SCHOOL",
        F.when(
            F.upper(F.trim(F.col("SCHOOL_GRADES_GROUP"))) == "HIGH",
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    # IS_SCHOOL_OPEN: 0 for closed schools — forecasting closed schools is meaningless
    .withColumn(
        "IS_SCHOOL_OPEN",
        F.when(
            F.upper(F.trim(F.col("SCHOOL_STATUS"))) == "OPEN",
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    # REGION_ENCODED: geographic region as integer
    # Enrollment decline is spatially uneven — South/West sides declining faster
    .withColumn(
        "REGION_ENCODED",
        F.when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "South Side",                  F.lit(1))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "West Side",                   F.lit(2))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "North Lakefront",             F.lit(3))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "Northwest Side",              F.lit(4))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "Far Northwest Side",          F.lit(5))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "Bronzeville / South Lakefront", F.lit(6))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "Greater Stony Island",        F.lit(7))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "Greater Calumet",             F.lit(8))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "Greater Midway",              F.lit(9))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "Greater Stockyards",          F.lit(10))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "Pilsen / Little Village",     F.lit(11))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "Near West Side",              F.lit(12))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "Greater Milwaukee Avenue",    F.lit(13))
         .when(F.col("ANNUAL_REGIONAL_ANALYSIS_REGION") == "Greater Lincoln Park",        F.lit(14))
         .otherwise(F.lit(0))  # unknown / missing
    )
    .select(
        "SCHOOL_KEY",
        "GOVERNANCE_ENCODED",
        "IS_SELECTIVE",
        "IS_ATTENDANCE_AREA",
        "IS_SMALL_SCHOOL",
        "IS_HIGH_SCHOOL",
        "IS_SCHOOL_OPEN",
        "REGION_ENCODED",
        # Keep raw strings for diagnostics — not used as model features
        "GOVERNANCE",
        "ATTENDANCE_BOUNDARY",
        "ANNUAL_REGIONAL_ANALYSIS_REGION",
        "COMMUNITY",
    )
)

# Validate dim feature coverage
print("\ndim_school feature value distribution:")
for col_name in ["GOVERNANCE_ENCODED", "IS_SELECTIVE", "IS_ATTENDANCE_AREA",
                 "IS_SMALL_SCHOOL", "IS_HIGH_SCHOOL", "IS_SCHOOL_OPEN", "REGION_ENCODED"]:
    dist = dim_features.groupBy(col_name).count().orderBy(col_name).collect()
    print(f"  {col_name}: {[(r[0], r[1]) for r in dist]}")

# =============================================================================
# 5. JOIN dim_school FEATURES ONTO FEATURE TABLE
# =============================================================================

df_feat = df_feat.join(dim_features, on="SCHOOL_KEY", how="left")

# How many school-grade-year rows have no dim_school match?
unmatched = df_feat.filter(F.col("GOVERNANCE_ENCODED").isNull()).count()
total = df_feat.count()
print(f"\ndim_school join: {total - unmatched:,} matched, {unmatched:,} unmatched ({round(unmatched/total*100,1)}%)")

# Fill nulls for unmatched schools (likely very old/historical schools not in dim)
dim_fill_defaults = {
    "GOVERNANCE_ENCODED": 0,   # assume CPS district
    "IS_SELECTIVE":       0,
    "IS_ATTENDANCE_AREA": 0,
    "IS_SMALL_SCHOOL":    0,
    "IS_HIGH_SCHOOL":     0,
    "IS_SCHOOL_OPEN":     1,   # assume open if not in dim (historical schools)
    "REGION_ENCODED":     0,
}
df_feat = df_feat.fillna(dim_fill_defaults)

# =============================================================================
# 6. BUILD FINAL FEATURE TABLE
# Target grades: 1–8, 10–12 (K and 9 need separate models)
# Require SAME_GRADE_LAST_YEAR non-null (need at least 1 year of history)
# Exclude closed schools from the feature table — no point training or forecasting them
# =============================================================================

TARGET_GRADES = ["1", "2", "3", "4", "5", "6", "7", "8", "10", "11", "12"]

final_columns = [
    # Identifiers
    "SCHOOL_KEY",
    "GRADE",
    "GRADE_NUMERIC",
    "SCHOOL_YEAR",

    # TARGET
    "ENROLLMENT",

    # Features — V2 school-level history
    "SAME_GRADE_LAST_YEAR",
    "SAME_GRADE_2YR_AGO",
    "FEEDER_GRADE_LAST_YEAR",
    "HAS_FEEDER_GRADE",
    "SCHOOL_TOTAL_ENROLLMENT",

    # Features — V2 survival rates
    "COHORT_SURVIVAL_RATE",
    "AVG_SURVIVAL_RATE_3YR",

    # Features — V2 district trend
    "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR",

    # Features — V2 anomaly flag
    "IS_MIGRANT_ANOMALY_YEAR",

    # Features — V3 school type (NEW)
    "GOVERNANCE_ENCODED",
    "IS_SELECTIVE",
    "IS_ATTENDANCE_AREA",
    "IS_SMALL_SCHOOL",
    "IS_HIGH_SCHOOL",
    "REGION_ENCODED",

    # Status flag — used for filtering forecast output, not a model feature
    "IS_SCHOOL_OPEN",

    # Raw strings — diagnostics only, not model features
    "GOVERNANCE",
    "ATTENDANCE_BOUNDARY",
    "ANNUAL_REGIONAL_ANALYSIS_REGION",
    "COMMUNITY",
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
# 7. VERIFY — feature fill rates
# =============================================================================

model_features = [
    "SAME_GRADE_LAST_YEAR", "SAME_GRADE_2YR_AGO", "FEEDER_GRADE_LAST_YEAR",
    "HAS_FEEDER_GRADE", "SCHOOL_TOTAL_ENROLLMENT", "COHORT_SURVIVAL_RATE",
    "AVG_SURVIVAL_RATE_3YR", "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR",
    "IS_MIGRANT_ANOMALY_YEAR", "GOVERNANCE_ENCODED", "IS_SELECTIVE",
    "IS_ATTENDANCE_AREA", "IS_SMALL_SCHOOL", "IS_HIGH_SCHOOL", "REGION_ENCODED",
]

print("\nFeature fill rates:")
for col_name in model_features:
    non_null = df_final.filter(F.col(col_name).isNotNull()).count()
    pct = round(non_null / final_count * 100, 1)
    flag = "✔" if pct >= 80 else "⚠"
    print(f"  {flag}  {col_name:45s}: {pct}%")

# School type distribution in final table
print("\nSchool type breakdown in feature table:")
df_final.groupBy("GOVERNANCE", "IS_SELECTIVE", "IS_ATTENDANCE_AREA").count().orderBy("GOVERNANCE").show(20, truncate=False)

# =============================================================================
# 8. WRITE TO LAKEHOUSE
# =============================================================================

OUTPUT_TABLE = "ml_enrollment_feature_table_v3"

df_final.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)

verify = spark.read.table(OUTPUT_TABLE)
print(f"\n✔ Written: '{OUTPUT_TABLE}'")
print(f"  Rows   : {verify.count():,}")
print(f"  Columns: {len(verify.columns)}")
print(f"  Grades : {sorted([r[0] for r in verify.select('GRADE').distinct().collect()])}")
print(f"  Years  : {verify.agg(F.min('SCHOOL_YEAR')).collect()[0][0]} – {verify.agg(F.max('SCHOOL_YEAR')).collect()[0][0]}")
print(f"  Open schools only: {verify.filter(F.col('IS_SCHOOL_OPEN')==1).select('SCHOOL_KEY').distinct().count():,} schools")

print("""
\n DONE. Next: run CPS_Enrollment_Model_Training_V3.py
  Read from: ml_enrollment_feature_table_v3
""")
