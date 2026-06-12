# =============================================================================
# STAGE-1 (FEATURE ENGINEERING) — DROP-IN FIX CELLS
#
# Two correctness fixes to paste into the Fabric notebook:
#
#   FIX-A  Calendar-aware lags.  F.lag() is POSITIONAL: it grabs the previous
#          ROW, not necessarily the previous YEAR.  When a school x grade has a
#          gap year (closure / grade reconfig), "last year" silently points at
#          a 2- or 3-year-old row.  We null any lag whose year gap != expected.
#
#   FIX-B  Train-only target-encoded school effect.  Replaces the meaningless
#          numeric SCHOOL_KEY with each school's historical survival behaviour,
#          computed from TRAINING YEARS ONLY (no leakage).
#
# Paste order:
#   FIX-A  -> replace cell 5 (3a same-grade lags) AND the lag block in cell 7.
#             Cell 9's school/district lags get the same guard.  All shown below
#             as ready-to-paste replacements.
#   FIX-B  -> NEW cell, paste AFTER the feature table is split into train_df /
#             val_df (i.e. after cell 15), and add "SCHOOL_EFFECT" to
#             FEATURE_COLS in cell 17 (remove "SCHOOL_KEY").
# =============================================================================


# =============================================================================
# FIX-A — CALENDAR-AWARE LAGS
# =============================================================================
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

w_sg = Window.partitionBy("SCHOOL_KEY", "GRADE").orderBy(TIME_COL)

# Helper: a lag is only valid if the row it points at is exactly `n` years back.
def calendar_lag(colname, n):
    """lag(colname, n) but NULL unless the lagged row is exactly n years prior."""
    lagged_val  = F.lag(colname, n).over(w_sg)
    lagged_year = F.lag(TIME_COL, n).over(w_sg)
    return F.when(F.col(TIME_COL) - lagged_year == n, lagged_val).otherwise(F.lit(None))


# ---- REPLACEMENT for cell 5, block "3a. Same-grade lags" --------------------
# (everything else in cell 5 — the feeder joins — is already calendar-correct,
#  because those join on explicit year+1 / year+2, so leave them as-is.)
df_feat = (
    df_agg
    .withColumn("SAME_GRADE_LAST_YEAR", calendar_lag(LABEL_COL, 1))   # was F.lag(LABEL_COL,1)
    .withColumn("SAME_GRADE_2YR_AGO",   calendar_lag(LABEL_COL, 2))   # was F.lag(LABEL_COL,2)
)
# ... then keep the rest of cell 5 unchanged (feeder_lookup join, HAS_FEEDER_GRADE,
#     coalesce, feeder_2yr_lookup join) exactly as written. ...


# ---- REPLACEMENT for the lag block in cell 7 (Cohort Survival Rate) ---------
# _SR_CURRENT is computed exactly as before (unchanged):
#   df_feat = df_feat.withColumn("_SR_CURRENT", F.when(... ENROLLMENT/FEEDER ...))
# Only the LAGS change — use calendar_lag so a gap year can't masquerade as t-1:
df_feat = df_feat.withColumn("COHORT_SURVIVAL_RATE", calendar_lag("_SR_CURRENT", 1))
df_feat = (
    df_feat
    .withColumn("_SR_LAG2", calendar_lag("_SR_CURRENT", 2))
    .withColumn("_SR_LAG3", calendar_lag("_SR_CURRENT", 3))
    .withColumn(
        "AVG_SURVIVAL_RATE_3YR",
        (
            F.coalesce(F.col("COHORT_SURVIVAL_RATE"), F.lit(0)) +
            F.coalesce(F.col("_SR_LAG2"),             F.lit(0)) +
            F.coalesce(F.col("_SR_LAG3"),             F.lit(0))
        ) / F.greatest(
            F.when(F.col("COHORT_SURVIVAL_RATE").isNotNull(), 1).otherwise(0) +
            F.when(F.col("_SR_LAG2").isNotNull(),             1).otherwise(0) +
            F.when(F.col("_SR_LAG3").isNotNull(),             1).otherwise(0),
            F.lit(1)
        )
    )
    .drop("_SR_CURRENT", "_SR_LAG2", "_SR_LAG3")
)


# ---- REPLACEMENT for the lag blocks in cell 9 -------------------------------
# School total: guard the 1-year gap on its OWN window (per school, all grades).
w_school = Window.partitionBy("SCHOOL_KEY").orderBy(TIME_COL)
school_totals = (
    school_totals_raw          # built exactly as before: groupBy(SCHOOL_KEY, year).sum
    .withColumn(
        "SCHOOL_TOTAL_LAST_YEAR",
        F.when(
            F.col(TIME_COL) - F.lag(TIME_COL, 1).over(w_school) == 1,
            F.lag("_school_total", 1).over(w_school)
        ).otherwise(F.lit(None))
    )
    .select("SCHOOL_KEY", TIME_COL, "SCHOOL_TOTAL_LAST_YEAR")
)

# District-grade total: same guard on the per-grade window. (District-grade
# rarely has gaps, but guard it for consistency.)
w_dg = Window.partitionBy("GRADE").orderBy(TIME_COL)
district_grade_totals = (
    district_grade_totals      # built exactly as before: groupBy(GRADE, year).sum
    .withColumn(
        "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR",
        F.when(
            F.col(TIME_COL) - F.lag(TIME_COL, 1).over(w_dg) == 1,
            F.lag("_dg_total", 1).over(w_dg)
        ).otherwise(F.lit(None))
    )
    .drop("_dg_total")
)
# ... rest of cell 9 (the joins + migrant flag) unchanged.

print("FIX-A applied: lags are now calendar-aware (gap years -> NULL, "
      "imputed downstream by the train-only medians).")


# =============================================================================
# FIX-B — TRAIN-ONLY TARGET-ENCODED SCHOOL EFFECT  (replaces raw SCHOOL_KEY)
#
# Each school gets ONE number = its historical survival behaviour, learned from
# TRAINING years only.  This gives the model real per-school identity that also
# generalises to the validation/forecast years and to unseen schools.
#
# Paste this AFTER cell 15 (after train_df / val_df are created), and update
# FEATURE_COLS in cell 17:  remove "SCHOOL_KEY", add "SCHOOL_EFFECT".
# =============================================================================

# District-wide mean survival rate from TRAIN years — the fallback / smoothing
# target for schools with little or no history.
global_sr = (
    train_df
    .filter(F.col("COHORT_SURVIVAL_RATE").isNotNull())
    .agg(F.avg("COHORT_SURVIVAL_RATE").alias("g"))
    .collect()[0]["g"]
)
global_sr = float(global_sr) if global_sr is not None else 1.0
SMOOTHING = 20.0     # shrink small-school estimates toward the global mean
                     # (a school with few training rows shouldn't get an extreme value)

# Per-school stats from TRAIN ONLY: mean survival rate + how many rows support it.
school_stats = (
    train_df
    .filter(F.col("COHORT_SURVIVAL_RATE").isNotNull())
    .groupBy("SCHOOL_KEY")
    .agg(
        F.avg("COHORT_SURVIVAL_RATE").alias("_school_mean"),
        F.count(F.lit(1)).alias("_n"),
    )
    # Bayesian / additive smoothing: blend the school's own mean with the global
    # mean, weighted by how much data the school has.
    .withColumn(
        "SCHOOL_EFFECT",
        (F.col("_school_mean") * F.col("_n") + F.lit(global_sr) * F.lit(SMOOTHING))
        / (F.col("_n") + F.lit(SMOOTHING))
    )
    .select("SCHOOL_KEY", "SCHOOL_EFFECT")
)

# Attach to BOTH train and val (same train-derived map -> no leakage).
# Schools absent from training (new schools) fall back to the global mean.
train_df = train_df.join(school_stats, on="SCHOOL_KEY", how="left") \
                   .withColumn("SCHOOL_EFFECT",
                               F.coalesce(F.col("SCHOOL_EFFECT"), F.lit(global_sr)))
val_df   = val_df.join(school_stats, on="SCHOOL_KEY", how="left") \
                 .withColumn("SCHOOL_EFFECT",
                             F.coalesce(F.col("SCHOOL_EFFECT"), F.lit(global_sr)))

print(f"FIX-B applied: SCHOOL_EFFECT built from train only "
      f"(global fallback = {global_sr:.3f}, smoothing = {SMOOTHING:.0f}).")
print("  -> In cell 17 FEATURE_COLS: remove 'SCHOOL_KEY', add 'SCHOOL_EFFECT'.")

# NOTE for the walk-forward (Part B): the per-fold loop must rebuild SCHOOL_EFFECT
# from that fold's TRAIN slice (years < T) and attach to its test slice, using the
# same code above with `train`/`test` in place of `train_df`/`val_df`. Otherwise
# the backtest would leak future school behaviour into earlier folds.
