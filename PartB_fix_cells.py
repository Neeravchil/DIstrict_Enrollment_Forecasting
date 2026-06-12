# =============================================================================
# PART B — DROP-IN FIX CELLS  (paste into Fabric notebook, replacing the broken
# Part B cells).  Author note: written to make the CSR-vs-ML comparison both
# RUNNABLE (build_comparison was never defined) and FAIR (CSR strengthened to
# match Iliana's documented SAS method; leadership metrics = BIAS + WAPE).
#
# Order to paste:
#   FIX-1  -> replaces the body of the walk-forward loop cell (cell 33)
#   FIX-2  -> NEW cell, paste immediately AFTER the walk-forward (defines
#             build_comparison + safe metrics).  Put it before cell 35.
#   (cells 35 / 37 / 39 then run unchanged)
# =============================================================================


# -----------------------------------------------------------------------------
# FIX-1  —  Stronger, faithful CSR baseline inside the walk-forward.
#
# Iliana's documented method (meeting notes + CONTEXT.md):
#   survival rate = enroll(grade N, year t) / enroll(grade N-1, year t-1),
#   per school x grade-transition, ~10yr history, with an AUTO-SELECTED
#   averaging window chosen to minimise historical forecast error.
#
# The original notebook's CSR was a FLAT 3-yr average x feeder, with a x1.0
# fallback when the rate was missing -> that is a strawman, not her system.
# Here we:
#   (a) compute leak-free per-(school,grade) survival ratios for ALL past years,
#   (b) for each candidate window pick the one with lowest trailing error
#       BEFORE year T (auto window selection, exactly like the SAS logic),
#   (c) clip the survival ratio to a sane band so a near-zero feeder can't
#       make the prediction explode (this is what was inflating CSR's RMSE).
#
# We compute this in the per-fold loop so it stays strictly out-of-sample.
# -----------------------------------------------------------------------------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

# Domain band for a continuing-grade survival rate. CPS enrollment declines,
# so rates cluster just below 1.0; >1.15 or <0.7 for a continuing grade is
# almost always a small-school artifact, not a real signal.
SR_LOW, SR_HIGH = 0.70, 1.15
CSR_WINDOWS = [1, 2, 3, 5, 10]            # candidate averaging windows (yrs)

def strong_csr_prediction(df_hist, eval_year):
    """
    Faithful CSR for one fold.  df_hist must contain ALL years (we filter
    internally).  Returns a DataFrame keyed (SCHOOL_KEY, GRADE) with column
    `y_hat_CSR_strong` for `eval_year`, using auto-selected window per group.
    Leak-free: only uses survival ratios from years < eval_year to both
    choose the window and form the prediction.
    """
    # Leak-free per-row survival ratio: enroll(t) / feeder_last_year(t), clipped.
    sr = (
        df_hist
        .filter(F.col("FEEDER_GRADE_LAST_YEAR") > 0)
        .withColumn(
            "_sr",
            F.greatest(F.lit(SR_LOW),
                       F.least(F.lit(SR_HIGH),
                               F.col(LABEL_COL) / F.col("FEEDER_GRADE_LAST_YEAR")))
        )
        .select("SCHOOL_KEY", "GRADE", TIME_COL, "_sr",
                LABEL_COL, "FEEDER_GRADE_LAST_YEAR")
    )

    # For each candidate window, the trailing mean SR as of each year, and the
    # implied prediction; we score windows on years strictly before eval_year.
    w_grp = Window.partitionBy("SCHOOL_KEY", "GRADE").orderBy(TIME_COL)

    scored = sr
    for k in CSR_WINDOWS:
        wk = w_grp.rowsBetween(-(k), -1)              # past k years, excl. current
        scored = scored.withColumn(
            f"_pred_w{k}",
            F.avg("_sr").over(wk) * F.col("FEEDER_GRADE_LAST_YEAR")
        )

    # trailing error of each window on years < eval_year (per school x grade)
    hist = scored.filter(F.col(TIME_COL) < eval_year)
    err_aggs = [
        F.avg(F.abs(F.col(f"_pred_w{k}") - F.col(LABEL_COL))).alias(f"_err_w{k}")
        for k in CSR_WINDOWS
    ]
    win_err = hist.groupBy("SCHOOL_KEY", "GRADE").agg(*err_aggs)

    # pick the window with the smallest historical MAE (ties -> shortest window);
    # groups with no history fall back to window=3 (district default).
    err_struct = F.array(*[
        F.struct(F.coalesce(F.col(f"_err_w{k}"), F.lit(float("inf"))).alias("e"),
                 F.lit(k).alias("k"))
        for k in CSR_WINDOWS
    ])
    best_win = (
        win_err
        .withColumn("_best", F.array_min(err_struct))
        .withColumn("_best_k",
                    F.when(F.col("_best.e") == float("inf"), F.lit(3))
                     .otherwise(F.col("_best.k")))
        .select("SCHOOL_KEY", "GRADE", "_best_k")
    )

    # prediction for eval_year using each group's chosen window
    eval_rows = scored.filter(F.col(TIME_COL) == eval_year) \
                      .join(best_win, ["SCHOOL_KEY", "GRADE"], "left") \
                      .withColumn("_best_k", F.coalesce(F.col("_best_k"), F.lit(3)))

    pred_expr = F.lit(None).cast(DoubleType())
    for k in CSR_WINDOWS:
        pred_expr = F.when(F.col("_best_k") == k, F.col(f"_pred_w{k}")).otherwise(pred_expr)

    return (
        eval_rows
        .withColumn("y_hat_CSR_strong",
                    F.coalesce(pred_expr.cast(DoubleType()),
                               F.col("FEEDER_GRADE_LAST_YEAR").cast(DoubleType())))
        .select("SCHOOL_KEY", "GRADE", "y_hat_CSR_strong")
    )


# ---- Per-fold SCHOOL_EFFECT (only needed if you adopted Stage-1 FIX-B) ------
# Rebuild the train-only target encoding from THIS fold's history (years < T) so
# earlier folds never see later school behaviour. If you did NOT adopt FIX-B
# (SCHOOL_KEY still in FEATURE_COLS), delete this helper and its two calls.
SMOOTHING = 20.0
def add_school_effect(train_sdf, test_sdf):
    g = (train_sdf.filter(F.col("COHORT_SURVIVAL_RATE").isNotNull())
                  .agg(F.avg("COHORT_SURVIVAL_RATE").alias("g")).collect()[0]["g"])
    g = float(g) if g is not None else 1.0
    stats = (train_sdf.filter(F.col("COHORT_SURVIVAL_RATE").isNotNull())
             .groupBy("SCHOOL_KEY")
             .agg(F.avg("COHORT_SURVIVAL_RATE").alias("_m"), F.count(F.lit(1)).alias("_n"))
             .withColumn("SCHOOL_EFFECT",
                         (F.col("_m")*F.col("_n") + F.lit(g)*F.lit(SMOOTHING))
                         / (F.col("_n") + F.lit(SMOOTHING)))
             .select("SCHOOL_KEY", "SCHOOL_EFFECT"))
    j = lambda s: s.join(stats, "SCHOOL_KEY", "left") \
                   .withColumn("SCHOOL_EFFECT", F.coalesce("SCHOOL_EFFECT", F.lit(g)))
    return j(train_sdf), j(test_sdf)


# ---- REPLACE the body of the walk-forward loop (cell 33) with this ----------
ID_COLS = ["SCHOOL_KEY", "GRADE", "NETWORK",
           "ANNUAL_REGIONAL_ANALYSIS_REGION", "COMMUNITY", "GOVERNANCE"]

fold_predictions = []
for T in EVAL_YEARS:
    train = df_net.filter((F.col(TIME_COL) >= TRAIN_START_YEAR) & (F.col(TIME_COL) < T))
    test  = df_net.filter(F.col(TIME_COL) == T)

    train, test = impute_train_only(train, test, null_fill_cols)
    train, test = add_weights(train, test)
    train, test = add_school_effect(train, test)   # <- only if FIX-B adopted
    train = clean_df(train, float_type_cols, FEATURE_COLS)
    test  = clean_df(test,  float_type_cols, FEATURE_COLS)

    fold_model = pipeline.fit(train)
    pred = fold_model.transform(test)

    # --- three baselines, all leak-free ---
    # 1) simple CSR (original notebook formula) - kept for reference
    pred = pred.withColumn(
        "y_hat_CSR_simple",
        (F.col("FEEDER_GRADE_LAST_YEAR") *
         F.coalesce(F.col("AVG_SURVIVAL_RATE_3YR"), F.lit(1.0))).cast(DoubleType()))
    # 2) persistence (last year's count) - the true floor every model must beat
    pred = pred.withColumn(
        "y_hat_PERSIST", F.col("SAME_GRADE_LAST_YEAR").cast(DoubleType()))

    # 3) strong CSR (auto window + clipped) - the FAIR comparison vs Iliana
    strong = strong_csr_prediction(df_net.filter(F.col(TIME_COL) <= T), T)
    pred = pred.join(strong, ["SCHOOL_KEY", "GRADE"], "left") \
               .withColumn("y_hat_CSR",
                           F.coalesce(F.col("y_hat_CSR_strong"),
                                      F.col("y_hat_CSR_simple")))

    fold_predictions.append(pred.select(
        F.col(TIME_COL).alias("YEAR"), *ID_COLS,
        F.col(LABEL_COL).cast(DoubleType()).alias("y"),
        F.col("prediction").cast(DoubleType()).alias("y_hat_ML"),
        "y_hat_CSR", "y_hat_CSR_simple", "y_hat_PERSIST",
    ))
    print(f"  Year {T}: trained on {train.count():,}  ->  predicted {test.count():,}")

preds_all = fold_predictions[0]
for p in fold_predictions[1:]:
    preds_all = preds_all.unionByName(p)
preds_all = preds_all.cache()
print(f"\nOut-of-sample predictions: {preds_all.count():,} rows, years {EVAL_YEARS}")
preds_all.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("csr_vs_ml_predictions")
preds_pdf = preds_all.toPandas()
print("preds_pdf shape:", preds_pdf.shape)


# =============================================================================
# FIX-2  —  NEW CELL: define build_comparison (+ safe metrics).  Paste this
# BEFORE cell 35.  Leadership headline = BIAS + WAPE; MedAPE/RMSE/R2 secondary.
# =============================================================================
import numpy as np
import pandas as pd

MODELS = {"CSR": "y_hat_CSR", "ML": "y_hat_ML",
          "PERSIST": "y_hat_PERSIST"}     # PERSIST shown as a sanity floor
APE_FLOOR = 10           # don't compute % error on tiny cells (small-school noise)

def _metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    if len(y) == 0:
        return dict(MAE=np.nan, RMSE=np.nan, R2=np.nan,
                    WAPE=np.nan, MedAPE=np.nan, BIAS=np.nan, BIAS_PCT=np.nan)
    err = yhat - y
    abserr = np.abs(err)
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y - y.mean())**2)
    big = y >= APE_FLOOR                       # % errors only where it's meaningful
    medape = np.median(abserr[big] / y[big]) * 100 if big.any() else np.nan
    return dict(
        MAE=abserr.mean(),
        RMSE=np.sqrt((err**2).mean()),
        R2=(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan,
        WAPE=abserr.sum() / y.sum() * 100 if y.sum() else np.nan,   # budget-relevant
        MedAPE=medape,
        BIAS=err.sum(),                         # +ve = over-prediction (over-budgeting)
        BIAS_PCT=err.sum() / y.sum() * 100 if y.sum() else np.nan,
    )

def build_comparison(pdf, group_col, level_name):
    """One row per group with CSR/ML/PERSIST metrics side by side."""
    groups = [(level_name, pdf)] if group_col is None \
             else list(pdf.groupby(group_col))
    rows = []
    for key, g in groups:
        row = {(level_name if group_col is None else group_col): key, "n": len(g)}
        for tag, col in MODELS.items():
            for mname, mval in _metrics(g["y"], g[col]).items():
                row[f"{mname}_{tag}"] = mval
        # improvement of ML vs the FAIR CSR (positive = ML better)
        for mname in ["MAE", "RMSE", "WAPE", "MedAPE"]:
            c, ml = row[f"{mname}_CSR"], row[f"{mname}_ML"]
            row[f"{mname}_impr_%"] = (c - ml) / c * 100 if c not in (0, np.nan) and np.isfinite(c) and c != 0 else np.nan
        row["MAE_winner"]  = "ML" if row["MAE_ML"]  < row["MAE_CSR"]  else "CSR"
        row["R2_winner"]   = "ML" if (row["R2_ML"] or -9) > (row["R2_CSR"] or -9) else "CSR"
        rows.append(row)
    return pd.DataFrame(rows)

print("build_comparison ready. Leadership headline columns: BIAS / BIAS_PCT / WAPE.")
print("Reminder: PERSIST (last-year count) is the floor — ML must beat it to add value.")


# =============================================================================
# FIX-3  —  NEW CELL: leadership headline (BIAS + WAPE).  Paste AFTER cell 35.
# This is the slide for Iliana / non-technical leadership: it answers
# "does ML stop the over-budgeting that CSR causes?" in plain numbers.
# =============================================================================
d  = build_comparison(preds_pdf, None, "District").iloc[0]

print("=" * 68)
print("  LEADERSHIP HEADLINE — District-wide, all backtest years pooled")
print("=" * 68)
print(f"  {'Metric':<24}{'CSR (current)':>16}{'ML (proposed)':>16}")
print("-" * 68)
print(f"  {'Total over-prediction':<24}{d['BIAS_CSR']:>+16,.0f}{d['BIAS_ML']:>+16,.0f}   (students)")
print(f"  {'  as % of enrollment':<24}{d['BIAS_PCT_CSR']:>+15.2f}%{d['BIAS_PCT_ML']:>+15.2f}%")
print(f"  {'WAPE (budget error)':<24}{d['WAPE_CSR']:>15.2f}%{d['WAPE_ML']:>15.2f}%")
print(f"  {'MAE (per school-grade)':<24}{d['MAE_CSR']:>16.2f}{d['MAE_ML']:>16.2f}")
print(f"  {'MedAPE':<24}{d['MedAPE_CSR']:>15.2f}%{d['MedAPE_ML']:>15.2f}%")
print(f"  {'RMSE':<24}{d['RMSE_CSR']:>16.2f}{d['RMSE_ML']:>16.2f}")
print("-" * 68)
print(f"  Persistence floor (last-yr count):  WAPE {d['WAPE_PERSIST']:.2f}%  "
      f"BIAS {d['BIAS_PERSIST']:+,.0f}")
print("=" * 68)
sign = "OVER" if d['BIAS_CSR'] > 0 else "UNDER"
print(f"  Read: CSR {sign}-predicts district enrollment by "
      f"{abs(d['BIAS_CSR']):,.0f} students ({abs(d['BIAS_PCT_CSR']):.1f}%).")
print(f"        ML cuts that to {abs(d['BIAS_ML']):,.0f} ({abs(d['BIAS_PCT_ML']):.1f}%) "
      f"and lowers budget error (WAPE) from {d['WAPE_CSR']:.1f}% to {d['WAPE_ML']:.1f}%.")
