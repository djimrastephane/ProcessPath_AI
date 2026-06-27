"""ProcessPath_AI — FastAPI prediction service.

Three endpoints, three models, one JSON response:
  POST /predict  — early warning + remaining time + survival
  GET  /health   — liveness check
  GET  /models   — model metadata and performance metrics
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "app" / "model"

bundles: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundles["prefix"] = joblib.load(MODEL_DIR / "prefix_k8.joblib")
    bundles["reg"]    = joblib.load(MODEL_DIR / "remaining_time_k8.joblib")
    bundles["cox"]    = joblib.load(MODEL_DIR / "survival_cox_k8.joblib")
    yield
    bundles.clear()


app = FastAPI(
    title="ProcessPath_AI",
    description=(
        "Process mining prediction API for multi-stage approval workflows. "
        "Trained on BPI Challenge 2020 (7,065 travel permit cases). "
        "Three deployed models: early warning classifier (AUC 0.810), "
        "remaining time regressor (MAE 12.4d), Cox PH survival model (c=0.814)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / response schemas ─────────────────────────────────────────────

class CaseFeatures(BaseModel):
    """Features derived from the first k=8 events of a case prefix."""

    n_events: int = Field(8, ge=1, le=8,
        description="Number of events observed so far (1–8)")
    elapsed_days: float = Field(10.0, ge=0.0,
        description="Days elapsed from first to last event in the prefix "
                    "(used by remaining-time and survival models; not early warning)")
    n_rejections: int = Field(0, ge=0,
        description="Count of REJECTED activities in the prefix")
    n_reminders: int = Field(0, ge=0,
        description="Count of Send Reminder activities in the prefix")
    n_approvals: int = Field(2, ge=0,
        description="Count of APPROVED activities in the prefix")
    case_start_month: int = Field(6, ge=1, le=12,
        description="Calendar month when the case was opened (1=Jan … 12=Dec)")
    case_start_dow: int = Field(0, ge=0, le=6,
        description="Day-of-week when the case was opened (0=Monday … 6=Sunday)")
    has_send_reminder: bool = Field(False,
        description="True if a Send Reminder event has already occurred")
    has_final_approved: bool = Field(False,
        description="True if Permit FINAL_APPROVED by SUPERVISOR is in the prefix")
    has_any_rejection: bool = Field(False,
        description="True if any REJECTED activity is in the prefix")
    has_any_approval: bool = Field(True,
        description="True if any APPROVED activity is in the prefix")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "n_events": 8,
                    "elapsed_days": 12.5,
                    "n_rejections": 1,
                    "n_reminders": 0,
                    "n_approvals": 2,
                    "case_start_month": 3,
                    "case_start_dow": 1,
                    "has_send_reminder": False,
                    "has_final_approved": False,
                    "has_any_rejection": True,
                    "has_any_approval": True,
                }
            ]
        }
    }


class EarlyWarningResult(BaseModel):
    probability_late: float = Field(description="P(case duration > 101 days)")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    threshold_days: int = Field(101, description="Threshold used to define 'late'")
    model_auc: float = Field(0.810, description="Leakage-free temporal CV AUC")


class RemainingTimeResult(BaseModel):
    point_estimate_days: float
    p10_days: float = Field(description="Optimistic bound (10th percentile)")
    p50_days: float = Field(description="Median prediction")
    p90_days: float = Field(description="Pessimistic bound (90th percentile)")
    interval_width_days: float = Field(description="P90 − P10")
    mae_holdout: float = Field(description="Model MAE on holdout set (days)")


class SurvivalResult(BaseModel):
    prob_complete_by_30d: float
    prob_complete_by_60d: float
    prob_complete_by_90d: float
    prob_complete_by_180d: float
    median_survival_days: float
    concordance_index: float = Field(description="Cox model concordance (survival AUC)")


class PredictResponse(BaseModel):
    early_warning: EarlyWarningResult
    remaining_time: RemainingTimeResult
    survival: SurvivalResult
    meta: dict


# ── Feature construction ───────────────────────────────────────────────────

def _build_row(features: CaseFeatures, feat_cols: list[str],
               all_cols: list[str] | None = None) -> pd.DataFrame:
    """Build a single-row DataFrame.  all_cols overrides feat_cols for zeroing."""
    feat_cols = all_cols or feat_cols
    """Zero-fill all model columns then populate from the request."""
    row: dict = {c: 0 for c in feat_cols}

    row["n_events_prefix"]  = features.n_events
    row["n_rejections"]     = features.n_rejections
    row["n_reminders"]      = features.n_reminders
    row["n_approvals"]      = features.n_approvals
    row["elapsed_days"]     = features.elapsed_days
    row["case_start_month"] = features.case_start_month
    row["case_start_dow"]   = features.case_start_dow

    if "has_Send_Reminder" in row:
        row["has_Send_Reminder"] = int(features.has_send_reminder)
    if "has_Permit_FINAL_APPROVED_by_SUPERVISOR" in row:
        row["has_Permit_FINAL_APPROVED_by_SUPERVISOR"] = int(features.has_final_approved)
    if features.has_any_rejection:
        for col in ("has_Declaration_REJECTED_by_DIRECTOR",
                    "has_Declaration_REJECTED_by_EMPLOYEE",
                    "has_Permit_REJECTED_by_ADMINISTRATION"):
            if col in row:
                row[col] = 1
                break
    if features.has_any_approval:
        for col in ("has_Declaration_APPROVED_by_SUPERVISOR",
                    "has_Declaration_APPROVED_by_BUDGET_OWNER"):
            if col in row:
                row[col] = 1
                break

    return pd.DataFrame([row])[feat_cols]


def _impute(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Impute and return only the columns the model expects.

    The Cox bundle's imputer was fitted on a broader feature set (pre
    variance-filtering); after imputation we slice to the Cox feature_cols.
    """
    imp = bundle["imputer"]
    imp_cols = list(getattr(imp, "feature_names_in_", bundle["feature_cols"]))
    # Ensure X has exactly the columns the imputer expects
    X_full = X.reindex(columns=imp_cols, fill_value=0)
    X_imp  = pd.DataFrame(imp.transform(X_full), columns=imp_cols)
    # Slice to the columns the model actually uses
    return X_imp[bundle["feature_cols"]]


def _survival_at(sf: pd.DataFrame, days: float) -> float:
    """P(complete by `days`) = 1 − S(days) for a single-case survival function."""
    idx = int(np.searchsorted(sf.index.values, days, side="right")) - 1
    idx = max(0, min(idx, len(sf) - 1))
    return round(float(1.0 - sf.iloc[idx, 0]), 4)


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    loaded = list(bundles.keys())
    if len(loaded) < 3:
        raise HTTPException(status_code=503, detail="Models not fully loaded")
    return {"status": "ok", "models_loaded": loaded}


@app.get("/models", tags=["ops"])
def model_info():
    p = bundles.get("prefix", {})
    r = bundles.get("reg",    {})
    c = bundles.get("cox",    {})
    return {
        "early_warning": {
            "file":        "prefix_k8.joblib",
            "type":        "XGBoost classifier",
            "k":           8,
            "auc_cv":      p.get("honest_auc_cv"),
            "n_features":  len(p.get("feature_cols", [])),
            "note":        p.get("note"),
        },
        "remaining_time": {
            "file":              "remaining_time_k8.joblib",
            "type":              "XGBoost regression + quantile (P10/P50/P90)",
            "k":                 r.get("k", 8),
            "mae_holdout":       r.get("mae_holdout"),
            "r2_holdout":        r.get("r2_holdout"),
            "coverage_p10_p90":  r.get("coverage_p10_p90"),
            "n_features":        len(r.get("feature_cols", [])),
        },
        "survival": {
            "file":             "survival_cox_k8.joblib",
            "type":             "Cox Proportional Hazards",
            "concordance":      c.get("concordance_idx"),
            "n_cases":          c.get("n_cases"),
            "n_censored":       c.get("n_censored"),
            "censoring_rate":   c.get("censoring_rate"),
            "median_survival":  c.get("median_survival"),
            "n_features":       len(c.get("feature_cols", [])),
        },
    }


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(features: CaseFeatures):
    """
    Return early warning, remaining time, and survival predictions for a single
    case described by its first k=8 prefix events.

    - **early_warning**: P(case takes > 101 days) with LOW / MEDIUM / HIGH label
    - **remaining_time**: point estimate + P10 / P50 / P90 quantile bounds
    - **survival**: P(complete by 30 / 60 / 90 / 180 days) from Cox PH model
    """
    # ── Early warning ──────────────────────────────────────────────────────
    pb = bundles["prefix"]
    X_p = _build_row(features, pb["feature_cols"])
    X_p_imp = _impute(pb, X_p)
    prob_late = float(pb["model"].predict_proba(X_p_imp)[0, 1])
    risk_level = "HIGH" if prob_late >= 0.7 else ("MEDIUM" if prob_late >= 0.4 else "LOW")

    # ── Remaining time ─────────────────────────────────────────────────────
    rb = bundles["reg"]
    X_r = _build_row(features, rb["feature_cols"])
    X_r_imp = _impute(rb, X_r)
    pt  = max(0.0, float(rb["model_point"].predict(X_r_imp)[0]))
    p10 = max(0.0, float(rb["model_p10"].predict(X_r_imp)[0]))
    p50 = max(0.0, float(rb["model_p50"].predict(X_r_imp)[0]))
    p90 = max(0.0, float(rb["model_p90"].predict(X_r_imp)[0]))

    # ── Survival ───────────────────────────────────────────────────────────
    cb = bundles["cox"]
    imp_cols = list(getattr(cb["imputer"], "feature_names_in_", cb["feature_cols"]))
    X_c = _build_row(features, cb["feature_cols"], all_cols=imp_cols)
    X_c_imp = _impute(cb, X_c)
    sf = cb["model"].predict_survival_function(X_c_imp)  # lifelines → DataFrame

    # Median: first time where S(t) ≤ 0.5
    below = sf.index[sf.iloc[:, 0] <= 0.5]
    median_t = float(below[0]) if len(below) else float(cb.get("median_survival", 72.4))

    return PredictResponse(
        early_warning=EarlyWarningResult(
            probability_late=round(prob_late, 4),
            risk_level=risk_level,
            threshold_days=101,
            model_auc=pb.get("honest_auc_cv", 0.810),
        ),
        remaining_time=RemainingTimeResult(
            point_estimate_days=round(pt, 1),
            p10_days=round(p10, 1),
            p50_days=round(p50, 1),
            p90_days=round(p90, 1),
            interval_width_days=round(max(0.0, p90 - p10), 1),
            mae_holdout=rb.get("mae_holdout", 12.4),
        ),
        survival=SurvivalResult(
            prob_complete_by_30d=_survival_at(sf, 30),
            prob_complete_by_60d=_survival_at(sf, 60),
            prob_complete_by_90d=_survival_at(sf, 90),
            prob_complete_by_180d=_survival_at(sf, 180),
            median_survival_days=round(median_t, 1),
            concordance_index=cb.get("concordance_idx", 0.814),
        ),
        meta={
            "k": 8,
            "dataset": "BPI Challenge 2020 — Travel permits",
            "models": ["prefix_k8", "remaining_time_k8", "survival_cox_k8"],
        },
    )
