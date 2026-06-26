"""Train and serialize the k=8 prefix model for the Streamlit app."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

ROOT = Path(__file__).parent.parent
DATA = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "app" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LONG_THRESHOLD_DAYS = 101.1
PREFIX_K = 8
RANDOM_STATE = 42

def load_event_df():
    from src.load_event_log import load_xes_log
    log = load_xes_log(ROOT / "data" / "raw" / "PermitLog.xes", legacy=True)
    rows = []
    for trace in log:
        cid = trace.attributes.get("concept:name", "")
        dept = trace.attributes.get("case:OrganizationalEntity", "")
        budget = trace.attributes.get("case:RequestedBudget", np.nan)
        for evt in trace:
            rows.append({
                "case_id": cid,
                "activity": str(evt["concept:name"]),
                "timestamp": evt["time:timestamp"],
                "org": str(evt.get("org:resource", "")),
                "dept": dept,
                "case:RequestedBudget": budget,
            })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    return df

def make_prefix_features(df, k):
    act_enc = LabelEncoder()
    act_enc.fit(df["activity"])

    ACTS = [
        "Send Reminder", "Declaration REJECTED by DIRECTOR",
        "Declaration REJECTED by EMPLOYEE", "Declaration REJECTED by BUDGET OWNER",
        "Permit REJECTED by ADMINISTRATION", "Permit REJECTED by BUDGET OWNER",
        "Permit REJECTED by PRE_APPROVER", "Permit REJECTED by DIRECTOR",
        "Permit REJECTED by MISSING", "Declaration APPROVED by DIRECTOR",
        "Declaration APPROVED by BUDGET OWNER", "Declaration APPROVED by SUPERVISOR",
        "Permit APPROVED by BUDGET OWNER", "Permit APPROVED by PRE_APPROVER",
        "Permit FINAL_APPROVED by SUPERVISOR",
    ]
    flag_cols = [f"has_{a.replace(' ', '_').replace(':', '_')}" for a in ACTS]

    case_starts = df.groupby("case_id")["timestamp"].min()
    case_ends   = df.groupby("case_id")["timestamp"].max()
    dur = (case_ends - case_starts).dt.total_seconds() / 86400
    is_long = (dur >= LONG_THRESHOLD_DAYS).astype(int)

    rows = []
    for cid, grp in df.groupby("case_id"):
        prefix = grp.head(k)
        t0 = prefix["timestamp"].iloc[0]
        elapsed = (prefix["timestamp"].iloc[-1] - t0).total_seconds() / 86400
        feats = {
            "case_id": cid,
            "elapsed_days": elapsed,
            "n_events_prefix": len(prefix),
            "n_rejections": prefix["activity"].str.contains("REJECTED").sum(),
            "n_reminders": (prefix["activity"] == "Send Reminder").sum(),
            "n_approvals": prefix["activity"].str.contains("APPROVED").sum(),
            "first_act_enc": act_enc.transform([prefix["activity"].iloc[0]])[0],
            "org_encoded": hash(str(prefix["org"].iloc[-1])) % 1000,
            "case:RequestedBudget": prefix["case:RequestedBudget"].iloc[0],
            "case_start_month": t0.month,
            "case_start_dow": t0.dayofweek,
        }
        acts_in_prefix = set(prefix["activity"])
        for act, col in zip(ACTS, flag_cols):
            feats[col] = int(act in acts_in_prefix)
        feats["is_long"] = is_long.get(cid, 0)
        rows.append(feats)

    return pd.DataFrame(rows)

print("Loading event log...")
df = load_event_df()
print(f"  {len(df):,} events, {df['case_id'].nunique():,} cases")

print(f"Building prefix features k={PREFIX_K}...")
feat_df = make_prefix_features(df, PREFIX_K)

feat_cols = [c for c in feat_df.columns if c not in ("case_id", "is_long")]
X = feat_df[feat_cols]
y = feat_df["is_long"]

X = X.drop(columns=[c for c in X.columns if X[c].isna().all()])
feat_cols = X.columns.tolist()

imp = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imp.fit_transform(X), columns=feat_cols)

print("Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
)
model.fit(X_imp, y)

joblib.dump({"model": model, "imputer": imp, "feature_cols": feat_cols},
            MODEL_DIR / "prefix_k8.joblib")
print(f"Saved to {MODEL_DIR / 'prefix_k8.joblib'}")
