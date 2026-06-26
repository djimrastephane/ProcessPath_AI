"""ProcessPath_AI — Streamlit app."""
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

ROOT  = Path(__file__).parent.parent
T     = ROOT / "outputs" / "tables"
F     = ROOT / "outputs" / "figures"
MODEL = ROOT / "app" / "model" / "prefix_k8.joblib"

st.set_page_config(
    page_title="ProcessPath_AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ ProcessPath_AI")
st.sidebar.caption("BPI Challenge 2020 · TU/e Travel Permits")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Bottlenecks", "Conformance", "Early Warning"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption("7,065 cases · 86,581 events · 51 activities · 18 months")

# ── Cached loaders ─────────────────────────────────────────────────────────
@st.cache_data
def load_table(name):
    return pd.read_csv(T / name)

@st.cache_resource
def load_model():
    return joblib.load(MODEL)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Process Overview")

    dur   = load_table("case_durations.csv")
    feats = load_table("features.csv")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Cases",        "7,065")
    col2.metric("Events",       "86,581")
    col3.metric("Activities",   "51")
    col4.metric("Variants",     "1,478")
    col5.metric("Median duration", f"{dur['duration_days'].median():.0f}d")

    st.markdown("---")
    st.subheader("Summary Dashboard")
    st.image(str(F / "report_dashboard.png"), use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Case Duration Distribution")
        st.image(str(F / "case_duration_distribution.png"), use_container_width=True)

    with col_r:
        st.subheader("Monthly Event Volume")
        st.image(str(F / "monthly_event_volume.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("Top 20 Process Variants")
    st.image(str(F / "top20_variants.png"), use_container_width=True)

    with st.expander("Key findings"):
        st.markdown("""
- **991 cases (14%)** are permanently stuck — last event is `Send Reminder`; median duration **134d vs 63d** for resolved cases
- **17.1% compliance violations** — 746 Type A (departed before permit submitted), 583 Type B (departed before approval)
- **69% of case duration is voluntary scheduling** — admin steps process 87.8% of cases same-day
- **Early warning model at k=8**: AUC **0.967** (temporal CV), deployable at `Permit FINAL_APPROVED`
- **Data drift confirmed**: `elapsed_days` halved from 2017Q1 → 2018Q4; standard k-fold overstates AUC by +0.048
        """)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — BOTTLENECKS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Bottlenecks":
    st.title("Bottleneck Analysis")

    wait  = load_table("bottleneck_waiting_time.csv")
    comb  = load_table("bottleneck_combined.csv")
    stuck = load_table("stuck_cases.csv")
    dept  = load_table("bottleneck_by_department.csv")
    svr   = load_table("stuck_vs_resolved_duration.csv")

    wait["median_wait_d"] = wait["median_wait_h"] / 24
    wait["p95_wait_d"]    = wait["p95_wait_h"]    / 24

    col1, col2, col3 = st.columns(3)
    col1.metric("Stuck cases",         "991",   "14% of all cases", delta_color="inverse")
    col2.metric("Median — stuck",      "134d",  "+71d vs resolved",  delta_color="inverse")
    col3.metric("Median — resolved",   "63d")

    st.markdown("---")

    # Waiting time table
    st.subheader("Waiting Time by Activity")
    n = st.slider("Show top N activities", 5, 30, 15)
    top = wait.nlargest(n, "median_wait_d")[
        ["concept:name", "count", "median_wait_d", "p95_wait_d"]
    ].rename(columns={
        "concept:name": "Activity",
        "count": "Occurrences",
        "median_wait_d": "Median wait (days)",
        "p95_wait_d": "P95 wait (days)",
    }).reset_index(drop=True)
    st.dataframe(top.style.format({"Median wait (days)": "{:.1f}", "P95 wait (days)": "{:.1f}"}),
                 use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Combined Bottleneck View")
        st.image(str(F / "bottleneck_combined.png"), use_container_width=True)
    with col_r:
        st.subheader("Stuck vs Resolved Duration")
        st.image(str(F / "stuck_vs_resolved_duration.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("Scheduling vs Admin Delay Decomposition")
    st.image(str(F / "employee_wait_split.png"), use_container_width=True)
    with st.expander("Interpretation"):
        st.markdown("""
The delay between `Declaration SUBMITTED` and `Declaration FINAL_APPROVED` splits into:
- **Employee scheduling gap** — time from trip end to declaration submission (voluntary)
- **Admin processing time** — actual system processing (87.8% of cases resolved same-day)

Most of the observed duration is scheduling behaviour, not process inefficiency.
        """)

    st.markdown("---")
    st.subheader("Bottleneck by Department")
    st.image(str(F / "bottleneck_by_department.png"), use_container_width=True)

    with st.expander("Stuck cases — raw data"):
        st.dataframe(stuck, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — CONFORMANCE
# ══════════════════════════════════════════════════════════════════════════
elif page == "Conformance":
    st.title("Conformance Analysis")

    conf   = load_table("conformance_replay.csv")
    vio    = load_table("violation_by_department.csv")
    vio_a  = load_table("violation_type_a.csv")
    vio_b  = load_table("violation_type_b.csv")
    dev    = load_table("conformance_deviation_activities.csv")

    fit_mean    = conf["fitness"].mean()
    fit_perfect = (conf["fitness"] == 1.0).mean()
    n_vio       = (conf["fitness"] < 1.0).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean trace fitness",  f"{fit_mean:.3f}")
    col2.metric("Perfect-fit cases",   f"{fit_perfect:.1%}")
    col3.metric("Cases with violations", f"{n_vio:,}", delta_color="inverse")
    col4.metric("Violation rate",      f"{n_vio/len(conf):.1%}", delta_color="inverse")

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Trace Fitness Distribution")
        st.image(str(F / "conformance_fitness.png"), use_container_width=True)
    with col_r:
        st.subheader("Compliance Duration Impact")
        st.image(str(F / "compliance_duration.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("Travel-Ordering Violations by Department")
    vio_display = vio.rename(columns={
        "case:OrganizationalEntity": "Department",
        "type_a": "Type A (departed before submit)",
        "type_b": "Type B (departed before approval)",
        "compliant": "Compliant",
        "total": "Total",
        "pct_violation": "Violation rate",
    })
    dept_filter = st.multiselect(
        "Filter departments",
        options=vio_display["Department"].tolist(),
        default=vio_display["Department"].tolist(),
    )
    filtered = vio_display[vio_display["Department"].isin(dept_filter)]
    st.dataframe(
        filtered.style.format({"Violation rate": "{:.1%}"}),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Conformance by Department")
    st.image(str(F / "conformance_by_department.png"), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Type A violations")
        with st.expander("Show cases"):
            st.dataframe(vio_a, use_container_width=True)
    with col_r:
        st.subheader("Type B violations")
        with st.expander("Show cases"):
            st.dataframe(vio_b, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — EARLY WARNING
# ══════════════════════════════════════════════════════════════════════════
elif page == "Early Warning":
    st.title("Early Warning Model")
    st.caption("Predicts whether a case will take >101 days (long) based on the first 8 events.")

    prefix_auc = load_table("prefix_auc_results.csv")
    kfold_vs_t = load_table("temporal_vs_kfold.csv")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model",                  "XGBoost k=8")
    col2.metric("AUC — deployable",       "0.810", help="Without elapsed_days (leakage-free)")
    col3.metric("AUC — naïve (leaky)",    "0.967", help="elapsed_days included; inflated by temporal leakage")
    col4.metric("Threshold",              "101.1 days (P67)")

    st.info(
        "**Leakage note (Notebook 10):** `elapsed_days` alone scores AUC 0.833 — for slow cases "
        "the 8-event window already spans enough calendar time to reveal the outcome. "
        "The model is trained and deployed **without** `elapsed_days`. Honest AUC = **0.810**.",
        icon="⚠️",
    )

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("AUC vs Prefix Length")
        st.image(str(F / "prefix_auc_curve.png"), use_container_width=True)
    with col_r:
        st.subheader("k-fold vs Temporal CV Bias")
        st.image(str(F / "temporal_vs_kfold_bias.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("SHAP Feature Importance")
    col_l, col_r = st.columns(2)
    with col_l:
        st.caption("Complete model")
        st.image(str(F / "shap_summary_complete.png"), use_container_width=True)
    with col_r:
        st.caption("Prefix k=5 model")
        st.image(str(F / "shap_summary_prefix5.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("Temporal Stability")
    col_l, col_r = st.columns(2)
    with col_l:
        st.image(str(F / "temporal_auc_heatmap.png"), use_container_width=True)
    with col_r:
        st.image(str(F / "temporal_feature_drift.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("Single-Case Prediction")
    st.caption(
        "Enter the first 8 events of a case. `elapsed_days` is excluded (temporal leakage). "
        "Model AUC = 0.810."
    )

    bundle = load_model()
    model_k8  = bundle["model"]
    imputer   = bundle["imputer"]
    feat_cols = bundle["feature_cols"]

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        n_rejections   = c1.number_input("Rejections in prefix",     0, 20, 0)
        n_reminders    = c2.number_input("Send Reminders in prefix", 0, 10, 0)
        n_approvals    = c3.number_input("Approvals in prefix",      0, 20, 2)

        c4, c5 = st.columns(2)
        n_events       = c4.number_input("Events in prefix (≤8)",   1,  8, 8)
        start_month    = c5.number_input("Case start month (1–12)", 1, 12, 6)

        c6, c7 = st.columns(2)
        has_reminder   = c6.checkbox("Has Send Reminder in prefix")
        has_final_app  = c7.checkbox("Has Permit FINAL_APPROVED in prefix")

        c8, c9 = st.columns(2)
        has_rejected   = c8.checkbox("Has any REJECTED activity in prefix")
        has_approved   = c9.checkbox("Has any APPROVED activity in prefix")

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = {c: 0 for c in feat_cols}
        row["n_events_prefix"]    = n_events
        row["n_rejections"]       = n_rejections
        row["n_reminders"]        = n_reminders
        row["n_approvals"]        = n_approvals
        row["case_start_month"]   = start_month
        row["has_Send_Reminder"]  = int(has_reminder)
        row["has_Permit_FINAL_APPROVED_by_SUPERVISOR"] = int(has_final_app)
        if has_rejected:
            row["has_Declaration_REJECTED_by_DIRECTOR"] = 1
        if has_approved:
            row["has_Declaration_APPROVED_by_SUPERVISOR"] = 1

        X_row = pd.DataFrame([row])[feat_cols]
        X_imp = pd.DataFrame(imputer.transform(X_row), columns=feat_cols)
        prob  = model_k8.predict_proba(X_imp)[0, 1]
        label = "LONG  (>101 days)" if prob >= 0.5 else "SHORT  (≤101 days)"
        color = "🔴" if prob >= 0.5 else "🟢"

        st.markdown("---")
        rc1, rc2 = st.columns(2)
        rc1.metric("Prediction", f"{color} {label}")
        rc2.metric("P(long)", f"{prob:.1%}")

        # SHAP waterfall
        explainer = shap.TreeExplainer(model_k8)
        sv = explainer(X_imp)
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    with st.expander("Temporal CV results table"):
        st.dataframe(kfold_vs_t, use_container_width=True)
    with st.expander("Training size sensitivity"):
        st.image(str(F / "temporal_training_size.png"), use_container_width=True)
