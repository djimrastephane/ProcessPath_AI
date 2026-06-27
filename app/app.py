"""ProcessPath_AI — Streamlit app."""
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

ROOT        = Path(__file__).parent.parent
T           = ROOT / "outputs" / "tables"
F           = ROOT / "outputs" / "figures"
MODEL       = ROOT / "app" / "model" / "prefix_k8.joblib"
MODEL_REG   = ROOT / "app" / "model" / "remaining_time_k8.joblib"
MODEL_COX   = ROOT / "app" / "model" / "survival_cox_k8.joblib"

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
    ["Overview", "Bottlenecks", "Conformance", "Early Warning", "Remaining Time", "Survival Analysis", "Violation Root Cause"],
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

@st.cache_resource
def load_reg_model():
    return joblib.load(MODEL_REG)

@st.cache_resource
def load_cox_model():
    return joblib.load(MODEL_COX)

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
- **Early warning model at k=8**: AUC **0.810** (leakage-free, `elapsed_days` excluded) — deployable at `Permit FINAL_APPROVED`
- **Data drift confirmed**: `elapsed_days` halved from 2017Q1 → 2018Q4; standard k-fold overstates AUC by +0.048
- **Temporal leakage identified**: `elapsed_days` alone scores AUC 0.833 and was excluded from the deployed model (Notebook 10)
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
        st.caption(
            "⚠️ These curves include `elapsed_days`. k=8 drops from 0.967 → **0.810** "
            "when `elapsed_days` is removed (see Leakage Audit below)."
        )
    with col_r:
        st.subheader("k-fold vs Temporal CV Bias")
        st.image(str(F / "temporal_vs_kfold_bias.png"), use_container_width=True)
        st.caption("⚠️ Temporal CV AUC at k=8 (0.967) includes `elapsed_days`. Honest value = **0.810**.")

    st.markdown("---")
    st.subheader("Leakage Audit (Notebook 10)")
    col_l, col_r = st.columns(2)
    with col_l:
        st.image(str(F / "leakage_ablation_auc.png"), use_container_width=True)
        st.caption("Ablation: removing `elapsed_days` drops AUC by −0.157. Reminders: no change.")
    with col_r:
        st.image(str(F / "leakage_elapsed_days_dist.png"), use_container_width=True)
        st.caption("`elapsed_days` alone scores AUC 0.833 — distributions are almost non-overlapping.")

    st.markdown("---")
    st.subheader("Calibration")
    st.image(str(F / "leakage_calibration.png"), use_container_width=True)
    st.caption("Brier score 0.066, skill score 0.70. Raw probabilities are reliable; Platt scaling not needed.")

    st.markdown("---")
    st.subheader("SHAP Feature Importance")
    col_l, col_r = st.columns(2)
    with col_l:
        st.caption("Complete model (retrospective — not deployed)")
        st.image(str(F / "shap_summary_complete.png"), use_container_width=True)
    with col_r:
        st.caption("Prefix k=5 model (includes `elapsed_days`)")
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

# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — REMAINING TIME
# ══════════════════════════════════════════════════════════════════════════
elif page == "Remaining Time":
    st.title("Remaining Time Prediction")
    st.caption(
        "Predicts how many days remain in a case after observing the first 8 events. "
        "XGBoost regression with P10 / P50 / P90 quantile uncertainty bounds."
    )

    reg_bundle = load_reg_model()
    reg_tcv    = load_table("remaining_time_temporal_cv.csv")
    reg_prefix = load_table("remaining_time_by_prefix.csv")

    mae_cv  = reg_bundle["mae_temporal_cv"]
    r2_cv   = reg_bundle["r2_temporal_cv"]
    mae_h   = reg_bundle["mae_holdout"]
    r2_h    = reg_bundle["r2_holdout"]
    cov     = reg_bundle["coverage_p10_p90"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Model",                "XGBoost k=8")
    col2.metric("MAE (holdout)",        f"{mae_h:.1f}d",  help="Mean absolute error on 20% holdout")
    col3.metric("MAE (temporal CV)",    f"{mae_cv:.1f}d", help="Mean across Q1–Q4 2018 expanding-window folds")
    col4.metric("R² (temporal CV)",     f"{r2_cv:.3f}")
    col5.metric("P10–P90 coverage",     f"{cov:.1%}",     help="Fraction of actuals inside the 80% prediction interval")

    st.info(
        "`elapsed_days` **is included** in this model — unlike the early warning classifier, "
        "it is not leaky here. For regression the question is *how much time is left*, "
        "and elapsed time is genuine signal about how fast the case has been moving.",
        icon="ℹ️",
    )

    st.markdown("---")
    st.subheader("MAE vs Prefix Length")
    col_l, col_r = st.columns(2)
    with col_l:
        st.image(str(F / "remaining_time_mae_curve.png"), use_container_width=True)
        st.caption("MAE drops and R² rises as more events are observed. k=8 is the sweet spot for early intervention.")
    with col_r:
        st.subheader("MAE by prefix length")
        st.dataframe(
            reg_prefix[["k", "MAE", "RMSE", "R2", "baseline_MAE"]]
            .rename(columns={"k": "Prefix k", "baseline_MAE": "Baseline MAE"})
            .style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}", "R2": "{:.3f}", "Baseline MAE": "{:.1f}"}),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Predicted vs Actual  |  Residuals")
    st.image(str(F / "remaining_time_pred_vs_actual.png"), use_container_width=True)
    st.caption(
        "Left: each dot is a test case. Perfect predictions lie on the red diagonal. "
        "Right: residuals are right-skewed — the model under-predicts for very long cases, "
        "which is common when tail durations are rare in training data."
    )

    st.markdown("---")
    st.subheader("Quantile Prediction Intervals (P10 / P50 / P90)")
    st.image(str(F / "remaining_time_quantile_intervals.png"), use_container_width=True)
    st.caption(
        f"Blue band = 80% prediction interval. Red dots = actual remaining days. "
        f"Coverage {cov:.1%} (target 80%). Case managers see a best-case / median / worst-case estimate."
    )

    st.markdown("---")
    st.subheader("SHAP Feature Importance")
    st.image(str(F / "remaining_time_shap_beeswarm.png"), use_container_width=True)
    st.caption(
        "`elapsed_days` is the dominant driver — slow-moving cases have more time left. "
        "`n_rejections` and rejection-flag features push remaining time up significantly."
    )

    st.markdown("---")
    st.subheader("Single-Case Prediction")
    st.caption("Enter case features to get a point estimate plus P10–P90 uncertainty range.")

    feat_cols_r = reg_bundle["feature_cols"]
    imputer_r   = reg_bundle["imputer"]
    m_point     = reg_bundle["model_point"]
    m_p10       = reg_bundle["model_p10"]
    m_p50       = reg_bundle["model_p50"]
    m_p90       = reg_bundle["model_p90"]

    with st.form("reg_prediction_form"):
        c1, c2, c3 = st.columns(3)
        elapsed_d      = c1.number_input("Elapsed days so far",        0.0, 500.0, 10.0, step=1.0)
        n_rejections_r = c2.number_input("Rejections in prefix",       0, 20, 0)
        n_reminders_r  = c3.number_input("Send Reminders in prefix",   0, 10, 0)

        c4, c5, c6 = st.columns(3)
        n_approvals_r  = c4.number_input("Approvals in prefix",        0, 20, 2)
        n_events_r     = c5.number_input("Events in prefix (≤8)",      1,  8, 8)
        start_month_r  = c6.number_input("Case start month (1–12)",    1, 12, 6)

        c7, c8 = st.columns(2)
        has_reminder_r  = c7.checkbox("Has Send Reminder in prefix")
        has_final_r     = c8.checkbox("Has Permit FINAL_APPROVED in prefix")

        c9, c10 = st.columns(2)
        has_rejected_r  = c9.checkbox("Has any REJECTED activity in prefix")
        has_approved_r  = c10.checkbox("Has any APPROVED activity in prefix")

        submitted_r = st.form_submit_button("Predict remaining time", type="primary")

    if submitted_r:
        row_r = {c: 0 for c in feat_cols_r}
        row_r["elapsed_days"]        = elapsed_d
        row_r["n_events_prefix"]     = n_events_r
        row_r["n_rejections"]        = n_rejections_r
        row_r["n_reminders"]         = n_reminders_r
        row_r["n_approvals"]         = n_approvals_r
        row_r["case_start_month"]    = start_month_r
        if "has_Send_Reminder" in feat_cols_r:
            row_r["has_Send_Reminder"] = int(has_reminder_r)
        if "has_Permit_FINAL_APPROVED_by_SUPERVISOR" in feat_cols_r:
            row_r["has_Permit_FINAL_APPROVED_by_SUPERVISOR"] = int(has_final_r)
        if has_rejected_r and "has_Declaration_REJECTED_by_DIRECTOR" in feat_cols_r:
            row_r["has_Declaration_REJECTED_by_DIRECTOR"] = 1
        if has_approved_r and "has_Declaration_APPROVED_by_SUPERVISOR" in feat_cols_r:
            row_r["has_Declaration_APPROVED_by_SUPERVISOR"] = 1

        X_r    = pd.DataFrame([row_r])[feat_cols_r]
        X_r_imp = pd.DataFrame(imputer_r.transform(X_r), columns=feat_cols_r)

        pred_point = max(0.0, float(m_point.predict(X_r_imp)[0]))
        pred_p10   = max(0.0, float(m_p10.predict(X_r_imp)[0]))
        pred_p50   = max(0.0, float(m_p50.predict(X_r_imp)[0]))
        pred_p90   = max(0.0, float(m_p90.predict(X_r_imp)[0]))

        st.markdown("---")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Point estimate",  f"{pred_point:.1f} days")
        rc2.metric("P10 (optimistic)", f"{pred_p10:.1f} days")
        rc3.metric("P50 (median)",     f"{pred_p50:.1f} days")
        rc4.metric("P90 (pessimistic)", f"{pred_p90:.1f} days")

        # simple bar showing the interval
        fig_r, ax_r = plt.subplots(figsize=(8, 1.8))
        ax_r.barh(0, pred_p90 - pred_p10, left=pred_p10,
                  height=0.4, color='#2c7bb6', alpha=0.35, label='P10–P90 interval')
        ax_r.axvline(pred_point, color='#d7191c', linewidth=2, label=f'Point: {pred_point:.1f}d')
        ax_r.axvline(pred_p50,   color='#2c7bb6', linewidth=2, linestyle='--',
                     label=f'P50: {pred_p50:.1f}d')
        ax_r.set_xlabel('Remaining days')
        ax_r.set_yticks([])
        ax_r.legend(loc='upper right', fontsize=8)
        ax_r.set_title('Prediction interval')
        plt.tight_layout()
        st.pyplot(fig_r, use_container_width=True)
        plt.close()

        # SHAP waterfall
        explainer_r = shap.TreeExplainer(m_point)
        sv_r = explainer_r(X_r_imp)
        fig_s, _ = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(sv_r[0], max_display=12, show=False)
        st.pyplot(fig_s, use_container_width=True)
        plt.close()

    st.markdown("---")
    with st.expander("Temporal CV results by fold"):
        st.dataframe(
            reg_tcv[["fold", "n_train", "n_test", "MAE", "RMSE", "R2", "baseline_MAE"]]
            .style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}", "R2": "{:.3f}", "baseline_MAE": "{:.1f}"}),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════
# PAGE 6 — SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Survival Analysis":
    st.title("Survival Analysis")
    st.caption(
        "Kaplan-Meier curves and Cox Proportional Hazards model on all 7,065 cases — "
        "including the 991 stuck (right-censored) cases excluded from earlier models."
    )

    cox_bundle = load_cox_model()
    surv_dept  = load_table("survival_by_department.csv")
    surv_summ  = load_table("survival_summary.csv")
    cox_hr     = load_table("survival_cox_hazard_ratios.csv")

    concordance   = cox_bundle["concordance_idx"]
    n_censored    = cox_bundle["n_censored"]
    cens_rate     = cox_bundle["censoring_rate"]
    median_surv   = cox_bundle["median_survival"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total cases",          "7,065",  help="All cases including stuck")
    col2.metric("Censored (stuck)",     f"{n_censored:,}", f"{cens_rate:.0%} of all cases",
                delta_color="inverse")
    col3.metric("KM median survival",   f"{median_surv:.0f}d",
                help="Median days until completion (accounting for censoring)")
    col4.metric("Cox concordance",      f"{concordance:.3f}",
                help="Survival equivalent of AUC — ability to rank cases by speed of completion")

    st.info(
        "**What is right-censoring?** The 991 stuck cases (last event = `Send Reminder`) "
        "never completed during the observation window. We know they lasted *at least* X days, "
        "but not how much longer. Kaplan-Meier and Cox models handle this correctly; "
        "simply excluding them would bias duration estimates downward.",
        icon="ℹ️",
    )

    st.markdown("---")
    st.subheader("Kaplan-Meier Survival Curves")
    st.image(str(F / "survival_km_curves.png"), use_container_width=True)
    st.caption(
        "Left: overall curve — the plateau after ~200d reflects censored cases that haven't completed. "
        "Centre: survival varies substantially across departments. "
        "Right: cases with any rejection take significantly longer to complete (log-rank p ≈ 0)."
    )

    st.markdown("---")
    st.subheader("Survival by Department")
    col_l, col_r = st.columns(2)
    with col_l:
        st.dataframe(
            surv_dept.rename(columns={
                "department": "Department",
                "n_cases": "Cases",
                "n_censored": "Censored",
                "censoring_rate": "Censoring rate",
                "median_survival_days": "Median survival (days)",
            }).style.format({
                "Censoring rate": "{:.1%}",
                "Median survival (days)": "{:.0f}",
            }),
            use_container_width=True,
        )
    with col_r:
        st.markdown(
            "Departments with high censoring rates have a large proportion of permanently stuck cases. "
            "Median survival is estimated by Kaplan-Meier, which accounts for censoring — "
            "it is more accurate than a simple mean of completed durations."
        )

    st.markdown("---")
    st.subheader("Cox Proportional Hazards — Hazard Ratios")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.image(str(F / "survival_cox_hazard_ratios.png"), use_container_width=True)
        st.caption(
            "HR > 1 (red) = feature associated with **faster** completion. "
            "HR < 1 (blue) = feature associated with **slower** completion / higher stuck risk. "
            f"Concordance index: {concordance:.3f}."
        )
    with col_r:
        sig_hr = cox_hr[cox_hr["p"] < 0.05].sort_values("HR", ascending=False).head(10)
        if not sig_hr.empty:
            st.dataframe(
                sig_hr[["covariate", "HR", "HR_lo", "HR_hi", "p"]]
                .rename(columns={
                    "covariate": "Feature",
                    "HR_lo": "HR low 95%",
                    "HR_hi": "HR high 95%",
                })
                .style.format({"HR": "{:.3f}", "HR low 95%": "{:.3f}", "HR high 95%": "{:.3f}", "p": "{:.2e}"}),
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("KM Curves by Cox Risk Group")
    st.image(str(F / "survival_km_risk_groups.png"), use_container_width=True)
    st.caption(
        "Cases ranked by their Cox partial hazard score and split into tertiles. "
        "The high-risk group (red) shows a long tail and high censoring — "
        "these are the cases most likely to get permanently stuck. "
        "Low-risk cases (green) complete quickly with almost no censoring."
    )

# ══════════════════════════════════════════════════════════════════════════
# PAGE 7 — VIOLATION ROOT CAUSE
# ══════════════════════════════════════════════════════════════════════════
elif page == "Violation Root Cause":
    st.title("Violation Root Cause Analysis")
    st.caption(
        "17.1% of cases have a travel-ordering violation — departed before permit submitted (Type A) "
        "or before approval (Type B). This page identifies which departments, budgets, and case "
        "characteristics drive those violations."
    )

    rca_summ  = load_table("violation_rca_summary.csv")
    rca_dept  = load_table("violation_rca_by_department.csv")
    rca_segs  = load_table("violation_rca_segments.csv")

    dt_auc  = rca_summ["dt_cv_auc"].iloc[0]
    xgb_auc = rca_summ["xgb_cv_auc"].iloc[0]
    n_vio   = int(rca_summ["n_violations"].iloc[0])
    n_a     = int(rca_summ["n_type_a"].iloc[0])
    n_b     = int(rca_summ["n_type_b"].iloc[0])
    top_dept      = rca_summ["top_risk_dept"].iloc[0]
    top_dept_rate = rca_summ["top_risk_dept_rate"].iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total violations",   f"{n_vio:,}",  "17.1% of cases", delta_color="inverse")
    col2.metric("Type A",             f"{n_a:,}",    "departed before submit", delta_color="inverse")
    col3.metric("Type B",             f"{n_b:,}",    "departed before approval", delta_color="inverse")
    col4.metric("Decision tree AUC",  dt_auc,        help="5-fold CV, depth=4 — interpretable rules")
    col5.metric("XGBoost AUC",        xgb_auc,       help="5-fold CV — best predictive performance")

    st.markdown("---")
    st.subheader("Violation Rate by Department, Budget & Season")
    st.image(str(F / "violation_rca_distributions.png"), use_container_width=True)
    st.caption(
        "Left: red bars = departments above the 17.1% baseline with 95% CI. "
        "Centre: violation rate rises with budget quartile — higher-budget trips are more likely to violate. "
        "Right: Q1 cases have the highest violation rate, suggesting year-start scheduling pressure."
    )

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("Department Risk Exposure")
        st.image(str(F / "violation_rca_dept_exposure.png"), use_container_width=True)
        st.caption(
            "Bar = absolute violation count (where to intervene for most impact). "
            "Dot = violation rate (where behaviour is worst). "
            f"Highest-volume violator: **{top_dept}** ({top_dept_rate} rate)."
        )
    with col_r:
        st.subheader("Department breakdown")
        st.dataframe(
            rca_dept.rename(columns={"dept": "Department", "n": "Cases",
                                     "vio": "Violations", "rate": "Rate", "ci": "±95% CI"})
            .sort_values("Rate", ascending=False)
            .style.format({"Rate": "{:.1%}", "±95% CI": "{:.3f}"}),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Decision Tree — Interpretable Rules (depth=4)")
    st.image(str(F / "violation_rca_decision_tree.png"), use_container_width=True)
    st.caption(
        f"Each leaf shows the violation proportion. CV AUC = {dt_auc}. "
        "Orange/red leaves are high-risk segments — the split features identify the strongest "
        "case-level predictors of travel-ordering violations."
    )

    st.markdown("---")
    st.subheader("SHAP Feature Importance (XGBoost)")
    st.image(str(F / "violation_rca_shap.png"), use_container_width=True)
    st.caption(
        f"XGBoost CV AUC = {xgb_auc}. Features ranked by mean |SHAP value|. "
        "Red = high feature value pushes toward violation; blue = low value. "
        "Duration and department are consistently the dominant drivers."
    )

    st.markdown("---")
    with st.expander("Decision tree leaf segments (high → low violation rate)"):
        st.dataframe(
            rca_segs.style.format({
                "violation_rate": "{:.1%}",
                "pct_of_all": "{:.1%}",
                "avg_duration": "{:.0f}",
                "avg_budget": "{:.0f}",
            }),
            use_container_width=True,
        )
