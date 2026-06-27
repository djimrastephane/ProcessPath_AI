"""ProcessPath_AI — Streamlit operational decision support app."""
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

ROOT      = Path(__file__).parent.parent
T         = ROOT / "outputs" / "tables"
F         = ROOT / "outputs" / "figures"
MODEL     = ROOT / "app" / "model" / "prefix_k8.joblib"
MODEL_REG = ROOT / "app" / "model" / "remaining_time_k8.joblib"
MODEL_COX = ROOT / "app" / "model" / "survival_cox_k8.joblib"

# ── Business-friendly activity labels ──────────────────────────────────────
BUSINESS_LABELS = {
    "Permit SUBMITTED by EMPLOYEE":                     "Permit Submission",
    "Permit SAVED by EMPLOYEE":                         "Permit Draft Saved",
    "Permit FOR_APPROVAL by SUPERVISOR":                "Supervisor Review",
    "Permit APPROVED by ADMINISTRATION":                "Administration Approval",
    "Permit APPROVED by BUDGET OWNER":                  "Budget Owner Approval",
    "Permit FINAL_APPROVED by SUPERVISOR":              "Supervisor Final Approval",
    "Permit FINAL_APPROVED by DIRECTOR":                "Director Final Approval",
    "Permit REJECTED by ADMINISTRATION":                "Administration Rejection",
    "Permit REJECTED by BUDGET OWNER":                  "Budget Owner Rejection",
    "Permit REJECTED by SUPERVISOR":                    "Supervisor Rejection",
    "Send Reminder":                                    "Escalation Reminder",
    "Start trip":                                       "Trip Start",
    "End trip":                                         "Trip End",
    "Declaration SUBMITTED by EMPLOYEE":                "Employee Declaration",
    "Declaration SAVED by EMPLOYEE":                    "Declaration Draft Saved",
    "Declaration APPROVED by SUPERVISOR":               "Supervisor Approval",
    "Declaration APPROVED by BUDGET OWNER":             "Budget Owner Approval",
    "Declaration APPROVED by PRE_APPROVER":             "Pre-approver Approval",
    "Declaration APPROVED by ADMINISTRATION":           "Administration Approval",
    "Declaration FINAL_APPROVED by SUPERVISOR":         "Supervisor Final Approval",
    "Declaration FINAL_APPROVED by DIRECTOR":           "Director Final Approval",
    "Declaration REJECTED by DIRECTOR":                 "Director Rejection",
    "Declaration REJECTED by EMPLOYEE":                 "Employee Withdrawal",
    "Declaration REJECTED by BUDGET OWNER":             "Budget Owner Rejection",
    "Declaration REJECTED by PRE_APPROVER":             "Pre-approver Rejection",
    "Request For Payment SUBMITTED by EMPLOYEE":        "Payment Request",
    "Request For Payment SAVED by EMPLOYEE":            "Payment Draft Saved",
    "Request For Payment APPROVED by BUDGET OWNER":     "Payment Budget Approval",
    "Request For Payment FINAL_APPROVED by SUPERVISOR": "Payment Supervisor Approval",
    "Request For Payment FINAL_APPROVED by DIRECTOR":   "Payment Director Approval",
    "Payment Handled":                                  "Payment Processed",
    "Request Payment":                                  "Payment Request",
}

# ── Intervention recommendations (keyed by technical activity name) ──────────
INTERVENTIONS = {
    "Send Reminder": (
        "Introduce automatic escalation after 14 days of inactivity; notify line manager if no response within 48 h.",
        "High",
    ),
    "Start trip": (
        "Enforce a minimum 14-day submission lead time; block trip requests submitted less than 14 days before departure.",
        "High",
    ),
    "Permit SUBMITTED by EMPLOYEE": (
        "Trigger automated administration notification within 2 business days of permit submission.",
        "High",
    ),
    "Permit APPROVED by ADMINISTRATION": (
        "Set a 2-business-day SLA for administration approval; trigger automated escalation on breach.",
        "High",
    ),
    "Permit FOR_APPROVAL by SUPERVISOR": (
        "Set a 3-day SLA for supervisor review; auto-approve recurring low-risk trip profiles.",
        "Medium",
    ),
    "Declaration SUBMITTED by EMPLOYEE": (
        "Require declaration submission within 5 days of trip end; send automated reminder at day 3.",
        "Medium",
    ),
    "Declaration APPROVED by SUPERVISOR": (
        "Batch-process routine approvals; target same-day turnaround for declarations under €500.",
        "Medium",
    ),
    "End trip": (
        "Automate trip-end notification to immediately trigger the declaration workflow.",
        "Medium",
    ),
    "Request For Payment SUBMITTED by EMPLOYEE": (
        "Auto-generate payment request after final declaration approval; remove manual step.",
        "Medium",
    ),
    "Permit FINAL_APPROVED by SUPERVISOR": (
        "Confirm permit is routed to administration automatically after supervisor sign-off.",
        "Medium",
    ),
    "Declaration REJECTED by DIRECTOR": (
        "Provide employees with a pre-submission checklist to reduce avoidable rejections.",
        "Low",
    ),
    "Permit REJECTED by ADMINISTRATION": (
        "Clarify submission requirements; add inline form validation to catch errors early.",
        "Low",
    ),
    "Payment Handled": (
        "Integrate payment processing with ERP to eliminate the manual handover step.",
        "Low",
    ),
}

_CONF_LABEL = {
    "High":   "✅ High",
    "Medium": "⚠️ Medium",
    "Low":    "❗ Low",
}

# ── Transition classification layer ─────────────────────────────────────────
# Keys use TECHNICAL activity names from the event log.
# Values: service_time | administrative_wait | scheduling_delay | escalation_delay | unclassified

TRANSITION_TYPES: dict[tuple[str, str], str] = {
    # ── Productive execution (service time) ──────────────────────────────────
    ("Start trip",                                 "End trip"):                                     "service_time",
    # ── Scheduling delays (employee owns the gap) ────────────────────────────
    ("Permit FINAL_APPROVED by SUPERVISOR",        "Start trip"):                                   "scheduling_delay",
    ("Permit FINAL_APPROVED by DIRECTOR",          "Start trip"):                                   "scheduling_delay",
    ("Permit APPROVED by BUDGET OWNER",            "Start trip"):                                   "scheduling_delay",
    ("Permit APPROVED by ADMINISTRATION",          "Start trip"):                                   "scheduling_delay",
    ("End trip",                                   "Declaration SUBMITTED by EMPLOYEE"):             "scheduling_delay",
    ("End trip",                                   "Declaration SAVED by EMPLOYEE"):                "scheduling_delay",
    ("Declaration FINAL_APPROVED by SUPERVISOR",   "Request For Payment SUBMITTED by EMPLOYEE"):    "scheduling_delay",
    ("Declaration FINAL_APPROVED by DIRECTOR",     "Request For Payment SUBMITTED by EMPLOYEE"):    "scheduling_delay",
    # ── Administrative wait (approval chain) ─────────────────────────────────
    ("Permit SAVED by EMPLOYEE",                   "Permit SUBMITTED by EMPLOYEE"):                 "administrative_wait",
    ("Permit SUBMITTED by EMPLOYEE",               "Permit FOR_APPROVAL by SUPERVISOR"):            "administrative_wait",
    ("Permit FOR_APPROVAL by SUPERVISOR",          "Permit APPROVED by ADMINISTRATION"):            "administrative_wait",
    ("Permit FOR_APPROVAL by SUPERVISOR",          "Permit FINAL_APPROVED by SUPERVISOR"):          "administrative_wait",
    ("Permit APPROVED by ADMINISTRATION",          "Permit FINAL_APPROVED by SUPERVISOR"):          "administrative_wait",
    ("Permit APPROVED by ADMINISTRATION",          "Permit FINAL_APPROVED by DIRECTOR"):            "administrative_wait",
    ("Permit APPROVED by ADMINISTRATION",          "Permit APPROVED by BUDGET OWNER"):              "administrative_wait",
    ("Permit APPROVED by BUDGET OWNER",            "Permit FINAL_APPROVED by SUPERVISOR"):          "administrative_wait",
    ("Permit APPROVED by BUDGET OWNER",            "Permit FINAL_APPROVED by DIRECTOR"):            "administrative_wait",
    ("Declaration SUBMITTED by EMPLOYEE",          "Declaration APPROVED by PRE_APPROVER"):         "administrative_wait",
    ("Declaration SUBMITTED by EMPLOYEE",          "Declaration APPROVED by SUPERVISOR"):           "administrative_wait",
    ("Declaration SUBMITTED by EMPLOYEE",          "Declaration FINAL_APPROVED by SUPERVISOR"):     "administrative_wait",
    ("Declaration SUBMITTED by EMPLOYEE",          "Declaration FINAL_APPROVED by DIRECTOR"):       "administrative_wait",
    ("Declaration APPROVED by PRE_APPROVER",       "Declaration APPROVED by SUPERVISOR"):           "administrative_wait",
    ("Declaration APPROVED by SUPERVISOR",         "Declaration APPROVED by BUDGET OWNER"):         "administrative_wait",
    ("Declaration APPROVED by SUPERVISOR",         "Declaration FINAL_APPROVED by SUPERVISOR"):     "administrative_wait",
    ("Declaration APPROVED by BUDGET OWNER",       "Declaration FINAL_APPROVED by SUPERVISOR"):     "administrative_wait",
    ("Declaration APPROVED by BUDGET OWNER",       "Declaration FINAL_APPROVED by DIRECTOR"):       "administrative_wait",
    ("Request For Payment SUBMITTED by EMPLOYEE",  "Request For Payment APPROVED by BUDGET OWNER"): "administrative_wait",
    ("Request For Payment SUBMITTED by EMPLOYEE",  "Request For Payment FINAL_APPROVED by SUPERVISOR"): "administrative_wait",
    ("Request For Payment SUBMITTED by EMPLOYEE",  "Request For Payment FINAL_APPROVED by DIRECTOR"): "administrative_wait",
    ("Request For Payment APPROVED by BUDGET OWNER","Request For Payment FINAL_APPROVED by SUPERVISOR"): "administrative_wait",
    ("Request For Payment FINAL_APPROVED by SUPERVISOR", "Payment Handled"):                        "administrative_wait",
    ("Request For Payment FINAL_APPROVED by DIRECTOR",   "Payment Handled"):                        "administrative_wait",
    # ── Rejection loops (correction cycle = admin wait) ──────────────────────
    ("Permit REJECTED by ADMINISTRATION",          "Permit SUBMITTED by EMPLOYEE"):                 "administrative_wait",
    ("Permit REJECTED by SUPERVISOR",              "Permit SUBMITTED by EMPLOYEE"):                 "administrative_wait",
    ("Permit REJECTED by BUDGET OWNER",            "Permit SUBMITTED by EMPLOYEE"):                 "administrative_wait",
    ("Declaration REJECTED by DIRECTOR",           "Declaration SUBMITTED by EMPLOYEE"):            "administrative_wait",
    ("Declaration REJECTED by EMPLOYEE",           "Declaration SUBMITTED by EMPLOYEE"):            "administrative_wait",
    ("Declaration REJECTED by BUDGET OWNER",       "Declaration SUBMITTED by EMPLOYEE"):            "administrative_wait",
    ("Declaration REJECTED by PRE_APPROVER",       "Declaration SUBMITTED by EMPLOYEE"):            "administrative_wait",
    # ── Escalation delays (reminder loops) ───────────────────────────────────
    ("Permit FOR_APPROVAL by SUPERVISOR",          "Send Reminder"):                                "escalation_delay",
    ("Declaration SUBMITTED by EMPLOYEE",          "Send Reminder"):                                "escalation_delay",
    ("Send Reminder",                              "Permit FOR_APPROVAL by SUPERVISOR"):            "escalation_delay",
    ("Send Reminder",                              "Permit APPROVED by ADMINISTRATION"):            "escalation_delay",
    ("Send Reminder",                              "Permit FINAL_APPROVED by SUPERVISOR"):          "escalation_delay",
    ("Send Reminder",                              "Declaration APPROVED by SUPERVISOR"):           "escalation_delay",
    ("Send Reminder",                              "Declaration FINAL_APPROVED by SUPERVISOR"):     "escalation_delay",
    ("Send Reminder",                              "Declaration SUBMITTED by EMPLOYEE"):            "escalation_delay",
}

TRANSITION_OWNERS: dict[tuple[str, str], str] = {
    ("Start trip",                                 "End trip"):                                     "Employee",
    ("Permit FINAL_APPROVED by SUPERVISOR",        "Start trip"):                                   "Employee",
    ("Permit FINAL_APPROVED by DIRECTOR",          "Start trip"):                                   "Employee",
    ("Permit APPROVED by BUDGET OWNER",            "Start trip"):                                   "Employee",
    ("Permit APPROVED by ADMINISTRATION",          "Start trip"):                                   "Employee",
    ("End trip",                                   "Declaration SUBMITTED by EMPLOYEE"):             "Employee",
    ("Declaration FINAL_APPROVED by SUPERVISOR",   "Request For Payment SUBMITTED by EMPLOYEE"):    "Employee",
    ("Permit SAVED by EMPLOYEE",                   "Permit SUBMITTED by EMPLOYEE"):                 "Employee",
    ("Permit SUBMITTED by EMPLOYEE",               "Permit FOR_APPROVAL by SUPERVISOR"):            "Supervisor",
    ("Permit FOR_APPROVAL by SUPERVISOR",          "Permit APPROVED by ADMINISTRATION"):            "Administration",
    ("Permit FOR_APPROVAL by SUPERVISOR",          "Permit FINAL_APPROVED by SUPERVISOR"):          "Supervisor",
    ("Permit APPROVED by ADMINISTRATION",          "Permit FINAL_APPROVED by SUPERVISOR"):          "Supervisor",
    ("Permit APPROVED by ADMINISTRATION",          "Permit FINAL_APPROVED by DIRECTOR"):            "Director",
    ("Permit APPROVED by ADMINISTRATION",          "Permit APPROVED by BUDGET OWNER"):              "Budget Owner",
    ("Permit APPROVED by BUDGET OWNER",            "Permit FINAL_APPROVED by SUPERVISOR"):          "Supervisor",
    ("Declaration SUBMITTED by EMPLOYEE",          "Declaration APPROVED by PRE_APPROVER"):         "Pre-approver",
    ("Declaration SUBMITTED by EMPLOYEE",          "Declaration APPROVED by SUPERVISOR"):           "Supervisor",
    ("Declaration SUBMITTED by EMPLOYEE",          "Declaration FINAL_APPROVED by SUPERVISOR"):     "Supervisor",
    ("Declaration APPROVED by PRE_APPROVER",       "Declaration APPROVED by SUPERVISOR"):           "Supervisor",
    ("Declaration APPROVED by SUPERVISOR",         "Declaration APPROVED by BUDGET OWNER"):         "Budget Owner",
    ("Declaration APPROVED by SUPERVISOR",         "Declaration FINAL_APPROVED by SUPERVISOR"):     "Supervisor",
    ("Declaration APPROVED by BUDGET OWNER",       "Declaration FINAL_APPROVED by SUPERVISOR"):     "Supervisor",
    ("Request For Payment SUBMITTED by EMPLOYEE",  "Request For Payment APPROVED by BUDGET OWNER"): "Budget Owner",
    ("Request For Payment SUBMITTED by EMPLOYEE",  "Request For Payment FINAL_APPROVED by SUPERVISOR"): "Supervisor",
    ("Request For Payment FINAL_APPROVED by SUPERVISOR", "Payment Handled"):                        "Finance",
    ("Send Reminder",                              "Permit APPROVED by ADMINISTRATION"):            "Administration",
    ("Send Reminder",                              "Declaration APPROVED by SUPERVISOR"):           "Supervisor",
}

TYPE_COLORS  = {
    "service_time":       "#2c7bb6",
    "administrative_wait":"#e08b00",
    "scheduling_delay":   "#d7191c",
    "escalation_delay":   "#7b2d8b",
    "unclassified":       "#aaaaaa",
}
TYPE_ICONS   = {
    "service_time":       "🔵",
    "administrative_wait":"🟠",
    "scheduling_delay":   "🔴",
    "escalation_delay":   "🟣",
    "unclassified":       "⚪",
}
TYPE_LABELS  = {
    "service_time":       "Productive execution",
    "administrative_wait":"Administrative wait",
    "scheduling_delay":   "Scheduling delay",
    "escalation_delay":   "Escalation delay",
    "unclassified":       "Unclassified",
}

def _classify_transition(from_act: str, to_act: str) -> str:
    """Explicit mapping first; then heuristic inference."""
    key = (from_act, to_act)
    if key in TRANSITION_TYPES:
        return TRANSITION_TYPES[key]
    fa, ta = from_act.lower(), to_act.lower()
    if ("reminder" in fa or "escalation" in fa or
            "reminder" in ta or "escalation" in ta):
        return "escalation_delay"
    if (any(w in fa for w in ("start", "begin", "check in", "open")) and
            any(w in ta for w in ("end", "finish", "check out", "close"))):
        return "service_time"
    if (any(w in fa for w in ("final_approved", "approved")) and
            any(w in ta for w in ("start", "begin", "execute", "submitted by employee"))):
        return "scheduling_delay"
    if ("rejected" in fa and "submitted" in ta):
        return "administrative_wait"
    if (any(w in fa for w in ("submitted", "for_approval", "saved", "approved")) and
            any(w in ta for w in ("approved", "final_approved", "handled"))):
        return "administrative_wait"
    return "unclassified"

def _get_owner(from_act: str, to_act: str, ttype: str) -> str:
    """Explicit mapping first; then heuristic from destination activity name."""
    key = (from_act, to_act)
    if key in TRANSITION_OWNERS:
        return TRANSITION_OWNERS[key]
    if ttype in ("service_time", "scheduling_delay"):
        return "Employee"
    ta = to_act.lower()
    if "administration" in ta:  return "Administration"
    if "director" in ta:        return "Director"
    if "supervisor" in ta:      return "Supervisor"
    if "budget owner" in ta:    return "Budget Owner"
    if "pre_approver" in ta:    return "Pre-approver"
    if "payment" in ta:         return "Finance"
    return "Unknown"

# Activities that are destinations of service_time transitions → avoidable delay = 0
_SERVICE_TIME_DESTS: set[str] = {
    to_act for (_, to_act), ttype in TRANSITION_TYPES.items()
    if ttype == "service_time"
}

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
    [
        "Overview", "Bottlenecks", "Conformance", "Early Warning",
        "Remaining Time", "Survival Analysis", "Violation Root Cause",
        "Process Variants",
    ],
    index=0,
)
st.sidebar.markdown("---")
use_biz_labels = st.sidebar.toggle(
    "Business labels", value=True,
    help="Show friendly activity names instead of raw event log terminology",
)
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Cost Parameters")
cost_per_day = st.sidebar.number_input(
    "Cost per waiting day (€)", min_value=0, max_value=50000, value=180, step=10,
    help="Fully-loaded cost of one person-day of delay (salary + overhead + opportunity cost)",
)
avg_travel_cost = st.sidebar.number_input(
    "Avg travel cost per case (€)", min_value=0, max_value=200000, value=2500, step=100,
    help="Average total travel spend per travel permit case",
)
avg_salary_cost = st.sidebar.number_input(
    "Avg salary cost per case/day (€)", min_value=0, max_value=10000, value=180, step=10,
    help="Average daily salary cost per case (used as alternative cost basis)",
)

# Live cost preview (uses pre-computed avoidable total from Bottlenecks page or fallback)
_AVOIDABLE_DAYS_FALLBACK = 69_000  # approximate; updated live on Bottlenecks page
_annual_impact = _AVOIDABLE_DAYS_FALLBACK * cost_per_day
st.sidebar.markdown(
    f"**Est. annual impact**  \n"
    f"≈ €{_annual_impact/1e6:.1f}M  \n"
    f"<small>({_AVOIDABLE_DAYS_FALLBACK:,} avoidable days × €{cost_per_day}/day)</small>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.caption("7,065 cases · 86,581 events · 51 activities · 18 months")


# ── Label helpers ──────────────────────────────────────────────────────────
def lbl(name: str) -> str:
    if use_biz_labels:
        return BUSINESS_LABELS.get(str(name), str(name))
    return str(name)

def lbl_col(s: pd.Series) -> pd.Series:
    if use_biz_labels:
        return s.map(lambda x: BUSINESS_LABELS.get(str(x), str(x)))
    return s


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
# PAGE 2 — BOTTLENECKS (operational decision support)
# ══════════════════════════════════════════════════════════════════════════
elif page == "Bottlenecks":
    st.title("Bottleneck Analysis")

    wait      = load_table("bottleneck_waiting_time.csv")
    comb      = load_table("bottleneck_combined.csv")
    stuck     = load_table("stuck_cases.csv")
    dept      = load_table("bottleneck_by_department.csv")
    svr       = load_table("stuck_vs_resolved_duration.csv")
    stuck_dept= load_table("stuck_by_department.csv")
    variants_df = load_table("variants.csv")

    # ── Derived metrics ────────────────────────────────────────────────────
    wait["median_wait_d"] = wait["median_wait_h"] / 24
    wait["mean_wait_d"]   = wait["mean_wait_h"]   / 24
    wait["p75_wait_d"]    = wait["p75_wait_h"]    / 24
    wait["p95_wait_d"]    = wait["p95_wait_h"]    / 24

    # Impact score: median wait × occurrences (delay-days)
    wait["impact_score_d"] = wait["median_wait_d"] * wait["count"]

    # P25 via log-normal symmetry: p25 ≈ median² / p75
    safe_p75             = wait["p75_wait_h"].replace(0, np.nan)
    wait["p25_wait_h"]   = (wait["median_wait_h"] ** 2 / safe_p75).fillna(wait["median_wait_h"] * 0.5)
    wait["p25_wait_d"]   = wait["p25_wait_h"] / 24

    # Avoidable delay
    wait["avoidable_d"]       = (wait["median_wait_d"] - wait["p25_wait_d"]).clip(lower=0)
    wait["total_actual_d"]    = wait["mean_wait_d"]  * wait["count"]
    wait["total_avoidable_d"] = wait["avoidable_d"]  * wait["count"]
    wait["total_baseline_d"]  = wait["p25_wait_d"]   * wait["count"]

    # Service-time destinations are productive work — avoidable delay = 0
    _svc_mask = wait["concept:name"].isin(_SERVICE_TIME_DESTS)
    wait.loc[_svc_mask, "avoidable_d"]       = 0.0
    wait.loc[_svc_mask, "total_avoidable_d"] = 0.0

    # Department impact: total_delay_d = mean_wait_d × n_events
    dept.columns = ["department", "n_events", "median_wait_h", "mean_wait_h",
                    "median_wait_d", "mean_wait_d"]
    dept["total_delay_d"] = dept["mean_wait_d"] * dept["n_events"]
    dept["share_pct"]     = dept["total_delay_d"] / dept["total_delay_d"].sum()
    dept = dept.merge(
        stuck_dept.rename(columns={"case:OrganizationalEntity": "department"}),
        on="department", how="left",
    )
    dept["stuck_count"] = dept["stuck_count"].fillna(0).astype(int)

    # Header KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Stuck cases",       "991",  "14% of all cases",  delta_color="inverse")
    col2.metric("Median — stuck",    "134d", "+71d vs resolved",  delta_color="inverse")
    col3.metric("Median — resolved", "63d")

    # ── ROI callout (computed here so it's visible before the tabs) ───────────
    _roi_df    = wait.sort_values("impact_score_d", ascending=False)
    _top_act   = _roi_df.iloc[0]
    _top_name  = lbl(_top_act["concept:name"])
    _top_days  = _top_act["impact_score_d"]
    _top_cost  = _top_days * cost_per_day
    _all_avoid = wait["total_avoidable_d"].sum()
    _top_avoid_days = _top_act["avoidable_d"] * _top_act["count"]
    _top_avoid_pct  = _top_avoid_days / _all_avoid if _all_avoid > 0 else 0
    _top_avoid_cost = _top_avoid_days * cost_per_day

    st.markdown("---")
    st.markdown(
        f"""
<div style="background:#fff3cd;border-left:5px solid #d7191c;padding:14px 18px;border-radius:6px;margin-bottom:8px">
<b>💡 Highest ROI intervention: <em>{_top_name}</em></b><br>
Fixing this single activity removes&nbsp;&nbsp;
<b>{_top_avoid_days:,.0f} delay-days</b>&nbsp;·&nbsp;
<b>≈ €{_top_avoid_cost/1e6:.1f}M annually</b>&nbsp;·&nbsp;
<b>{_top_avoid_pct:.0%} of total avoidable delay</b>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Activity Impact",
        "🔀 Process Flow",
        "🏢 Department Impact",
        "⏱ Avoidable Delay",
        "🎯 Priority Matrix",
        "📋 Detail",
    ])

    # ── Tab 1: Activity Impact ─────────────────────────────────────────────
    with tab1:
        st.subheader("Which bottlenecks matter most?")
        st.caption(
            "**Impact Score = median wait × case count** (delay-days). "
            "A rare slow step and a common fast step both score low. "
            "Only steps that are both slow and frequent rank at the top."
        )

        impact_df = wait.copy()
        impact_df["Activity"] = lbl_col(impact_df["concept:name"])
        impact_df = impact_df.sort_values("impact_score_d", ascending=False)

        top_n = st.slider("Show top N activities", 5, 20, 12, key="impact_n")
        display = impact_df.head(top_n)[
            ["Activity", "count", "median_wait_d", "impact_score_d"]
        ].rename(columns={
            "count":          "Cases",
            "median_wait_d":  "Median Wait (days)",
            "impact_score_d": "Impact Score (delay-days)",
        }).reset_index(drop=True)

        st.dataframe(
            display.style
            .format({"Median Wait (days)": "{:.1f}", "Impact Score (delay-days)": "{:,.0f}"})
            .background_gradient(subset=["Impact Score (delay-days)"], cmap="Reds"),
            use_container_width=True,
        )
        st.download_button(
            "⬇ Export CSV",
            data=display.to_csv(index=False),
            file_name="activity_impact.csv",
            mime="text/csv",
            key="dl_impact",
        )

        chart_data = impact_df.head(top_n).sort_values("impact_score_d")
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.45)))
        colors = ["#d7191c" if i == len(chart_data) - 1 else "#2c7bb6"
                  for i in range(len(chart_data))]
        ax.barh(chart_data["Activity"], chart_data["impact_score_d"],
                color=colors, alpha=0.85)
        ax.set_xlabel("Impact Score (delay-days = median wait × case count)")
        ax.set_title(f"Top {top_n} Activities by Total Delay Impact")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Intervention recommendations ───────────────────────────────────
        st.markdown("---")
        st.subheader("💡 Recommended Interventions")
        st.caption(
            "Recommendations for the top-N activities by impact score. "
            "**Confidence** reflects case count: ✅ ≥500 · ⚠️ 50–499 · ❗ <50."
        )

        count_lookup = wait.set_index("concept:name")["count"].to_dict()
        rows = []
        for tech_name, (rec, priority) in INTERVENTIONS.items():
            n_cases = int(count_lookup.get(tech_name, 0))
            if n_cases == 0:
                continue
            conf = "✅ High" if n_cases >= 500 else ("⚠️ Medium" if n_cases >= 50 else "❗ Low")
            rows.append({
                "Activity":       lbl(tech_name),
                "Cases":          n_cases,
                "Priority":       priority,
                "Confidence":     conf,
                "Recommendation": rec,
            })

        if rows:
            int_df = (
                pd.DataFrame(rows)
                .sort_values(["Priority", "Cases"], ascending=[True, False])
                .reset_index(drop=True)
            )
            # Sort order: High priority first
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            int_df["_ord"] = int_df["Priority"].map(priority_order)
            int_df = int_df.sort_values(["_ord", "Cases"], ascending=[True, False]).drop(
                columns="_ord"
            ).reset_index(drop=True)

            st.dataframe(
                int_df.style.apply(
                    lambda col: [
                        "background-color:#fde8e8" if v == "High"
                        else ("background-color:#fff3cd" if v == "Medium" else "")
                        for v in col
                    ] if col.name == "Priority" else [""] * len(col),
                    axis=0,
                ).format({"Cases": "{:,}"}),
                use_container_width=True,
                height=min(35 * len(int_df) + 38, 500),
            )
            st.download_button(
                "⬇ Export interventions CSV",
                data=int_df.to_csv(index=False),
                file_name="intervention_recommendations.csv",
                mime="text/csv",
                key="dl_interv",
            )

    # ── Tab 2: Value-Stream Analysis ──────────────────────────────────────
    with tab2:
        st.subheader("Value-Stream Analysis")
        st.caption(
            "Each arrow is coloured by transition type. "
            "Wait times reflect the delay **before entering** the destination activity. "
            "Only non-productive delays are counted as avoidable."
        )

        top_n_flow = st.slider("Show top N steps", 5, 15, 9, key="flow_n")
        top_variant_str = variants_df.iloc[0]["variant"]
        steps = [s.strip() for s in top_variant_str.split("->")][:top_n_flow]

        wait_lookup = wait.set_index("concept:name")["median_wait_d"].to_dict()

        # waits[i] = wait BEFORE steps[i] starts = delay on incoming transition to steps[i]
        # Edge from steps[i] → steps[i+1] carries wait = waits[i+1]
        waits      = [wait_lookup.get(s, 0.0) for s in steps]
        edge_waits = waits[1:]   # wait on each edge = wait before entering destination

        # Classify and assign ownership for each edge
        edge_types  = [_classify_transition(steps[i], steps[i + 1]) for i in range(len(steps) - 1)]
        edge_owners = [_get_owner(steps[i], steps[i + 1], edge_types[i]) for i in range(len(steps) - 1)]

        # ── DOT graph ─────────────────────────────────────────────────────
        dot_lines = [
            "digraph {",
            "  rankdir=TB;",
            '  node [shape=box style="rounded,filled" fillcolor="#f5f8fa" '
            'color="#555555" fontsize=11 margin="0.25,0.12"];',
            '  edge [fontsize=9];',
        ]
        for i, step in enumerate(steps):
            dot_lines.append(f'  n{i} [label="{lbl(step)}"];')

        for i, (ttype, w) in enumerate(zip(edge_types, edge_waits)):
            color    = TYPE_COLORS.get(ttype, "#aaaaaa")
            icon     = TYPE_ICONS.get(ttype, "⚪")
            lbl_type = TYPE_LABELS.get(ttype, ttype).replace("_", " ")
            w_lbl    = f"{w:.1f}d" if w > 0 else "—"
            dot_lines.append(
                f'  n{i} -> n{i+1} [label=" {w_lbl} " '
                f'color="{color}" fontcolor="{color}" penwidth=2.0];'
            )
        dot_lines.append("}")
        st.graphviz_chart("\n".join(dot_lines))

        # ── Legend ─────────────────────────────────────────────────────────
        leg_cols = st.columns(len(TYPE_LABELS))
        for col, (ttype, label) in zip(leg_cols, TYPE_LABELS.items()):
            col.markdown(f"{TYPE_ICONS[ttype]} **{label}**")

        st.markdown("---")

        # ── Process time decomposition ─────────────────────────────────────
        st.subheader("Process Time Decomposition")
        st.caption("Based on median wait at each transition on the most common path.")

        type_days: dict[str, float] = {t: 0.0 for t in TYPE_LABELS}
        for ttype, w in zip(edge_types, edge_waits):
            type_days[ttype] = type_days.get(ttype, 0.0) + w
        total_path_d = sum(type_days.values())

        pct = lambda v: f"{v / total_path_d:.0%}" if total_path_d > 0 else "—"

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("🔵 Productive",   f"{type_days['service_time']:.1f}d",       pct(type_days["service_time"]))
        d2.metric("🟠 Admin wait",   f"{type_days['administrative_wait']:.1f}d", pct(type_days["administrative_wait"]))
        d3.metric("🔴 Scheduling",   f"{type_days['scheduling_delay']:.1f}d",    pct(type_days["scheduling_delay"]))
        d4.metric("🟣 Escalation",   f"{type_days['escalation_delay']:.1f}d",    pct(type_days["escalation_delay"]))
        d5.metric("⚪ Unclassified", f"{type_days['unclassified']:.1f}d",        pct(type_days["unclassified"]))

        st.markdown("---")

        # ── Value-added / waste ratios ──────────────────────────────────────
        avoidable_d  = total_path_d - type_days["service_time"]
        va_ratio     = type_days["service_time"] / total_path_d if total_path_d > 0 else 0
        waste_ratio  = avoidable_d / total_path_d if total_path_d > 0 else 0
        cost_waste   = avoidable_d * cost_per_day

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Value-added ratio",   f"{va_ratio:.0%}",
                  help="Productive execution ÷ total path duration")
        r2.metric("Waste ratio",         f"{waste_ratio:.0%}",
                  delta_color="inverse",
                  help="Non-productive time ÷ total path duration")
        r3.metric("Total path time",     f"{total_path_d:.1f}d",
                  help="Sum of median transition waits on most common path")
        r4.metric("Est. cost of waste",
                  f"€{cost_waste/1e6:.1f}M" if cost_waste >= 1e6 else f"€{cost_waste:,.0f}",
                  delta_color="inverse",
                  help="Avoidable path days × cost per day (sidebar)")

        st.markdown("---")

        # ── Automated recommendations ───────────────────────────────────────
        dominant_type = max(type_days, key=type_days.get) if total_path_d > 0 else "unclassified"
        _REC = {
            "scheduling_delay": (
                "🔴 **Large scheduling delays detected.**  \n"
                "Most time is lost between approval and the employee starting their trip.  \n"
                "**Actions:** Enforce minimum submission lead time · "
                "Automate reminders after final approval · "
                "Set booking windows · SLA enforcement on trip-start lag"
            ),
            "administrative_wait": (
                "🟠 **Administration bottleneck detected.**  \n"
                "Approval processing delays dominate the critical path.  \n"
                "**Actions:** Set 2-day SLA per approval step · "
                "Balance workload across approvers · "
                "Auto-approve low-risk/recurring profiles · "
                "Escalate breaches to line manager"
            ),
            "escalation_delay": (
                "🟣 **High escalation frequency detected.**  \n"
                "Reminder loops indicate systematic non-response to pending actions.  \n"
                "**Actions:** Mandatory field validation before submission · "
                "Auto-escalate after 14 days of inactivity · "
                "Track approval ownership with daily digest · "
                "Flag stalled cases in department dashboard"
            ),
            "service_time": (
                "🔵 **This path is dominated by productive execution time.**  \n"
                "Administrative and scheduling overhead is relatively low.  \n"
                "**Focus:** Protect this path from additional administrative steps. "
                "Monitor escalation rate as an early-warning signal."
            ),
            "unclassified": (
                "⚪ **Transition types are partially unclassified for this path.**  \n"
                "Extend `TRANSITION_TYPES` for more precise value-stream analysis."
            ),
        }
        st.info(_REC.get(dominant_type, _REC["unclassified"]))

        st.markdown("---")

        # ── Detailed transition table ───────────────────────────────────────
        with st.expander("Transition detail"):
            flow_rows = []
            for i in range(len(steps) - 1):
                ttype = edge_types[i]
                w     = edge_waits[i]
                flow_rows.append({
                    "From":          lbl(steps[i]),
                    "To":            lbl(steps[i + 1]),
                    "Wait (days)":   round(w, 1),
                    "Type":          TYPE_ICONS.get(ttype, "⚪") + " " + TYPE_LABELS.get(ttype, ttype),
                    "Owner":         edge_owners[i],
                    "Avoidable":     "No" if ttype == "service_time" else "Yes",
                    "Est. cost (€)": 0 if ttype == "service_time" else round(w * cost_per_day, 0),
                })
            flow_df = pd.DataFrame(flow_rows)
            st.dataframe(
                flow_df.style.format({"Wait (days)": "{:.1f}", "Est. cost (€)": "€{:,.0f}"}),
                use_container_width=True,
            )
            st.download_button(
                "⬇ Export CSV",
                data=flow_df.to_csv(index=False),
                file_name="value_stream_transitions.csv",
                mime="text/csv",
                key="dl_flow",
            )

    # ── Tab 3: Department Impact ───────────────────────────────────────────
    with tab3:
        st.subheader("Which team owns the most delay?")
        st.caption(
            "Delay Days = mean wait × transitions attributed to each department. "
            "Cost = Delay Days × sidebar rate. Ranks by total business impact."
        )

        # Enrich with case counts from violation_by_department.csv
        try:
            vio_dept = load_table("violation_by_department.csv").rename(
                columns={"case:OrganizationalEntity": "department", "total": "cases"}
            )[["department", "cases"]]
        except Exception:
            vio_dept = pd.DataFrame(columns=["department", "cases"])

        dept_sorted = dept.sort_values("total_delay_d", ascending=False).copy()
        dept_sorted = dept_sorted.merge(vio_dept, on="department", how="left")
        dept_sorted["cases"] = dept_sorted["cases"].fillna(0).astype(int)
        dept_sorted["delay_cost_€"] = dept_sorted["total_delay_d"] * cost_per_day
        dept_sorted["share_pct_disp"] = (dept_sorted["share_pct"] * 100).round(1)

        # ── Business impact table ──────────────────────────────────────────
        disp = dept_sorted[[
            "department", "cases", "total_delay_d", "delay_cost_€",
            "stuck_count", "share_pct_disp",
        ]].rename(columns={
            "department":      "Department",
            "cases":           "Cases",
            "total_delay_d":   "Delay (days)",
            "delay_cost_€":    "Cost (€)",
            "stuck_count":     "Stuck",
            "share_pct_disp":  "Share %",
        }).reset_index(drop=True)

        st.dataframe(
            disp.style
            .format({
                "Cases":       "{:,}",
                "Delay (days)":"{:,.0f}",
                "Cost (€)":    "€{:,.0f}",
                "Share %":     "{:.1f}%",
            })
            .background_gradient(subset=["Cost (€)"],    cmap="Reds")
            .background_gradient(subset=["Delay (days)"], cmap="YlOrRd"),
            use_container_width=True,
        )
        st.download_button(
            "⬇ Export CSV",
            data=disp.to_csv(index=False),
            file_name="department_impact.csv",
            mime="text/csv",
            key="dl_dept",
        )

        st.markdown("---")

        # ── Plotly bar chart ───────────────────────────────────────────────
        plot_dept = dept_sorted.head(10).sort_values("delay_cost_€")
        fig_dept = px.bar(
            plot_dept,
            x="delay_cost_€",
            y="department",
            orientation="h",
            color="delay_cost_€",
            color_continuous_scale=["#2c7bb6", "#fdae61", "#d7191c"],
            labels={"delay_cost_€": "Cost (€)", "department": "Department"},
            title="Top 10 Departments by Delay Cost",
            hover_data={"cases": True, "total_delay_d": ":.0f", "stuck_count": True},
        )
        fig_dept.update_layout(
            showlegend=False, coloraxis_showscale=False,
            xaxis_tickformat=",.0f",
            height=380, margin=dict(t=50, b=30, l=10),
        )
        fig_dept.update_xaxes(title="Estimated delay cost (€)")
        st.plotly_chart(fig_dept, use_container_width=True)

    # ── Tab 4: Avoidable Delay + Cost Impact ──────────────────────────────────
    with tab4:
        st.subheader("How much delay is avoidable — and what does it cost?")
        st.caption(
            "Baseline = estimated p25 of observed wait times "
            "(log-normal: p25 ≈ median² / p75). "
            "Avoidable = actual − baseline, summed across all cases. "
            "Cost parameters are set in the sidebar."
        )

        total_actual    = wait["total_actual_d"].sum()
        total_baseline  = wait["total_baseline_d"].sum()
        total_avoidable = wait["total_avoidable_d"].sum()
        pct_avoidable   = total_avoidable / total_actual if total_actual > 0 else 0

        # Cost computations (use sidebar cost_per_day)
        cost_total     = total_actual    * cost_per_day
        cost_avoidable = total_avoidable * cost_per_day
        cost_baseline  = total_baseline  * cost_per_day

        # Top bottleneck savings
        avoid_df = wait.copy()
        avoid_df["Activity"] = lbl_col(avoid_df["concept:name"])
        avoid_df = avoid_df.sort_values("total_avoidable_d", ascending=False)
        top_bneck_name  = avoid_df.iloc[0]["Activity"] if len(avoid_df) else "—"
        top_bneck_saved = avoid_df.iloc[0]["total_avoidable_d"] * cost_per_day if len(avoid_df) else 0

        # ── Delay KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total delay (days)",     f"{total_actual:,.0f}")
        m2.metric("Expected delay (days)",  f"{total_baseline:,.0f}")
        m3.metric("Avoidable delay (days)", f"{total_avoidable:,.0f}", delta_color="inverse")
        m4.metric("Avoidable %",            f"{pct_avoidable:.0%}",   delta_color="inverse")

        st.markdown("---")
        st.subheader("💰 Estimated Cost Impact")
        st.caption(
            f"Using **€{cost_per_day:,}/day** (set in sidebar). "
            "Adjust parameters to model different scenarios."
        )

        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Annual cost of all delay",
            f"€{cost_total/1e6:.1f}M" if cost_total >= 1e6 else f"€{cost_total:,.0f}",
        )
        k2.metric(
            "Estimated avoidable cost",
            f"€{cost_avoidable/1e6:.1f}M" if cost_avoidable >= 1e6 else f"€{cost_avoidable:,.0f}",
            delta_color="inverse",
        )
        k3.metric(
            f"Savings: fix '{top_bneck_name}'",
            f"€{top_bneck_saved/1e6:.1f}M" if top_bneck_saved >= 1e6 else f"€{top_bneck_saved:,.0f}",
        )

        # ── Cost waterfall (Plotly)
        wf_labels  = ["Total delay cost", "Expected (baseline)", "Avoidable savings"]
        wf_values  = [cost_total, -cost_baseline, -cost_avoidable]
        wf_colors  = ["#2c7bb6", "#2ca25f", "#d7191c"]
        fig_wf = go.Figure(go.Bar(
            x=wf_labels, y=[cost_total, cost_baseline, cost_avoidable],
            marker_color=wf_colors, opacity=0.85,
            text=[
                f"€{cost_total/1e6:.1f}M",
                f"€{cost_baseline/1e6:.1f}M",
                f"€{cost_avoidable/1e6:.1f}M",
            ],
            textposition="outside",
        ))
        fig_wf.update_layout(
            title="Cost breakdown — annual delay cost",
            yaxis_title="Cost (€)",
            yaxis_tickformat=",.0f",
            showlegend=False,
            height=320,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown("---")

        # ── Per-activity cost table (top 15)
        avoid_top = avoid_df.head(15).copy()
        avoid_top["cost_avoidable_€"] = avoid_top["total_avoidable_d"] * cost_per_day

        fig3, ax3 = plt.subplots(figsize=(11, 5))
        x = np.arange(len(avoid_top))
        ax3.bar(x, avoid_top["total_actual_d"],
                label="Total actual delay", color="#abd9e9", alpha=0.9, width=0.6)
        ax3.bar(x, avoid_top["total_baseline_d"],
                label="Expected (baseline)", color="#2c7bb6", alpha=0.9, width=0.6)
        ax3.set_xticks(x)
        ax3.set_xticklabels(avoid_top["Activity"], rotation=35, ha="right", fontsize=8)
        ax3.set_ylabel("Delay (days)")
        ax3.set_title("Actual vs Expected Delay — gap is avoidable delay")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax3.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        with st.expander("Avoidable delay & cost by activity"):
            export_avoid = avoid_top[[
                "Activity", "count", "median_wait_d", "p25_wait_d",
                "avoidable_d", "total_avoidable_d", "cost_avoidable_€",
            ]].rename(columns={
                "count":              "Cases",
                "median_wait_d":      "Median Wait (days)",
                "p25_wait_d":         "Baseline p25 (days)",
                "avoidable_d":        "Avoidable/Case (days)",
                "total_avoidable_d":  "Total Avoidable (days)",
                "cost_avoidable_€":   "Avoidable Cost (€)",
            }).reset_index(drop=True)
            st.dataframe(
                export_avoid.style
                .format({
                    "Median Wait (days)":   "{:.1f}",
                    "Baseline p25 (days)":  "{:.1f}",
                    "Avoidable/Case (days)":"{:.1f}",
                    "Total Avoidable (days)":"{:,.0f}",
                    "Avoidable Cost (€)":   "€{:,.0f}",
                })
                .background_gradient(subset=["Avoidable Cost (€)"], cmap="Reds"),
                use_container_width=True,
            )
            st.download_button(
                "⬇ Export CSV",
                data=export_avoid.to_csv(index=False),
                file_name="avoidable_delay_cost.csv",
                mime="text/csv",
                key="dl_avoid",
            )

    # ── Tab 5: Priority Matrix ─────────────────────────────────────────────
    with tab5:
        st.subheader("Where should management act first?")
        st.caption(
            "Each bubble is one activity. "
            "**X-axis** = how often it occurs (frequency). "
            "**Y-axis** = how long it makes people wait (median days). "
            "**Bubble size** = impact score (total delay-days). "
            "Quadrant boundaries = median frequency and median wait across all activities."
        )

        pm_df = wait.copy()
        pm_df["Activity"] = lbl_col(pm_df["concept:name"])

        freq_med  = pm_df["count"].median()
        delay_med = pm_df["median_wait_d"].median()

        def _quadrant(row):
            hi_f = row["count"]        >= freq_med
            hi_d = row["median_wait_d"] >= delay_med
            if   hi_f and hi_d:     return "Act Now  (high freq · high delay)"
            elif not hi_f and hi_d: return "Investigate  (low freq · high delay)"
            elif hi_f and not hi_d: return "Monitor  (high freq · low delay)"
            else:                   return "Ignore  (low freq · low delay)"

        pm_df["Quadrant"] = pm_df.apply(_quadrant, axis=1)
        pm_df["cost_impact_€"] = pm_df["avoidable_d"] * pm_df["count"] * cost_per_day

        q_colors = {
            "Act Now  (high freq · high delay)":       "#d7191c",
            "Investigate  (low freq · high delay)":    "#fdae61",
            "Monitor  (high freq · low delay)":        "#2c7bb6",
            "Ignore  (low freq · low delay)":          "#aaaaaa",
        }

        fig_pm = px.scatter(
            pm_df,
            x="count",
            y="median_wait_d",
            size="impact_score_d",
            color="Quadrant",
            color_discrete_map=q_colors,
            hover_name="Activity",
            hover_data={
                "count":          True,
                "median_wait_d":  ":.1f",
                "impact_score_d": ":,.0f",
                "cost_impact_€":  ":,.0f",
                "Quadrant":       False,
            },
            labels={
                "count":          "Occurrence count",
                "median_wait_d":  "Median wait (days)",
                "impact_score_d": "Impact score (delay-days)",
            },
            title="Bottleneck Prioritization Matrix",
            size_max=55,
        )
        fig_pm.add_vline(
            x=freq_med, line_dash="dash", line_color="#555555",
            annotation_text=f"Median freq: {freq_med:.0f}",
            annotation_position="top right",
        )
        fig_pm.add_hline(
            y=delay_med, line_dash="dash", line_color="#555555",
            annotation_text=f"Median wait: {delay_med:.1f}d",
            annotation_position="right",
        )
        fig_pm.update_layout(
            height=560,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Occurrence count (frequency)",
            yaxis_title="Median wait (days)",
        )
        st.plotly_chart(fig_pm, use_container_width=True)

        col_guide_l, col_guide_r = st.columns(2)
        with col_guide_l:
            st.markdown("""
| Quadrant | Action |
|---|---|
| 🔴 **Act Now** — high freq · high delay | Immediate process redesign or SLA enforcement |
| 🟡 **Investigate** — low freq · high delay | Root cause analysis; may be a rare exception path |
| 🔵 **Monitor** — high freq · low delay | Efficient; watch for regression |
| ⚪ **Ignore** — low freq · low delay | No intervention needed |
            """)
        with col_guide_r:
            act_now = pm_df[pm_df["Quadrant"].str.startswith("Act Now")].sort_values(
                "impact_score_d", ascending=False
            )
            if not act_now.empty:
                st.markdown("**Top 'Act Now' activities:**")
                st.dataframe(
                    act_now[["Activity", "count", "median_wait_d", "impact_score_d"]]
                    .head(6)
                    .rename(columns={
                        "count":          "Freq",
                        "median_wait_d":  "Wait (days)",
                        "impact_score_d": "Impact",
                    })
                    .reset_index(drop=True)
                    .style.format({"Wait (days)": "{:.1f}", "Impact": "{:,.0f}"}),
                    use_container_width=True,
                )

        with st.expander("Full priority matrix table"):
            pm_export = pm_df[[
                "Quadrant", "Activity", "count", "median_wait_d",
                "impact_score_d", "cost_impact_€",
            ]].sort_values(["Quadrant", "impact_score_d"], ascending=[True, False]).rename(columns={
                "count":          "Frequency",
                "median_wait_d":  "Median Wait (days)",
                "impact_score_d": "Impact Score",
                "cost_impact_€":  "Est. Avoidable Cost (€)",
            }).reset_index(drop=True)
            st.dataframe(
                pm_export.style.format({
                    "Median Wait (days)": "{:.1f}",
                    "Impact Score":       "{:,.0f}",
                    "Est. Avoidable Cost (€)": "€{:,.0f}",
                }),
                use_container_width=True,
            )
            st.download_button(
                "⬇ Export CSV",
                data=pm_export.to_csv(index=False),
                file_name="priority_matrix.csv",
                mime="text/csv",
                key="dl_pm",
            )

    # ── Tab 6: Detail (existing content preserved) ─────────────────────────
    with tab6:
        n = st.slider("Show top N activities", 5, 30, 15, key="detail_n")
        wait["median_wait_d2"] = wait["median_wait_h"] / 24
        wait["p95_wait_d2"]    = wait["p95_wait_h"]    / 24
        wait["Activity"]       = lbl_col(wait["concept:name"])
        top = wait.nlargest(n, "median_wait_d2")[[
            "Activity", "count", "median_wait_d2", "p95_wait_d2"
        ]].rename(columns={
            "count":          "Occurrences",
            "median_wait_d2": "Median wait (days)",
            "p95_wait_d2":    "P95 wait (days)",
        }).reset_index(drop=True)
        st.dataframe(
            top.style.format({"Median wait (days)": "{:.1f}", "P95 wait (days)": "{:.1f}"}),
            use_container_width=True,
        )

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
    col1.metric("Mean trace fitness",     f"{fit_mean:.3f}")
    col2.metric("Perfect-fit cases",      f"{fit_perfect:.1%}")
    col3.metric("Cases with violations",  f"{n_vio:,}", delta_color="inverse")
    col4.metric("Violation rate",         f"{n_vio/len(conf):.1%}", delta_color="inverse")

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
        "type_a":      "Type A (departed before submit)",
        "type_b":      "Type B (departed before approval)",
        "compliant":   "Compliant",
        "total":       "Total",
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
    col1.metric("Model",               "XGBoost k=8")
    col2.metric("AUC — deployable",    "0.810", help="Without elapsed_days (leakage-free)")
    col3.metric("AUC — naïve (leaky)", "0.967", help="elapsed_days included; inflated by temporal leakage")
    col4.metric("Threshold",           "101.1 days (P67)")

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

    bundle    = load_model()
    model_k8  = bundle["model"]
    imputer   = bundle["imputer"]
    feat_cols = bundle["feature_cols"]

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        n_rejections  = c1.number_input("Rejections in prefix",     0, 20, 0)
        n_reminders   = c2.number_input("Send Reminders in prefix", 0, 10, 0)
        n_approvals   = c3.number_input("Approvals in prefix",      0, 20, 2)

        c4, c5 = st.columns(2)
        n_events    = c4.number_input("Events in prefix (≤8)",   1,  8, 8)
        start_month = c5.number_input("Case start month (1–12)", 1, 12, 6)

        c6, c7 = st.columns(2)
        has_reminder  = c6.checkbox("Has Send Reminder in prefix")
        has_final_app = c7.checkbox("Has Permit FINAL_APPROVED in prefix")

        c8, c9 = st.columns(2)
        has_rejected = c8.checkbox("Has any REJECTED activity in prefix")
        has_approved = c9.checkbox("Has any APPROVED activity in prefix")

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

    mae_cv = reg_bundle["mae_temporal_cv"]
    r2_cv  = reg_bundle["r2_temporal_cv"]
    mae_h  = reg_bundle["mae_holdout"]
    r2_h   = reg_bundle["r2_holdout"]
    cov    = reg_bundle["coverage_p10_p90"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Model",             "XGBoost k=8")
    col2.metric("MAE (holdout)",     f"{mae_h:.1f}d",  help="Mean absolute error on 20% holdout")
    col3.metric("MAE (temporal CV)", f"{mae_cv:.1f}d", help="Mean across Q1–Q4 2018 expanding-window folds")
    col4.metric("R² (temporal CV)",  f"{r2_cv:.3f}")
    col5.metric("P10–P90 coverage",  f"{cov:.1%}",     help="Fraction of actuals inside the 80% prediction interval")

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

    st.markdown("---")
    st.subheader("Quantile Prediction Intervals (P10 / P50 / P90)")
    st.image(str(F / "remaining_time_quantile_intervals.png"), use_container_width=True)
    st.caption(
        f"Blue band = 80% prediction interval. Red dots = actual remaining days. "
        f"Coverage {cov:.1%} (target 80%)."
    )

    st.markdown("---")
    st.subheader("SHAP Feature Importance")
    st.image(str(F / "remaining_time_shap_beeswarm.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("Single-Case Prediction")

    feat_cols_r = reg_bundle["feature_cols"]
    imputer_r   = reg_bundle["imputer"]
    m_point     = reg_bundle["model_point"]
    m_p10       = reg_bundle["model_p10"]
    m_p50       = reg_bundle["model_p50"]
    m_p90       = reg_bundle["model_p90"]

    with st.form("reg_prediction_form"):
        c1, c2, c3 = st.columns(3)
        elapsed_d      = c1.number_input("Elapsed days so far",       0.0, 500.0, 10.0, step=1.0)
        n_rejections_r = c2.number_input("Rejections in prefix",      0, 20, 0)
        n_reminders_r  = c3.number_input("Send Reminders in prefix",  0, 10, 0)

        c4, c5, c6 = st.columns(3)
        n_approvals_r = c4.number_input("Approvals in prefix",       0, 20, 2)
        n_events_r    = c5.number_input("Events in prefix (≤8)",     1,  8, 8)
        start_month_r = c6.number_input("Case start month (1–12)",   1, 12, 6)

        c7, c8 = st.columns(2)
        has_reminder_r = c7.checkbox("Has Send Reminder in prefix")
        has_final_r    = c8.checkbox("Has Permit FINAL_APPROVED in prefix")

        c9, c10 = st.columns(2)
        has_rejected_r = c9.checkbox("Has any REJECTED activity in prefix")
        has_approved_r = c10.checkbox("Has any APPROVED activity in prefix")

        submitted_r = st.form_submit_button("Predict remaining time", type="primary")

    if submitted_r:
        row_r = {c: 0 for c in feat_cols_r}
        row_r["elapsed_days"]     = elapsed_d
        row_r["n_events_prefix"]  = n_events_r
        row_r["n_rejections"]     = n_rejections_r
        row_r["n_reminders"]      = n_reminders_r
        row_r["n_approvals"]      = n_approvals_r
        row_r["case_start_month"] = start_month_r
        if "has_Send_Reminder" in feat_cols_r:
            row_r["has_Send_Reminder"] = int(has_reminder_r)
        if "has_Permit_FINAL_APPROVED_by_SUPERVISOR" in feat_cols_r:
            row_r["has_Permit_FINAL_APPROVED_by_SUPERVISOR"] = int(has_final_r)
        if has_rejected_r and "has_Declaration_REJECTED_by_DIRECTOR" in feat_cols_r:
            row_r["has_Declaration_REJECTED_by_DIRECTOR"] = 1
        if has_approved_r and "has_Declaration_APPROVED_by_SUPERVISOR" in feat_cols_r:
            row_r["has_Declaration_APPROVED_by_SUPERVISOR"] = 1

        X_r     = pd.DataFrame([row_r])[feat_cols_r]
        X_r_imp = pd.DataFrame(imputer_r.transform(X_r), columns=feat_cols_r)

        pred_point = max(0.0, float(m_point.predict(X_r_imp)[0]))
        pred_p10   = max(0.0, float(m_p10.predict(X_r_imp)[0]))
        pred_p50   = max(0.0, float(m_p50.predict(X_r_imp)[0]))
        pred_p90   = max(0.0, float(m_p90.predict(X_r_imp)[0]))

        st.markdown("---")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Point estimate",    f"{pred_point:.1f} days")
        rc2.metric("P10 (optimistic)",  f"{pred_p10:.1f} days")
        rc3.metric("P50 (median)",      f"{pred_p50:.1f} days")
        rc4.metric("P90 (pessimistic)", f"{pred_p90:.1f} days")

        fig_r, ax_r = plt.subplots(figsize=(8, 1.8))
        ax_r.barh(0, pred_p90 - pred_p10, left=pred_p10,
                  height=0.4, color="#2c7bb6", alpha=0.35, label="P10–P90 interval")
        ax_r.axvline(pred_point, color="#d7191c", linewidth=2, label=f"Point: {pred_point:.1f}d")
        ax_r.axvline(pred_p50,   color="#2c7bb6", linewidth=2, linestyle="--",
                     label=f"P50: {pred_p50:.1f}d")
        ax_r.set_xlabel("Remaining days")
        ax_r.set_yticks([])
        ax_r.legend(loc="upper right", fontsize=8)
        ax_r.set_title("Prediction interval")
        plt.tight_layout()
        st.pyplot(fig_r, use_container_width=True)
        plt.close()

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

    concordance = cox_bundle["concordance_idx"]
    n_censored  = cox_bundle["n_censored"]
    cens_rate   = cox_bundle["censoring_rate"]
    median_surv = cox_bundle["median_survival"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total cases",        "7,065",  help="All cases including stuck")
    col2.metric("Censored (stuck)",   f"{n_censored:,}", f"{cens_rate:.0%} of all cases",
                delta_color="inverse")
    col3.metric("KM median survival", f"{median_surv:.0f}d",
                help="Median days until completion (accounting for censoring)")
    col4.metric("Cox concordance",    f"{concordance:.3f}",
                help="Survival equivalent of AUC")

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

    st.markdown("---")
    st.subheader("Survival by Department")
    col_l, col_r = st.columns(2)
    with col_l:
        st.dataframe(
            surv_dept.rename(columns={
                "department":            "Department",
                "n_cases":               "Cases",
                "n_censored":            "Censored",
                "censoring_rate":        "Censoring rate",
                "median_survival_days":  "Median survival (days)",
            }).style.format({
                "Censoring rate":        "{:.1%}",
                "Median survival (days)":"{:.0f}",
            }),
            use_container_width=True,
        )
    with col_r:
        st.markdown(
            "Departments with high censoring rates have a large proportion of permanently stuck cases. "
            "Median survival is estimated by Kaplan-Meier, which accounts for censoring."
        )

    st.markdown("---")
    st.subheader("Cox Proportional Hazards — Hazard Ratios")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.image(str(F / "survival_cox_hazard_ratios.png"), use_container_width=True)
        st.caption(
            f"HR > 1 (red) = faster completion. HR < 1 (blue) = slower / higher stuck risk. "
            f"Concordance index: {concordance:.3f}."
        )
    with col_r:
        sig_hr = cox_hr[cox_hr["p"] < 0.05].sort_values("HR", ascending=False).head(10)
        if not sig_hr.empty:
            st.dataframe(
                sig_hr[["covariate", "HR", "HR_lo", "HR_hi", "p"]]
                .rename(columns={
                    "covariate": "Feature",
                    "HR_lo":     "HR low 95%",
                    "HR_hi":     "HR high 95%",
                })
                .style.format({"HR": "{:.3f}", "HR low 95%": "{:.3f}",
                               "HR high 95%": "{:.3f}", "p": "{:.2e}"}),
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("KM Curves by Cox Risk Group")
    st.image(str(F / "survival_km_risk_groups.png"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 7 — VIOLATION ROOT CAUSE
# ══════════════════════════════════════════════════════════════════════════
elif page == "Violation Root Cause":
    st.title("Violation Root Cause Analysis")
    st.caption(
        "44.9% of cases have a conformance violation (fitness < 1.0). "
        "This page identifies which departments, budgets, and case characteristics drive violations, "
        "and investigates the rejection paradox and lead time findings."
    )

    rca_summ = load_table("violation_rca_summary.csv")
    rca_dept = load_table("violation_rca_by_department.csv")
    rca_segs = load_table("violation_rca_segments.csv")

    dt_auc        = rca_summ["dt_cv_auc"].iloc[0]
    xgb_auc       = rca_summ["xgb_cv_auc"].iloc[0]
    n_vio         = int(rca_summ["n_violations"].iloc[0])
    n_a           = int(rca_summ["n_type_a"].iloc[0])
    n_b           = int(rca_summ["n_type_b"].iloc[0])
    top_dept      = rca_summ["top_risk_dept"].iloc[0]
    top_dept_rate = rca_summ["top_risk_dept_rate"].iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total violations",  f"{n_vio:,}", "44.9% of cases", delta_color="inverse")
    col2.metric("Type A",            f"{n_a:,}",   "departed before submit", delta_color="inverse")
    col3.metric("Type B",            f"{n_b:,}",   "departed before approval", delta_color="inverse")
    col4.metric("Decision tree AUC", dt_auc,       help="5-fold CV, depth=4 — interpretable rules")
    col5.metric("XGBoost AUC",       xgb_auc,      help="5-fold CV — best predictive performance")

    rca_tab1, rca_tab2, rca_tab3 = st.tabs([
        "📊 Root Cause Models",
        "🔍 Rejection Paradox",
        "⏱ Lead Time Analysis",
    ])

    with rca_tab1:
        st.subheader("Violation Rate by Department, Budget & Season")
        st.image(str(F / "violation_rca_distributions.png"), use_container_width=True)

        st.markdown("---")
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.subheader("Department Risk Exposure")
            st.image(str(F / "violation_rca_dept_exposure.png"), use_container_width=True)
        with col_r:
            st.subheader("Department breakdown")
            st.dataframe(
                rca_dept.rename(columns={
                    "dept": "Department", "n": "Cases",
                    "vio": "Violations", "rate": "Rate", "ci": "±95% CI",
                })
                .sort_values("Rate", ascending=False)
                .style.format({"Rate": "{:.1%}", "±95% CI": "{:.3f}"}),
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("Decision Tree — Interpretable Rules (depth=4)")
        st.image(str(F / "violation_rca_decision_tree.png"), use_container_width=True)

        st.markdown("---")
        st.subheader("SHAP Feature Importance (XGBoost)")
        st.image(str(F / "violation_rca_shap.png"), use_container_width=True)
        st.caption(
            f"XGBoost CV AUC = {xgb_auc}. Duration and department are the dominant drivers."
        )

        with st.expander("Decision tree leaf segments"):
            st.dataframe(
                rca_segs.style.format({
                    "violation_rate": "{:.1%}",
                    "pct_of_all":     "{:.1%}",
                    "avg_duration":   "{:.0f}",
                    "avg_budget":     "{:.0f}",
                }),
                use_container_width=True,
            )

    with rca_tab2:
        st.subheader("The Rejection Paradox")
        st.info(
            "The SHAP plot shows high N rejections pushing violation risk **left** (protective). "
            "Counterintuitively, more rejections correlates with *lower* violation probability — "
            "but the raw bucket rates tell a more nuanced story.",
            icon="🔍",
        )
        st.image(str(F / "violation_rejection_paradox.png"), use_container_width=True)
        st.image(str(F / "violation_activity_gap.png"), use_container_width=True)
        st.markdown("""
**Key findings:**
- **1-rejection group**: 98.1% violation rate (154 of 157 cases) — the sharpest signal in the dataset
- **Zero-rejection violators** submitted permits (99.2%) but `Permit APPROVED by ADMINISTRATION` is present in only 44% vs 100% of compliant cases — the missing gate is administration approval, not permit submission
- **SHAP direction**: in cases with many events and adequate approvals, 2+ rejections signals a proper correction cycle — a conditional interaction, not a marginal effect
        """)

        with st.expander("Rejection bucket statistics"):
            try:
                rej_df = load_table("violation_rejection_paradox.csv")
                st.dataframe(rej_df.style.format({
                    "vio_rate": "{:.1%}", "type_a_rate": "{:.1%}",
                    "type_b_rate": "{:.1%}", "pct_cases": "{:.1%}",
                }), use_container_width=True)
            except Exception:
                st.caption("Table not available.")

    with rca_tab3:
        st.subheader("Lead Time Analysis")
        st.info(
            "Does submitting permits earlier prevent violations? "
            "The data answers: **no** — and the reason why reveals the real intervention target.",
            icon="⏱",
        )
        st.image(str(F / "violation_lead_time.png"), use_container_width=True)
        st.markdown("""
**Key findings:**
- Violators submit permits **earlier** than compliant cases (median 35d vs 16d before trip)
- Administration approves permits in **0 days** (median) when they act — no backlog
- Violation rate **increases** with longer lead time — submitting earlier does not help
- **Root cause**: routing/awareness gap — permits enter the system but administration is never notified or does not action them

**Recommended intervention:** automated SLA alert when `Permit SUBMITTED by EMPLOYEE` has no `Permit APPROVED by ADMINISTRATION` response within 2 business days.
        """)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 8 — PROCESS VARIANTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Process Variants":
    st.title("Process Variants")
    st.caption(
        "How do different case paths compare in duration and stuck rate? "
        "Understanding variant behaviour helps identify which paths need intervention."
    )

    variants_df = load_table("variants.csv")
    features_df = load_table("features.csv")

    # Per-variant stats from features (variant_rank 1 = most common)
    var_stats = (
        features_df.groupby("variant_rank")
        .agg(
            median_duration=("duration_days", "median"),
            mean_duration=("duration_days",   "mean"),
            stuck_rate=("ends_with_reminder",  "mean"),
        )
        .reset_index()
        .rename(columns={"variant_rank": "rank"})
    )
    variants_df = variants_df.reset_index().rename(columns={"index": "rank"})
    variants_df["rank"] = variants_df["rank"] + 1
    merged = variants_df.merge(var_stats, on="rank", how="left")
    merged["n_steps"] = merged["variant"].apply(lambda x: len(x.split("->")))

    def fmt_path(path_str: str) -> str:
        steps = [s.strip() for s in path_str.split("->")]
        if use_biz_labels:
            steps = [BUSINESS_LABELS.get(s, s) for s in steps]
        return " → ".join(steps)

    merged["path_display"] = merged["variant"].apply(fmt_path)

    # Summary KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique variants",      f"{len(merged):,}")
    col2.metric("Top variant share",    f"{merged['pct'].iloc[0]:.1f}%")
    col3.metric("Top 10 variants cover",f"{merged.head(10)['pct'].sum():.1f}%")

    st.markdown("---")

    top_n_v = st.slider("Show top N variants", 5, 20, 10)

    # Comparison table
    vdisp = merged.head(top_n_v)[[
        "rank", "case_count", "pct", "n_steps",
        "median_duration", "stuck_rate", "path_display",
    ]].rename(columns={
        "rank":            "#",
        "case_count":      "Cases",
        "pct":             "Freq %",
        "n_steps":         "Steps",
        "median_duration": "Median Duration (days)",
        "stuck_rate":      "Stuck Rate",
        "path_display":    "Process Path",
    }).reset_index(drop=True)

    st.dataframe(
        vdisp.style
        .format({
            "Freq %":                "{:.1f}%",
            "Median Duration (days)":"{:.0f}",
            "Stuck Rate":            "{:.1%}",
        })
        .background_gradient(subset=["Stuck Rate"],            cmap="Reds")
        .background_gradient(subset=["Median Duration (days)"], cmap="YlOrRd"),
        use_container_width=True,
    )
    st.download_button(
        "⬇ Export CSV",
        data=vdisp.to_csv(index=False),
        file_name="process_variants.csv",
        mime="text/csv",
        key="dl_variants",
    )

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Variant Frequency")
        plot_v = merged.head(top_n_v).copy()
        plot_v["short_label"] = [f"V{r}" for r in plot_v["rank"]]
        fig_v, ax_v = plt.subplots(figsize=(5, max(3, top_n_v * 0.4)))
        ax_v.barh(plot_v["short_label"][::-1], plot_v["case_count"][::-1],
                  color="#2c7bb6", alpha=0.85)
        ax_v.set_xlabel("Number of cases")
        ax_v.set_title(f"Top {top_n_v} Variants")
        plt.tight_layout()
        st.pyplot(fig_v, use_container_width=True)
        plt.close()

    with col_r:
        st.subheader("Duration vs Stuck Rate")
        plot_sc = merged.head(top_n_v).dropna(subset=["median_duration", "stuck_rate"])
        fig_sc, ax_sc = plt.subplots(figsize=(5, 4))
        sc = ax_sc.scatter(
            plot_sc["median_duration"], plot_sc["stuck_rate"],
            s=plot_sc["case_count"] / 4,
            c=plot_sc["stuck_rate"], cmap="RdYlGn_r",
            alpha=0.8, vmin=0, vmax=plot_sc["stuck_rate"].max() or 0.5,
        )
        for _, row in plot_sc.iterrows():
            ax_sc.annotate(f"V{int(row['rank'])}",
                           (row["median_duration"], row["stuck_rate"]),
                           fontsize=7.5, ha="center", va="bottom")
        plt.colorbar(sc, ax=ax_sc, label="Stuck rate")
        ax_sc.set_xlabel("Median duration (days)")
        ax_sc.set_ylabel("Stuck rate")
        ax_sc.set_title("Duration vs Stuck Rate\n(bubble = case count)")
        plt.tight_layout()
        st.pyplot(fig_sc, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.subheader("Inspect a Variant")

    options = merged.head(top_n_v)["rank"].tolist()
    selected_rank = st.selectbox(
        "Select variant",
        options=options,
        format_func=lambda r: (
            f"V{r} — {int(merged[merged['rank']==r]['case_count'].values[0]):,} cases  "
            f"({merged[merged['rank']==r]['pct'].values[0]:.1f}%)"
        ),
    )

    sel = merged[merged["rank"] == selected_rank].iloc[0]
    raw_steps = [s.strip() for s in sel["variant"].split("->")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cases",            f"{int(sel['case_count']):,}")
    c2.metric("Frequency",        f"{sel['pct']:.1f}%")
    c3.metric("Median duration",
              f"{sel['median_duration']:.0f}d" if not pd.isna(sel.get("median_duration")) else "—")
    c4.metric("Stuck rate",
              f"{sel['stuck_rate']:.1%}" if not pd.isna(sel.get("stuck_rate")) else "—")

    st.markdown("**Process path:**")
    for step in raw_steps:
        icon = "🔴" if "Reminder" in step else "▸"
        display = lbl(step)
        tech    = f"  `{step}`" if not use_biz_labels else ""
        st.markdown(f"&nbsp;&nbsp;{icon} {display}{tech}", unsafe_allow_html=True)
