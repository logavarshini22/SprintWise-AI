
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="SprintWise AI | Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED = [
    "Sprint",
    "Planned_Story_Points",
    "Completed_Story_Points",
    "Planned_Tasks",
    "Completed_Tasks",
    "Bugs",
    "Blocked_Tasks",
]

NUMERIC = REQUIRED[1:]
RISK_COLORS = {
    "Low": "#35D0A0",
    "Medium": "#F6BD60",
    "High": "#FF6B81",
}

# =========================================================
# PREMIUM DARK THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080B14;
    --surface: #101522;
    --surface2: #151C2D;
    --border: #252E43;
    --text: #F1F5FF;
    --muted: #95A2BB;
    --violet: #8B7CFF;
    --cyan: #42D9F4;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: var(--bg);
    color: var(--text);
}
[data-testid="stHeader"] {
    background: rgba(8,11,20,.94);
}
[data-testid="stSidebar"] {
    background: #0C101C;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text);
}
.block-container {
    max-width: 1550px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
    font-weight: 800 !important;
    letter-spacing: -.03em;
}
p, label {
    color: #C1CADB;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg,#151C2D,#101522);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 8px 26px rgba(0,0,0,.16);
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: .82rem;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #F5F7FF !important;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}
.hero {
    background:
      radial-gradient(ellipse at 90% 0%,rgba(139,124,255,.24),transparent 45%),
      radial-gradient(ellipse at 0% 100%,rgba(66,217,244,.10),transparent 45%),
      linear-gradient(135deg,#141A2B,#0E1320);
    border: 1px solid #303751;
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 24px;
}
.eyebrow {
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: .15em;
    font-size: .72rem;
    font-weight: 800;
    margin-bottom: 9px;
}
.hero-title {
    color: #FFFFFF;
    font-family: 'Manrope', sans-serif;
    font-size: 2.4rem;
    line-height: 1.15;
    font-weight: 800;
    margin-bottom: 12px;
}
.hero-copy {
    color: #AEBBD2;
    font-size: 1rem;
    line-height: 1.7;
    max-width: 850px;
}
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0D1220;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 5px;
    gap: 5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 10px 13px;
    color: #AAB6CD;
    font-weight: 700;
}
.stTabs [aria-selected="true"] {
    background: #282348 !important;
    color: #DCD7FF !important;
}
.stButton button, .stDownloadButton button {
    border-radius: 10px;
    font-weight: 700;
    border: 1px solid #51468C;
    background: linear-gradient(135deg,#7765E8,#5848B7);
    color: white;
}
.stButton button:hover, .stDownloadButton button:hover {
    border-color: var(--cyan);
    color: white;
}
[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #485573;
    border-radius: 13px;
    padding: 12px;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}
[data-testid="stAlert"] {
    border-radius: 12px;
}
div[data-baseweb="select"] > div {
    background: #151C2D;
    border-color: #303A52;
    color: #F1F5FF;
}
hr {
    border-color: var(--border);
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">AGILE DELIVERY INTELLIGENCE</div>
    <div class="hero-title">SprintWise AI</div>
    <div class="hero-copy">
        Understand sprint performance, delivery velocity, quality signals,
        blockers, and project risk using your uploaded project data.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR: REAL CSV UPLOAD, NO DEMO MODE
# =========================================================
with st.sidebar:
    st.markdown("## 📊 SprintWise AI")
    st.caption("SPRINT ANALYTICS WORKSPACE")
    st.divider()
    st.markdown("### 01 · Upload your data")
    uploaded = st.file_uploader(
        "Choose sprint CSV",
        type=["csv"],
        help="Upload one row per sprint. No sample data is loaded automatically.",
    )
    with st.expander("Required CSV columns"):
        st.code(", ".join(REQUIRED), language="text")
    st.divider()
    st.caption("Academic analytics prototype")

if uploaded is None:
    st.markdown("### Start your analysis")
    st.write(
        "Upload your CSV using the sidebar. SprintWise AI will validate "
        "your file and build the dashboard from your actual data."
    )
    a, b, c = st.columns(3)
    for col, num, title, description in [
        (a, "01", "Upload", "Select your sprint dataset."),
        (b, "02", "Analyze", "Calculate delivery and risk indicators."),
        (c, "03", "Present", "Explore charts and download results."),
    ]:
        with col:
            st.markdown(f"""
            <div class="panel">
                <div class="eyebrow">{num}</div>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# =========================================================
# CSV READING AND VALIDATION
# =========================================================
try:
    uploaded.seek(0)
    raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

raw.columns = [str(col).strip() for col in raw.columns]

if raw.columns.duplicated().any():
    st.error("Your CSV has duplicate column names. Rename the duplicate columns and upload it again.")
    st.stop()

missing = [col for col in REQUIRED if col not in raw.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.info("Required columns: " + ", ".join(REQUIRED))
    st.stop()

df = raw[REQUIRED].copy()
df["Sprint"] = df["Sprint"].astype("string").str.strip()

if df.empty:
    st.error("The CSV has no sprint records.")
    st.stop()

if df["Sprint"].isna().any() or df["Sprint"].eq("").any():
    st.error("Sprint names cannot be blank.")
    st.stop()

if df["Sprint"].duplicated().any():
    duplicates = df.loc[
        df["Sprint"].duplicated(keep=False), "Sprint"
    ].unique().tolist()
    st.error("Sprint names must be unique. Duplicates: " + ", ".join(duplicates))
    st.stop()

validation_errors = []
for col in NUMERIC:
    original = df[col]
    converted = pd.to_numeric(original, errors="coerce")

    if converted.isna().any():
        validation_errors.append(f"{col} contains blank or non-numeric values.")
    elif not np.isfinite(converted.to_numpy(dtype=float)).all():
        validation_errors.append(f"{col} contains invalid infinite values.")
    elif (converted < 0).any():
        validation_errors.append(f"{col} cannot contain negative values.")

    df[col] = converted

if validation_errors:
    st.error("Please fix these CSV issues:")
    for error in validation_errors:
        st.write(f"- {error}")
    st.stop()

warnings = []

for col in NUMERIC:
    if not np.allclose(df[col], np.round(df[col])):
        warnings.append(f"{col} contains decimal values; values have been rounded to whole numbers.")
    df[col] = df[col].round().astype(int)

if (df["Planned_Story_Points"] <= 0).any():
    st.error("Planned_Story_Points must be greater than zero.")
    st.stop()

if (df["Planned_Tasks"] <= 0).any():
    st.error("Planned_Tasks must be greater than zero.")
    st.stop()

if (df["Completed_Story_Points"] > df["Planned_Story_Points"]).any():
    warnings.append("Completed story points exceed planned points in some rows. Review possible scope changes.")
if (df["Completed_Tasks"] > df["Planned_Tasks"]).any():
    warnings.append("Completed tasks exceed planned tasks in some rows. Review possible scope changes.")
if (df["Blocked_Tasks"] > df["Planned_Tasks"]).any():
    warnings.append("Blocked tasks exceed planned tasks in some rows. Review the source data.")

for warning in warnings:
    st.warning(warning)

st.success(f"CSV validated successfully · {len(df)} sprint records loaded")

st.caption(
    "Sprint order follows the row order in your CSV. Arrange the rows "
    "chronologically before uploading so the latest sprint is identified correctly."
)

# =========================================================
# FILTERS
# =========================================================
st.sidebar.markdown("### 02 · Dashboard filters")
minimum_completion = st.sidebar.slider(
    "Minimum completion percentage",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
)

# =========================================================
# METRIC CALCULATIONS
# =========================================================
df["Completion_%"] = (
    df["Completed_Story_Points"] / df["Planned_Story_Points"] * 100
).clip(upper=100)

df["Task_Completion_%"] = (
    df["Completed_Tasks"] / df["Planned_Tasks"] * 100
).clip(upper=100)

df["Velocity"] = df["Completed_Story_Points"]

df["Remaining_Story_Points"] = (
    df["Planned_Story_Points"] - df["Completed_Story_Points"]
).clip(lower=0)

df["Risk_Score"] = (
    (100 - df["Completion_%"]) * 0.45
    + df["Bugs"] * 4
    + df["Blocked_Tasks"] * 6
).clip(0, 100)

def classify_risk(score):
    if score >= 60:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"

df["Risk"] = df["Risk_Score"].apply(classify_risk)

df["Data_Quality"] = np.where(
    (df["Completed_Story_Points"] > df["Planned_Story_Points"])
    | (df["Completed_Tasks"] > df["Planned_Tasks"]),
    "Review over-plan values",
    "OK",
)

view = df[df["Completion_%"] >= minimum_completion].copy()

if view.empty:
    st.warning("No sprints match this completion filter. Lower the threshold.")
    st.stop()

latest = view.iloc[-1]
overall_completion = min(
    100,
    view["Completed_Story_Points"].sum()
    / view["Planned_Story_Points"].sum() * 100,
)
average_velocity = view["Velocity"].mean()
total_bugs = int(view["Bugs"].sum())
total_blocked = int(view["Blocked_Tasks"].sum())
high_risk = int((view["Risk"] == "High").sum())

# =========================================================
# KPI CARDS
# =========================================================
st.markdown('<div class="eyebrow">EXECUTIVE SNAPSHOT</div>', unsafe_allow_html=True)
st.markdown("### Sprint performance at a glance")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Overall completion", f"{overall_completion:.1f}%")
k2.metric("Average velocity", f"{average_velocity:.1f} SP")
k3.metric("Total bugs", total_bugs)
k4.metric("Blocked tasks", total_blocked)
k5.metric("High-risk sprints", high_risk)

risk_bg = {
    "Low": "#12372F",
    "Medium": "#3A2C16",
    "High": "#401E2A",
}
risk_fg = {
    "Low": "#6BE2BC",
    "Medium": "#FFD17A",
    "High": "#FF8FA0",
}

st.markdown(f"""
<div class="panel">
    <div class="eyebrow">LATEST MATCHING SPRINT · CSV ROW ORDER</div>
    <h2>{latest['Sprint']}</h2>
    <span style="background:{risk_bg[latest['Risk']]};
                 color:{risk_fg[latest['Risk']]};
                 padding:5px 10px;border-radius:20px;font-weight:800;">
        {latest['Risk'].upper()} RISK
    </span>
    <p>Completion: {latest['Completion_%']:.1f}% ·
       Velocity: {latest['Velocity']} story points ·
       Risk score: {latest['Risk_Score']:.1f}/100</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# PLOTLY CHART STYLE
# =========================================================
def style_chart(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="#101522",
        plot_bgcolor="#101522",
        font=dict(family="DM Sans", color="#C1CADB"),
        margin=dict(l=15, r=15, t=55, b=15),
        title_font=dict(family="Manrope", size=16, color="#F1F5FF"),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(gridcolor="#252E43", title=None, zeroline=False),
    )
    return fig

# =========================================================
# DASHBOARD TABS
# =========================================================
overview, delivery, risk_tab, recommendations, data_tab, guide = st.tabs([
    "Overview",
    "Delivery analytics",
    "Risk & quality",
    "Recommendations",
    "Data & export",
    "Project guide",
])

with overview:
    st.markdown("### Delivery overview")
    left, right = st.columns(2)

    with left:
        fig = px.line(
            view, x="Sprint", y="Velocity",
            markers=True, title="Sprint velocity",
        )
        fig.update_traces(
            line=dict(color="#8B7CFF", width=3),
            marker=dict(color="#42D9F4", size=8),
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        points = view.melt(
            id_vars="Sprint",
            value_vars=["Planned_Story_Points", "Completed_Story_Points"],
            var_name="Type",
            value_name="Story points",
        )
        fig = px.bar(
            points, x="Sprint", y="Story points",
            color="Type", barmode="group",
            title="Planned vs completed story points",
            color_discrete_map={
                "Planned_Story_Points": "#56617A",
                "Completed_Story_Points": "#8B7CFF",
            },
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.line(
            view, x="Sprint", y="Completion_%",
            markers=True, title="Completion trend",
        )
        fig.update_traces(line=dict(color="#35D0A0", width=3))
        fig.add_hline(
            y=80, line_dash="dash", line_color="#F6BD60",
            annotation_text="80% reference",
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        quality = view.melt(
            id_vars="Sprint",
            value_vars=["Bugs", "Blocked_Tasks"],
            var_name="Indicator",
            value_name="Count",
        )
        fig = px.bar(
            quality, x="Sprint", y="Count",
            color="Indicator", barmode="group",
            title="Bugs and blocked tasks",
            color_discrete_map={
                "Bugs": "#FF6B81",
                "Blocked_Tasks": "#F6BD60",
            },
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("### Key observations")
    if overall_completion < 80:
        st.warning("Overall completion is below the 80% reference threshold.")
    else:
        st.success("Overall completion meets or exceeds the 80% reference threshold.")

    if high_risk:
        st.warning(f"{high_risk} sprint(s) have high rule-based risk scores.")
    else:
        st.info("No selected sprint has a high rule-based risk score.")

with delivery:
    st.markdown("### Delivery analytics")
    left, right = st.columns(2)

    with left:
        fig = px.bar(
            view, x="Sprint", y="Task_Completion_%",
            title="Task completion by sprint",
        )
        fig.update_traces(marker_color="#8B7CFF")
        fig.add_hline(y=80, line_dash="dash", line_color="#F6BD60")
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        fig = px.bar(
            view, x="Sprint", y="Remaining_Story_Points",
            title="Remaining planned story points",
        )
        fig.update_traces(marker_color="#42D9F4")
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown("#### Sprint delivery table")
    cols = [
        "Sprint", "Planned_Story_Points", "Completed_Story_Points",
        "Completion_%", "Planned_Tasks", "Completed_Tasks",
        "Task_Completion_%", "Velocity", "Data_Quality",
    ]
    st.dataframe(view[cols].round(1), use_container_width=True, hide_index=True)

with risk_tab:
    st.markdown("### Risk and quality analysis")
    left, right = st.columns([1.3, 1])

    with left:
        fig = px.bar(
            view, x="Sprint", y="Risk_Score", color="Risk",
            color_discrete_map=RISK_COLORS,
            hover_data=["Completion_%", "Bugs", "Blocked_Tasks"],
            range_y=[0, 100],
            title="Rule-based sprint risk score",
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        counts = (
            view["Risk"].value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .rename_axis("Risk")
            .reset_index(name="Sprints")
        )
        fig = px.pie(
            counts, names="Risk", values="Sprints",
            color="Risk", hole=0.62,
            color_discrete_map=RISK_COLORS,
            title="Risk category distribution",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.info(
        "Risk formula: 0.45 × (100 − completion %) + 4 × bugs + "
        "6 × blocked tasks, capped at 100. Low: below 35; "
        "Medium: 35–59.9; High: 60 or above. This is a heuristic score, "
        "not a calibrated probability of sprint failure."
    )
    st.dataframe(
        view[[
            "Sprint", "Completion_%", "Bugs", "Blocked_Tasks",
            "Risk_Score", "Risk", "Data_Quality",
        ]].round(1),
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# RECOMMENDATIONS
# =========================================================
def make_recommendations(row, avg_velocity):
    result = []

    if row["Completion_%"] < 80:
        result.append((
            "DELIVERY",
            "Completion is below 80%. Review unfinished stories and reconsider sprint scope.",
        ))
    if row["Bugs"] >= 5:
        result.append((
            "QUALITY",
            "Bug count is elevated. Investigate recurring defects and reserve time for fixes and regression testing.",
        ))
    if row["Blocked_Tasks"] >= 3:
        result.append((
            "DEPENDENCIES",
            "Several tasks are blocked. Identify dependencies, assign owners, and agree on resolution dates.",
        ))
    if row["Velocity"] < avg_velocity:
        result.append((
            "VELOCITY",
            "Velocity is below the selected-sprint average. Review capacity, unplanned work, and dependency delays.",
        ))
    if row["Task_Completion_%"] < 80:
        result.append((
            "PLANNING",
            "Task completion is below 80%. Break large stories into smaller tasks and clarify acceptance criteria.",
        ))
    if not result:
        result.append((
            "STABLE",
            "Current indicators are relatively stable. Continue monitoring delivery, bugs, and blockers.",
        ))
    return result

with recommendations:
    st.markdown("### Actionable recommendations")
    st.caption(
        f"Based on {latest['Sprint']}, the last matching row in your CSV."
    )

    for category, message in make_recommendations(latest, average_velocity):
        st.markdown(f"""
        <div class="panel">
            <div class="eyebrow">{category}</div>
            <p style="font-size:1rem;color:#E1E7F4">{message}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Recommendations by sprint")
    for _, row in view.iterrows():
        with st.expander(
            f"{row['Sprint']} · {row['Risk']} risk · "
            f"{row['Completion_%']:.1f}% completion"
        ):
            for category, message in make_recommendations(row, average_velocity):
                st.write(f"**{category}:** {message}")

# =========================================================
# DATA TABLE AND DOWNLOADS
# =========================================================
with data_tab:
    st.markdown("### Analyzed sprint data")
    st.dataframe(view.round(2), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download analyzed CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="sprintwise_analysis.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇ Download cleaned source CSV",
        data=df[REQUIRED].to_csv(index=False).encode("utf-8"),
        file_name="sprintwise_cleaned_source.csv",
        mime="text/csv",
    )

# =========================================================
# FACULTY PRESENTATION GUIDE
# =========================================================
with guide:
    st.markdown("### Explain the project")
    st.markdown("#### Objective")
    st.write(
        "SprintWise AI transforms sprint CSV data into project delivery "
        "metrics, risk indicators, charts, and actionable recommendations."
    )

    st.markdown("#### Processing pipeline")
    st.code(
        "CSV upload → validation → metric calculation → risk scoring "
        "→ interactive dashboard → recommendations → CSV export",
        language="text",
    )

    st.markdown("#### Core formulas")
    st.latex(
        r"\text{Completion (\%)}="
        r"\frac{\text{Completed Story Points}}"
        r"{\text{Planned Story Points}}\times100"
    )
    st.latex(
        r"\text{Task Completion (\%)}="
        r"\frac{\text{Completed Tasks}}{\text{Planned Tasks}}\times100"
    )
    st.latex(
        r"\text{Risk}=\min(100,\;"
        r"0.45(100-\text{Completion})+4(\text{Bugs})"
        r"+6(\text{Blocked Tasks}))"
    )

    st.markdown("#### Important limitation")
    st.warning(
        "This version uses a rule-based risk score and threshold-based "
        "recommendations. It does not contain a trained Random Forest model. "
        "A real predictive model needs historical sprint records with "
        "independently recorded outcomes and evaluation on held-out data."
    )

    st.markdown("#### Suggested live demonstration")
    st.write(
        "1. Upload a CSV with several sprint records.\n"
        "2. Explain the data validation status and KPI cards.\n"
        "3. Compare velocity and completion charts.\n"
        "4. Explain the risk score and risk categories.\n"
        "5. Show recommendations linked to sprint metrics.\n"
        "6. Download the analyzed CSV."
    )

st.divider()
st.caption(
    "SprintWise AI · Academic prototype · Analysis depends on the "
    "accuracy and completeness of uploaded data."
)
