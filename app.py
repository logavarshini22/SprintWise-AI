import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="SprintWise AI | Sprint Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_COLUMNS = [
    "Sprint",
    "Planned_Story_Points",
    "Completed_Story_Points",
    "Planned_Tasks",
    "Completed_Tasks",
    "Bugs",
    "Blocked_Tasks",
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]
RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = {
    "Low": "#16A085",
    "Medium": "#E6A23C",
    "High": "#E05D5D",
}


# ============================== PREMIUM DESIGN ==============================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #F5F7FB;
    }

    [data-testid="stHeader"] {
        background: rgba(245,247,251,.92);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E7ECF3;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #17233C;
        letter-spacing: -.025em;
    }

    h1, h2, h3 {
        font-weight: 800 !important;
    }

    p, label {
        color: #52617A;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E7ECF3;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 18px rgba(30,50,90,.035);
    }

    [data-testid="stMetricLabel"] {
        color: #71809A;
        font-size: .83rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #17233C;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    .hero {
        background: linear-gradient(
            120deg, #17233C 0%, #253E69 58%, #3B5D91 100%
        );
        border-radius: 22px;
        padding: 30px 32px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 14px 32px rgba(27,47,82,.14);
    }

    .hero-kicker {
        color: #AFC6F3;
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .72rem;
        font-weight: 700;
        margin-bottom: 9px;
    }

    .hero-title {
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: 2.1rem;
        line-height: 1.15;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero-copy {
        color: #D9E3F5;
        font-size: 1rem;
        margin: 0;
        max-width: 850px;
    }

    .section-eyebrow {
        color: #75849D;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .7rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .panel {
        background: white;
        border: 1px solid #E7ECF3;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 10px;
        box-shadow: 0 4px 18px rgba(30,50,90,.025);
    }

    .panel-label {
        color: #71809A;
        font-size: .78rem;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .panel-value {
        color: #17233C;
        font-family: 'Manrope', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
    }

    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: .76rem;
        font-weight: 700;
    }

    .step-card {
        background: #FFFFFF;
        border: 1px solid #E7ECF3;
        border-radius: 15px;
        padding: 18px;
        min-height: 130px;
    }

    .step-num {
        display: inline-flex;
        width: 29px;
        height: 29px;
        border-radius: 9px;
        align-items: center;
        justify-content: center;
        background: #EAF0FC;
        color: #315AA6;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .step-title {
        color: #17233C;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .step-copy {
        color: #71809A;
        font-size: .84rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E7ECF3;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 12px 15px;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        color: #315AA6 !important;
    }

    .stButton button, .stDownloadButton button {
        border-radius: 10px;
        font-weight: 700;
    }

    [data-testid="stFileUploader"] {
        background: #F7F9FD;
        border: 1px dashed #C9D5E8;
        padding: 12px;
        border-radius: 14px;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================== DATA VALIDATION ==============================

def validate_and_clean(raw):
    errors = []
    warnings = []

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        return (
            None,
            ["Missing required columns: " + ", ".join(missing)],
            warnings,
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["Sprint"] = df["Sprint"].astype("string").str.strip()

    if df["Sprint"].isna().any() or df["Sprint"].eq("").any():
        errors.append("Sprint names cannot be blank.")

    if df["Sprint"].duplicated().any():
        duplicates = (
            df.loc[df["Sprint"].duplicated(), "Sprint"]
            .astype(str)
            .tolist()
        )
        errors.append(
            "Sprint names must be unique. Duplicates: "
            + ", ".join(duplicates)
        )

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if df[column].isna().any():
            errors.append(
                f"'{column}' contains blank or non-numeric values."
            )
        elif not np.isfinite(df[column].to_numpy(dtype=float)).all():
            errors.append(f"'{column}' contains invalid numeric values.")
        elif (df[column] < 0).any():
            errors.append(f"'{column}' cannot contain negative values.")

    if errors:
        return None, errors, warnings

    for column in NUMERIC_COLUMNS:
        if not np.allclose(df[column], np.round(df[column])):
            warnings.append(
                f"'{column}' contains decimals; values were rounded."
            )

        df[column] = df[column].round().astype(int)

    if (df["Planned_Story_Points"] <= 0).any():
        errors.append("'Planned_Story_Points' must be greater than zero.")

    if (df["Planned_Tasks"] <= 0).any():
        errors.append("'Planned_Tasks' must be greater than zero.")

    if (
        df["Completed_Story_Points"] > df["Planned_Story_Points"]
    ).any():
        warnings.append(
            "Some completed story points exceed planned points. "
            "Please verify the data."
        )

    if (df["Completed_Tasks"] > df["Planned_Tasks"]).any():
        warnings.append(
            "Some completed task counts exceed planned task counts."
        )

    if (df["Blocked_Tasks"] > df["Planned_Tasks"]).any():
        warnings.append(
            "Some blocked-task counts exceed planned task counts."
        )

    if errors:
        return None, errors, warnings

    return df.reset_index(drop=True), errors, warnings


# ============================== METRIC ENGINE ==============================

def calculate_metrics(data):
    df = data.copy()

    df["Completion_%"] = (
        df["Completed_Story_Points"]
        / df["Planned_Story_Points"]
        * 100
    )

    df["Task_Completion_%"] = (
        df["Completed_Tasks"]
        / df["Planned_Tasks"]
        * 100
    )

    df["Velocity"] = df["Completed_Story_Points"]

    df["Risk_Score"] = (
        (100 - df["Completion_%"]).clip(lower=0) * 0.45
        + df["Bugs"] * 4
        + df["Blocked_Tasks"] * 6
    ).clip(0, 100)

    def risk_label(score):
        if score >= 60:
            return "High"
        if score >= 35:
            return "Medium"
        return "Low"

    df["Risk"] = df["Risk_Score"].apply(risk_label)

    return df


# ============================ RECOMMENDATION ENGINE ============================

def recommendations_for(row, average_velocity):
    recommendations = []

    if row["Completion_%"] < 80:
        recommendations.append((
            "Delivery",
            "Completion is below 80%. Review unfinished stories "
            "and reconsider the sprint commitment."
        ))

    if row["Bugs"] >= 5:
        recommendations.append((
            "Quality",
            "Bug count is elevated. Investigate recurring defects "
            "and reserve capacity for fixes and regression testing."
        ))

    if row["Blocked_Tasks"] >= 3:
        recommendations.append((
            "Dependencies",
            "Several tasks are blocked. Identify dependencies, "
            "assign owners, and agree on resolution dates."
        ))

    if row["Velocity"] < average_velocity:
        recommendations.append((
            "Velocity",
            "Velocity is below the selected sprint average. "
            "Review capacity, unplanned work, and dependency delays."
        ))

    if row["Task_Completion_%"] < 80:
        recommendations.append((
            "Planning",
            "Task completion is below 80%. Break large stories "
            "into smaller tasks and clarify acceptance criteria."
        ))

    if not recommendations:
        recommendations.append((
            "Stable",
            "Indicators are relatively stable. Continue monitoring "
            "delivery, defects, and blockers."
        ))

    return recommendations


# ================================ CHART STYLE ================================

def chart_theme(fig, height=330):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color="#52617A",
            size=12,
        ),
        margin=dict(l=8, r=8, t=48, b=8),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(showgrid=False, linecolor="#E7ECF3", title=None),
        yaxis=dict(gridcolor="#EDF1F7", zeroline=False, title=None),
        title=dict(
            font=dict(
                size=15,
                color="#17233C",
                family="Manrope, sans-serif",
            )
        ),
    )
    return fig


# ================================= SIDEBAR =================================

with st.sidebar:
    st.markdown("## 📊 SprintWise AI")
    st.caption("SPRINT INTELLIGENCE WORKSPACE")
    st.divider()

    st.markdown("### 01 · Upload dataset")

    uploaded_file = st.file_uploader(
        "Choose a sprint CSV",
        type=["csv"],
        help="Upload one row per sprint.",
    )

    with st.expander("Required CSV format"):
        st.code(", ".join(REQUIRED_COLUMNS), language="text")
        st.caption(
            "Sprint names must be unique. Numeric values cannot be negative."
        )

    st.divider()
    st.markdown("### 02 · Analysis filters")
    st.caption("Filters become available after a file is uploaded.")

    st.markdown("---")
    st.caption("Academic prototype · Sprint analytics")


# =================================== HEADER ==================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Agile delivery intelligence</div>
        <div class="hero-title">SprintWise AI</div>
        <p class="hero-copy">
            Turn sprint data into a clear picture of delivery performance,
            quality signals, blockers, and project risk — in one workspace.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================== UPLOAD LANDING ==============================

if uploaded_file is None:
    st.markdown("### Start with your sprint data")
    st.write(
        "Upload a CSV from the left sidebar. "
        "The dashboard will be generated from your file. "
        "No sample data is loaded automatically."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">1</div>
                <div class="step-title">Upload</div>
                <div class="step-copy">
                    Choose your sprint CSV and check its columns.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">2</div>
                <div class="step-title">Analyze</div>
                <div class="step-copy">
                    Calculate completion, velocity, quality, and risk.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">3</div>
                <div class="step-title">Present results</div>
                <div class="step-copy">
                    Explore charts, read recommendations, and export results.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# ================================ READ THE CSV ================================

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

clean_df, errors, warnings = validate_and_clean(raw_df)

if errors:
    st.error("Please correct the uploaded file before analysis:")

    for error in errors:
        st.write(f"- {error}")

    st.info(
        "Check the required column names in the sidebar, "
        "correct the CSV, and upload it again."
    )
    st.stop()

for warning in warnings:
    st.warning(warning)

if clean_df.empty:
    st.error("The uploaded CSV contains no sprint records.")
    st.stop()

df = calculate_metrics(clean_df)

st.markdown(
    f'<span class="pill" style="background:#E5F7F1;color:#137A61;">'
    f'✓ DATA VALIDATED</span> '
    f'<span style="color:#71809A;font-size:.86rem;">'
    f'{len(df)} sprint records · '
    f'{len(REQUIRED_COLUMNS)} source columns</span>',
    unsafe_allow_html=True,
)


# ================================== FILTERS ==================================

with st.sidebar:
    st.markdown("### 02 · Refine analysis")

    risk_filter = st.multiselect(
        "Risk categories",
        RISK_ORDER,
        default=RISK_ORDER,
    )

    minimum_completion = st.slider(
        "Minimum completion",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        format="%d%%",
    )

view_df = df[
    df["Risk"].isin(risk_filter)
    & (df["Completion_%"] >= minimum_completion)
].copy()

if view_df.empty:
    st.warning(
        "No sprints match these filters. "
        "Adjust the risk categories or completion threshold."
    )
    st.stop()


# ============================== EXECUTIVE KPIs ==============================

latest = view_df.iloc[-1]

overall_completion = (
    view_df["Completed_Story_Points"].sum()
    / view_df["Planned_Story_Points"].sum()
    * 100
)

average_velocity = float(view_df["Velocity"].mean())
total_bugs = int(view_df["Bugs"].sum())
total_blocked = int(view_df["Blocked_Tasks"].sum())
high_risk_count = int((view_df["Risk"] == "High").sum())

st.markdown(
    '<div class="section-eyebrow">Executive snapshot</div>',
    unsafe_allow_html=True,
)

st.markdown("### Sprint performance at a glance")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Overall completion",
    f"{overall_completion:.1f}%",
    help="Completed story points divided by planned story points.",
)

m2.metric(
    "Average velocity",
    f"{average_velocity:.1f} SP",
    help="Average completed story points per selected sprint.",
)

m3.metric(
    "Bugs reported",
    str(total_bugs),
    help="Total reported bugs across selected sprints.",
)

m4.metric(
    "Blocked tasks",
    str(total_blocked),
    help="Total blocked-task reports across selected sprints.",
)

m5.metric(
    "High-risk sprints",
    str(high_risk_count),
    help="Number of sprints with a risk score of 60 or higher.",
)

risk_now = str(latest["Risk"])

risk_bg = {
    "Low": "#E5F7F1",
    "Medium": "#FFF3D9",
    "High": "#FDE9E8",
}[risk_now]

risk_fg = {
    "Low": "#137A61",
    "Medium": "#93630B",
    "High": "#B93838",
}[risk_now]

st.markdown(
    f'<div class="panel">'
    f'<div class="panel-label">LATEST SPRINT IN CURRENT VIEW</div>'
    f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
    f'<div class="panel-value">{latest["Sprint"]}</div>'
    f'<span class="pill" style="background:{risk_bg};color:{risk_fg};">'
    f'{risk_now.upper()} RISK</span>'
    f'<span style="color:#71809A;">'
    f'Completion {latest["Completion_%"]:.1f}% · '
    f'Velocity {latest["Velocity"]:.0f} SP · '
    f'Risk score {latest["Risk_Score"]:.1f}/100'
    f'</span></div></div>',
    unsafe_allow_html=True,
)


# ================================= MAIN TABS =================================

(
    tab_overview,
    tab_delivery,
    tab_risk,
    tab_recs,
    tab_data,
    tab_guide,
) = st.tabs([
    "Overview",
    "Delivery analytics",
    "Risk & quality",
    "Recommendations",
    "Data & export",
    "Explain this project",
])


# ================================ OVERVIEW ==================================

with tab_overview:
    st.markdown("### Performance overview")

    left, right = st.columns(2)

    with left:
        fig = px.line(
            view_df,
            x="Sprint",
            y="Velocity",
            markers=True,
            title="Velocity by sprint",
        )

        fig.update_traces(
            line=dict(color="#3D6FC4", width=3),
            marker=dict(size=8),
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True,
        )

    with right:
        plot_df = view_df.melt(
            id_vars="Sprint",
            value_vars=[
                "Planned_Story_Points",
                "Completed_Story_Points",
            ],
            var_name="Series",
            value_name="Story points",
        )

        fig = px.bar(
            plot_df,
            x="Sprint",
            y="Story points",
            color="Series",
            barmode="group",
            title="Planned vs completed story points",
            color_discrete_map={
                "Planned_Story_Points": "#C7D4E8",
                "Completed_Story_Points": "#3D6FC4",
            },
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True,
        )

    left, right = st.columns(2)

    with left:
        fig = px.line(
            view_df,
            x="Sprint",
            y="Completion_%",
            markers=True,
            title="Story-point completion trend",
        )

        fig.update_traces(
            line=dict(color="#16A085", width=3),
            marker=dict(size=8),
        )

        fig.add_hline(
            y=80,
            line_dash="dash",
            line_color="#E6A23C",
            annotation_text="80% reference",
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True,
        )

    with right:
        quality = view_df.melt(
            id_vars="Sprint",
            value_vars=["Bugs", "Blocked_Tasks"],
            var_name="Indicator",
            value_name="Count",
        )

        fig = px.bar(
            quality,
            x="Sprint",
            y="Count",
            color="Indicator",
            barmode="group",
            title="Quality and blocker signals",
            color_discrete_map={
                "Bugs": "#E05D5D",
                "Blocked_Tasks": "#E6A23C",
            },
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True,
        )

    st.markdown("### What stands out")

    if overall_completion < 80:
        st.warning(
            f"Overall completion is {overall_completion:.1f}%, "
            "below the 80% reference threshold."
        )
    else:
        st.success(
            f"Overall completion is {overall_completion:.1f}%, "
            "meeting or exceeding the 80% reference threshold."
        )

    if high_risk_count:
        st.warning(
            f"{high_risk_count} selected sprint(s) have a high "
            "rule-based risk score."
        )
    else:
        st.info(
            "No selected sprint is classified as high risk "
            "by the current scoring rules."
        )

    if total_blocked:
        st.info(
            f"There are {total_blocked} blocked-task reports "
            "across the selected sprints."
        )


# ============================ DELIVERY ANALYTICS ============================

with tab_delivery:
    st.markdown("### Delivery analytics")

    left, right = st.columns(2)

    with left:
        task_fig = px.bar(
            view_df,
            x="Sprint",
            y="Task_Completion_%",
            title="Task completion by sprint",
        )

        task_fig.update_traces(marker_color="#6C8FD7")

        task_fig.add_hline(
            y=80,
            line_dash="dash",
            line_color="#E6A23C",
        )

        st.plotly_chart(
            chart_theme(task_fig),
            use_container_width=True,
        )

    with right:
        completion_gap = view_df.copy()

        completion_gap["Uncompleted_Story_Points"] = (
            completion_gap["Planned_Story_Points"]
            - completion_gap["Completed_Story_Points"]
        ).clip(lower=0)

        gap_fig = px.bar(
            completion_gap,
            x="Sprint",
            y="Uncompleted_Story_Points",
            title="Remaining planned story points",
        )

        gap_fig.update_traces(marker_color="#8C9BB5")

        st.plotly_chart(
            chart_theme(gap_fig),
            use_container_width=True,
        )

    st.markdown("#### Sprint-by-sprint delivery table")

    delivery_cols = [
        "Sprint",
        "Planned_Story_Points",
        "Completed_Story_Points",
        "Completion_%",
        "Planned_Tasks",
        "Completed_Tasks",
        "Task_Completion_%",
        "Velocity",
    ]

    st.dataframe(
        view_df[delivery_cols].round(1),
        use_container_width=True,
        hide_index=True,
    )


# ============================== RISK & QUALITY ==============================

with tab_risk:
    st.markdown("### Risk and quality analysis")

    left, right = st.columns([1.3, 1])

    with left:
        fig = px.bar(
            view_df,
            x="Sprint",
            y="Risk_Score",
            color="Risk",
            category_orders={"Risk": RISK_ORDER},
            color_discrete_map=RISK_COLORS,
            hover_data=[
                "Completion_%",
                "Bugs",
                "Blocked_Tasks",
            ],
            range_y=[0, 100],
            title="Rule-based risk score by sprint",
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True,
        )

    with right:
        risk_counts = (
            view_df["Risk"]
            .value_counts()
            .reindex(RISK_ORDER, fill_value=0)
            .reset_index()
        )

        risk_counts.columns = ["Risk", "Sprint count"]

        fig = px.pie(
            risk_counts,
            names="Risk",
            values="Sprint count",
            hole=0.66,
            color="Risk",
            color_discrete_map=RISK_COLORS,
            title="Risk category distribution",
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True,
        )

    st.info(
        "Risk score is a transparent heuristic, not a calibrated "
        "failure probability. Score = 0.45 × completion shortfall "
        "+ 4 × bugs + 6 × blocked tasks, capped at 100. "
        "Low: below 35; Medium: 35–59.9; High: 60 or above."
    )

    risk_cols = [
        "Sprint",
        "Completion_%",
        "Bugs",
        "Blocked_Tasks",
        "Risk_Score",
        "Risk",
    ]

    st.dataframe(
        view_df[risk_cols].round(1),
        use_container_width=True,
        hide_index=True,
    )


# =========================== RECOMMENDATIONS ================================

with tab_recs:
    st.markdown("### Actionable recommendations")

    st.caption(
        "Recommendations are based on the latest sprint in the current view: "
        f"{latest['Sprint']}."
    )

    for category, recommendation in recommendations_for(
        latest,
        average_velocity,
    ):
        st.markdown(
            f'<div class="panel">'
            f'<div class="section-eyebrow">{category}</div>'
            f'<div style="color:#25334D;font-weight:600;">'
            f'{recommendation}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Recommendations for each sprint")

    for _, row in view_df.iterrows():
        with st.expander(
            f"{row['Sprint']} · {row['Risk']} risk · "
            f"{row['Completion_%']:.1f}% completion"
        ):
            for category, recommendation in recommendations_for(
                row,
                average_velocity,
            ):
                st.markdown(
                    f"**{category}:** {recommendation}"
                )


# =============================== DATA & EXPORT ==============================

with tab_data:
    st.markdown("### Uploaded and analyzed data")

    st.caption(
        "The table includes calculated metrics. "
        "Download the filtered analysis or cleaned source data."
    )

    st.dataframe(
        view_df.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download analyzed CSV",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name="sprintwise_analysis.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇️ Download cleaned source CSV",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name="sprintwise_cleaned_source.csv",
        mime="text/csv",
    )


# ============================= PROJECT EXPLANATION ==========================

with tab_guide:
    st.markdown("### How to explain SprintWise AI to faculty")

    st.markdown("#### Project objective")

    st.write(
        "SprintWise AI is a sprint analytics dashboard that transforms "
        "project-management CSV data into delivery metrics, risk indicators, "
        "visual summaries, and practical recommendations."
    )

    st.markdown("#### Processing pipeline")

    st.code(
        "CSV upload → validation → metric calculation → risk scoring "
        "→ visual analytics → recommendations → CSV export",
        language="text",
    )

    st.markdown("#### Key formulas")

    st.latex(
        r"\text{Completion (\%)} = "
        r"\frac{\text{Completed Story Points}}"
        r"{\text{Planned Story Points}} \times 100"
    )

    st.latex(
        r"\text{Task Completion (\%)} = "
        r"\frac{\text{Completed Tasks}}"
        r"{\text{Planned Tasks}} \times 100"
    )

    st.latex(
        r"\text{Risk} = \min\left(100,\;"
        r"0.45(100-\text{Completion})"
        r"+4(\text{Bugs})+6(\text{Blocked Tasks})\right)"
    )

    st.markdown("#### Important AI limitation")

    st.warning(
        "This version uses a rule-based risk score and "
        "rule-triggered recommendations. It does not claim a validated "
        "machine-learning prediction. A genuine predictive model requires "
        "historical sprint records with independently recorded outcomes, "
        "a held-out test set, and evaluation metrics."
    )

    st.markdown("#### Suggested live-demo sequence")

    st.markdown(
        """
        1. Upload a prepared CSV containing multiple sprint records.
        2. Show the validation status and KPI cards.
        3. Compare velocity and completion charts.
        4. Open Risk & quality and explain the risk formula.
        5. Open Recommendations and connect each suggestion to the data.
        6. Download the analyzed CSV to demonstrate the output workflow.
        """
    )


# ================================= FOOTER ===================================

st.divider()

st.caption(
    "SprintWise AI · Academic prototype · "
    "Findings depend on the accuracy and completeness of the uploaded data."
)
