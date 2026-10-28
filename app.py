
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="SprintWise AI | Sprint Analytics",
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


def validate_and_clean(raw: pd.DataFrame):
    """Return cleaned data, errors, and non-fatal warnings."""
    errors, warnings = [], []
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return None, errors, warnings

    df = df[REQUIRED_COLUMNS].copy()
    df["Sprint"] = df["Sprint"].astype("string").str.strip()

    if df["Sprint"].isna().any() or df["Sprint"].eq("").any():
        errors.append("Sprint names cannot be blank.")
    if df["Sprint"].duplicated().any():
        duplicates = df.loc[
            df["Sprint"].duplicated(), "Sprint"
        ].astype(str).tolist()
        errors.append(
            "Sprint names must be unique. Duplicate(s): "
            + ", ".join(duplicates)
        )

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            errors.append(
                f"Column '{col}' contains blank or non-numeric values."
            )
        elif not np.isfinite(df[col].to_numpy(dtype=float)).all():
            errors.append(f"Column '{col}' contains invalid numeric values.")
        elif (df[col] < 0).any():
            errors.append(f"Column '{col}' cannot contain negative values.")

    if errors:
        return None, errors, warnings

    for col in NUMERIC_COLUMNS:
        if not np.allclose(df[col], np.round(df[col])):
            warnings.append(
                f"'{col}' contains decimal values; values were rounded to whole numbers."
            )
        df[col] = df[col].round().astype(int)

    if (df["Planned_Story_Points"] <= 0).any():
        errors.append("Planned_Story_Points must be greater than zero.")
    if (df["Planned_Tasks"] <= 0).any():
        errors.append("Planned_Tasks must be greater than zero.")
    if (df["Completed_Story_Points"] > df["Planned_Story_Points"]).any():
        warnings.append(
            "Some completed story points exceed planned points. Check the source data."
        )
    if (df["Completed_Tasks"] > df["Planned_Tasks"]).any():
        warnings.append(
            "Some completed task counts exceed planned tasks. Check the source data."
        )
    if (df["Blocked_Tasks"] > df["Planned_Tasks"]).any():
        warnings.append(
            "Some blocked task counts exceed planned tasks. Check the source data."
        )

    if errors:
        return None, errors, warnings
    return df.reset_index(drop=True), errors, warnings


def calculate_metrics(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["Completion_%"] = (
        df["Completed_Story_Points"]
        / df["Planned_Story_Points"]
        * 100
    )
    df["Task_Completion_%"] = (
        df["Completed_Tasks"] / df["Planned_Tasks"] * 100
    )
    df["Velocity"] = df["Completed_Story_Points"]

    # Transparent heuristic score, not an ML-generated probability.
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


def recommendations_for(row, average_velocity):
    recommendations = []

    if row["Completion_%"] < 80:
        recommendations.append(
            "Completion is below 80%. Review unfinished stories and reconsider the sprint commitment."
        )

    if row["Bugs"] >= 5:
        recommendations.append(
            "Bug count is elevated. Investigate recurring defects and reserve capacity for fixes and regression testing."
        )

    if row["Blocked_Tasks"] >= 3:
        recommendations.append(
            "Several tasks are blocked. Identify dependencies, assign owners, and agree on resolution dates."
        )

    if row["Velocity"] < average_velocity:
        recommendations.append(
            "Velocity is below the average for this uploaded dataset. Review capacity, unplanned work, and dependencies."
        )

    if row["Task_Completion_%"] < 80:
        recommendations.append(
            "Task completion is below 80%. Break large stories into smaller tasks and clarify acceptance criteria."
        )

    if not recommendations:
        recommendations.append(
            "Indicators are relatively stable. Continue monitoring delivery, defects, and blockers."
        )

    return recommendations


# Styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        padding: 14px 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 SprintWise AI")
st.markdown("### AI-Driven Sprint Progress Analytics")
st.write(
    "Upload your sprint CSV to calculate delivery metrics, inspect risk "
    "indicators, review recommendations, and download the analyzed report."
)

# CSV upload
with st.sidebar:
    st.header("📂 Upload sprint data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Your file must contain all seven required columns.",
    )

    st.divider()
    st.subheader("Required columns")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")
    st.caption(
        "CSV values should describe one sprint per row. Sprint names must be unique."
    )
    st.caption("For privacy, upload only data you are authorized to share.")

if uploaded_file is None:
    st.info("👈 Upload a CSV file using the sidebar to begin the analysis.")

    st.subheader("How it works")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Upload**\n\nChoose your sprint CSV file.")
    c2.markdown("**2. Analyze**\n\nCalculate metrics and identify risk signals.")
    c3.markdown("**3. Report**\n\nExplore charts and download results.")
    st.stop()

# Read uploaded CSV safely
try:
    raw_df = pd.read_csv(uploaded_file)
except (
    pd.errors.ParserError,
    UnicodeDecodeError,
    ValueError,
    OSError,
) as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

clean_df, errors, warnings = validate_and_clean(raw_df)

if errors:
    st.error("Please correct the following issues in your CSV:")
    for error in errors:
        st.write(f"- {error}")
    st.stop()

for warning in warnings:
    st.warning(warning)

if clean_df.empty:
    st.error("The uploaded CSV contains no sprint records.")
    st.stop()

# Calculate metrics
df = calculate_metrics(clean_df)

st.success(f"Successfully loaded and validated {len(df)} sprint record(s).")

with st.expander("Preview uploaded source data"):
    st.dataframe(clean_df, use_container_width=True, hide_index=True)

# Dashboard filters
with st.sidebar:
    st.divider()
    st.subheader("🔎 Filters")

    risk_filter = st.multiselect(
        "Risk levels",
        RISK_ORDER,
        default=RISK_ORDER,
    )

    minimum_completion = st.slider(
        "Minimum completion (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    )

view_df = df[
    df["Risk"].isin(risk_filter)
    & (df["Completion_%"] >= minimum_completion)
].copy()

if view_df.empty:
    st.warning(
        "No sprints match these filters. Adjust the filters to view results."
    )
    st.stop()

latest = view_df.iloc[-1]
planned_total = view_df["Planned_Story_Points"].sum()

overall_completion = (
    view_df["Completed_Story_Points"].sum() / planned_total * 100
    if planned_total > 0
    else 0
)

average_velocity = view_df["Velocity"].mean()
high_risk_count = int((view_df["Risk"] == "High").sum())

# KPI metrics
st.subheader("📌 Sprint overview")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Overall completion", f"{overall_completion:.1f}%")
m2.metric("Average velocity", f"{average_velocity:.1f} SP")
m3.metric("Total bugs", int(view_df["Bugs"].sum()))
m4.metric("Blocked tasks", int(view_df["Blocked_Tasks"].sum()))
m5.metric("High-risk sprints", high_risk_count)

st.caption(
    f"Latest sprint in the filtered view: {latest['Sprint']} · "
    f"Completion: {latest['Completion_%']:.1f}% · "
    f"Rule-based risk: {latest['Risk']} "
    f"({latest['Risk_Score']:.1f}/100)"
)

# Dashboard tabs
(
    tab_overview,
    tab_risk,
    tab_recommendations,
    tab_data,
    tab_method,
) = st.tabs(
    [
        "📈 Overview",
        "🚦 Risk analysis",
        "💡 Recommendations",
        "📋 Data & export",
        "ℹ️ Methodology",
    ]
)

# Overview charts
with tab_overview:
    left, right = st.columns(2)

    with left:
        st.subheader("Sprint velocity")
        fig = px.line(
            view_df,
            x="Sprint",
            y="Velocity",
            markers=True,
            title="Completed story points per sprint",
        )
        fig.update_layout(
            xaxis_title="Sprint",
            yaxis_title="Story points",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Planned vs completed story points")

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
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Completion trend")
        fig = px.line(
            view_df,
            x="Sprint",
            y="Completion_%",
            markers=True,
        )
        fig.add_hline(
            y=80,
            line_dash="dash",
            annotation_text="80% reference",
        )
        fig.update_layout(yaxis_title="Completion (%)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Bugs and blocked tasks")

        quality_df = view_df.melt(
            id_vars="Sprint",
            value_vars=["Bugs", "Blocked_Tasks"],
            var_name="Indicator",
            value_name="Count",
        )

        fig = px.bar(
            quality_df,
            x="Sprint",
            y="Count",
            color="Indicator",
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

# Risk analysis
with tab_risk:
    st.subheader("Sprint risk analysis")

    fig = px.bar(
        view_df,
        x="Sprint",
        y="Risk_Score",
        color="Risk",
        category_orders={"Risk": RISK_ORDER},
        hover_data=[
            "Completion_%",
            "Bugs",
            "Blocked_Tasks",
        ],
        range_y=[0, 100],
    )

    fig.update_layout(yaxis_title="Rule-based score (0–100)")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Risk is calculated using a transparent heuristic, not a calibrated "
        "probability or a validated ML prediction. Low: below 35; "
        "Medium: 35–59.9; High: 60 or above."
    )

    columns = [
        "Sprint",
        "Completion_%",
        "Task_Completion_%",
        "Velocity",
        "Bugs",
        "Blocked_Tasks",
        "Risk_Score",
        "Risk",
    ]

    st.dataframe(
        view_df[columns].round(1),
        use_container_width=True,
        hide_index=True,
    )

# Recommendations
with tab_recommendations:
    st.subheader("Recommendations")

    st.write(
        f"Recommendations for the latest sprint in this view: **{latest['Sprint']}**"
    )

    for item in recommendations_for(latest, average_velocity):
        st.info(item)

    st.divider()
    st.subheader("Recommendations by sprint")

    for _, row in view_df.iterrows():
        with st.expander(
            f"{row['Sprint']} · {row['Risk']} risk · "
            f"{row['Completion_%']:.1f}% completion"
        ):
            for item in recommendations_for(row, average_velocity):
                st.write(f"- {item}")

# Data table and export
with tab_data:
    st.subheader("Analyzed uploaded data")

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

    st.caption(
        "The export includes calculated metrics and rule-based risk categories "
        "for filtered rows."
    )

# Methodology
with tab_method:
    st.subheader("How the analysis works")

    st.markdown(
        """
        - **Completion:** completed story points divided by planned story points,
          multiplied by 100.
        - **Velocity:** completed story points for each sprint.
        - **Task completion:** completed tasks divided by planned tasks,
          multiplied by 100.
        - **Risk score:** a transparent heuristic combining the shortfall from
          full completion, bug count, and blocked-task count.
        - **Recommendations:** rule-based suggestions triggered by the
          uploaded metrics.

        This version does not claim validated machine-learning predictions.
        To demonstrate a defensible Random Forest, we need historical sprint
        records with independently defined outcome labels and a proper
        held-out evaluation.
        """
    )

    st.caption(
        "SprintWise AI · Academic prototype · Results depend on the quality "
        "of uploaded data."
    )

st.divider()
st.caption(
    "SprintWise AI · Analysis is based only on the CSV uploaded for this session."
)
