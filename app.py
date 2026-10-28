import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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
FEATURE_COLUMNS = ["Completion_%", "Task_Completion_%", "Bugs", "Blocked_Tasks"]
RISK_ORDER = ["Low", "Medium", "High"]

SAMPLE_DATA = pd.DataFrame(
    {
        "Sprint": [f"Sprint {i}" for i in range(1, 13)],
        "Planned_Story_Points": [40, 45, 50, 45, 50, 48, 52, 50, 55, 55, 58, 60],
        "Completed_Story_Points": [38, 40, 32, 25, 46, 44, 39, 48, 41, 30, 54, 43],
        "Planned_Tasks": [20, 22, 25, 23, 24, 24, 26, 25, 27, 28, 29, 30],
        "Completed_Tasks": [19, 20, 17, 13, 23, 22, 20, 24, 21, 16, 27, 22],
        "Bugs": [2, 3, 7, 9, 2, 3, 6, 1, 5, 8, 2, 7],
        "Blocked_Tasks": [1, 1, 4, 6, 0, 1, 3, 0, 3, 5, 1, 4],
    }
)


def validate_and_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Return cleaned data, fatal errors, and non-fatal warnings."""
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
        duplicates = df.loc[df["Sprint"].duplicated(), "Sprint"].astype(str).tolist()
        errors.append("Sprint names must be unique. Duplicate(s): " + ", ".join(duplicates))

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            errors.append(f"Column '{col}' contains blank or non-numeric values.")
        elif (df[col] < 0).any():
            errors.append(f"Column '{col}' cannot contain negative values.")
        elif not np.all(np.isfinite(df[col].to_numpy(dtype=float))):
            errors.append(f"Column '{col}' contains invalid numeric values.")

    if errors:
        return None, errors, warnings

    for col in ["Planned_Story_Points", "Completed_Story_Points", "Planned_Tasks", "Completed_Tasks", "Bugs", "Blocked_Tasks"]:
        if not np.allclose(df[col], np.round(df[col])):
            warnings.append(f"'{col}' contains decimals; values were rounded to whole numbers.")
            df[col] = df[col].round().astype(int)
        else:
            df[col] = df[col].astype(int)

    if (df["Planned_Story_Points"] <= 0).any():
        errors.append("Planned_Story_Points must be greater than zero.")
    if (df["Planned_Tasks"] <= 0).any():
        errors.append("Planned_Tasks must be greater than zero.")
    if (df["Completed_Story_Points"] > df["Planned_Story_Points"]).any():
        warnings.append("Some completed story-point values exceed planned points. Completion is shown as reported, without clipping.")
    if (df["Completed_Tasks"] > df["Planned_Tasks"]).any():
        warnings.append("Some completed-task values exceed planned tasks. Task completion is shown as reported.")
    if (df["Blocked_Tasks"] > df["Planned_Tasks"]).any():
        warnings.append("Some blocked-task counts exceed planned tasks; please verify the source data.")

    if errors:
        return None, errors, warnings
    return df.reset_index(drop=True), errors, warnings


def calculate_metrics(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["Completion_%"] = np.where(
        df["Planned_Story_Points"] > 0,
        df["Completed_Story_Points"] / df["Planned_Story_Points"] * 100,
        np.nan,
    )
    df["Task_Completion_%"] = np.where(
        df["Planned_Tasks"] > 0,
        df["Completed_Tasks"] / df["Planned_Tasks"] * 100,
        np.nan,
    )
    df["Velocity"] = df["Completed_Story_Points"]
    # Transparent heuristic score; this is not a learned probability.
    df["Risk_Score"] = (
        (100 - df["Completion_%"]).clip(lower=0) * 0.45
        + df["Bugs"] * 4
        + df["Blocked_Tasks"] * 6
    ).clip(0, 100)

    def label(score: float) -> str:
        if score >= 60:
            return "High"
        if score >= 35:
            return "Medium"
        return "Low"

    df["Risk"] = df["Risk_Score"].apply(label)
    return df


def recommendations_for(row: pd.Series, average_velocity: float) -> list[str]:
    recommendations = []
    if row["Completion_%"] < 80:
        recommendations.append("Completion is below 80%. Review unfinished stories and re-check whether the sprint commitment was realistic.")
    if row["Bugs"] >= 5:
        recommendations.append("Bug count is elevated. Review recurring defects and reserve capacity for fixing and regression testing.")
    if row["Blocked_Tasks"] >= 3:
        recommendations.append("Several tasks are blocked. Identify the top dependencies, assign owners, and agree on resolution dates.")
    if row["Velocity"] < average_velocity:
        recommendations.append("Velocity is below the dataset average. Review team capacity, unplanned work, and dependency delays.")
    if row["Task_Completion_%"] < 80:
        recommendations.append("Task completion is below 80%. Break large stories into smaller, trackable tasks and clarify acceptance criteria.")
    if not recommendations:
        recommendations.append("The selected sprint has relatively stable indicators. Continue monitoring delivery, defects, and blockers.")
    return recommendations


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: 14px 16px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 SprintWise AI")
st.markdown("### AI-Driven Sprint Progress Analytics")
st.caption("Explore sprint delivery, velocity, quality signals, blockers, and risk indicators.")

with st.sidebar:
    st.header("📂 Data source")
    data_mode = st.radio("Choose data", ["Built-in demo data", "Upload a CSV"], index=0)
    uploaded_file = None
    if data_mode == "Upload a CSV":
        uploaded_file = st.file_uploader("Choose sprint CSV", type=["csv"])
        st.caption("Required columns are listed below.")
    st.divider()
    st.subheader("Required CSV columns")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")
    st.download_button(
        "⬇️ Download sample CSV",
        SAMPLE_DATA.to_csv(index=False).encode("utf-8"),
        file_name="sprintwise_ai_demo_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("Demo records are synthetic and intended for software demonstration.")

if data_mode == "Upload a CSV" and uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not read this CSV file: {exc}")
        st.stop()
elif data_mode == "Upload a CSV":
    st.info("Upload a CSV file to analyze it, or select **Built-in demo data** to explore the dashboard.")
    st.stop()
else:
    raw_df = SAMPLE_DATA.copy()

clean_df, errors, warnings = validate_and_clean(raw_df)
if errors:
    st.error("The file needs to be corrected before analysis:")
    for error in errors:
        st.write(f"- {error}")
    st.info("Use the sample CSV download in the sidebar to check the expected format.")
    st.stop()
for warning in warnings:
    st.warning(warning)

df = calculate_metrics(clean_df)

st.success(f"Loaded {len(df)} sprint records successfully.")
with st.expander("Preview source data"):
    st.dataframe(clean_df, use_container_width=True, hide_index=True)

# Filters
st.sidebar.divider()
st.sidebar.subheader("🔎 Dashboard filters")
risk_filter = st.sidebar.multiselect("Risk levels", RISK_ORDER, default=RISK_ORDER)
minimum_completion = st.sidebar.slider("Minimum completion (%)", 0, 100, 0, 5)
view_df = df[df["Risk"].isin(risk_filter) & (df["Completion_%"] >= minimum_completion)].copy()

if view_df.empty:
    st.warning("No sprints match these filters. Adjust the risk levels or completion threshold.")
    st.stop()

latest = view_df.iloc[-1]
overall_completion = (
    view_df["Completed_Story_Points"].sum() / view_df["Planned_Story_Points"].sum() * 100
    if view_df["Planned_Story_Points"].sum() > 0 else 0
)
avg_velocity = view_df["Velocity"].mean()
total_bugs = int(view_df["Bugs"].sum())
total_blocked = int(view_df["Blocked_Tasks"].sum())
high_risk_count = int((view_df["Risk"] == "High").sum())

st.subheader("📌 Sprint overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Overall completion", f"{overall_completion:.1f}%")
m2.metric("Average velocity", f"{avg_velocity:.1f} SP")
m3.metric("Total bugs", f"{total_bugs}")
m4.metric("Blocked tasks", f"{total_blocked}")
m5.metric("High-risk sprints", f"{high_risk_count}")

st.caption(
    f"Latest sprint in the filtered view: **{latest['Sprint']}** · "
    f"Completion **{latest['Completion_%']:.1f}%** · "
    f"Heuristic risk **{latest['Risk']}** (score {latest['Risk_Score']:.1f}/100)"
)

tab_overview, tab_risk, tab_recommendations, tab_data, tab_model = st.tabs(
    ["📈 Overview", "🚦 Risk analysis", "💡 Recommendations", "📋 Data & export", "🤖 ML notes"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Velocity by sprint")
        fig = px.line(view_df, x="Sprint", y="Velocity", markers=True, title="Completed story points")
        fig.update_layout(xaxis_title="Sprint", yaxis_title="Story points")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Planned vs completed")
        plot_df = view_df.melt(
            id_vars="Sprint",
            value_vars=["Planned_Story_Points", "Completed_Story_Points"],
            var_name="Series",
            value_name="Story points",
        )
        fig = px.bar(plot_df, x="Sprint", y="Story points", color="Series", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Completion trend")
        fig = px.line(view_df, x="Sprint", y="Completion_%", markers=True)
        fig.add_hline(y=80, line_dash="dash", annotation_text="80% reference")
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
        fig = px.bar(quality_df, x="Sprint", y="Count", color="Indicator", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

with tab_risk:
    st.subheader("Risk score by sprint")
    fig = px.bar(
        view_df,
        x="Sprint",
        y="Risk_Score",
        color="Risk",
        category_orders={"Risk": RISK_ORDER},
        hover_data=["Completion_%", "Bugs", "Blocked_Tasks"],
        range_y=[0, 100],
    )
    fig.update_layout(yaxis_title="Heuristic score (0–100)")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Risk score is a transparent rule-based indicator, not a calibrated probability of sprint failure. "
        "Low: below 35; Medium: 35–59.9; High: 60 or above."
    )
    display_cols = ["Sprint", "Completion_%", "Task_Completion_%", "Velocity", "Bugs", "Blocked_Tasks", "Risk_Score", "Risk"]
    risk_table = view_df[display_cols].copy()
    st.dataframe(risk_table.round(1), use_container_width=True, hide_index=True)

with tab_recommendations:
    st.subheader("Recommendations for the latest sprint in this view")
    st.write(f"Selected sprint: **{latest['Sprint']}**")
    for item in recommendations_for(latest, avg_velocity):
        st.info(item)
    st.divider()
    st.subheader("Sprint-by-sprint recommendations")
    for _, row in view_df.iterrows():
        with st.expander(f"{row['Sprint']} · {row['Risk']} risk · {row['Completion_%']:.1f}% completion"):
            for item in recommendations_for(row, avg_velocity):
                st.write(f"- {item}")

with tab_data:
    st.subheader("Analyzed sprint data")
    st.dataframe(view_df.round(2), use_container_width=True, hide_index=True)
    export_bytes = view_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download analyzed CSV",
        data=export_bytes,
        file_name="sprintwise_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("Export includes calculated metrics and rule-based risk labels for the filtered rows.")

with tab_model:
    st.subheader("Machine-learning status")
    st.write(
        "This app uses a transparent rule-based score for its default risk labels. "
        "A Random Forest should only be described as a predictive model when it is trained and evaluated "
        "against independently labelled historical outcomes."
    )
    if "Risk_Label" in raw_df.columns:
        st.write("A `Risk_Label` column was found in the uploaded file.")
        st.caption("To use it responsibly, labels must represent observed outcomes and not be generated from the same features and score being evaluated.")
        st.warning("This version does not automatically train a model from Risk_Label; validate label quality and class counts before adding model evaluation.")
    else:
        st.write("No `Risk_Label` column was provided, so the app does not claim validated Random Forest prediction.")
        st.markdown(
            """
            **For a defensible Random Forest experiment**, add a `Risk_Label` column with historical outcome labels
            (`Low`, `Medium`, or `High`) based on a documented definition. Use enough examples per class, keep
            a held-out test set, and report metrics such as precision, recall, F1-score, and a confusion matrix.
            """
        )
    st.caption("The built-in demo data is synthetic. It demonstrates the dashboard workflow, not real-world model accuracy.")

st.divider()
st.caption("SprintWise AI · Academic prototype · Results depend on the quality and completeness of the supplied sprint data.")
