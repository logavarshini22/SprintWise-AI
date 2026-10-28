
import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================================
# SPRINTWISE AI — Sprint Performance & Risk Analytics
# CSV upload only. No demo data is loaded automatically.
# ==========================================================

st.set_page_config(
    page_title="SprintWise AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Premium dark theme ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background:
      radial-gradient(ellipse at 12% 0%, rgba(105,70,220,.18), transparent 32%),
      radial-gradient(ellipse at 92% 12%, rgba(0,194,255,.10), transparent 28%),
      #080b14;
    color: #edf1ff;
}
[data-testid="stHeader"] {
    background: rgba(8,11,20,.72);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#101528,#0b1020);
    border-right: 1px solid rgba(143,157,201,.15);
}
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -.035em;
}
h1 { font-size: 2.6rem !important; }
p, label { color: #aeb8d3; }

div[data-testid="stMetric"] {
    background: linear-gradient(145deg,rgba(23,31,55,.96),rgba(15,21,39,.96));
    border: 1px solid rgba(132,148,196,.17);
    padding: 18px 20px;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,.14);
}
div[data-testid="stMetricLabel"] { color: #9eabc9 !important; }
div[data-testid="stMetricValue"] {
    color: #f5f7ff !important;
    font-family: 'Space Grotesk', sans-serif;
}
.hero {
    padding: 26px 30px;
    border-radius: 24px;
    margin-bottom: 22px;
    border: 1px solid rgba(139,125,255,.28);
    background: linear-gradient(120deg,rgba(104,75,220,.19),
        rgba(18,29,52,.84) 52%,rgba(0,194,255,.08));
}
.eyebrow {
    color: #8e9cff;
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
}
.hero h1 { margin: 5px 0 8px 0; }
.hero p { margin-bottom: 0; max-width: 820px; line-height: 1.65; }
.pill {
    display: inline-block;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: .75rem;
    background: rgba(139,125,255,.13);
    border: 1px solid rgba(139,125,255,.26);
    color: #c8c2ff;
}
div.stButton > button, div.stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid rgba(139,125,255,.38);
    background: linear-gradient(100deg,#6e5ae6,#4c78db);
    color: white;
    font-weight: 700;
    min-height: 42px;
}
[data-testid="stFileUploader"] {
    background: rgba(18,25,45,.75);
    border: 1px dashed rgba(139,157,213,.42);
    border-radius: 18px;
    padding: 12px;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(132,148,196,.16);
    border-radius: 14px;
    overflow: hidden;
}
.footer {
    color: #657291;
    font-size: .78rem;
    text-align: center;
    padding: 28px 0 8px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- CSV schema ----------------
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


# ---------------- Data validation ----------------
def validate_csv(uploaded_file):
    """Return dataframe, validation errors, and warnings."""
    errors, warnings = [], []

    try:
        raw = pd.read_csv(uploaded_file)
    except Exception as exc:
        return None, [f"Could not read the CSV file: {exc}"], []

    if raw.empty:
        return raw, ["The CSV contains no data rows."], []

    raw.columns = [str(c).strip() for c in raw.columns]

    if raw.columns.duplicated().any():
        duplicates = raw.columns[raw.columns.duplicated()].tolist()
        errors.append(
            "Duplicate column names: " + ", ".join(duplicates)
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        errors.append(
            "Missing required columns: " + ", ".join(missing)
        )
        return raw, errors, warnings

    df = raw[REQUIRED_COLUMNS].copy()
    df["Sprint"] = df["Sprint"].astype("string").str.strip()

    if df["Sprint"].isna().any() or (df["Sprint"] == "").any():
        errors.append("Sprint names cannot be blank.")

    duplicates = (
        df.loc[df["Sprint"].duplicated(keep=False), "Sprint"]
        .dropna()
        .unique()
        .tolist()
    )
    if duplicates:
        errors.append(
            "Sprint names must be unique. Duplicates: "
            + ", ".join(map(str, duplicates))
        )

    for col in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid = converted.isna()

        if invalid.any():
            rows = (np.flatnonzero(invalid.to_numpy()) + 2).tolist()
            errors.append(
                f"{col} has blank or non-numeric values "
                f"near CSV row(s): {rows[:8]}"
            )

        df[col] = converted

    if not errors:
        if (df[NUMERIC_COLUMNS] < 0).any().any():
            errors.append("Numeric values cannot be negative.")

        if (df["Planned_Story_Points"] <= 0).any():
            errors.append(
                "Planned_Story_Points must be greater than zero."
            )

        if (df["Planned_Tasks"] <= 0).any():
            errors.append("Planned_Tasks must be greater than zero.")

        for col in NUMERIC_COLUMNS:
            if (df[col] % 1 != 0).any():
                warnings.append(
                    f"{col} contains decimal values. "
                    "Check whether decimals are intentional."
                )
                break

        if (
            df["Completed_Story_Points"]
            > df["Planned_Story_Points"]
        ).any():
            warnings.append(
                "Some sprints completed more story points than planned. "
                "Check for scope changes."
            )

        if (
            df["Completed_Tasks"] > df["Planned_Tasks"]
        ).any():
            warnings.append(
                "Some sprints completed more tasks than planned."
            )

        if (
            df["Blocked_Tasks"] > df["Planned_Tasks"]
        ).any():
            warnings.append(
                "Blocked tasks exceed planned tasks in at least one sprint."
            )

        if len(df) < 3:
            warnings.append(
                "Fewer than three sprints were uploaded. "
                "Trend comparisons may be limited."
            )

    return df, errors, warnings


# ---------------- Analytics and risk scoring ----------------
def analyze_sprints(
    df,
    completion_threshold,
    bug_threshold,
    blocked_threshold,
):
    out = df.copy()

    out["Completion_Rate_%"] = (
        out["Completed_Story_Points"]
        / out["Planned_Story_Points"] * 100
    ).round(1)

    out["Task_Completion_%"] = (
        out["Completed_Tasks"] / out["Planned_Tasks"] * 100
    ).round(1)

    out["Remaining_Story_Points"] = (
        out["Planned_Story_Points"]
        - out["Completed_Story_Points"]
    ).clip(lower=0)

    out["Scope_Overrun"] = (
        (out["Completed_Story_Points"] > out["Planned_Story_Points"])
        | (out["Completed_Tasks"] > out["Planned_Tasks"])
    )

    # Transparent rule-based risk score, not a trained ML model.
    score = (
        np.where(
            out["Completion_Rate_%"] < completion_threshold, 40, 0
        )
        + np.where(out["Bugs"] >= bug_threshold, 25, 0)
        + np.where(
            out["Blocked_Tasks"] >= blocked_threshold, 25, 0
        )
        + np.where(
            out["Task_Completion_%"] < completion_threshold, 10, 0
        )
    )

    out["Risk_Score"] = np.minimum(score, 100).astype(int)

    out["Risk_Level"] = np.select(
        [
            out["Risk_Score"] >= 60,
            out["Risk_Score"] >= 30,
        ],
        ["High", "Medium"],
        default="Low",
    )

    out["Data_Quality"] = np.where(
        out["Scope_Overrun"], "Review scope", "OK"
    )

    return out


# ---------------- Recommendation engine ----------------
def make_recommendations(
    df,
    completion_threshold,
    bug_threshold,
    blocked_threshold,
):
    recommendations = []

    avg_completion = df["Completion_Rate_%"].mean()
    avg_bugs = df["Bugs"].mean()
    avg_blocked = df["Blocked_Tasks"].mean()

    if avg_completion < completion_threshold:
        recommendations.append((
            "Delivery",
            "Reduce sprint commitment",
            f"Average story-point completion is {avg_completion:.1f}%, "
            f"below your {completion_threshold:.0f}% target. "
            "Plan fewer points or split large stories."
        ))
    else:
        recommendations.append((
            "Delivery",
            "Maintain a realistic commitment",
            f"Average story-point completion is {avg_completion:.1f}%. "
            "Review sprint-to-sprint variation before increasing capacity."
        ))

    if avg_bugs >= bug_threshold:
        recommendations.append((
            "Quality",
            "Add earlier quality checks",
            f"Average bugs per sprint are {avg_bugs:.1f}, at or above "
            f"the threshold of {bug_threshold}. Consider peer reviews, "
            "automated tests, and clearer acceptance criteria."
        ))
    else:
        recommendations.append((
            "Quality",
            "Keep quality practices consistent",
            f"Average bugs per sprint are {avg_bugs:.1f}. "
            "Continue regression testing and track bug severity."
        ))

    if avg_blocked >= blocked_threshold:
        recommendations.append((
            "Flow",
            "Resolve blockers earlier",
            f"Average blocked tasks are {avg_blocked:.1f}. "
            "Review dependencies during planning and assign blocker owners."
        ))
    else:
        recommendations.append((
            "Flow",
            "Monitor dependency delays",
            f"Average blocked tasks are {avg_blocked:.1f}. "
            "Consider tracking blocker age in future data."
        ))

    worst = df.sort_values(
        ["Risk_Score", "Completion_Rate_%"],
        ascending=[False, True],
    ).iloc[0]

    if worst["Risk_Level"] != "Low":
        recommendations.append((
            "Priority",
            f"Review {worst['Sprint']}",
            f"This sprint has {worst['Risk_Level'].lower()} risk "
            f"({int(worst['Risk_Score'])}/100) and "
            f"{worst['Completion_Rate_%']:.1f}% completion. "
            "Inspect blockers, scope changes, and bug causes."
        ))

    if df["Scope_Overrun"].any():
        recommendations.append((
            "Data",
            "Confirm scope changes",
            "Some sprints exceeded planned points or tasks. "
            "Confirm whether sprint scope changed."
        ))

    return recommendations


# ---------------- App header ----------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Agile intelligence · Sprint analytics</div>
  <h1>⚡ SprintWise AI</h1>
  <p>Turn your sprint history into clear delivery signals. Upload your
  own CSV to inspect completion, velocity, quality, blockers, risk,
  and practical next steps.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.markdown("## ⚡ SprintWise")
    st.caption("Performance workspace")
    st.markdown("---")
    st.markdown("### Analysis settings")

    completion_threshold = st.slider(
        "Minimum completion target (%)", 50, 100, 80, 5
    )
    bug_threshold = st.number_input(
        "Bug risk threshold",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
    )
    blocked_threshold = st.number_input(
        "Blocked-task risk threshold",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
    )

    st.markdown("---")
    st.markdown("### Required CSV columns")
    for column in REQUIRED_COLUMNS:
        st.code(column, language=None)

    st.caption("CSV upload required. No demo data is loaded.")


# ---------------- CSV upload workflow ----------------
uploaded = st.file_uploader(
    "Upload sprint history (CSV)",
    type=["csv"],
    help="Use the exact column names shown in the sidebar.",
)

if uploaded is None:
    st.info(
        "Upload a CSV file to begin. The dashboard will not show "
        "fabricated or demo results."
    )

    with st.expander("CSV format checklist"):
        st.write(
            "Your CSV must contain one row per sprint and all seven "
            "required columns. Column names are case-sensitive."
        )
        st.code(",".join(REQUIRED_COLUMNS), language="text")
        st.write(
            "Numeric values must be non-negative. Sprint names must be "
            "unique. Planned story points and planned tasks must be "
            "greater than zero."
        )

    st.stop()


source_df, errors, warnings = validate_csv(uploaded)

if errors:
    st.error("The uploaded CSV needs corrections before analysis.")
    for error in errors:
        st.markdown(f"- {error}")

    if source_df is not None:
        with st.expander("Preview uploaded data"):
            st.dataframe(
                source_df.head(30),
                use_container_width=True,
                hide_index=True,
            )
    st.stop()

for warning in warnings:
    st.warning(warning)


# ---------------- Run analytics ----------------
analyzed = analyze_sprints(
    source_df,
    completion_threshold,
    bug_threshold,
    blocked_threshold,
)

with st.sidebar:
    st.markdown("### Display filters")
    all_sprints = analyzed["Sprint"].astype(str).tolist()

    selected_sprints = st.multiselect(
        "Sprints to include",
        all_sprints,
        default=all_sprints,
    )

    risk_filter = st.multiselect(
        "Risk levels",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )

filtered = analyzed[
    analyzed["Sprint"].astype(str).isin(selected_sprints)
    & analyzed["Risk_Level"].isin(risk_filter)
].copy()

if filtered.empty:
    st.warning(
        "No sprints match your filters. Adjust the sprint or risk filters."
    )
    st.stop()

# Preserve original upload order in charts and tables.
order = {
    sprint: index
    for index, sprint in enumerate(all_sprints)
}
filtered["_order"] = filtered["Sprint"].astype(str).map(order)
filtered = (
    filtered.sort_values("_order")
    .drop(columns="_order")
)

avg_completion = filtered["Completion_Rate_%"].mean()
avg_task_completion = filtered["Task_Completion_%"].mean()
avg_velocity = filtered["Completed_Story_Points"].mean()
total_bugs = int(filtered["Bugs"].sum())
high_risk_count = int((filtered["Risk_Level"] == "High").sum())
scope_overruns = int(filtered["Scope_Overrun"].sum())

st.markdown(
    f'<span class="pill">ANALYSIS READY · {len(filtered)} SPRINTS</span>',
    unsafe_allow_html=True,
)

st.markdown("### Performance snapshot")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Story-point completion",
    f"{avg_completion:.1f}%",
    delta=f"{avg_completion - completion_threshold:+.1f} pp vs target",
)
k2.metric("Average velocity", f"{avg_velocity:.1f}")
k3.metric("Task completion", f"{avg_task_completion:.1f}%")
k4.metric("Bugs logged", f"{total_bugs}")
k5.metric(
    "High-risk sprints",
    f"{high_risk_count}",
    delta="Needs review" if high_risk_count else "No high-risk sprint",
    delta_color="inverse",
)


# Project Guide has been removed.
tabs = st.tabs([
    "Overview",
    "Delivery analytics",
    "Risk & quality",
    "Recommendations",
    "Data & export",
])


# ---------------- Overview tab ----------------
with tabs[0]:
    st.markdown("### Sprint performance")
    left, right = st.columns([1.55, 1])

    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Planned",
            x=filtered["Sprint"],
            y=filtered["Planned_Story_Points"],
            marker_color="#65708f",
        ))
        fig.add_trace(go.Bar(
            name="Completed",
            x=filtered["Sprint"],
            y=filtered["Completed_Story_Points"],
            marker_color="#8b7cff",
        ))
        fig.update_layout(
            barmode="group",
            title="Planned vs completed story points",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=55, b=10),
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        risk_counts = (
            filtered["Risk_Level"]
            .value_counts()
            .reindex(["High", "Medium", "Low"], fill_value=0)
            .reset_index()
        )
        risk_counts.columns = ["Risk level", "Sprints"]

        fig = px.pie(
            risk_counts,
            names="Risk level",
            values="Sprints",
            hole=.66,
            title="Risk distribution",
            color="Risk level",
            color_discrete_map={
                "High": "#ff5d7d",
                "Medium": "#ffbd69",
                "Low": "#36d6b5",
            },
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=55, b=10),
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sprint health table")

    display_columns = [
        "Sprint",
        "Completion_Rate_%",
        "Task_Completion_%",
        "Completed_Story_Points",
        "Bugs",
        "Blocked_Tasks",
        "Risk_Score",
        "Risk_Level",
    ]

    st.dataframe(
        filtered[display_columns].rename(columns={
            "Completion_Rate_%": "Completion (%)",
            "Task_Completion_%": "Task completion (%)",
            "Completed_Story_Points": "Velocity (points)",
            "Risk_Score": "Risk score",
            "Risk_Level": "Risk level",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ---------------- Delivery analytics tab ----------------
with tabs[1]:
    st.markdown("### Delivery analytics")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.line(
            filtered,
            x="Sprint",
            y="Completion_Rate_%",
            markers=True,
            title="Story-point completion trend",
            labels={
                "Completion_Rate_%": "Completion (%)",
                "Sprint": "Sprint",
            },
        )
        fig.add_hline(
            y=completion_threshold,
            line_dash="dash",
            line_color="#ffbd69",
            annotation_text="Target",
        )
        fig.update_traces(line_color="#8b7cff", marker_size=9)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            filtered,
            x="Sprint",
            y="Remaining_Story_Points",
            title="Uncompleted planned story points",
            labels={"Remaining_Story_Points": "Remaining points"},
        )
        fig.update_traces(marker_color="#35c7e8")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered["Sprint"],
        y=filtered["Completed_Tasks"],
        name="Completed tasks",
        mode="lines+markers",
        line=dict(color="#36d6b5", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=filtered["Sprint"],
        y=filtered["Planned_Tasks"],
        name="Planned tasks",
        mode="lines+markers",
        line=dict(color="#65708f", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Task delivery trend",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Velocity is the number of completed story points per sprint. "
        "Compare similarly sized sprints for useful trends."
    )


# ---------------- Risk & quality tab ----------------
with tabs[2]:
    st.markdown("### Risk & quality monitor")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Average bugs / sprint",
        f"{filtered['Bugs'].mean():.1f}",
        help=f"Configured threshold: {bug_threshold}",
    )
    c2.metric(
        "Average blocked tasks",
        f"{filtered['Blocked_Tasks'].mean():.1f}",
        help=f"Configured threshold: {blocked_threshold}",
    )
    c3.metric("Scope overruns", scope_overruns)

    fig = px.scatter(
        filtered,
        x="Completion_Rate_%",
        y="Bugs",
        size="Blocked_Tasks",
        color="Risk_Level",
        hover_name="Sprint",
        hover_data=["Risk_Score", "Task_Completion_%"],
        title="Completion vs bugs (bubble size = blocked tasks)",
        color_discrete_map={
            "High": "#ff5d7d",
            "Medium": "#ffbd69",
            "Low": "#36d6b5",
        },
        size_max=34,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)

    risk_table = filtered[[
        "Sprint",
        "Completion_Rate_%",
        "Bugs",
        "Blocked_Tasks",
        "Risk_Score",
        "Risk_Level",
        "Data_Quality",
    ]].rename(columns={
        "Completion_Rate_%": "Completion (%)",
        "Risk_Score": "Risk score",
        "Risk_Level": "Risk level",
        "Data_Quality": "Data check",
    })

    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("How the risk score is calculated"):
        st.write(
            f"- 40 points if completion is below {completion_threshold}%."
        )
        st.write(
            f"- 25 points if bugs meet or exceed {bug_threshold}."
        )
        st.write(
            f"- 25 points if blocked tasks meet or exceed {blocked_threshold}."
        )
        st.write(
            f"- 10 points if task completion is below {completion_threshold}%."
        )
        st.write(
            "High = 60–100, Medium = 30–59, Low = 0–29. "
            "This is a transparent rule-based indicator, not a trained "
            "machine-learning prediction."
        )


# ---------------- Recommendations tab ----------------
with tabs[3]:
    st.markdown("### Recommended next actions")
    st.caption(
        "Recommendations update when filters or risk thresholds change."
    )

    recommendations = make_recommendations(
        filtered,
        completion_threshold,
        bug_threshold,
        blocked_threshold,
    )

    for category, title, detail in recommendations:
        with st.container(border=True):
            st.markdown(
                f'<span class="pill">{category.upper()}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"#### {title}")
            st.write(detail)

    st.markdown("### Trend check")

    if len(filtered) >= 3:
        group_size = max(1, len(filtered) // 3)
        recent = filtered.tail(group_size)
        earlier = filtered.head(group_size)

        delta = (
            recent["Completion_Rate_%"].mean()
            - earlier["Completion_Rate_%"].mean()
        )

        if delta > 3:
            st.success(
                f"Recent completion is up {delta:.1f} percentage points "
                "compared with the earliest selected sprint group."
            )
        elif delta < -3:
            st.warning(
                f"Recent completion is down {abs(delta):.1f} percentage "
                "points compared with the earliest selected sprint group."
            )
        else:
            st.info(
                "Completion appears broadly stable between the earliest "
                "and most recent selected sprint groups."
            )
    else:
        st.info(
            "Select at least three sprints to compare early and recent trends."
        )


# ---------------- Data & export tab ----------------
with tabs[4]:
    st.markdown("### Validated data & export")
    st.write(
        f"**Rows analyzed:** {len(filtered)} · "
        f"**Rows uploaded:** {len(source_df)} · "
        f"**Required columns:** {len(REQUIRED_COLUMNS)}"
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    analyzed_csv = filtered.to_csv(index=False).encode("utf-8")
    clean_csv = source_df.to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            "⬇ Download analyzed results",
            data=analyzed_csv,
            file_name="sprintwise_analyzed_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        st.download_button(
            "⬇ Download validated source data",
            data=clean_csv,
            file_name="sprintwise_validated_source.csv",
            mime="text/csv",
            use_container_width=True,
        )

    report = [
        "SPRINTWISE AI — SPRINT SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Sprints included: {len(filtered)}",
        f"Average story-point completion: {avg_completion:.1f}%",
        f"Average velocity: {avg_velocity:.1f} story points",
        f"Average task completion: {avg_task_completion:.1f}%",
        f"Total bugs: {total_bugs}",
        f"High-risk sprints: {high_risk_count}",
        f"Scope overruns: {scope_overruns}",
        "",
        "Risk uses a threshold-based score, not a trained ML model.",
        "",
        "RECOMMENDATIONS:",
    ]

    report.extend([
        f"- [{category}] {title}: {detail}"
        for category, title, detail in make_recommendations(
            filtered,
            completion_threshold,
            bug_threshold,
            blocked_threshold,
        )
    ])

    st.download_button(
        "⬇ Download text summary report",
        data="\n".join(report),
        file_name="sprintwise_summary_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


st.markdown(
    '<div class="footer">SprintWise AI · Analytics from your uploaded data '
    '· Rule-based risk scoring</div>',
    unsafe_allow_html=True,
)
