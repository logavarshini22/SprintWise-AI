import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="SprintWise AI", page_icon="📊", layout="wide")

st.title("📊 SprintWise AI")
st.subheader("AI-Driven Sprint Progress Analytics")
st.write("Upload sprint data to analyze progress, velocity, completion, and sprint risk.")

# Sample dataset
sample = pd.DataFrame({
    "Sprint": ["Sprint 1","Sprint 2","Sprint 3","Sprint 4","Sprint 5","Sprint 6","Sprint 7","Sprint 8"],
    "Planned_Story_Points": [40,45,50,48,55,60,58,62],
    "Completed_Story_Points": [36,39,42,44,46,48,52,50],
    "Planned_Tasks": [20,22,25,24,28,30,29,31],
    "Completed_Tasks": [18,19,21,22,23,24,25,24],
    "Bugs": [2,3,4,3,5,7,6,8],
    "Blocked_Tasks": [1,2,2,1,3,4,3,5]
})

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = sample

required = ["Sprint","Planned_Story_Points","Completed_Story_Points",
            "Planned_Tasks","Completed_Tasks","Bugs","Blocked_Tasks"]

missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.info("Required columns: " + ", ".join(required))
    st.stop()

df["Completion_%"] = (df["Completed_Story_Points"] / df["Planned_Story_Points"] * 100).clip(0,100)
df["Task_Completion_%"] = (df["Completed_Tasks"] / df["Planned_Tasks"] * 100).clip(0,100)
df["Velocity"] = df["Completed_Story_Points"]
df["Risk_Score"] = (
    (100 - df["Completion_%"]) * 0.45
    + df["Bugs"] * 4
    + df["Blocked_Tasks"] * 6
).clip(0,100)

def risk_label(score):
    if score >= 60:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"

df["Risk"] = df["Risk_Score"].apply(risk_label)

# Train a small explainable ML model from the available sprint history.
# The model learns a risk class from completion, bugs and blocked tasks.
X = df[["Completion_%","Bugs","Blocked_Tasks"]]
y = df["Risk"]
if len(df) >= 6 and y.nunique() >= 2:
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
    model.fit(X, y)
    df["AI_Risk"] = model.predict(X)
else:
    df["AI_Risk"] = df["Risk"]

latest = df.iloc[-1]
avg_completion = df["Completion_%"].mean()
avg_velocity = df["Velocity"].mean()
latest_risk = latest["AI_Risk"]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Latest Completion", f"{latest['Completion_%']:.1f}%")
c2.metric("Average Velocity", f"{avg_velocity:.1f} SP")
c3.metric("Latest Bugs", int(latest["Bugs"]))
c4.metric("AI Risk", str(latest_risk))

st.divider()

left,right = st.columns(2)
with left:
    st.subheader("📈 Sprint Velocity")
    st.line_chart(df.set_index("Sprint")["Velocity"])
with right:
    st.subheader("✅ Completion Trend")
    st.line_chart(df.set_index("Sprint")["Completion_%"])

st.subheader("🚦 Sprint Risk Analysis")
display_cols = ["Sprint","Completion_%","Velocity","Bugs","Blocked_Tasks","Risk_Score","AI_Risk"]
st.dataframe(df[display_cols].round(1), use_container_width=True)

# AI-style recommendation engine
st.subheader("🤖 AI Recommendations")

recs = []
if latest["Completion_%"] < 80:
    recs.append("Completion is below 80%: review unfinished stories and consider reducing sprint scope.")
if latest["Bugs"] >= 5:
    recs.append("Bug count is elevated: allocate time for defect fixing and regression testing.")
if latest["Blocked_Tasks"] >= 3:
    recs.append("Several tasks are blocked: identify dependencies and assign owners for resolution.")
if latest["Velocity"] < avg_velocity:
    recs.append("Current velocity is below the historical average: investigate workload or dependency issues.")
if not recs:
    recs.append("Current sprint indicators are stable. Continue monitoring velocity, bugs and blockers.")

for r in recs:
    st.info("• " + r)

st.subheader("📋 Data")
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download analyzed CSV", csv, "sprintwise_analysis.csv", "text/csv")
