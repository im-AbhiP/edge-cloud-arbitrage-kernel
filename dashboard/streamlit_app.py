"""
Executive dashboard.

"Executives remember dashboards, not code."

Run with: streamlit run dashboard/streamlit_app.py
(Use PyCharm's terminal for this)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from pathlib import Path

st.set_page_config(
    page_title="Edge-Cloud AI Arbitrage Kernel", layout="wide"
)
st.title("🧠 Edge-Cloud AI Arbitrage Kernel")
st.caption("Hybrid Compute Governance Dashboard")

DB_PATH = Path(__file__).parent.parent / "data" / "logs.db"

if not DB_PATH.exists():
    st.error("No log data found. Run some tasks first!")
    st.stop()

conn = sqlite3.connect(str(DB_PATH))
df = pd.read_sql_query("SELECT * FROM call_logs", conn)
conn.close()

if df.empty:
    st.warning("No calls logged yet.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- TOP METRICS ---
col1, col2, col3, col4 = st.columns(4)

total_cost = df["estimated_cost_usd"].sum()
total_alt_cost = df["cloud_alternative_cost_usd"].sum()
cost_avoided = total_alt_cost - total_cost
edge_pct = (df["tier"] == "edge").mean() * 100

col1.metric("💰 Cost Avoided", f"${cost_avoided:.4f}")
col2.metric("🏠 Edge Processing", f"{edge_pct:.0f}%")
col3.metric("☁️ Cloud Spend", f"${total_cost:.4f}")
col4.metric("📊 Total Calls", len(df))

st.divider()

# --- CHARTS ---
left, right = st.columns(2)

with left:
    st.subheader("Compute Distribution")
    tier_counts = df["tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]
    fig = px.pie(tier_counts, values="Count", names="Tier",
                 color_discrete_map={"edge": "#2ecc71", "cloud": "#3498db"})
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Avg Latency by Model")
    latency = df.groupby("model_name")["latency_ms"].mean().reset_index()
    fig = px.bar(latency, x="model_name", y="latency_ms",
                 color="model_name",
                 labels={"latency_ms": "Avg Latency (ms)"})
    st.plotly_chart(fig, use_container_width=True)

# --- CUMULATIVE COST ---
st.subheader("Cumulative Cost Over Time")
df_sorted = df.sort_values("timestamp")
df_sorted["cum_cost"] = df_sorted["estimated_cost_usd"].cumsum()
df_sorted["cum_alt"] = df_sorted["cloud_alternative_cost_usd"].cumsum()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_sorted["timestamp"], y=df_sorted["cum_cost"],
    name="Actual (Hybrid)", line=dict(color="#2ecc71")))
fig.add_trace(go.Scatter(
    x=df_sorted["timestamp"], y=df_sorted["cum_alt"],
    name="Full Cloud Baseline", line=dict(color="#e74c3c", dash="dash")))
fig.update_layout(yaxis_title="Cumulative Cost (USD)")
st.plotly_chart(fig, use_container_width=True)

# --- GOVERNANCE ---
st.subheader("Governance Events")
st.write(f"🔒 Privacy enforcements: **{df['privacy_enforced'].sum()}**")
st.write(f"💸 Budget enforcements: **{df['budget_enforced'].sum()}**")

with st.expander("View Raw Logs"):
    st.dataframe(df[["timestamp", "model_name", "tier", "task_type",
                      "latency_ms", "estimated_cost_usd",
                      "privacy_enforced", "budget_enforced", "success"]])
