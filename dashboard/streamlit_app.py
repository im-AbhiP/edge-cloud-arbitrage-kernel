"""
Executive Dashboard for the Edge-Cloud AI Arbitrage Kernel.

WHY A DASHBOARD?
"Executives remember dashboards, not code."
This single page communicates everything about your system:
cost savings, compute distribution, latency tradeoffs,
governance enforcement, council deliberation behavior,
and detailed inference performance analysis.

WHAT IT SHOWS:
1. Top-line metrics: cost avoided, edge %, cloud spend, total calls
2. Compute distribution pie chart (edge vs cloud)
3. Average latency by model (bar chart)
4. Cumulative cost over time (actual vs full-cloud baseline)
5. Governance events (privacy + budget enforcement counts)
6. Per-model breakdown table
7. INFERENCE PERFORMANCE ANALYSIS (NEW):
   - Tokens per second by model
   - Prefill speed vs decode speed
   - Model load time comparison
   - Throughput over time
   - Performance efficiency metrics
8. Council deliberation stats
9. Raw log tables (expandable)

HOW TO RUN:
    In PyCharm terminal: streamlit run dashboard/streamlit_app.py
    Opens automatically at: http://localhost:8501
    To stop: Ctrl+C in the terminal

PREREQUISITES:
    - pip install streamlit plotly pandas
    - data/logs.db must exist (run test_smoke.py first)

STREAMLIT BASICS:
    Streamlit reruns the ENTIRE script top to bottom every time
    you interact with a widget or the source file changes.
    Think of it as a script that generates a web page,
    not a server that handles requests.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
# set_page_config MUST be the first Streamlit command.
# layout="wide" uses the full browser width.
# page_icon sets the favicon in the browser tab.
# =====================================================================

st.set_page_config(
    page_title="Edge-Cloud AI Arbitrage Kernel",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Edge-Cloud AI Arbitrage Kernel")
st.caption("Hybrid Compute Governance Dashboard — Council of Local LLMs")


# =====================================================================
# DATABASE CONNECTION AND DATA LOADING
# =====================================================================
# We connect to the SQLite database that logging_utils.py writes to.
# Path(__file__).parent.parent navigates from dashboard/ up to the
# project root, then into data/logs.db.
#
# Each load function handles its own connection and error handling.
# This is defensive programming — if one table is missing or corrupt,
# the rest of the dashboard still works.
# =====================================================================

DB_PATH = Path(__file__).parent.parent / "data" / "logs.db"

if not DB_PATH.exists():
    st.error(
        "No log data found at data/logs.db. "
        "Run test_smoke.py first to generate data."
    )
    st.stop()


def load_call_logs():
    """
    Load the call_logs table into a pandas DataFrame.

    WHY A FUNCTION?
    Streamlit reruns the entire script on every interaction.
    Wrapping database reads in functions makes it easy to add
    @st.cache_data later if performance becomes an issue.
    """
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query("SELECT * FROM call_logs", conn)
    except Exception as e:
        st.error(f"Error reading call_logs: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def load_deliberation_logs():
    """
    Load the council_deliberations table into a pandas DataFrame.

    Returns empty DataFrame if the table doesn't exist yet.
    This happens if the user hasn't run the council (only used
    the old rule-based router).
    """
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query(
            "SELECT * FROM council_deliberations", conn
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


# --- Load all data ---
df = load_call_logs()
delib_df = load_deliberation_logs()

if df.empty:
    st.warning(
        "No calls logged yet. Run test_smoke.py or main.py "
        "to generate data, then refresh this page."
    )
    st.stop()

# Convert timestamp strings to proper datetime objects
# This is needed for time-series charts to work correctly
# FIX: utc=True converts ALL timestamps to UTC timezone-aware.
# This handles the mix of old entries (no timezone from datetime.utcnow)
# and new entries (+00:00 from datetime.now(UTC)).
# Without utc=True, pandas creates a column with MIXED types
# (some tz-naive, some tz-aware) which cannot be sorted or compared.
df["timestamp"] = pd.to_datetime(
    df["timestamp"], format="ISO8601", utc=True
)
if not delib_df.empty and "timestamp" in delib_df.columns:
    delib_df["timestamp"] = pd.to_datetime(
        delib_df["timestamp"], format="ISO8601", utc=True
    )


# --- Compute derived performance columns ---
# These columns are calculated from raw data and used across
# multiple dashboard sections. Computing them once here avoids
# recalculating in every chart.

# Overall tokens per second: how many tokens produced per second
# This is total_tokens / (latency_ms / 1000)
# We use .clip(lower=1) to avoid division by zero
df["tokens_per_sec"] = (
    df["total_tokens"]
    / (df["latency_ms"].clip(lower=1) / 1000)
).round(1)

# Decode tokens per second: completion tokens generated per second
# This is a rough proxy for "generation speed"
df["decode_tokens_per_sec"] = (
    df["completion_tokens"]
    / (df["latency_ms"].clip(lower=1) / 1000)
).round(1)

# If we have Ollama's detailed timing data, compute precise speeds
# These columns may not exist in older databases
has_detailed_timing = (
    "prefill_ms" in df.columns
    and "decode_ms" in df.columns
    and df["prefill_ms"].notna().any()
)

if has_detailed_timing:
    # True prefill speed: prompt tokens / prefill time
    df["prefill_tok_per_sec"] = (
        df["prompt_tokens"]
        / (df["prefill_ms"].clip(lower=1) / 1000)
    ).round(1)

    # True decode speed: completion tokens / decode time
    df["true_decode_tok_per_sec"] = (
        df["completion_tokens"]
        / (df["decode_ms"].clip(lower=1) / 1000)
    ).round(1)


# =====================================================================
# SECTION 1: TOP-LINE METRICS
# =====================================================================
# These four numbers are the FIRST thing anyone sees.
# They answer the two most important questions:
#   "How much money did this system save?"
#   "How is compute distributed?"
#
# st.columns(4) creates 4 equal-width columns side by side.
# .metric() renders a big number with a label — ideal for KPIs.
# The 'help' parameter adds a tooltip (hover) explanation.
# =====================================================================

st.header("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

# Calculate top-line metrics
total_cost = df["estimated_cost_usd"].sum()
total_alt_cost = df["cloud_alternative_cost_usd"].sum()
cost_avoided = total_alt_cost - total_cost
edge_pct = (df["tier"] == "edge").mean() * 100

col1.metric(
    label="Cost Avoided",
    value=f"${cost_avoided:.4f}",
    help="Dollars saved compared to sending every call to cloud",
)
col2.metric(
    label="Edge Processing",
    value=f"{edge_pct:.0f}%",
    help="Percentage of calls handled by local models at $0 cost",
)
col3.metric(
    label="Cloud Spend",
    value=f"${total_cost:.4f}",
    help="Total money spent on cloud API calls",
)
col4.metric(
    label="Total Calls",
    value=f"{len(df)}",
    help="Total number of model inference calls logged",
)

st.divider()


# =====================================================================
# SECTION 2: COMPUTE DISTRIBUTION + LATENCY COMPARISON
# =====================================================================
# Two charts side by side.
# Left: Pie chart showing edge vs cloud call split.
# Right: Bar chart showing average latency per model.
#
# WHY THESE TWO CHARTS TOGETHER?
# The pie chart tells the "cost story" (most work is local = free).
# The bar chart tells the "speed story" (latency tradeoffs).
# Together they show awareness of both dimensions.
# =====================================================================

st.header("Compute Overview")

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Compute Distribution")

    # value_counts() counts how many rows have each unique value
    # reset_index() converts the Series to a DataFrame for Plotly
    tier_counts = df["tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]

    fig_pie = px.pie(
        tier_counts,
        values="Count",
        names="Tier",
        color_discrete_map={
            "edge": "#2ecc71",   # Green = local = free
            "cloud": "#3498db",  # Blue = cloud = costs money
        },
        hole=0.4,  # Makes it a donut chart (more modern look)
    )
    fig_pie.update_traces(
        textinfo="label+percent",
        textfont_size=14,
    )
    st.plotly_chart(fig_pie, width="stretch")

with chart_right:
    st.subheader("Average Latency by Model")

    # Group by model, calculate mean latency, sort ascending
    latency_by_model = (
        df.groupby("model_name")["latency_ms"]
        .mean()
        .reset_index()
        .sort_values("latency_ms", ascending=True)
    )

    fig_latency = px.bar(
        latency_by_model,
        x="model_name",
        y="latency_ms",
        color="model_name",
        labels={
            "model_name": "Model",
            "latency_ms": "Avg Latency (ms)",
        },
    )
    fig_latency.update_layout(showlegend=False)
    st.plotly_chart(fig_latency, width="stretch")

st.divider()


# =====================================================================
# SECTION 3: CUMULATIVE COST OVER TIME
# =====================================================================
# This is the "money chart" — the single most impactful visual.
#
# It shows two lines:
#   Green solid line: Your actual cost (hybrid edge+cloud routing)
#   Red dashed line:  What it WOULD have cost if everything went cloud
#
# The gap between the lines is your savings.
# Over time, this gap widens — which IS the entire ROI story.
#
# WHY CUMULATIVE instead of per-call?
# Individual call costs are tiny fractions of a cent ($0.000075).
# They don't look impressive as a chart. Cumulative cost shows
# the trend and total impact — which is what executives care about.
# =====================================================================

st.header("Cost Over Time")

df_sorted = df.sort_values("timestamp").copy()
df_sorted["cumulative_cost"] = df_sorted["estimated_cost_usd"].cumsum()
df_sorted["cumulative_alt_cost"] = (
    df_sorted["cloud_alternative_cost_usd"].cumsum()
)

fig_cost = go.Figure()

# Actual cost line (green, solid, with fill)
fig_cost.add_trace(
    go.Scatter(
        x=df_sorted["timestamp"],
        y=df_sorted["cumulative_cost"],
        name="Actual Cost (Hybrid)",
        line=dict(color="#2ecc71", width=3),
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.1)",
    )
)

# Full-cloud baseline line (red, dashed)
fig_cost.add_trace(
    go.Scatter(
        x=df_sorted["timestamp"],
        y=df_sorted["cumulative_alt_cost"],
        name="Full Cloud Baseline",
        line=dict(color="#e74c3c", width=2, dash="dash"),
    )
)

fig_cost.update_layout(
    yaxis_title="Cumulative Cost (USD)",
    xaxis_title="Time",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    height=400,
)

st.plotly_chart(fig_cost, width="stretch")

st.divider()


# =====================================================================
# SECTION 4: GOVERNANCE EVENTS
# =====================================================================
# Counts of how many times governance policies were enforced:
# - Privacy: sensitive data forced to local
# - Budget: cloud spend exceeded cap, forced to local
#
# WHY THIS MATTERS:
# In enterprise settings, the NUMBER of enforcement events is
# a compliance metric. It shows the governance layer is active
# and doing its job — not just theoretical.
# =====================================================================

st.header("Governance Events")

gov_left, gov_right = st.columns(2)

privacy_count = int(df["privacy_enforced"].sum())
budget_count = int(df["budget_enforced"].sum())

with gov_left:
    st.metric(
        label="Privacy Enforcements",
        value=privacy_count,
        help="Times HIGH-sensitivity data was forced to edge",
    )
    if privacy_count > 0:
        st.success(
            f"{privacy_count} sensitive task(s) kept local. "
            f"No private data sent to cloud."
        )
    else:
        st.info("No privacy enforcement events yet.")

with gov_right:
    st.metric(
        label="Budget Enforcements",
        value=budget_count,
        help="Times cloud spend was capped by budget policy",
    )
    if budget_count > 0:
        st.success(
            f"{budget_count} call(s) downgraded or blocked "
            f"due to budget limits."
        )
    else:
        st.info("No budget enforcement events yet.")

st.divider()


# =====================================================================
# SECTION 5: PER-MODEL BREAKDOWN TABLE
# =====================================================================
# A summary table showing each model's aggregate stats.
# Answers: "Which models are used, how fast, how expensive?"
# =====================================================================

st.header("Per-Model Breakdown")

if not df.empty:
    model_stats = (
        df.groupby(["model_name", "tier"])
        .agg(
            calls=("model_name", "count"),
            avg_latency_ms=("latency_ms", "mean"),
            total_tokens=("total_tokens", "sum"),
            total_cost=("estimated_cost_usd", "sum"),
            success_rate=("success", "mean"),
            avg_tok_per_sec=("tokens_per_sec", "mean"),
        )
        .reset_index()
    )

    model_stats["avg_latency_ms"] = model_stats["avg_latency_ms"].round(0)
    model_stats["total_cost"] = model_stats["total_cost"].round(6)
    model_stats["success_rate"] = (
        model_stats["success_rate"] * 100
    ).round(1)
    model_stats["avg_tok_per_sec"] = model_stats["avg_tok_per_sec"].round(1)

    st.dataframe(
        model_stats.rename(columns={
            "model_name": "Model",
            "tier": "Tier",
            "calls": "Calls",
            "avg_latency_ms": "Avg Latency (ms)",
            "total_tokens": "Total Tokens",
            "total_cost": "Total Cost ($)",
            "success_rate": "Success Rate (%)",
            "avg_tok_per_sec": "Avg Tok/s",
        }),
        width="stretch",
        hide_index=True,
    )

st.divider()


# =====================================================================
# SECTION 6: INFERENCE PERFORMANCE ANALYSIS
# =====================================================================
# THIS IS THE NEW SECTION — the "engineering depth" showcase.
#
# WHY INFERENCE PERFORMANCE METRICS MATTER:
# In the AI infrastructure world (especially at AMD, NVIDIA, etc.),
# inference performance is measured in specific ways:
#
# 1. TOKENS PER SECOND (tok/s):
#    How many tokens the model processes per second.
#    Higher = faster. This is the headline number.
#
# 2. PREFILL SPEED (prompt tok/s):
#    How fast the model processes INPUT tokens.
#    This is the "reading" phase — heavily parallelizable.
#    Limited by memory BANDWIDTH (how fast you can read weights).
#    Apple M1 Pro: ~200 GB/s unified memory bandwidth.
#
# 3. DECODE SPEED (generation tok/s):
#    How fast the model generates OUTPUT tokens one by one.
#    This is the "writing" phase — sequential (autoregressive).
#    Limited by memory LATENCY and compute throughput.
#    Typically 5-50x slower than prefill per token.
#
# 4. TIME TO FIRST TOKEN (TTFT):
#    How long until the first output token appears.
#    This is approximately: model_load_time + prefill_time.
#    Users perceive TTFT as "responsiveness."
#
# WHY THIS MATTERS AT AMD:
# AMD's MI300X has 192GB HBM3 at 5.3 TB/s bandwidth.
# The prefill vs decode tradeoff is exactly what their hardware
# team optimizes for. Showing you understand this in your dashboard
# is a VERY strong signal for AI infrastructure roles.
# =====================================================================

st.header("Inference Performance Analysis")

# --- 6.1: Overall Tokens Per Second by Model ---
# This is the "headline" performance metric.
# Higher tok/s = faster model = better user experience.

st.subheader("Tokens Per Second by Model")

# Filter to successful calls with tokens > 0
perf_df = df[
    (df["success"] == 1)
    & (df["total_tokens"] > 0)
    & (df["latency_ms"] > 0)
].copy()

if not perf_df.empty:
    # Calculate per-model average tokens/sec
    tok_per_sec_by_model = (
        perf_df.groupby("model_name")["tokens_per_sec"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    tok_per_sec_by_model.columns = [
        "Model", "Mean tok/s", "Median tok/s",
        "Min tok/s", "Max tok/s", "Calls"
    ]

    # Bar chart of mean tokens/sec
    fig_tps = px.bar(
        tok_per_sec_by_model,
        x="Model",
        y="Mean tok/s",
        color="Model",
        title="Average Tokens Per Second by Model (higher = faster)",
        text="Mean tok/s",
    )
    fig_tps.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_tps.update_layout(showlegend=False, yaxis_title="Tokens/sec")
    st.plotly_chart(fig_tps, width="stretch")

    # Show the detailed stats table
    st.dataframe(
        tok_per_sec_by_model.round(1),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No performance data available yet.")

st.divider()

# --- 6.2: Prefill vs Decode Speed ---
# This section ONLY shows if detailed Ollama timing is available.
# If not, we show the approximate version calculated from overall timing.

st.subheader("Prefill vs Decode Speed")

if has_detailed_timing:
    # We have actual Ollama prefill/decode timing!
    st.caption(
        "Measured by Ollama's internal timer. "
        "Prefill = processing input tokens (parallel, bandwidth-bound). "
        "Decode = generating output tokens (sequential, latency-bound)."
    )

    timing_df = perf_df[
        perf_df["prefill_ms"].notna()
        & perf_df["decode_ms"].notna()
        & (perf_df["prefill_ms"] > 0)
        & (perf_df["decode_ms"] > 0)
    ].copy()

    if not timing_df.empty:
        # Calculate per-model prefill and decode speeds
        prefill_decode_stats = (
            timing_df.groupby("model_name")
            .agg(
                avg_prefill_tok_s=("prefill_tok_per_sec", "mean"),
                avg_decode_tok_s=("true_decode_tok_per_sec", "mean"),
                avg_prefill_ms=("prefill_ms", "mean"),
                avg_decode_ms=("decode_ms", "mean"),
            )
            .reset_index()
        )

        # Grouped bar chart: prefill speed vs decode speed
        fig_pd = go.Figure()

        fig_pd.add_trace(go.Bar(
            name="Prefill (tok/s)",
            x=prefill_decode_stats["model_name"],
            y=prefill_decode_stats["avg_prefill_tok_s"],
            marker_color="#3498db",
            text=prefill_decode_stats["avg_prefill_tok_s"].round(0),
            textposition="outside",
        ))

        fig_pd.add_trace(go.Bar(
            name="Decode (tok/s)",
            x=prefill_decode_stats["model_name"],
            y=prefill_decode_stats["avg_decode_tok_s"],
            marker_color="#e74c3c",
            text=prefill_decode_stats["avg_decode_tok_s"].round(0),
            textposition="outside",
        ))

        fig_pd.update_layout(
            barmode="group",
            title="Prefill vs Decode Speed by Model",
            yaxis_title="Tokens/sec",
            xaxis_title="Model",
        )
        st.plotly_chart(fig_pd, width="stretch")

        # Time breakdown chart: how much time is prefill vs decode
        st.subheader("Time Breakdown: Prefill vs Decode vs Other")

        for _, row in prefill_decode_stats.iterrows():
            model = row["model_name"]
            p_ms = row["avg_prefill_ms"]
            d_ms = row["avg_decode_ms"]

            # Get average total latency for this model
            avg_total = perf_df[
                perf_df["model_name"] == model
            ]["latency_ms"].mean()

            other_ms = max(avg_total - p_ms - d_ms, 0)

            # Show as a horizontal stacked bar
            fig_breakdown = go.Figure()
            fig_breakdown.add_trace(go.Bar(
                y=[model], x=[p_ms], name="Prefill",
                orientation="h", marker_color="#3498db",
            ))
            fig_breakdown.add_trace(go.Bar(
                y=[model], x=[d_ms], name="Decode",
                orientation="h", marker_color="#e74c3c",
            ))
            fig_breakdown.add_trace(go.Bar(
                y=[model], x=[other_ms], name="Overhead",
                orientation="h", marker_color="#95a5a6",
            ))
            fig_breakdown.update_layout(
                barmode="stack",
                height=120,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Time (ms)",
                showlegend=True,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_breakdown, width="stretch")

    else:
        st.info("No calls with detailed timing data yet.")

else:
    # No detailed timing — show approximate decode speed
    st.caption(
        "Approximate speeds calculated from overall latency. "
        "For precise prefill/decode breakdown, update runtime/models.py "
        "to capture Ollama's internal timing fields."
    )

    if not perf_df.empty:
        approx_stats = (
            perf_df.groupby("model_name")
            .agg(
                avg_overall_tok_s=("tokens_per_sec", "mean"),
                avg_decode_tok_s=("decode_tokens_per_sec", "mean"),
            )
            .reset_index()
            .sort_values("avg_overall_tok_s", ascending=False)
        )

        fig_approx = go.Figure()
        fig_approx.add_trace(go.Bar(
            name="Overall (tok/s)",
            x=approx_stats["model_name"],
            y=approx_stats["avg_overall_tok_s"],
            marker_color="#2ecc71",
        ))
        fig_approx.add_trace(go.Bar(
            name="Generation (tok/s)",
            x=approx_stats["model_name"],
            y=approx_stats["avg_decode_tok_s"],
            marker_color="#e74c3c",
        ))
        fig_approx.update_layout(
            barmode="group",
            title="Approximate Inference Speed by Model",
            yaxis_title="Tokens/sec",
        )
        st.plotly_chart(fig_approx, width="stretch")

st.divider()

# --- 6.3: Model Load Time (if available) ---
# Shows how long it takes Ollama to load each model into RAM.
# This matters because on 16GB M1 Pro, model swapping is frequent
# during council deliberations.

if has_detailed_timing and "load_ms" in df.columns:
    load_df = df[df["load_ms"].notna() & (df["load_ms"] > 100)].copy()

    if not load_df.empty:
        st.subheader("Model Load Time")
        st.caption(
            "Time to load model into RAM. Higher for larger models. "
            "On 16GB M1 Pro, frequent model swapping during council "
            "deliberations increases this overhead."
        )

        load_stats = (
            load_df.groupby("model_name")["load_ms"]
            .agg(["mean", "max", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        load_stats.columns = [
            "Model", "Avg Load (ms)", "Max Load (ms)", "Loads"
        ]

        fig_load = px.bar(
            load_stats,
            x="Model",
            y="Avg Load (ms)",
            color="Model",
            title="Average Model Load Time (RAM swap overhead)",
            text="Avg Load (ms)",
        )
        fig_load.update_traces(
            texttemplate="%{text:.0f}ms",
            textposition="outside",
        )
        fig_load.update_layout(showlegend=False)
        st.plotly_chart(fig_load, width="stretch")

st.divider()

# --- 6.4: Throughput Over Time ---
# Shows how tokens/sec changes over time. Useful for spotting:
# - Performance degradation under load
# - The effect of model swapping (spikes when model is already loaded)
# - Thermal throttling on long sessions

st.subheader("Throughput Over Time")

if not perf_df.empty:
    # Only show edge models (cloud latency includes network, not comparable)
    edge_perf = perf_df[perf_df["tier"] == "edge"].copy()

    if not edge_perf.empty:
        fig_throughput = px.scatter(
            edge_perf.sort_values("timestamp"),
            x="timestamp",
            y="tokens_per_sec",
            color="model_name",
            title="Edge Model Throughput Over Time (tok/s)",
            labels={
                "tokens_per_sec": "Tokens/sec",
                "timestamp": "Time",
                "model_name": "Model",
            },
            hover_data=["total_tokens", "latency_ms"],
        )
        fig_throughput.update_layout(height=400)
        st.plotly_chart(fig_throughput, width="stretch")
    else:
        st.info("No edge model performance data available.")

st.divider()

# --- 6.5: Performance Efficiency Summary ---
# Key performance numbers in a clean metric row.

st.subheader("Performance Summary")

if not perf_df.empty:
    edge_data = perf_df[perf_df["tier"] == "edge"]
    cloud_data = perf_df[perf_df["tier"] == "cloud"]

    p1, p2, p3, p4 = st.columns(4)

    if not edge_data.empty:
        p1.metric(
            label="Edge Avg tok/s",
            value=f"{edge_data['tokens_per_sec'].mean():.0f}",
            help="Average tokens per second for local models",
        )
        p2.metric(
            label="Edge Median Latency",
            value=f"{edge_data['latency_ms'].median():.0f}ms",
            help="Median end-to-end latency for local calls",
        )
    else:
        p1.metric(label="Edge Avg tok/s", value="N/A")
        p2.metric(label="Edge Median Latency", value="N/A")

    if not cloud_data.empty:
        p3.metric(
            label="Cloud Avg tok/s",
            value=f"{cloud_data['tokens_per_sec'].mean():.0f}",
            help="Average tokens per second for cloud models",
        )
        p4.metric(
            label="Cloud Median Latency",
            value=f"{cloud_data['latency_ms'].median():.0f}ms",
            help="Median end-to-end latency for cloud calls",
        )
    else:
        p3.metric(label="Cloud Avg tok/s", value="N/A")
        p4.metric(label="Cloud Median Latency", value="N/A")

    # Total tokens processed
    total_tokens = df["total_tokens"].sum()
    total_prompt_tokens = df["prompt_tokens"].sum()
    total_completion_tokens = df["completion_tokens"].sum()

    t1, t2, t3 = st.columns(3)
    t1.metric(
        label="Total Tokens Processed",
        value=f"{total_tokens:,}",
    )
    t2.metric(
        label="Total Prompt Tokens",
        value=f"{total_prompt_tokens:,}",
        help="Input tokens (what you sent to models)",
    )
    t3.metric(
        label="Total Completion Tokens",
        value=f"{total_completion_tokens:,}",
        help="Output tokens (what models generated)",
    )

st.divider()


# =====================================================================
# SECTION 7: COUNCIL DELIBERATION STATS
# =====================================================================
# Shows how the council of local LLMs has been deliberating:
# vote breakdowns, model popularity, consensus rates.
#
# This section only appears if deliberation data exists.
# =====================================================================

st.header("Council Deliberation Stats")

if delib_df.empty:
    st.info(
        "No council deliberations logged yet. "
        "Run main.py or test_smoke.py to see stats here."
    )
else:
    # --- Top metrics ---
    total_deliberations = delib_df["deliberation_id"].nunique()

    delib_col1, delib_col2, delib_col3 = st.columns(3)
    delib_col1.metric(label="Total Deliberations", value=total_deliberations)

    # --- Routing vote breakdown ---
    routing_votes = delib_df[delib_df["phase"] == "routing_vote"].copy()

    if not routing_votes.empty:
        local_votes = routing_votes[
            routing_votes["vote_or_action"].isin(["local", "edge"])
        ].shape[0]
        cloud_votes = routing_votes[
            routing_votes["vote_or_action"] == "cloud"
        ].shape[0]

        delib_col2.metric(label="Local Votes (total)", value=int(local_votes))
        delib_col3.metric(label="Cloud Votes (total)", value=int(cloud_votes))

        # Routing vote pie chart
        st.subheader("Routing Vote Distribution")
        vote_counts = pd.DataFrame({
            "Vote": ["Local", "Cloud"],
            "Count": [local_votes, cloud_votes],
        })
        if local_votes + cloud_votes > 0:
            fig_votes = px.pie(
                vote_counts,
                values="Count",
                names="Vote",
                color_discrete_map={
                    "Local": "#2ecc71",
                    "Cloud": "#3498db",
                },
                hole=0.4,
            )
            st.plotly_chart(fig_votes, width="stretch")

    # --- Model selection popularity ---
    selection_votes = delib_df[
        delib_df["phase"] == "model_selection"
    ].copy()

    if not selection_votes.empty:
        st.subheader("Most Voted-For Local Models")
        model_vote_counts = (
            selection_votes["vote_or_action"]
            .value_counts()
            .reset_index()
        )
        model_vote_counts.columns = ["Model", "Times Voted For"]

        fig_model_pop = px.bar(
            model_vote_counts,
            x="Model",
            y="Times Voted For",
            color="Model",
            title="Council Model Selection Votes",
        )
        fig_model_pop.update_layout(showlegend=False)
        st.plotly_chart(fig_model_pop, width="stretch")

    # --- Consensus stats ---
    final_decisions = delib_df[
        delib_df["phase"] == "final_decision"
    ].copy()

    if not final_decisions.empty:
        st.subheader("Consensus Performance")

        cons_left, cons_right = st.columns(2)

        avg_iterations = final_decisions["iteration"].mean()
        cons_left.metric(
            label="Avg Iterations to Complete",
            value=f"{avg_iterations:.1f}",
            help="Average review rounds before acceptance.",
        )

        if "approved" in final_decisions.columns:
            consensus_count = final_decisions["approved"].sum()
            total_final = len(final_decisions)
            consensus_rate = (
                (consensus_count / total_final * 100)
                if total_final > 0 else 0
            )
            cons_right.metric(
                label="Consensus Rate",
                value=f"{consensus_rate:.0f}%",
            )

        local_decisions = final_decisions[
            final_decisions["vote_or_action"] == "local"
        ].shape[0]
        cloud_decisions = final_decisions[
            final_decisions["vote_or_action"] == "cloud"
        ].shape[0]
        st.write(
            f"**Final routing:** {local_decisions} local, "
            f"{cloud_decisions} cloud "
            f"(out of {len(final_decisions)} total)"
        )

    # --- Review quality by iteration ---
    review_data = delib_df[
        delib_df["phase"] == "answer_review"
    ].copy()

    if not review_data.empty and "confidence" in review_data.columns:
        st.subheader("Review Quality by Iteration")

        avg_by_iter = (
            review_data.groupby("iteration")["confidence"]
            .mean()
            .reset_index()
        )
        avg_by_iter.columns = ["Iteration", "Avg Quality Score"]

        fig_quality = px.bar(
            avg_by_iter,
            x="Iteration",
            y="Avg Quality Score",
            color="Avg Quality Score",
            color_continuous_scale="greens",
        )
        fig_quality.update_layout(
            yaxis_range=[0, 1],
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_quality, width="stretch")

        if len(avg_by_iter) > 1:
            first_score = avg_by_iter.iloc[0]["Avg Quality Score"]
            last_score = avg_by_iter.iloc[-1]["Avg Quality Score"]
            if last_score > first_score:
                st.success(
                    f"Quality improved by {last_score - first_score:.2f} "
                    f"across iterations."
                )

    st.divider()

    # --- Full deliberation log ---
    with st.expander("View Full Deliberation Logs"):
        display_columns = [
            col for col in [
                "deliberation_id", "timestamp", "phase",
                "model_name", "vote_or_action", "reasoning",
                "confidence", "iteration", "approved", "feedback",
            ]
            if col in delib_df.columns
        ]
        st.dataframe(
            delib_df[display_columns],
            width="stretch",
            hide_index=True,
        )


# =====================================================================
# SECTION 8: RAW CALL LOGS (Expandable)
# =====================================================================
# The full call_logs table in an expandable section.
# Includes the new timing columns if available.
# =====================================================================

with st.expander("View Raw Call Logs"):
    raw_cols = [
        col for col in [
            "timestamp", "model_name", "tier", "task_type",
            "latency_ms", "prompt_tokens", "completion_tokens",
            "estimated_cost_usd", "cloud_alternative_cost_usd",
            "privacy_enforced", "budget_enforced", "success",
            "prefill_ms", "decode_ms", "load_ms", "error",
        ]
        if col in df.columns
    ]
    st.dataframe(
        df[raw_cols],
        width="stretch",
        hide_index=True,
    )


# =====================================================================
# FOOTER
# =====================================================================

st.divider()
st.caption(
    "Edge-Cloud AI Arbitrage Kernel | "
    "Council of Local LLMs | "
    "Built with Streamlit + Plotly + SQLite"
)
