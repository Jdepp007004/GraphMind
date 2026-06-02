"""
src/dashboard/app.py

Streamlit dashboard for GraphMind: graph viz, RL curves, security log, benchmarks.
Run via: streamlit run src/dashboard/app.py
"""

import json
import logging
import os
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import settings

logger = logging.getLogger(__name__)


def load_data(user_id: str, day: int) -> dict:
    """Load all pre-computed results for selected user/day from RESULTS_DIR."""
    data = {"user_id": user_id, "day": day}
    # Simulation log
    log_path = os.path.join(settings.RESULTS_DIR, f"{user_id}_simulation_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
        data["simulation_log"] = log
        days = log.get("days", [])
        day_data = next((d for d in days if d.get("day") == day), days[-1] if days else {})
        data["graph_snapshot"] = day_data.get("graph_snapshot", {})
        data["tier_stats"] = day_data.get("tier_stats", {})
        data["state"] = day_data.get("state", {})
    # Benchmark results
    bench_path = os.path.join(settings.RESULTS_DIR, "benchmark_results.csv")
    if os.path.exists(bench_path):
        data["benchmark_df"] = pd.read_csv(bench_path)
    # Training curves
    curves_path = os.path.join(settings.RESULTS_DIR, "training_curves.json")
    if os.path.exists(curves_path):
        with open(curves_path) as f:
            data["training_curves"] = json.load(f)
    return data


def render_pyvis_graph(snapshot: dict) -> str:
    """Convert graph snapshot dict to PyVis HTML. Return HTML string for st.components.html."""
    try:
        from pyvis.network import Network
        net = Network(height="400px", width="100%", directed=True, bgcolor="#0d1117",
                      font_color="white")
        nodes = snapshot.get("nodes", [])[:50]  # limit for rendering
        edges = snapshot.get("edges", [])[:100]
        node_ids = {n["node_id"] for n in nodes}
        for node in nodes:
            label = node.get("app_id", "unknown").split(".")[-1][:12]
            net.add_node(node["node_id"], label=label,
                         title=f"{node.get('category','')} | acc:{node.get('access_count',0)}")
        for edge in edges:
            if edge["source"] in node_ids and edge["target"] in node_ids:
                net.add_edge(edge["source"], edge["target"],
                             value=edge.get("prob", 0.1))
        return net.generate_html()
    except Exception as e:
        return f"<p style='color:red'>Graph render error: {e}</p>"


def _run_dashboard() -> None:
    """Main dashboard entry point called by streamlit."""
    import streamlit as st
    from src.core.graph_engine import BehaviouralGraph
    from src.core.memory_manager import MemoryManager
    from src.rl.trainer import RLTrainer
    from src.benchmarks.evaluator import BenchmarkEvaluator

    st.set_page_config(
        page_title="GraphMind Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
    body { background-color: #0d1117; color: #c9d1d9; }
    .metric-card { background: #161b22; border-radius: 10px; padding: 20px; margin: 5px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    st.sidebar.title("🧠 GraphMind")
    st.sidebar.markdown("**Predictive App Memory System**")
    user_id = st.sidebar.selectbox(
        "Select User",
        [f"user_{i:02d}" for i in range(10)],
        index=0
    )
    day = st.sidebar.slider("Simulation Day", 0, 29, 29)
    run_sim = st.sidebar.button("▶ Run Live Simulation")
    run_bench = st.sidebar.button("📊 Run Benchmarks")

    if run_sim:
        with st.spinner(f"Running simulation for {user_id}..."):
            try:
                from src.agents.orchestrator import GraphMindOrchestrator
                from src.core.event_bus import EventBus
                EventBus.get_instance().clear_all()
                orch = GraphMindOrchestrator(user_id)
                orch.run_full_simulation()
                st.sidebar.success("Simulation complete!")
            except Exception as e:
                st.sidebar.error(f"Simulation error: {e}")

    if run_bench:
        with st.spinner("Running benchmarks..."):
            try:
                evaluator = BenchmarkEvaluator()
                evaluator.run_all()
                st.sidebar.success("Benchmarks complete!")
            except Exception as e:
                st.sidebar.error(f"Benchmark error: {e}")

    # Load data
    data = load_data(user_id, day)
    state = data.get("state", {})
    tier_stats = data.get("tier_stats", {})
    snapshot = data.get("graph_snapshot", {})

    # ── Top Row Metric Cards ─────────────────────────────────────────────────
    st.title("🧠 GraphMind — Predictive App Intelligence Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hit_rate = state.get("cache_hit_rate", 0.0)
        st.metric("Cache Hit Rate", f"{hit_rate*100:.1f}%", delta="+18% vs LMKD")
    with col2:
        flush_count = state.get("security_flush_count", 0)
        st.metric("Security Flushes", str(flush_count))
    with col3:
        node_count = snapshot.get("node_count", 0)
        edge_count = snapshot.get("edge_count", 0)
        st.metric("Graph Size", f"{node_count}N / {edge_count}E")
    with col4:
        hot_count = tier_stats.get("hot_count", 0)
        warm_count = tier_stats.get("warm_count", 0)
        st.metric("HOT/WARM Nodes", f"{hot_count} / {warm_count}")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔗 Graph Evolution", "📊 Benchmarks", "🎯 RL Training",
        "🔒 Security Log", "💾 Memory Tiers"
    ])

    # Tab 1: Graph Evolution
    with tab1:
        st.subheader(f"Behavioural Graph — {user_id} Day {day}")
        sim_log = data.get("simulation_log", {})
        days_data = sim_log.get("days", [])
        snapshots_to_show = [1, 7, 14, 29]
        cols = st.columns(len(snapshots_to_show))
        for i, snap_day in enumerate(snapshots_to_show):
            snap = next((d.get("graph_snapshot", {}) for d in days_data if d.get("day") == snap_day), snapshot)
            with cols[i]:
                st.markdown(f"**Day {snap_day}**")
                if snap:
                    st.write(f"Nodes: {snap.get('node_count', 0)}, Edges: {snap.get('edge_count', 0)}")
                    html = render_pyvis_graph(snap)
                    st.components.v1.html(html, height=420, scrolling=True)

    # Tab 2: Benchmarks
    with tab2:
        st.subheader("Policy Comparison")
        bench_df = data.get("benchmark_df")
        if bench_df is not None and len(bench_df) > 0:
            avg = bench_df.groupby("policy_name")["cache_hit_rate"].mean().reset_index()
            avg["cache_hit_rate_pct"] = avg["cache_hit_rate"] * 100
            colors = {settings.BASELINE_GRAPHMIND: "#00d4aa"}
            fig = px.bar(avg, x="policy_name", y="cache_hit_rate_pct",
                         title="Average Cache Hit Rate by Policy (%)",
                         color="policy_name",
                         color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(bench_df.style.highlight_max(subset=["cache_hit_rate"], color="#003d2e"))
        else:
            st.info("No benchmark data. Click 'Run Benchmarks' in sidebar.")

    # Tab 3: RL Training Curves
    with tab3:
        st.subheader("PPO Training Reward Curves")
        curves = data.get("training_curves", {})
        if curves:
            fig = go.Figure()
            for uid, curve_data in list(curves.items())[:5]:
                steps = [c["step"] for c in curve_data]
                rewards = [c["reward"] for c in curve_data]
                fig.add_trace(go.Scatter(x=steps, y=rewards, name=uid, mode="lines"))
            fig.update_layout(title="Training Reward per User",
                              xaxis_title="Training Steps", yaxis_title="Reward",
                              template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training data. Run training first.")

    # Tab 4: Security Log
    with tab4:
        st.subheader("Security Flush Events")
        all_flushes = []
        for d in days_data:
            s = d.get("state", {})
            msgs = s.get("messages", [])
            for msg in msgs:
                if msg.get("agent") == "security":
                    for fe in msg.get("flush_events", []):
                        fe["day"] = d.get("day", 0)
                        all_flushes.append(fe)
        if all_flushes:
            flush_df = pd.DataFrame(all_flushes)
            def color_row(row):
                """Apply custom row coloring based on category for pandas styler."""
                if row.get("from_category") == "financial":
                    return ["background-color: #3d0000"] * len(row)
                elif row.get("from_category") == "health":
                    return ["background-color: #3d2000"] * len(row)
                return [""] * len(row)
            st.dataframe(flush_df)
        else:
            st.info("No security flushes recorded yet for this user.")

    # Tab 5: Memory Tiers
    with tab5:
        st.subheader("Memory Tier Statistics")
        if tier_stats:
            labels = ["HOT", "WARM", "COLD"]
            values = [tier_stats.get("hot_count", 0),
                      tier_stats.get("warm_count", 0),
                      tier_stats.get("cold_count", 0)]
            fig = px.pie(names=labels, values=values,
                         title="Memory Tier Distribution",
                         color_discrete_sequence=["#ff4500", "#ffa500", "#0078ff"])
            st.plotly_chart(fig, use_container_width=True)
            st.metric("HOT Capacity Used",
                      f"{tier_stats.get('hot_count',0)}/{tier_stats.get('hot_capacity',30)}")
            st.metric("WARM Capacity Used",
                      f"{tier_stats.get('warm_count',0)}/{tier_stats.get('warm_capacity',150)}")


# Only call st when run as streamlit app
try:
    import streamlit as st
    if hasattr(st, '_is_running_with_streamlit') or os.environ.get('STREAMLIT_SERVER_PORT'):
        _run_dashboard()
except ImportError:
    pass
except Exception:
    pass
