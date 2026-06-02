"""
src/graph_playback/graph_animator.py

Produces dashboard-ready animation data from TimelineEngine frames.
Generates PyVis HTML snapshots for scrubbing and frame comparison.
"""

import logging
from typing import List, Optional, Dict, Any

from src.graph_playback.timeline_engine import TimelineEngine, TimelineFrame

logger = logging.getLogger(__name__)

# Category color map aligned with existing dashboard palette
CATEGORY_COLORS = {
    "social": "#4f94f0",
    "entertainment": "#f06050",
    "financial": "#e8b44e",
    "health": "#50c878",
    "gaming": "#c86090",
    "productivity": "#50c8c8",
    "utility": "#909090",
    "navigation": "#f07840",
    "shopping": "#b050f0",
    "enterprise": "#f0d050",
    "government": "#d0f050",
    "system": "#606060",
    "default": "#aaaaaa",
}


class GraphAnimator:
    """
    Converts TimelineFrame data into renderable graph frames for the dashboard.
    Produces PyVis HTML strings and Plotly-compatible data.
    """

    def __init__(self, timeline: TimelineEngine) -> None:
        self.timeline = timeline

    def render_frame_html(self, day: int,
                           height: str = "350px",
                           max_nodes: int = 50,
                           max_edges: int = 100) -> str:
        """
        Render the graph for a specific day as a PyVis HTML string.
        Returns an HTML error string on failure.
        Nodes are color-coded by category. Edge width scales with transition_prob.
        """
        frame = self.timeline.get_frame(day)
        if frame is None:
            return f"<p style='color:#888'>No data for day {day}</p>"
        return self._snapshot_to_html(frame, height, max_nodes, max_edges)

    def render_milestone_frames(self) -> List[Dict[str, Any]]:
        """
        Return a list of dicts with HTML + metadata for milestone days.
        Used by the dashboard Graph Evolution tab.
        """
        results = []
        for frame in self.timeline.get_milestone_frames():
            results.append({
                "day": frame.day,
                "html": self._snapshot_to_html(frame, "300px", 40, 80),
                "node_count": frame.node_count,
                "edge_count": frame.edge_count,
                "node_delta": frame.node_delta,
                "new_nodes": len(frame.new_node_ids),
                "cache_hit_rate": frame.cache_hit_rate,
                "drift_detected": frame.drift_detected,
            })
        return results

    def _snapshot_to_html(self, frame: TimelineFrame,
                           height: str, max_nodes: int, max_edges: int) -> str:
        """Convert a TimelineFrame to a PyVis HTML string."""
        try:
            from pyvis.network import Network
            net = Network(
                height=height, width="100%",
                directed=True,
                bgcolor="#0d1117",
                font_color="white"
            )
            nodes = frame.nodes[:max_nodes]
            edges = frame.edges[:max_edges]
            node_id_set = {n.get("node_id", "") for n in nodes}

            for node in nodes:
                nid = node.get("node_id", "")
                app_id = node.get("app_id", "unknown")
                category = node.get("category", "utility")
                access_count = node.get("access_count", 0)
                label = app_id.split(".")[-1][:14]
                color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["default"])
                size = min(40, 15 + access_count * 2)
                title = (f"App: {app_id}\n"
                         f"Category: {category}\n"
                         f"Accesses: {access_count}\n"
                         f"Day: {frame.day}")
                net.add_node(nid, label=label, color=color, size=size, title=title)

            for edge in edges:
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                prob = float(edge.get("prob", 0.1))
                if src in node_id_set and tgt in node_id_set:
                    net.add_edge(src, tgt, value=prob * 5,
                                 title=f"prob: {prob:.2f}")

            return net.generate_html()
        except ImportError:
            return "<p style='color:#888'>PyVis not installed. Run: pip install pyvis</p>"
        except Exception as e:
            logger.error(f"GraphAnimator render error: {e}")
            return f"<p style='color:red'>Render error: {e}</p>"

    def get_playback_frames_data(self) -> List[dict]:
        """
        Return lightweight frame data for a timeline scrubber.
        Does NOT include HTML (too heavy). Use render_frame_html() on demand.
        """
        return self.timeline.get_frames_dict()

    def get_growth_chart_data(self) -> dict:
        """Return growth time-series for Plotly line charts."""
        return self.timeline.get_growth_series()

    def get_category_evolution(self) -> Dict[str, List[int]]:
        """
        Return per-category node count evolution over days.
        Returns: {'social': [0, 2, 5, ...], 'financial': [0, 1, 2, ...], ...}
        """
        frames = self.timeline.get_all_frames()
        categories = set(CATEGORY_COLORS.keys()) - {"default"}
        evolution: Dict[str, List[int]] = {cat: [] for cat in categories}
        evolution["days"] = []

        for frame in frames:
            evolution["days"].append(frame.day)
            cat_counts = {cat: 0 for cat in categories}
            for node in frame.nodes:
                cat = node.get("category", "utility")
                if cat in cat_counts:
                    cat_counts[cat] += 1
            for cat in categories:
                evolution[cat].append(cat_counts[cat])

        return evolution
