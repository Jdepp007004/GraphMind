"""
src/agents/orchestrator.py

LangGraph state machine wiring all 5 agents together.
This is the top-level coordinator for one user's simulation.
"""

import json
import logging
import os
from typing import TypedDict, Optional, List, Dict, Any

from config import settings
from src.core.graph_engine import BehaviouralGraph
from src.core.memory_manager import MemoryManager
from src.prefetch.daemon import PrefetchDaemon
from src.security.context_boundary import ContextBoundaryEnforcer
from src.agents.graph_manager_agent import GraphManagerAgent
from src.agents.rl_trainer_agent import RLTrainerAgent
from src.agents.prefetch_agent import PrefetchAgent
from src.agents.drift_detector_agent import DriftDetectorAgent
from src.agents.security_agent import SecurityAgent

logger = logging.getLogger(__name__)


class GraphMindState(TypedDict):
    """State schema for the LangGraph state machine."""
    user_id: str
    current_day: int
    current_event: Optional[dict]
    battery: float
    kl_divergence: float
    cache_hit_rate: float
    security_flush_count: int
    last_agent: str
    messages: List[dict]


def _route_after_drift(state: GraphMindState) -> str:
    """Conditional edge: route to rl_trainer on drift, else prefetch."""
    if state.get("kl_divergence", 0.0) > settings.DRIFT_KL_THRESHOLD:
        return "rl_trainer"
    return "prefetch"


class GraphMindOrchestrator:
    """
    LangGraph state machine coordinating all 5 agents.
    Runs one full simulation day as one orchestration cycle.
    """

    def __init__(self, user_id: str) -> None:
        """
        Initialize all 5 agents and their dependencies:
            - BehaviouralGraph(user_id) -> shared across agents
            - MemoryManager(user_id, graph) -> shared
            - PrefetchDaemon(user_id, graph, memory_manager)
            - ContextBoundaryEnforcer(user_id, memory_manager)
            - GraphManagerAgent(graph, memory_manager)
            - RLTrainerAgent(user_id)
            - PrefetchAgent(daemon)
            - DriftDetectorAgent(user_id)
            - SecurityAgent(enforcer)
        Build the LangGraph graph using build_graph().
        """
        self.user_id = user_id
        self.graph = BehaviouralGraph(user_id)
        self.memory_manager = MemoryManager(user_id, self.graph)
        self.daemon = PrefetchDaemon(user_id, self.graph, self.memory_manager)
        self.enforcer = ContextBoundaryEnforcer(user_id, self.memory_manager)

        self.graph_manager_agent = GraphManagerAgent(self.graph, self.memory_manager)
        self.rl_trainer_agent = RLTrainerAgent(user_id)
        self.prefetch_agent = PrefetchAgent(self.daemon)
        self.drift_detector_agent = DriftDetectorAgent(user_id)
        self.security_agent = SecurityAgent(self.enforcer)

        self.compiled_graph = self.build_graph()
        logger.info(f"GraphMindOrchestrator initialized for {user_id}")

    def build_graph(self):
        """
        Build and compile the LangGraph StateGraph.

        Nodes: 'graph_manager', 'rl_trainer', 'prefetch', 'drift_detector', 'security'

        Edges (sequential with conditional):
            START -> graph_manager
            graph_manager -> drift_detector
            drift_detector -> rl_trainer (if kl_divergence > DRIFT_KL_THRESHOLD)
            drift_detector -> prefetch (if kl_divergence <= DRIFT_KL_THRESHOLD)
            rl_trainer -> prefetch
            prefetch -> security
            security -> END

        Returns compiled graph.
        """
        from langgraph.graph import StateGraph, END

        builder = StateGraph(GraphMindState)

        # Add nodes
        builder.add_node("graph_manager", self.graph_manager_agent.run)
        builder.add_node("drift_detector", self.drift_detector_agent.run)
        builder.add_node("rl_trainer", self.rl_trainer_agent.run)
        builder.add_node("prefetch", self.prefetch_agent.run)
        builder.add_node("security", self.security_agent.run)

        # Edges
        builder.set_entry_point("graph_manager")
        builder.add_edge("graph_manager", "drift_detector")
        builder.add_conditional_edges(
            "drift_detector",
            _route_after_drift,
            {
                "rl_trainer": "rl_trainer",
                "prefetch": "prefetch"
            }
        )
        builder.add_edge("rl_trainer", "prefetch")
        builder.add_edge("prefetch", "security")
        builder.add_edge("security", END)

        compiled = builder.compile()
        return compiled

    def run_day(self, day: int) -> GraphMindState:
        """
        Run one full simulation day through the state machine.
        Initializes state with current day and user context.
        Invokes the compiled LangGraph graph.
        Returns final state after all agents have run.
        """
        initial_state: GraphMindState = {
            "user_id": self.user_id,
            "current_day": day,
            "current_event": None,
            "battery": 100.0,
            "kl_divergence": 0.0,
            "cache_hit_rate": 0.0,
            "security_flush_count": len(self.enforcer.get_flush_log()),
            "last_agent": "",
            "messages": []
        }
        try:
            result = self.compiled_graph.invoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"Orchestrator run_day failed for {self.user_id} day {day}: {e}")
            initial_state["last_agent"] = "error"
            initial_state["messages"].append({"error": str(e)})
            return initial_state

    def run_full_simulation(self) -> List[GraphMindState]:
        """
        Run all SIMULATION_DAYS days sequentially.
        Returns list of daily state snapshots.
        Save snapshots to RESULTS_DIR/{user_id}_simulation_log.json.
        """
        from src.data.event_simulator import EventSimulator
        os.makedirs(settings.RESULTS_DIR, exist_ok=True)
        simulator = EventSimulator(self.user_id)
        daily_states = []

        for day in range(settings.SIMULATION_DAYS):
            # Replay all events for this day through the simulator
            day_events = simulator.step_day()
            # Run orchestration cycle
            state = self.run_day(day)
            # Get graph snapshot and tier stats
            snapshot = self.graph.get_graph_snapshot(day)
            tier_stats = self.memory_manager.get_tier_stats()
            # Prune and evict periodically
            if day % 7 == 0:
                self.graph.prune_weak_edges()
            self.graph.evict_stale_nodes(day)
            daily_states.append({
                "day": day,
                "state": dict(state),
                "graph_snapshot": snapshot,
                "tier_stats": tier_stats
            })
            logger.debug(f"Completed day {day} for {self.user_id}")

        log = {
            "user_id": self.user_id,
            "days": daily_states
        }
        log_path = os.path.join(settings.RESULTS_DIR, f"{self.user_id}_simulation_log.json")
        with open(log_path, "w") as f:
            json.dump(log, f)
        logger.info(f"Simulation log saved to {log_path}")
        return [d["state"] for d in daily_states]
