# GRAPHMIND — HARD CHECKER
# File 2 of 2: End-to-End Verification Script
# Run this AFTER the full project is implemented.
# It will tell you exactly what failed and what to do to fix it.

# ── HOW TO RUN ─────────────────────────────────────────────────────────────
# python GRAPHMIND_HARDCHECK.py
# python GRAPHMIND_HARDCHECK.py --phase 1        (check only Phase 1)
# python GRAPHMIND_HARDCHECK.py --phase 3        (check only Phase 3)
# python GRAPHMIND_HARDCHECK.py --verbose        (show all pass results too)
# python GRAPHMIND_HARDCHECK.py --fix-hints-only (only show fix instructions, no pass/fail)

import os
import sys
import json
import importlib
import traceback
import argparse
import subprocess
from pathlib import Path
from typing import Callable

# ── Checker Infrastructure ─────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = []   # list of {"phase": int, "check": str, "passed": bool, "error": str, "fix": str}
TOTAL_PASS = 0
TOTAL_FAIL = 0


def check(phase: int, name: str, fix_instruction: str):
    """
    Decorator for check functions.
    Usage:
        @check(1, "EventBus singleton", "Implement get_instance() as a classmethod with a lock...")
        def _():
            ...assert something...
    """
    def decorator(fn: Callable):
        def wrapper():
            global TOTAL_PASS, TOTAL_FAIL
            try:
                fn()
                RESULTS.append({"phase": phase, "check": name, "passed": True, "error": "", "fix": ""})
                TOTAL_PASS += 1
            except Exception as e:
                error_detail = traceback.format_exc()
                RESULTS.append({"phase": phase, "check": name, "passed": False, "error": str(e), "fix": fix_instruction, "traceback": error_detail})
                TOTAL_FAIL += 1
        return wrapper
    return decorator


def run_checks(phase_filter: int | None, verbose: bool, fix_hints_only: bool):
    """Run all registered checks and print results."""
    # All check functions are defined below and auto-collected
    all_check_functions = [
        # Phase 1
        check_env_setup,
        check_settings_file,
        check_taxonomy_file,
        check_event_bus_import,
        check_event_bus_singleton,
        check_event_bus_pubsub,
        check_event_bus_unsubscribe,
        check_event_bus_topics_defined,
        check_graph_engine_import,
        check_graph_node_class,
        check_graph_edge_class,
        check_behavioural_graph_class,
        check_graph_add_node,
        check_graph_add_edge,
        check_graph_update_edge_weights,
        check_graph_get_top_k,
        check_graph_prune_edges,
        check_graph_evict_stale,
        check_graph_serialization,
        check_graph_snapshot_schema,
        check_dataset_generator_import,
        check_dataset_files_exist,
        check_dataset_schema,
        check_dataset_event_count,
        check_dataset_metadata,
        # Phase 2
        check_context_encoder_import,
        check_context_encoder_output_shape,
        check_context_encoder_deterministic,
        check_memory_manager_import,
        check_memory_manager_hot_promote,
        check_memory_manager_hot_capacity,
        check_memory_manager_demote,
        check_memory_manager_flush_by_category,
        check_memory_manager_tier_stats,
        check_event_simulator_import,
        check_event_simulator_loads,
        check_event_simulator_step_publishes,
        check_event_simulator_day_advance,
        # Phase 3
        check_reward_import,
        check_reward_compute_positive,
        check_reward_penalizes_thrash,
        check_reward_penalizes_battery,
        check_reward_episode_summary,
        check_rl_env_import,
        check_rl_env_instantiate,
        check_rl_env_observation_space,
        check_rl_env_action_space,
        check_rl_env_reset,
        check_rl_env_step,
        check_rl_trainer_import,
        check_rl_model_exists,
        check_rl_model_loadable,
        # Phase 4
        check_prefetch_daemon_import,
        check_prefetch_daemon_instantiate,
        check_context_boundary_import,
        check_context_boundary_transition_detect,
        check_context_boundary_flush_correct,
        check_context_boundary_no_false_positive,
        check_drift_detector_import,
        check_drift_detector_zero_data,
        check_drift_detector_divergent_data,
        check_orchestrator_import,
        check_orchestrator_instantiate,
        check_orchestrator_run_day,
        check_orchestrator_state_schema,
        check_langgraph_graph_built,
        # Phase 5
        check_baselines_import,
        check_baseline_lmkd,
        check_baseline_art,
        check_baseline_lru,
        check_baseline_bixby,
        check_evaluator_import,
        check_benchmark_results_exist,
        check_benchmark_results_schema,
        check_graphmind_beats_lmkd,
        check_graphmind_beats_bixby,
        check_security_flushes_recorded,
        check_graph_node_count_stable,
        check_simulation_logs_exist,
        check_dashboard_importable,
        # Submission
        check_readme_filled,
        check_license_exists,
        check_docs_folder,
        check_ax_md_exists,
        check_agents_md_exists,
        check_src_folder_structure,
        check_no_circular_imports,
        check_all_functions_have_docstrings,
    ]

    for fn in all_check_functions:
        fn()

    # Print results
    print("\n" + "="*70)
    print("GRAPHMIND HARD CHECK RESULTS")
    print("="*70)

    current_phase = 0
    for r in RESULTS:
        if phase_filter and r["phase"] != phase_filter:
            continue
        if r["phase"] != current_phase:
            current_phase = r["phase"]
            print(f"\n── PHASE {current_phase} ──────────────────────────────────")

        if fix_hints_only:
            if not r["passed"]:
                print(f"\n  ✗ {r['check']}")
                print(f"    ERROR: {r['error']}")
                print(f"    FIX:   {r['fix']}")
            continue

        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        if r["passed"]:
            if verbose:
                print(f"  {status}  {r['check']}")
        else:
            print(f"  {status}  {r['check']}")
            print(f"    ERROR: {r['error']}")
            print(f"    FIX:   {r['fix']}")

    print("\n" + "="*70)
    applicable = [r for r in RESULTS if not phase_filter or r["phase"] == phase_filter]
    passed = sum(1 for r in applicable if r["passed"])
    failed = sum(1 for r in applicable if not r["passed"])
    print(f"TOTAL: {passed} passed, {failed} failed out of {len(applicable)} checks")
    if failed == 0:
        print("ALL CHECKS PASSED — ready for submission!")
    else:
        print(f"\n{failed} checks failed. Fix the issues above before submitting.")
        print("Re-run: python GRAPHMIND_HARDCHECK.py --fix-hints-only")
    print("="*70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 CHECKS
# ─────────────────────────────────────────────────────────────────────────────

@check(1, "Python version >= 3.11", 
       "Use Python 3.11 or newer. Check 'python --version'.")
def check_env_setup():
    assert sys.version_info >= (3, 11), f"Python {sys.version_info} < 3.11"


@check(1, "config/settings.py importable and has all required constants",
       "Create config/settings.py exactly as specified in GRAPHMIND_BUILD_SPEC.md Section 4. "
       "Ensure all constants are defined: NUM_USERS, HOT_TIER_CAPACITY, EDGE_PRUNE_THRESHOLD, "
       "REWARD_ALPHA, BASELINE_LMKD, etc.")
def check_settings_file():
    from config import settings
    required = [
        "NUM_USERS", "SIMULATION_DAYS", "NODE_EMBEDDING_DIM", "HOT_TIER_CAPACITY",
        "WARM_TIER_CAPACITY", "EDGE_PRUNE_THRESHOLD", "NODE_EVICTION_DAYS",
        "PPO_TOTAL_TIMESTEPS", "REWARD_ALPHA", "REWARD_BETA", "REWARD_GAMMA",
        "REWARD_DELTA", "REWARD_EPSILON", "PREFETCH_INTERVAL_MINUTES",
        "DRIFT_KL_THRESHOLD", "SENSITIVE_CATEGORIES", "CONSUMER_CATEGORIES",
        "BASELINE_LMKD", "BASELINE_ART", "BASELINE_LRU", "BASELINE_BIXBY",
        "BASELINE_GRAPHMIND", "GEMMA_MODEL_ID", "DATA_DIR", "SYNTHETIC_DIR",
        "APP_TAXONOMY_PATH", "USERS_DIR", "RESULTS_DIR", "RL_MODELS_DIR"
    ]
    missing = [c for c in required if not hasattr(settings, c)]
    assert not missing, f"Missing constants in settings.py: {missing}"
    assert settings.NUM_USERS == 10, f"NUM_USERS must be 10, got {settings.NUM_USERS}"
    assert settings.HOT_TIER_CAPACITY == 30
    assert settings.EDGE_PRUNE_THRESHOLD == 0.05
    assert settings.NODE_EMBEDDING_DIM == 64


@check(1, "data/app_taxonomy.json exists and has required structure",
       "Create data/app_taxonomy.json as specified in GRAPHMIND_BUILD_SPEC.md. "
       "Each entry must have 'name' and 'category' keys. "
       "Categories must include at least: social, financial, health, enterprise, entertainment.")
def check_taxonomy_file():
    from config.settings import APP_TAXONOMY_PATH
    assert os.path.exists(APP_TAXONOMY_PATH), f"File not found: {APP_TAXONOMY_PATH}"
    with open(APP_TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    assert len(taxonomy) >= 10, f"Taxonomy has only {len(taxonomy)} entries, need >= 10"
    sample = next(iter(taxonomy.values()))
    assert "name" in sample, "Each taxonomy entry must have 'name' key"
    assert "category" in sample, "Each taxonomy entry must have 'category' key"
    categories = {v["category"] for v in taxonomy.values()}
    required_cats = {"social", "financial", "health", "enterprise", "entertainment"}
    missing_cats = required_cats - categories
    assert not missing_cats, f"Missing categories in taxonomy: {missing_cats}"


@check(1, "src.core.event_bus importable",
       "Create src/core/event_bus.py. Ensure src/core/__init__.py exists (can be empty). "
       "The module must import without error.")
def check_event_bus_import():
    from src.core import event_bus
    assert hasattr(event_bus, 'EventBus'), "EventBus class not found in event_bus.py"


@check(1, "EventBus is a singleton",
       "Implement EventBus.get_instance() as a classmethod. Use a class-level _instance variable "
       "and a threading.Lock() to ensure thread safety. Two calls to get_instance() must return "
       "the exact same object (use 'is' comparison).")
def check_event_bus_singleton():
    from src.core.event_bus import EventBus
    EventBus._instance = None  # Reset for test
    a = EventBus.get_instance()
    b = EventBus.get_instance()
    assert a is b, "EventBus.get_instance() returned different objects — singleton broken"


@check(1, "EventBus publish/subscribe works correctly",
       "In subscribe(), append the callback to a dict[topic] = list of callbacks. "
       "In publish(), iterate callbacks for the topic and call each with payload. "
       "The callback MUST be called synchronously before publish() returns.")
def check_event_bus_pubsub():
    from src.core.event_bus import EventBus
    bus = EventBus.get_instance()
    bus.clear_all()
    received = []
    def cb(payload): received.append(payload)
    bus.subscribe("test_topic", cb)
    bus.publish("test_topic", {"timestamp": 1.0, "value": 42})
    assert len(received) == 1, f"Expected 1 callback call, got {len(received)}"
    assert received[0]["value"] == 42
    bus.clear_all()


@check(1, "EventBus unsubscribe works correctly",
       "In unsubscribe(), remove the specific callback from the topic's list. "
       "After unsubscribe, publishing to that topic must NOT call the removed callback.")
def check_event_bus_unsubscribe():
    from src.core.event_bus import EventBus
    bus = EventBus.get_instance()
    bus.clear_all()
    received = []
    def cb(payload): received.append(payload)
    bus.subscribe("test_topic", cb)
    bus.unsubscribe("test_topic", cb)
    bus.publish("test_topic", {"timestamp": 1.0})
    assert len(received) == 0, "Callback still called after unsubscribe"
    bus.clear_all()


@check(1, "EventBus topic constants are defined",
       "Define all TOPIC_* constants at module level in event_bus.py. "
       "e.g. TOPIC_APP_LAUNCHED = 'app_launched'. See spec Section 4 for full list.")
def check_event_bus_topics_defined():
    from src.core import event_bus
    required_topics = [
        "TOPIC_APP_LAUNCHED", "TOPIC_APP_CLOSED", "TOPIC_BATTERY_UPDATED",
        "TOPIC_CACHE_HIT", "TOPIC_CACHE_MISS", "TOPIC_DRIFT_DETECTED",
        "TOPIC_SECURITY_FLUSH", "TOPIC_PREFETCH_TRIGGERED", "TOPIC_NODE_PROMOTED"
    ]
    missing = [t for t in required_topics if not hasattr(event_bus, t)]
    assert not missing, f"Missing topic constants: {missing}"


@check(1, "src.core.graph_engine importable",
       "Create src/core/graph_engine.py with GraphNode, GraphEdge, BehaviouralGraph classes. "
       "Ensure networkx is installed: pip install networkx==3.3")
def check_graph_engine_import():
    from src.core import graph_engine
    assert hasattr(graph_engine, 'GraphNode')
    assert hasattr(graph_engine, 'GraphEdge')
    assert hasattr(graph_engine, 'BehaviouralGraph')


@check(1, "GraphNode has all required fields",
       "GraphNode must have fields: node_id(str), embedding(np.ndarray), app_id(str), "
       "time_bucket(int), battery_bucket(int), context_flags(dict), last_seen_day(int), "
       "access_count(int), category(str). Use dataclass or __init__ with all these fields.")
def check_graph_node_class():
    import numpy as np
    from src.core.graph_engine import GraphNode
    node = GraphNode(
        node_id="test-001",
        embedding=np.zeros(64),
        app_id="com.instagram.android",
        time_bucket=10,
        battery_bucket=3,
        context_flags={"headphones": False, "calendar_near": False, "weekend": True},
        last_seen_day=0,
        access_count=1,
        category="social"
    )
    assert node.node_id == "test-001"
    assert node.embedding.shape == (64,)
    assert node.category == "social"


@check(1, "GraphEdge has all required fields",
       "GraphEdge must have fields: source_id(str), target_id(str), transition_prob(float), "
       "time_sensitivity(float), battery_cost(float). Use dataclass or __init__.")
def check_graph_edge_class():
    from src.core.graph_engine import GraphEdge
    edge = GraphEdge(
        source_id="a", target_id="b",
        transition_prob=0.5, time_sensitivity=0.3, battery_cost=0.1
    )
    assert edge.transition_prob == 0.5


@check(1, "BehaviouralGraph class is importable and has correct attributes",
       "BehaviouralGraph must be importable from src.core.graph_engine.")
def check_behavioural_graph_class():
    from src.core.graph_engine import BehaviouralGraph
    g = BehaviouralGraph("user_test")
    assert g.user_id == "user_test", "BehaviouralGraph should store user_id as an attribute"


@check(1, "BehaviouralGraph add_node() and get_node() work correctly",
       "In add_node(), store the node in self.graph (nx.DiGraph) using node.node_id as key. "
       "In get_node(), return self.graph.nodes[node_id]['data'] or equivalent. "
       "If node_id already exists in add_node(), update last_seen_day and access_count only.")
def check_graph_add_node():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    node = GraphNode("n1", np.zeros(64), "com.instagram.android", 10, 3,
                     {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social")
    g.add_node(node)
    assert g.get_node("n1") is not None, "get_node() returned None after add_node()"
    assert g.get_node("n1").app_id == "com.instagram.android"
    assert g.node_count() == 1
    EventBus.get_instance().clear_all()


@check(1, "BehaviouralGraph add_edge() and get_edges_from() work correctly",
       "add_edge() must store edge data on the nx.DiGraph edge. "
       "get_edges_from() must return a list of GraphEdge objects for all outgoing edges. "
       "Raise ValueError if source_id or target_id not in graph.")
def check_graph_add_edge():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    for nid in ["n1", "n2"]:
        g.add_node(GraphNode(nid, np.zeros(64), f"app_{nid}", 5, 2,
                             {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social"))
    g.add_edge("n1", "n2", 0.3, 0.5, 0.2)
    edges = g.get_edges_from("n1")
    assert len(edges) == 1, f"Expected 1 edge, got {len(edges)}"
    assert edges[0].transition_prob == 0.3
    # Test ValueError for missing nodes
    try:
        g.add_edge("n1", "nonexistent", 0.1, 0.1, 0.1)
        assert False, "Should have raised ValueError for missing target node"
    except ValueError:
        pass
    EventBus.get_instance().clear_all()


@check(1, "BehaviouralGraph update_edge_weights() clamps to [0,1]",
       "In update_edge_weights(), apply additive delta then clamp: "
       "new_val = max(0.0, min(1.0, old_val + delta)). "
       "Raise ValueError if edge does not exist.")
def check_graph_update_edge_weights():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    for nid in ["n1", "n2"]:
        g.add_node(GraphNode(nid, np.zeros(64), "app", 5, 2,
                             {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social"))
    g.add_edge("n1", "n2", 0.9, 0.5, 0.2)
    g.update_edge_weights("n1", "n2", delta_prob=0.5, delta_time=0.0, delta_battery=0.0)  # Should clamp to 1.0
    edges = g.get_edges_from("n1")
    assert edges[0].transition_prob == 1.0, f"Expected 1.0 after clamping, got {edges[0].transition_prob}"
    g.update_edge_weights("n1", "n2", delta_prob=-2.0, delta_time=0.0, delta_battery=0.0)  # Should clamp to 0.0
    edges = g.get_edges_from("n1")
    assert edges[0].transition_prob == 0.0
    EventBus.get_instance().clear_all()


@check(1, "BehaviouralGraph get_top_k_next_nodes() returns correct ordering",
       "get_top_k_next_nodes() must score each edge as: "
       "score = transition_prob - (battery_cost * (1 - battery_level/100)) "
       "Then sort descending and return top-k target node_ids. "
       "With battery < BATTERY_SUPPRESS_THRESHOLD, k must be halved.")
def check_graph_get_top_k():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    nodes = ["src", "high_prob", "low_prob", "mid_prob"]
    for nid in nodes:
        g.add_node(GraphNode(nid, np.zeros(64), "app", 5, 2,
                             {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social"))
    g.add_edge("src", "high_prob", 0.9, 0.3, 0.1)
    g.add_edge("src", "low_prob", 0.1, 0.3, 0.1)
    g.add_edge("src", "mid_prob", 0.5, 0.3, 0.1)
    result = g.get_top_k_next_nodes("src", 2, 80.0)
    assert result[0] == "high_prob", f"Expected high_prob first, got {result[0]}"
    assert "low_prob" not in result, "low_prob should not be in top-2"
    EventBus.get_instance().clear_all()


@check(1, "BehaviouralGraph prune_weak_edges() removes edges below threshold",
       "In prune_weak_edges(), iterate all edges and remove any where transition_prob < 0.05. "
       "Return the count of removed edges. Use list(self.graph.edges()) to avoid "
       "RuntimeError: dictionary changed size during iteration.")
def check_graph_prune_edges():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    for nid in ["n1", "n2", "n3"]:
        g.add_node(GraphNode(nid, np.zeros(64), "app", 5, 2,
                             {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social"))
    g.add_edge("n1", "n2", 0.03, 0.5, 0.1)  # below threshold → should be pruned
    g.add_edge("n1", "n3", 0.5, 0.5, 0.1)   # above threshold → should remain
    pruned = g.prune_weak_edges()
    assert pruned == 1, f"Expected 1 pruned edge, got {pruned}"
    assert g.edge_count() == 1
    EventBus.get_instance().clear_all()


@check(1, "BehaviouralGraph evict_stale_nodes() removes nodes inactive for 45+ days",
       "In evict_stale_nodes(current_day), for each node check: "
       "(current_day - node.last_seen_day) > NODE_EVICTION_DAYS (45). "
       "Also remove all edges connected to evicted nodes. "
       "Iterate over list(self.graph.nodes()) to avoid modification-during-iteration error.")
def check_graph_evict_stale():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    old = GraphNode("old_node", np.zeros(64), "app", 5, 2,
                    {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social")
    fresh = GraphNode("fresh_node", np.zeros(64), "app", 5, 2,
                      {"headphones": False, "calendar_near": False, "weekend": False}, 40, 1, "social")
    g.add_node(old)
    g.add_node(fresh)
    g.add_edge("old_node", "fresh_node", 0.5, 0.3, 0.1)
    evicted = g.evict_stale_nodes(50)  # old_node: 50-0=50 > 45, fresh_node: 50-40=10 ≤ 45
    assert evicted == 1, f"Expected 1 eviction, got {evicted}"
    assert g.get_node("old_node") is None
    assert g.get_node("fresh_node") is not None
    assert g.edge_count() == 0  # edge to evicted node must also be removed
    EventBus.get_instance().clear_all()


@check(1, "BehaviouralGraph save_to_disk() and load_from_disk() roundtrip",
       "save_to_disk() must use pickle.dump() on a serializable representation. "
       "load_from_disk() must restore exact same graph state (same nodes, edges, weights). "
       "Test: save, create new empty graph, load, verify nodes/edges match.")
def check_graph_serialization():
    import numpy as np
    import tempfile
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    node = GraphNode("n1", np.ones(64), "com.instagram.android", 10, 3,
                     {"headphones": True, "calendar_near": False, "weekend": False}, 5, 7, "social")
    g.add_node(node)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    g.save_to_disk(path)
    g2 = BehaviouralGraph("user_test")
    g2.load_from_disk(path)
    assert g2.node_count() == 1, f"After load, node count = {g2.node_count()}, expected 1"
    loaded_node = g2.get_node("n1")
    assert loaded_node is not None
    assert loaded_node.app_id == "com.instagram.android"
    assert loaded_node.access_count == 7
    os.unlink(path)
    EventBus.get_instance().clear_all()


@check(1, "BehaviouralGraph get_graph_snapshot() returns correct schema",
       "get_graph_snapshot() must return a dict with keys: day, user_id, node_count, edge_count, "
       "nodes (list of dicts with node_id/app_id/category/access_count), edges (list of dicts). "
       "Must be JSON-serializable (no numpy arrays, no custom objects in the returned dict).")
def check_graph_snapshot_schema():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    g.add_node(GraphNode("n1", np.zeros(64), "com.test.app", 5, 2,
                         {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social"))
    snapshot = g.get_graph_snapshot(day=5)
    required_keys = {"day", "user_id", "node_count", "edge_count", "nodes", "edges"}
    assert required_keys.issubset(set(snapshot.keys())), f"Missing keys: {required_keys - set(snapshot.keys())}"
    assert snapshot["day"] == 5
    assert snapshot["node_count"] == 1
    assert isinstance(snapshot["nodes"], list)
    assert "app_id" in snapshot["nodes"][0]
    # Must be JSON serializable
    json.dumps(snapshot)  # Will raise if not serializable
    EventBus.get_instance().clear_all()


@check(1, "src.data.dataset_generator importable",
       "Create src/data/dataset_generator.py. Ensure src/data/__init__.py exists. "
       "USER_PROFILES list must be defined at module level with exactly 10 entries.")
def check_dataset_generator_import():
    from src.data import dataset_generator
    assert hasattr(dataset_generator, 'USER_PROFILES')
    assert hasattr(dataset_generator, 'DatasetGenerator')
    assert len(dataset_generator.USER_PROFILES) == 10, \
        f"USER_PROFILES must have 10 entries, has {len(dataset_generator.USER_PROFILES)}"
    ids = [p["user_id"] for p in dataset_generator.USER_PROFILES]
    expected = [f"user_{i:02d}" for i in range(10)]
    assert ids == expected, f"user_ids must be user_00..user_09, got {ids}"


@check(1, "All 10 synthetic user dataset files exist",
       "Run: python scripts/generate_dataset.py \n"
       "This creates data/synthetic/users/user_00.json through user_09.json. "
       "If Gemma is not available, the fallback rule-based generator will be used automatically.")
def check_dataset_files_exist():
    from config.settings import USERS_DIR, NUM_USERS
    for i in range(NUM_USERS):
        path = os.path.join(USERS_DIR, f"user_{i:02d}.json")
        assert os.path.exists(path), f"Missing dataset file: {path}"


@check(1, "Dataset event schema is correct",
       "Each event in the dataset must be a dict with keys: "
       "day(int), timestamp(float), app_id(str), battery(float), time_bucket(int), "
       "headphones(bool), calendar_event_in_mins(int|null), weekend(bool), category(str). "
       "Check user_00.json for correct structure.")
def check_dataset_schema():
    from config.settings import USERS_DIR
    path = os.path.join(USERS_DIR, "user_00.json")
    with open(path) as f:
        events = json.load(f)
    assert isinstance(events, list), "user_00.json must contain a JSON list of events"
    assert len(events) > 0
    required_keys = {"day", "timestamp", "app_id", "battery", "time_bucket",
                     "headphones", "calendar_event_in_mins", "weekend", "category"}
    sample = events[0]
    missing = required_keys - set(sample.keys())
    assert not missing, f"Event missing keys: {missing}"
    assert isinstance(sample["day"], int)
    assert isinstance(sample["battery"], float)
    assert 0.0 <= sample["battery"] <= 100.0, f"Battery out of range: {sample['battery']}"
    assert 0 <= sample["time_bucket"] <= 47


@check(1, "Dataset has sufficient events per user",
       "Each user must have at least 1000 events (SIMULATION_DAYS * min events per day). "
       "Check that generate_dataset.py is generating enough events per day.")
def check_dataset_event_count():
    from config.settings import USERS_DIR, NUM_USERS
    for i in range(NUM_USERS):
        path = os.path.join(USERS_DIR, f"user_{i:02d}.json")
        with open(path) as f:
            events = json.load(f)
        assert len(events) >= 1000, f"user_{i:02d} has only {len(events)} events, need >= 1000"


@check(1, "Dataset metadata.json exists and is correct",
       "Run scripts/generate_dataset.py to create data/synthetic/metadata.json. "
       "It must have: num_users(10), days_per_user(30), total_events(int), "
       "generation_mode('gemma' or 'fallback'), created_at(ISO string).")
def check_dataset_metadata():
    from config.settings import SYNTHETIC_DIR
    meta_path = os.path.join(SYNTHETIC_DIR, "metadata.json")
    assert os.path.exists(meta_path), f"Missing: {meta_path}"
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta.get("num_users") == 10
    assert meta.get("days_per_user") == 30
    assert "total_events" in meta
    assert meta.get("generation_mode") in ("gemma", "fallback")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 CHECKS
# ─────────────────────────────────────────────────────────────────────────────

@check(2, "src.data.context_encoder importable with correct architecture",
       "Create src/data/context_encoder.py with ContextEncoder class. "
       "The MLP must have input_dim=35, hidden=128, output=64. "
       "APP_ID_VOCAB must be defined at module level with exactly 30 entries.")
def check_context_encoder_import():
    from src.data import context_encoder
    assert hasattr(context_encoder, 'ContextEncoder')
    assert hasattr(context_encoder, 'APP_ID_VOCAB')
    assert len(context_encoder.APP_ID_VOCAB) == 30, \
        f"APP_ID_VOCAB must have 30 entries, has {len(context_encoder.APP_ID_VOCAB)}"


@check(2, "ContextEncoder.encode() returns shape (64,) numpy array",
       "encode() must: 1) one-hot encode app_id using APP_ID_VOCAB (size 30), "
       "2) normalize time_bucket by /47, battery by /100, "
       "3) concatenate into tensor(35,), 4) pass through MLP, 5) return numpy(64,). "
       "Total input dims: 30 (one-hot) + 1 (time) + 1 (battery) + 1 (headphones) + 1 (calendar) + 1 (weekend) = 35")
def check_context_encoder_output_shape():
    import numpy as np
    from src.data.context_encoder import ContextEncoder
    enc = ContextEncoder()
    event = {"app_id": "com.instagram.android", "time_bucket": 10, "battery": 75.0,
             "headphones": False, "calendar_event_in_mins": None, "weekend": False}
    result = enc.encode(event)
    assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
    assert result.shape == (64,), f"Expected shape (64,), got {result.shape}"


@check(2, "ContextEncoder.encode() is deterministic",
       "Same input event must always produce identical output embedding. "
       "The encoder is in eval mode and uses no dropout, so output should be identical.")
def check_context_encoder_deterministic():
    import numpy as np
    from src.data.context_encoder import ContextEncoder
    enc = ContextEncoder()
    event = {"app_id": "com.slack.android", "time_bucket": 20, "battery": 50.0,
             "headphones": True, "calendar_event_in_mins": 15, "weekend": False}
    r1 = enc.encode(event)
    r2 = enc.encode(event)
    assert np.allclose(r1, r2), "encode() is not deterministic — check for random dropout or random ops"


@check(2, "src.core.memory_manager importable",
       "Create src/core/memory_manager.py with MemoryManager class. "
       "It must import BehaviouralGraph from src.core.graph_engine. "
       "The __init__ must create SQLite connection at COLD_DB_PATH.")
def check_memory_manager_import():
    from src.core import memory_manager
    assert hasattr(memory_manager, 'MemoryManager')


@check(2, "MemoryManager.promote_to_hot() works correctly",
       "promote_to_hot(node_id) must: 1) check if already in HOT (return True if yes), "
       "2) if HOT full, evict LRU to WARM first, 3) add node_id to HOT dict, "
       "4) publish TOPIC_NODE_PROMOTED. "
       "If node_id not found anywhere, return False.")
def check_memory_manager_hot_promote():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    from src.core.memory_manager import MemoryManager
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    node = GraphNode("n1", np.zeros(64), "com.test.app", 5, 2,
                     {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social")
    g.add_node(node)
    mm = MemoryManager("user_test", g)
    result = mm.promote_to_hot("n1")
    assert result is True, "promote_to_hot returned False for a valid node"
    assert mm.is_in_hot("n1"), "Node not in HOT after promote_to_hot"
    EventBus.get_instance().clear_all()


@check(2, "MemoryManager respects HOT_TIER_CAPACITY limit",
       "When HOT tier reaches HOT_TIER_CAPACITY (30) nodes, adding one more must "
       "evict the least-recently-used node to WARM first. "
       "After eviction, HOT count must stay at HOT_TIER_CAPACITY (not exceed it).")
def check_memory_manager_hot_capacity():
    import numpy as np
    from config.settings import HOT_TIER_CAPACITY
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    from src.core.memory_manager import MemoryManager
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_cap_test")
    mm = MemoryManager("user_cap_test", g)
    # Fill HOT beyond capacity
    for i in range(HOT_TIER_CAPACITY + 5):
        nid = f"node_{i}"
        g.add_node(GraphNode(nid, np.zeros(64), "app", 5, 2,
                             {"headphones": False, "calendar_near": False, "weekend": False}, i, 1, "social"))
        mm.promote_to_hot(nid)
    stats = mm.get_tier_stats()
    assert stats["hot_count"] <= HOT_TIER_CAPACITY, \
        f"HOT tier exceeded capacity: {stats['hot_count']} > {HOT_TIER_CAPACITY}"
    EventBus.get_instance().clear_all()


@check(2, "MemoryManager.demote_from_hot() moves node to WARM",
       "demote_from_hot(node_id) must move the node from HOT dict to WARM OrderedDict. "
       "After demotion, is_in_hot() must return False and is_in_warm() must return True.")
def check_memory_manager_demote():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    from src.core.memory_manager import MemoryManager
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    node = GraphNode("n1", np.zeros(64), "com.test.app", 5, 2,
                     {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social")
    g.add_node(node)
    mm = MemoryManager("user_test", g)
    mm.promote_to_hot("n1")
    result = mm.demote_from_hot("n1")
    assert result is True
    assert not mm.is_in_hot("n1"), "Node still in HOT after demotion"
    assert mm.is_in_warm("n1"), "Node not in WARM after demotion"
    EventBus.get_instance().clear_all()


@check(2, "MemoryManager.flush_hot_by_category() removes only matching category nodes",
       "flush_hot_by_category('financial') must remove ALL nodes in HOT whose "
       "GraphNode.category == 'financial'. Nodes of other categories must remain. "
       "Returns list of flushed node_ids.")
def check_memory_manager_flush_by_category():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    from src.core.memory_manager import MemoryManager
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    fin_node = GraphNode("fin_1", np.zeros(64), "com.hdfcbank.new", 5, 2,
                         {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "financial")
    soc_node = GraphNode("soc_1", np.zeros(64), "com.instagram.android", 8, 2,
                         {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "social")
    g.add_node(fin_node)
    g.add_node(soc_node)
    mm = MemoryManager("user_test", g)
    mm.promote_to_hot("fin_1")
    mm.promote_to_hot("soc_1")
    flushed = mm.flush_hot_by_category("financial")
    assert "fin_1" in flushed, "Financial node not flushed"
    assert not mm.is_in_hot("fin_1"), "Financial node still in HOT"
    assert mm.is_in_hot("soc_1"), "Social node incorrectly flushed"
    EventBus.get_instance().clear_all()


@check(2, "MemoryManager.get_tier_stats() returns correct schema",
       "get_tier_stats() must return a dict with keys: "
       "hot_count, warm_count, cold_count, hot_capacity, warm_capacity. "
       "hot_capacity must equal HOT_TIER_CAPACITY (30), warm_capacity must equal WARM_TIER_CAPACITY (150).")
def check_memory_manager_tier_stats():
    import numpy as np
    from config.settings import HOT_TIER_CAPACITY, WARM_TIER_CAPACITY
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph
    from src.core.memory_manager import MemoryManager
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    mm = MemoryManager("user_test", g)
    stats = mm.get_tier_stats()
    assert set(stats.keys()) == {"hot_count", "warm_count", "cold_count", "hot_capacity", "warm_capacity"}
    assert stats["hot_capacity"] == HOT_TIER_CAPACITY
    assert stats["warm_capacity"] == WARM_TIER_CAPACITY
    EventBus.get_instance().clear_all()


@check(2, "src.data.event_simulator importable and loads user file",
       "Create src/data/event_simulator.py with EventSimulator class. "
       "__init__(user_id) must load events from data/synthetic/users/{user_id}.json. "
       "Raises FileNotFoundError if file doesn't exist.")
def check_event_simulator_import():
    from src.data.event_simulator import EventSimulator
    sim = EventSimulator("user_00")
    assert len(sim.events) > 0, "EventSimulator loaded 0 events"


@check(2, "EventSimulator loads events correctly and initializes current_day",
       "EventSimulator should initialize current_day to 0 and load events list.")
def check_event_simulator_loads():
    from src.data.event_simulator import EventSimulator
    sim = EventSimulator("user_00")
    assert sim.current_day == 0, "EventSimulator should start current_day at 0"
    assert isinstance(sim.events, list), "sim.events should be a list of events"
    assert len(sim.events) > 0, "sim.events should not be empty"


@check(2, "EventSimulator.step() publishes TOPIC_APP_LAUNCHED",
       "step() must call EventBus.get_instance().publish(TOPIC_APP_LAUNCHED, event_payload). "
       "The payload must match the OS event schema (see Section 6 Contract 1). "
       "Subscribing to TOPIC_APP_LAUNCHED before calling step() should trigger the callback.")
def check_event_simulator_step_publishes():
    from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED
    from src.data.event_simulator import EventSimulator
    EventBus.get_instance().clear_all()
    received = []
    EventBus.get_instance().subscribe(TOPIC_APP_LAUNCHED, lambda p: received.append(p))
    sim = EventSimulator("user_00")
    result = sim.step()
    assert result is not None, "step() returned None on first call"
    assert len(received) == 1, f"Expected 1 published event, got {len(received)}"
    assert "app_id" in received[0], "Published event missing 'app_id' key"
    assert "battery" in received[0], "Published event missing 'battery' key"
    EventBus.get_instance().clear_all()


@check(2, "EventSimulator.step_day() advances day counter",
       "step_day() must process all events with event['day'] == self.current_day, "
       "then increment self.current_day. "
       "Calling step_day() twice must result in self.current_day == 2.")
def check_event_simulator_day_advance():
    from src.core.event_bus import EventBus
    from src.data.event_simulator import EventSimulator
    EventBus.get_instance().clear_all()
    sim = EventSimulator("user_00")
    assert sim.current_day == 0
    sim.step_day()
    assert sim.current_day == 1, f"Expected current_day=1 after step_day(), got {sim.current_day}"
    sim.step_day()
    assert sim.current_day == 2
    EventBus.get_instance().clear_all()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 CHECKS
# ─────────────────────────────────────────────────────────────────────────────

@check(3, "src.rl.reward importable with correct function signatures",
       "Create src/rl/reward.py with compute_reward() and compute_episode_summary() functions. "
       "Ensure src/rl/__init__.py exists.")
def check_reward_import():
    from src.rl import reward
    assert hasattr(reward, 'compute_reward')
    assert hasattr(reward, 'compute_episode_summary')


@check(3, "compute_reward() returns positive value for good cache hit",
       "With cache_hits=9, cache_misses=1, thrash_events=0, battery_consumed=0.5, "
       "friction_saved=8, the reward should be positive (good performance). "
       "Verify formula: R = α*hit_rate + β*speed - γ*thrash - δ*battery + ε*friction")
def check_reward_compute_positive():
    from src.rl.reward import compute_reward
    r = compute_reward(cache_hits=9, cache_misses=1, thrash_events=0,
                       battery_consumed=0.5, friction_saved=8, step_duration_seconds=1.0)
    assert isinstance(r, float), f"compute_reward must return float, got {type(r)}"
    assert r > 0, f"Expected positive reward for good performance, got {r}"


@check(3, "compute_reward() penalizes high thrash events",
       "With high thrash (10 events) and low cache hits, reward must be less than "
       "the same scenario with 0 thrash events. The γ*thrash_rate term must reduce reward.")
def check_reward_penalizes_thrash():
    from src.rl.reward import compute_reward
    no_thrash = compute_reward(5, 5, 0, 1.0, 3, 1.0)
    high_thrash = compute_reward(5, 5, 10, 1.0, 3, 1.0)
    assert high_thrash < no_thrash, \
        f"High thrash ({high_thrash:.3f}) should be less than no thrash ({no_thrash:.3f})"


@check(3, "compute_reward() penalizes high battery cost",
       "With battery_consumed=5.0 (max penalty) vs battery_consumed=0.0, "
       "the high battery version must have lower reward. δ*battery_cost must reduce reward.")
def check_reward_penalizes_battery():
    from src.rl.reward import compute_reward
    low_battery = compute_reward(5, 5, 0, 0.0, 3, 1.0)
    high_battery = compute_reward(5, 5, 0, 5.0, 3, 1.0)
    assert high_battery < low_battery, \
        f"High battery cost ({high_battery:.3f}) should be less than low ({low_battery:.3f})"


@check(3, "compute_episode_summary() returns correct schema",
       "compute_episode_summary(rewards) must return dict with keys: "
       "mean, min, max, total, steps. All values must be correct for input list.")
def check_reward_episode_summary():
    from src.rl.reward import compute_episode_summary
    rewards = [1.0, 2.0, 3.0, 0.5, -0.5]
    summary = compute_episode_summary(rewards)
    assert set(summary.keys()) == {"mean", "min", "max", "total", "steps"}
    assert summary["steps"] == 5
    assert abs(summary["mean"] - 1.2) < 0.001
    assert summary["min"] == -0.5
    assert summary["max"] == 3.0
    assert abs(summary["total"] - 6.0) < 0.001


@check(3, "src.rl.environment.GraphMindEnv importable",
       "Create src/rl/environment.py with GraphMindEnv class inheriting from gymnasium.Env. "
       "It must define observation_space (Box, shape=(68,)) and action_space (Discrete(31)).")
def check_rl_env_import():
    from src.rl.environment import GraphMindEnv
    assert GraphMindEnv is not None


@check(3, "GraphMindEnv instantiates without error for user_00",
       "GraphMindEnv('user_00') must construct successfully. "
       "It needs the dataset file to exist (Phase 1) and graph/memory manager to initialize. "
       "Check that all imports in environment.py match the Connector Registry.")
def check_rl_env_instantiate():
    from src.core.event_bus import EventBus
    from src.rl.environment import GraphMindEnv
    EventBus.get_instance().clear_all()
    env = GraphMindEnv("user_00")
    assert env is not None
    EventBus.get_instance().clear_all()


@check(3, "GraphMindEnv observation_space is Box of shape (68,)",
       "Set self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(68,), dtype=np.float32). "
       "The 68 dims: 35 context embedding + 30 HOT occupancy + 3 state signals.")
def check_rl_env_observation_space():
    import gymnasium
    import numpy as np
    from src.core.event_bus import EventBus
    from src.rl.environment import GraphMindEnv
    EventBus.get_instance().clear_all()
    env = GraphMindEnv("user_00")
    assert isinstance(env.observation_space, gymnasium.spaces.Box)
    assert env.observation_space.shape == (68,), f"Expected shape (68,), got {env.observation_space.shape}"
    EventBus.get_instance().clear_all()


@check(3, "GraphMindEnv action_space is Discrete(31)",
       "Set self.action_space = gymnasium.spaces.Discrete(31). "
       "Actions 0-28: node priority, 29: prune cycle, 30: emergency demote.")
def check_rl_env_action_space():
    import gymnasium
    from src.core.event_bus import EventBus
    from src.rl.environment import GraphMindEnv
    EventBus.get_instance().clear_all()
    env = GraphMindEnv("user_00")
    assert isinstance(env.action_space, gymnasium.spaces.Discrete)
    assert env.action_space.n == 31, f"Expected Discrete(31), got Discrete({env.action_space.n})"
    EventBus.get_instance().clear_all()


@check(3, "GraphMindEnv.reset() returns observation of shape (68,)",
       "reset() must return tuple (observation, info_dict). "
       "observation must be np.ndarray of shape (68,) and dtype float32.")
def check_rl_env_reset():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.rl.environment import GraphMindEnv
    EventBus.get_instance().clear_all()
    env = GraphMindEnv("user_00")
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray), f"Expected ndarray, got {type(obs)}"
    assert obs.shape == (68,), f"Expected shape (68,), got {obs.shape}"
    assert obs.dtype == np.float32
    EventBus.get_instance().clear_all()


@check(3, "GraphMindEnv.step() returns correct 5-tuple",
       "step(action) must return (obs, reward, terminated, truncated, info). "
       "obs: ndarray(68,), reward: float, terminated: bool, truncated: bool (always False), info: dict. "
       "info must have keys: cache_hits, cache_misses, day.")
def check_rl_env_step():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.rl.environment import GraphMindEnv
    EventBus.get_instance().clear_all()
    env = GraphMindEnv("user_00")
    env.reset()
    result = env.step(29)  # no-op action
    assert len(result) == 5, f"step() must return 5 values, got {len(result)}"
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (68,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert "cache_hits" in info
    env.close()
    EventBus.get_instance().clear_all()


@check(3, "src.rl.trainer importable",
       "Create src/rl/trainer.py with RLTrainer class. "
       "Ensure stable_baselines3 is installed: pip install stable-baselines3==2.3.2")
def check_rl_trainer_import():
    from src.rl.trainer import RLTrainer
    trainer = RLTrainer()
    assert trainer is not None


@check(3, "PPO model file exists for at least user_00",
       "Run: python scripts/train_rl.py --user user_00 --timesteps 50000 \n"
       "This creates models/rl_policies/user_00_ppo.zip. "
       "Reduce timesteps to 10000 for quick test. The model file must exist.")
def check_rl_model_exists():
    from config.settings import RL_MODELS_DIR
    path = os.path.join(RL_MODELS_DIR, "user_00_ppo.zip")
    assert os.path.exists(path), f"PPO model not found at {path}. Run training first."


@check(3, "Trained PPO model is loadable and can predict actions",
       "RLTrainer.load_policy('user_00') must return a stable_baselines3.PPO object. "
       "Calling model.predict(observation) with a valid obs must return (action, _states). "
       "action must be an integer in range [0, 30].")
def check_rl_model_loadable():
    import numpy as np
    from src.rl.trainer import RLTrainer
    trainer = RLTrainer()
    model = trainer.load_policy("user_00")
    assert model is not None, "load_policy returned None — model file may be corrupted"
    obs = np.zeros((68,), dtype=np.float32)
    action, _ = model.predict(obs)
    action = int(action)
    assert 0 <= action <= 30, f"Predicted action {action} out of valid range [0, 30]"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 CHECKS
# ─────────────────────────────────────────────────────────────────────────────

@check(4, "src.prefetch.daemon importable",
       "Create src/prefetch/daemon.py with PrefetchDaemon class. "
       "Ensure src/prefetch/__init__.py exists. "
       "Do NOT start the scheduler in __init__(), only in start().")
def check_prefetch_daemon_import():
    from src.prefetch.daemon import PrefetchDaemon
    assert PrefetchDaemon is not None


@check(4, "PrefetchDaemon instantiates without starting scheduler",
       "PrefetchDaemon(user_id, graph, memory_manager) must create the object "
       "WITHOUT starting APScheduler. self.scheduler must be None after __init__. "
       "Only start() should initiate scheduling.")
def check_prefetch_daemon_instantiate():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph
    from src.core.memory_manager import MemoryManager
    from src.prefetch.daemon import PrefetchDaemon
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    mm = MemoryManager("user_test", g)
    daemon = PrefetchDaemon("user_test", g, mm)
    assert daemon.scheduler is None, \
        "scheduler must be None before start() is called — do not auto-start in __init__"
    EventBus.get_instance().clear_all()


@check(4, "src.security.context_boundary importable",
       "Create src/security/context_boundary.py with ContextBoundaryEnforcer class. "
       "Ensure src/security/__init__.py exists.")
def check_context_boundary_import():
    from src.security.context_boundary import ContextBoundaryEnforcer
    assert ContextBoundaryEnforcer is not None


@check(4, "ContextBoundaryEnforcer correctly detects sensitive→consumer transition",
       "check_transition('financial', 'social') must return True. "
       "check_transition('health', 'entertainment') must return True. "
       "These are sensitive→consumer transitions that require cache flushing.")
def check_context_boundary_transition_detect():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph
    from src.core.memory_manager import MemoryManager
    from src.security.context_boundary import ContextBoundaryEnforcer
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    mm = MemoryManager("user_test", g)
    enforcer = ContextBoundaryEnforcer("user_test", mm)
    assert enforcer.check_transition("financial", "social") is True
    assert enforcer.check_transition("health", "entertainment") is True
    assert enforcer.check_transition("enterprise", "shopping") is True
    EventBus.get_instance().clear_all()


@check(4, "ContextBoundaryEnforcer.enforce_boundary() flushes HOT and logs event",
       "enforce_boundary('financial', 'social', 1000.0) must: "
       "1) call memory_manager.flush_hot_by_category() for each SENSITIVE category, "
       "2) return a flush_event dict (not None), "
       "3) append to self.flush_log, "
       "4) publish TOPIC_SECURITY_FLUSH. "
       "Test: add financial node to HOT, trigger boundary, verify it's gone from HOT.")
def check_context_boundary_flush_correct():
    import numpy as np
    from src.core.event_bus import EventBus, TOPIC_SECURITY_FLUSH
    from src.core.graph_engine import BehaviouralGraph, GraphNode
    from src.core.memory_manager import MemoryManager
    from src.security.context_boundary import ContextBoundaryEnforcer
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    fin_node = GraphNode("fin_1", np.zeros(64), "com.hdfcbank.new", 5, 2,
                         {"headphones": False, "calendar_near": False, "weekend": False}, 0, 1, "financial")
    g.add_node(fin_node)
    mm = MemoryManager("user_test", g)
    mm.promote_to_hot("fin_1")
    enforcer = ContextBoundaryEnforcer("user_test", mm)
    flush_events_received = []
    EventBus.get_instance().subscribe(TOPIC_SECURITY_FLUSH, lambda p: flush_events_received.append(p))
    result = enforcer.enforce_boundary("financial", "social", 1000.0)
    assert result is not None, "enforce_boundary returned None — should return flush_event dict"
    assert not mm.is_in_hot("fin_1"), "Financial node still in HOT after boundary enforcement"
    assert len(enforcer.get_flush_log()) == 1
    assert len(flush_events_received) == 1, "TOPIC_SECURITY_FLUSH not published"
    EventBus.get_instance().clear_all()


@check(4, "ContextBoundaryEnforcer does NOT flush on non-sensitive transitions",
       "check_transition('social', 'financial') must return False (consumer→sensitive is not a threat). "
       "check_transition('entertainment', 'productivity') must return False. "
       "Only SENSITIVE→CONSUMER triggers flush, not the reverse.")
def check_context_boundary_no_false_positive():
    import numpy as np
    from src.core.event_bus import EventBus
    from src.core.graph_engine import BehaviouralGraph
    from src.core.memory_manager import MemoryManager
    from src.security.context_boundary import ContextBoundaryEnforcer
    EventBus.get_instance().clear_all()
    g = BehaviouralGraph("user_test")
    mm = MemoryManager("user_test", g)
    enforcer = ContextBoundaryEnforcer("user_test", mm)
    assert enforcer.check_transition("social", "financial") is False, \
        "Consumer→sensitive should NOT trigger flush (no sensitive data in HOT to protect)"
    assert enforcer.check_transition("entertainment", "productivity") is False
    assert enforcer.check_transition("social", "social") is False
    EventBus.get_instance().clear_all()


@check(4, "src.agents.drift_detector_agent importable",
       "Create src/agents/drift_detector_agent.py with DriftDetectorAgent class. "
       "Ensure src/agents/__init__.py exists.")
def check_drift_detector_import():
    from src.agents.drift_detector_agent import DriftDetectorAgent
    assert DriftDetectorAgent is not None


@check(4, "DriftDetectorAgent.compute_kl_divergence() returns 0.0 with no data",
       "With empty transition_history and recent_window, compute_kl_divergence() "
       "must return 0.0 (insufficient data). Do not raise an exception.")
def check_drift_detector_zero_data():
    from src.core.event_bus import EventBus
    from src.agents.drift_detector_agent import DriftDetectorAgent
    EventBus.get_instance().clear_all()
    agent = DriftDetectorAgent("user_test")
    result = agent.compute_kl_divergence()
    assert result == 0.0, f"Expected 0.0 with no data, got {result}"
    EventBus.get_instance().clear_all()


@check(4, "DriftDetectorAgent detects genuine distribution shift",
       "Fill transition_history with 100 events for app_A, app_B, app_C in equal proportion. "
       "Then fill recent_window with 100 events for only app_D, app_E (completely different). "
       "compute_kl_divergence() must return > DRIFT_KL_THRESHOLD (0.3).")
def check_drift_detector_divergent_data():
    from config.settings import DRIFT_KL_THRESHOLD
    from src.core.event_bus import EventBus, TOPIC_APP_LAUNCHED
    from src.agents.drift_detector_agent import DriftDetectorAgent
    EventBus.get_instance().clear_all()
    agent = DriftDetectorAgent("user_test")
    # Inject historical pattern (app_A, app_B, app_C cycling)
    for i in range(150):
        apps = ["com.appA.android", "com.appB.android", "com.appC.android"]
        agent.transition_history.append(apps[i % 3])
    # Inject recent drift (completely different apps)
    for i in range(100):
        apps = ["com.appD.android", "com.appE.android"]
        agent.recent_window.append(apps[i % 2])
    kl = agent.compute_kl_divergence()
    assert kl > DRIFT_KL_THRESHOLD, \
        f"KL divergence {kl:.3f} not > threshold {DRIFT_KL_THRESHOLD} — drift not detected"
    EventBus.get_instance().clear_all()


@check(4, "src.agents.orchestrator importable",
       "Create src/agents/orchestrator.py with GraphMindState TypedDict and "
       "GraphMindOrchestrator class. Ensure langgraph is installed: pip install langgraph==0.1.14")
def check_orchestrator_import():
    from src.agents.orchestrator import GraphMindOrchestrator, GraphMindState
    assert GraphMindOrchestrator is not None


@check(4, "GraphMindOrchestrator instantiates without error for user_00",
       "GraphMindOrchestrator('user_00') must initialize all 5 agents and build the LangGraph. "
       "Common failure: agent __init__ raises because a dependency (graph, memory_manager) "
       "is not passed correctly. Check orchestrator.__init__ creates all dependencies in order.")
def check_orchestrator_instantiate():
    from src.core.event_bus import EventBus
    from src.agents.orchestrator import GraphMindOrchestrator
    EventBus.get_instance().clear_all()
    orch = GraphMindOrchestrator("user_00")
    assert orch is not None
    EventBus.get_instance().clear_all()


@check(4, "GraphMindOrchestrator.run_day() returns GraphMindState with all keys",
       "run_day(0) must invoke the LangGraph state machine and return a dict "
       "with all required GraphMindState keys: user_id, current_day, kl_divergence, "
       "cache_hit_rate, security_flush_count, last_agent, messages.")
def check_orchestrator_run_day():
    from src.core.event_bus import EventBus
    from src.agents.orchestrator import GraphMindOrchestrator
    EventBus.get_instance().clear_all()
    orch = GraphMindOrchestrator("user_00")
    state = orch.run_day(0)
    required = {"user_id", "current_day", "kl_divergence", "cache_hit_rate",
                "security_flush_count", "last_agent", "messages"}
    assert required.issubset(set(state.keys())), \
        f"Missing state keys: {required - set(state.keys())}"
    assert state["user_id"] == "user_00"
    assert state["current_day"] == 0
    EventBus.get_instance().clear_all()


@check(4, "GraphMindState TypedDict has correct field types",
       "GraphMindState must be a TypedDict with fields: "
       "user_id(str), current_day(int), current_event(dict|None), battery(float), "
       "kl_divergence(float), cache_hit_rate(float), security_flush_count(int), "
       "last_agent(str), messages(list). "
       "Check that all agent run() functions return updated state with all keys preserved.")
def check_orchestrator_state_schema():
    from src.agents.orchestrator import GraphMindState
    import typing
    hints = typing.get_type_hints(GraphMindState)
    required = {"user_id", "current_day", "current_event", "battery", "kl_divergence",
                "cache_hit_rate", "security_flush_count", "last_agent", "messages"}
    missing = required - set(hints.keys())
    assert not missing, f"GraphMindState missing fields: {missing}"


@check(4, "LangGraph graph is built with correct 5 nodes",
       "orchestrator.build_graph() must create a StateGraph with nodes: "
       "'graph_manager', 'rl_trainer', 'prefetch', 'drift_detector', 'security'. "
       "The compiled graph must be stored as self.compiled_graph.")
def check_langgraph_graph_built():
    from src.core.event_bus import EventBus
    from src.agents.orchestrator import GraphMindOrchestrator
    EventBus.get_instance().clear_all()
    orch = GraphMindOrchestrator("user_00")
    assert hasattr(orch, 'compiled_graph'), "orchestrator missing 'compiled_graph' attribute"
    assert orch.compiled_graph is not None
    EventBus.get_instance().clear_all()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 CHECKS
# ─────────────────────────────────────────────────────────────────────────────

@check(5, "src.benchmarks.baselines importable with all 4 baseline classes",
       "Create src/benchmarks/baselines.py with: BaselinePolicy, LMKDReactiveBaseline, "
       "ARTStaticProfileBaseline, UsageStatsLRUBaseline, BixbyFrequencyBaseline. "
       "Ensure src/benchmarks/__init__.py exists.")
def check_baselines_import():
    from src.benchmarks import baselines
    for cls in ["BaselinePolicy", "LMKDReactiveBaseline", "ARTStaticProfileBaseline",
                "UsageStatsLRUBaseline", "BixbyFrequencyBaseline"]:
        assert hasattr(baselines, cls), f"Missing class: {cls}"


@check(5, "LMKDReactiveBaseline.predict_next_apps() returns list of app_ids",
       "LMKDReactiveBaseline must track recency. After update() with 3 apps, "
       "predict_next_apps() must return a list of string app_ids (not empty).")
def check_baseline_lmkd():
    from src.benchmarks.baselines import LMKDReactiveBaseline
    bl = LMKDReactiveBaseline()
    for app in ["com.instagram.android", "com.whatsapp", "com.spotify.music"]:
        bl.update({"app_id": app, "time_bucket": 10, "battery": 80.0, "day": 1, "weekend": False})
    result = bl.predict_next_apps("com.instagram.android", {"time_bucket": 10, "battery": 80.0})
    assert isinstance(result, list), "predict_next_apps must return a list"
    assert len(result) > 0, "predict_next_apps returned empty list"
    assert all(isinstance(a, str) for a in result), "All predictions must be strings"


@check(5, "ARTStaticProfileBaseline profile is FROZEN after Day 7",
       "ARTStaticProfileBaseline must call build_profile() on first 7 days of events. "
       "After build_profile(), the predictions must NOT change even if update() is called "
       "with new events (simulate ART's static AOT profile behavior).")
def check_baseline_art():
    from src.benchmarks.baselines import ARTStaticProfileBaseline
    bl = ARTStaticProfileBaseline()
    # Build profile from days 0-6
    events = []
    for day in range(7):
        for _ in range(10):
            events.append({"app_id": "com.instagram.android", "time_bucket": 10,
                           "battery": 80.0, "day": day, "weekend": False})
    bl.build_profile(events)
    predictions_before = bl.predict_next_apps("any", {"time_bucket": 10, "battery": 80.0})
    # Add new events (post-profile)
    for _ in range(20):
        bl.update({"app_id": "com.totally.different.app", "time_bucket": 10, "battery": 80.0, "day": 15, "weekend": False})
    predictions_after = bl.predict_next_apps("any", {"time_bucket": 10, "battery": 80.0})
    assert predictions_before == predictions_after, \
        "ART profile predictions changed after build — profile must be FROZEN"


@check(5, "UsageStatsLRUBaseline updates and predicts from recency",
       "After 5 updates, predict_next_apps must return top-5 most recently used apps. "
       "The most recently used app should appear first in predictions.")
def check_baseline_lru():
    from src.benchmarks.baselines import UsageStatsLRUBaseline
    bl = UsageStatsLRUBaseline()
    apps = ["app_a", "app_b", "app_c", "app_d", "app_e"]
    for app in apps:
        bl.update({"app_id": app, "time_bucket": 5, "battery": 60.0, "day": 1, "weekend": False})
    result = bl.predict_next_apps("app_e", {"time_bucket": 5, "battery": 60.0})
    assert len(result) > 0
    assert result[0] == "app_e", f"Most recent app should be first, got {result[0]}"


@check(5, "BixbyFrequencyBaseline uses time+day context for predictions",
       "BixbyFrequencyBaseline must track frequency per (time_bucket, weekend) pair. "
       "After 20 events all at time_bucket=10 on weekdays, predicting at time_bucket=10 weekday "
       "must return those apps. Predicting at time_bucket=20 should return different/empty results.")
def check_baseline_bixby():
    from src.benchmarks.baselines import BixbyFrequencyBaseline
    bl = BixbyFrequencyBaseline()
    for _ in range(20):
        bl.update({"app_id": "com.instagram.android", "time_bucket": 10,
                   "battery": 80.0, "day": 1, "weekend": False})
    result_morning = bl.predict_next_apps("any", {"time_bucket": 10, "battery": 80.0, "weekend": False})
    result_night = bl.predict_next_apps("any", {"time_bucket": 40, "battery": 80.0, "weekend": False})
    assert "com.instagram.android" in result_morning, "Should predict instagram at morning time bucket"
    # Night predictions should NOT be driven by morning data
    assert result_morning != result_night or len(result_night) == 0, \
        "Bixby should have different predictions for different time buckets"


@check(5, "src.benchmarks.evaluator importable",
       "Create src/benchmarks/evaluator.py with BenchmarkEvaluator class. "
       "Ensure all 4 baseline classes import correctly from src.benchmarks.baselines.")
def check_evaluator_import():
    from src.benchmarks.evaluator import BenchmarkEvaluator
    evaluator = BenchmarkEvaluator()
    assert evaluator is not None


@check(5, "results/benchmark_results.csv exists with correct structure",
       "Run: python scripts/run_benchmarks.py \n"
       "This creates results/benchmark_results.csv. "
       "Must have 50+ rows (10 users × 5 policies × at least 1 day each) and required columns.")
def check_benchmark_results_exist():
    import pandas as pd
    from config.settings import RESULTS_DIR
    path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    assert os.path.exists(path), f"Benchmark results not found at {path}. Run: python scripts/run_benchmarks.py"
    df = pd.read_csv(path)
    assert len(df) >= 50, f"Expected >= 50 rows, got {len(df)}"


@check(5, "benchmark_results.csv has all required columns",
       "The CSV must have columns: user_id, policy_name, day, cache_hit_rate, "
       "launch_speed_gain_pct, thrash_rate, battery_overhead_pct, graph_node_count. "
       "Check BenchmarkEvaluator.run_all() return value and CSV save logic.")
def check_benchmark_results_schema():
    import pandas as pd
    from config.settings import RESULTS_DIR
    path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    df = pd.read_csv(path)
    required_cols = {"user_id", "policy_name", "day", "cache_hit_rate",
                     "launch_speed_gain_pct", "thrash_rate", "battery_overhead_pct"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing CSV columns: {missing}"
    policies = set(df["policy_name"].unique())
    from config.settings import BASELINE_LMKD, BASELINE_ART, BASELINE_LRU, BASELINE_BIXBY, BASELINE_GRAPHMIND
    expected_policies = {BASELINE_LMKD, BASELINE_ART, BASELINE_LRU, BASELINE_BIXBY, BASELINE_GRAPHMIND}
    missing_policies = expected_policies - policies
    assert not missing_policies, f"Missing policies in results: {missing_policies}"


@check(5, "GraphMind outperforms LMKD baseline in cache hit rate (>=8/10 users)",
       "This is a critical KPI check. If GraphMind is NOT outperforming LMKD for >= 8 users, "
       "check: 1) Is the RL policy actually trained (models/rl_policies/*.zip exist)? "
       "2) Is the pre-fetch daemon running during simulation? "
       "3) Are edge weights being updated by RL? Check src/rl/environment.py step() logic.")
def check_graphmind_beats_lmkd():
    import pandas as pd
    from config.settings import RESULTS_DIR, BASELINE_LMKD, BASELINE_GRAPHMIND
    path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    df = pd.read_csv(path)
    per_user_avg = df.groupby(["user_id", "policy_name"])["cache_hit_rate"].mean().reset_index()
    users = df["user_id"].unique()
    wins = 0
    for user in users:
        gm_rate = per_user_avg[(per_user_avg["user_id"] == user) & (per_user_avg["policy_name"] == BASELINE_GRAPHMIND)]["cache_hit_rate"].values
        lm_rate = per_user_avg[(per_user_avg["user_id"] == user) & (per_user_avg["policy_name"] == BASELINE_LMKD)]["cache_hit_rate"].values
        if len(gm_rate) > 0 and len(lm_rate) > 0 and gm_rate[0] > lm_rate[0]:
            wins += 1
    assert wins >= 8, \
        f"GraphMind only beats LMKD for {wins}/10 users. Need >= 8. Check RL training quality."


@check(5, "GraphMind outperforms Bixby baseline in cache hit rate (>=8/10 users)",
       "If GraphMind is NOT beating Bixby, check that the RL agent is using behavioral graph "
       "transition patterns. Bixby uses frequency per time-bucket — your graph uses actual "
       "transition chains which should be strictly more information. "
       "Check that prefetch daemon is promoting correct nodes based on graph edge weights.")
def check_graphmind_beats_bixby():
    import pandas as pd
    from config.settings import RESULTS_DIR, BASELINE_BIXBY, BASELINE_GRAPHMIND
    path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    df = pd.read_csv(path)
    per_user_avg = df.groupby(["user_id", "policy_name"])["cache_hit_rate"].mean().reset_index()
    users = df["user_id"].unique()
    wins = 0
    for user in users:
        gm = per_user_avg[(per_user_avg["user_id"] == user) & (per_user_avg["policy_name"] == BASELINE_GRAPHMIND)]["cache_hit_rate"].values
        bx = per_user_avg[(per_user_avg["user_id"] == user) & (per_user_avg["policy_name"] == BASELINE_BIXBY)]["cache_hit_rate"].values
        if len(gm) > 0 and len(bx) > 0 and gm[0] > bx[0]:
            wins += 1
    assert wins >= 8, f"GraphMind only beats Bixby for {wins}/10 users. Need >= 8."


@check(5, "Security flushes are recorded in simulation logs",
       "Each user's simulation log must have at least some security flush events. "
       "If flush_count is 0 for ALL users, check: 1) Does the synthetic dataset contain "
       "transitions from financial/health/enterprise apps to social apps? "
       "2) Is ContextBoundaryEnforcer subscribed to TOPIC_APP_LAUNCHED? "
       "3) Is SecurityAgent being called in the LangGraph pipeline?")
def check_security_flushes_recorded():
    from config.settings import RESULTS_DIR, NUM_USERS
    total_flushes = 0
    users_with_flushes = 0
    for i in range(NUM_USERS):
        log_path = os.path.join(RESULTS_DIR, f"user_{i:02d}_simulation_log.json")
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            log = json.load(f)
        user_flushes = sum(d.get("state", {}).get("security_flush_count", 0) for d in log.get("days", []))
        if user_flushes > 0:
            users_with_flushes += 1
        total_flushes += user_flushes
    assert users_with_flushes >= 5, \
        f"Only {users_with_flushes}/10 users have security flush events. " \
        f"Ensure synthetic dataset has sensitive→consumer app transitions."


@check(5, "Graph node count stays < 1000 for all users after 30 days",
       "If graph grows beyond 1000 nodes, check: 1) Is prune_weak_edges() being called? "
       "2) Is evict_stale_nodes() being called? 3) Is GraphManagerAgent calling prune on day % 7? "
       "Reduce NODE_EVICTION_DAYS or lower EDGE_PRUNE_THRESHOLD if still too large.")
def check_graph_node_count_stable():
    from config.settings import RESULTS_DIR, NUM_USERS
    for i in range(NUM_USERS):
        log_path = os.path.join(RESULTS_DIR, f"user_{i:02d}_simulation_log.json")
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            log = json.load(f)
        last_day = log.get("days", [{}])[-1]
        node_count = last_day.get("graph_snapshot", {}).get("node_count", 0)
        assert node_count < 1000, \
            f"user_{i:02d} graph has {node_count} nodes after 30 days (limit: 1000). " \
            f"Check pruning and eviction are running."


@check(5, "Simulation log files exist for all 10 users",
       "Run: for i in $(seq -f '%02g' 0 9); do python scripts/run_simulation.py --user user_${i}; done \n"
       "Each user must have a results/user_XX_simulation_log.json file.")
def check_simulation_logs_exist():
    from config.settings import RESULTS_DIR, NUM_USERS
    for i in range(NUM_USERS):
        path = os.path.join(RESULTS_DIR, f"user_{i:02d}_simulation_log.json")
        assert os.path.exists(path), f"Missing simulation log: {path}"


@check(5, "src.dashboard.app importable",
       "Create src/dashboard/app.py. It must import without error when loaded as a module. "
       "Do NOT put streamlit calls at module level — wrap them in if __name__ == '__main__' "
       "or inside functions that are only called by streamlit. "
       "All imports in dashboard/app.py must follow the Connector Registry.")
def check_dashboard_importable():
    spec = importlib.util.spec_from_file_location("dashboard_app",
                                                   PROJECT_ROOT / "src" / "dashboard" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass  # Streamlit may call sys.exit — that's OK for this check


# ─────────────────────────────────────────────────────────────────────────────
# SUBMISSION CHECKS
# ─────────────────────────────────────────────────────────────────────────────

@check(6, "README.md exists and is not the template placeholder",
       "Fill in README.md with: problem statement number (3), team name (GraphMind), "
       "member names, institute, video links (must be real YouTube URLs), "
       "model links (must be real HuggingFace URLs), dataset links. "
       "Remove all placeholder text like '*Member 1 Name*'.")
def check_readme_filled():
    readme_path = PROJECT_ROOT / "README.md"
    assert readme_path.exists(), "README.md not found in project root"
    content = readme_path.read_text()
    assert "GraphMind" in content, "README.md must mention project name 'GraphMind'"
    assert "*Member 1 Name*" not in content, "README.md still has placeholder '*Member 1 Name*' — fill it in"
    assert "Problem Statement Number" in content or "03" in content, "README.md must include problem statement number"


@check(6, "LICENSE file (Apache 2.0) exists",
       "Create a LICENSE file in the project root with Apache 2.0 text. "
       "Guide: https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-license-to-a-repository "
       "All code must be released under Apache 2.0 per hackathon rules.")
def check_license_exists():
    license_path = PROJECT_ROOT / "LICENSE"
    assert license_path.exists(), "LICENSE file not found"
    content = license_path.read_text()
    assert "Apache" in content, "LICENSE must be Apache 2.0"


@check(6, "docs/ folder has all required files",
       "Create docs/ folder with: architecture.md, installation.md, user_guide.md, ax.md. "
       "ax.md is SPECIFICALLY REQUIRED by the submission template. "
       "Each file must be non-empty (> 200 chars).")
def check_docs_folder():
    docs_path = PROJECT_ROOT / "docs"
    assert docs_path.exists(), "docs/ folder not found"
    required = ["architecture.md", "installation.md", "user_guide.md", "ax.md"]
    for f in required:
        p = docs_path / f
        assert p.exists(), f"Missing required doc: docs/{f}"
        assert len(p.read_text()) > 200, f"docs/{f} is too short — must have substantial content"


@check(6, "docs/ax.md covers all 8 agentic practice areas",
       "docs/ax.md must mention all 8 agentic practices from the hackathon guidelines: "
       "agentic workflows, reasoning & planning, tool use/chaining, coding assistants, "
       "MCP servers, memory/context handling, multi-agent orchestration, practical problem-solving. "
       "Also must include 'What Worked' and 'What Did NOT Work' sections.")
def check_ax_md_exists():
    ax_path = PROJECT_ROOT / "docs" / "ax.md"
    content = ax_path.read_text().lower()
    required_topics = ["agentic workflow", "reasoning", "tool", "memory", "multi-agent", "orchestrat"]
    missing = [t for t in required_topics if t not in content]
    assert not missing, f"docs/ax.md missing topics: {missing}"
    assert "did not work" in content or "what did not work" in content, \
        "docs/ax.md must have a 'What Did NOT Work' section — judges specifically look for this"


@check(6, "agents.md exists in project root",
       "Create agents.md in the project root (not in docs/). "
       "This is the agentic AI practices summary for the submission. "
       "Content can mirror docs/ax.md.")
def check_agents_md_exists():
    agents_md = PROJECT_ROOT / "agents.md"
    assert agents_md.exists(), "agents.md not found in project root"
    assert len(agents_md.read_text()) > 200, "agents.md must have substantial content"


@check(6, "src/ folder has correct structure with all __init__.py files",
       "Every directory under src/ must have an __init__.py file for Python package imports. "
       "Missing __init__.py causes ModuleNotFoundError. "
       "Check: src/, src/core/, src/data/, src/rl/, src/prefetch/, src/security/, "
       "src/agents/, src/benchmarks/, src/dashboard/")
def check_src_folder_structure():
    required_inits = [
        "src/__init__.py", "src/core/__init__.py", "src/data/__init__.py",
        "src/rl/__init__.py", "src/prefetch/__init__.py", "src/security/__init__.py",
        "src/agents/__init__.py", "src/benchmarks/__init__.py", "src/dashboard/__init__.py"
    ]
    missing = [p for p in required_inits if not (PROJECT_ROOT / p).exists()]
    assert not missing, f"Missing __init__.py files: {missing}"


@check(6, "No circular imports between modules",
       "Circular imports cause ImportError at runtime. "
       "The import hierarchy must follow the tier order in the Connector Registry (Section 3): "
       "config → core → data → rl → prefetch/security → agents → benchmarks → dashboard. "
       "A tier must NEVER import from a higher tier. "
       "If you get ImportError, trace which module is importing which and fix the direction.")
def check_no_circular_imports():
    modules_to_check = [
        "config.settings",
        "src.core.event_bus",
        "src.core.graph_engine",
        "src.core.memory_manager",
        "src.data.context_encoder",
        "src.data.event_simulator",
        "src.rl.reward",
        "src.prefetch.daemon",
        "src.security.context_boundary",
        "src.benchmarks.baselines",
    ]
    for mod_name in modules_to_check:
        try:
            importlib.import_module(mod_name)
        except ImportError as e:
            raise AssertionError(f"Circular import or missing dependency in {mod_name}: {e}")


@check(6, "All public functions in src/ have docstrings",
       "Every function and class defined in src/ must have a docstring (triple-quoted string "
       "immediately after the def/class line). Judges and the hardchecker verify this. "
       "Use: grep -rn 'def ' src/ | grep -v '__' to find functions, then verify each has a docstring.")
def check_all_functions_have_docstrings():
    import ast
    violations = []
    for py_file in (PROJECT_ROOT / "src").rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError as e:
            violations.append(f"SyntaxError in {py_file}: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("__") and node.name.endswith("__"):
                    continue  # skip dunder methods
                if not (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant)):
                    violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{node.lineno} "
                                      f"function '{node.name}' missing docstring")
    assert not violations, "Missing docstrings:\n" + "\n".join(violations[:20])


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphMind Hard Checker")
    parser.add_argument("--phase", type=int, default=None,
                        help="Check only a specific phase (1-6). Omit to check all.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all PASS results, not just failures.")
    parser.add_argument("--fix-hints-only", action="store_true",
                        help="Only show fix instructions for failed checks.")
    args = parser.parse_args()

    run_checks(
        phase_filter=args.phase,
        verbose=args.verbose,
        fix_hints_only=args.fix_hints_only
    )
