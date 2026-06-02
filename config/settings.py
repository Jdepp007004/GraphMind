# config/settings.py
"""
GraphMind — Single source of truth for all project constants.
All other modules must import from here. No magic numbers elsewhere.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
USERS_DIR = os.path.join(SYNTHETIC_DIR, "users")
BASE_GRAPHS_DIR = os.path.join(DATA_DIR, "base_graphs")
APP_TAXONOMY_PATH = os.path.join(DATA_DIR, "app_taxonomy.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ── Dataset ────────────────────────────────────────────────────────────────
NUM_USERS = 10
SIMULATION_DAYS = 30
EVENTS_PER_DAY_MEAN = 80          # mean number of app events per simulated day
EVENTS_PER_DAY_STD = 20
RANDOM_SEED = 42

# ── Graph Engine ───────────────────────────────────────────────────────────
NODE_EMBEDDING_DIM = 64
EDGE_PRUNE_THRESHOLD = 0.05       # delete edge if transition prob < 5%
NODE_EVICTION_DAYS = 15           # evict node from COLD if inactive this many days
MAX_NODES_COLD = 2000             # hard cap on COLD graph size

# ── Memory Manager ─────────────────────────────────────────────────────────
HOT_TIER_CAPACITY = 30            # max nodes in HOT (simulated RAM)
WARM_TIER_CAPACITY = 150          # max nodes in WARM (simulated cache)
COLD_DB_PATH = os.path.join(DATA_DIR, "cold_graph.db")

# ── RL Training ────────────────────────────────────────────────────────────
PPO_TOTAL_TIMESTEPS = 200_000
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
RL_MODELS_DIR = os.path.join(MODELS_DIR, "rl_policies")

# Reward weights
REWARD_ALPHA = 1.0    # cache hit rate weight
REWARD_BETA = 0.8     # launch speed gain weight
REWARD_GAMMA = 0.5    # thrash penalty weight
REWARD_DELTA = 0.3    # battery cost weight
REWARD_EPSILON = 0.4  # friction saved weight

# ── Pre-fetch Daemon ───────────────────────────────────────────────────────
PREFETCH_INTERVAL_MINUTES = 15
PREFETCH_TOP_K = 5                # number of nodes to pre-warm each cycle
BATTERY_SUPPRESS_THRESHOLD = 20  # percent — suppress aggressive pre-fetch below this

# ── Drift Detection ────────────────────────────────────────────────────────
DRIFT_WINDOW_SIZE = 100           # number of recent transitions to track
DRIFT_KL_THRESHOLD = 0.3          # KL divergence above this triggers learning rate spike
DRIFT_LR_SPIKE_MULTIPLIER = 5.0   # multiply learning rate by this on drift

# ── Security ───────────────────────────────────────────────────────────────
UNKNOWN_SENSITIVE_CATEGORY = "unknown_sensitive"
SENSITIVE_CATEGORIES = ["financial", "health", "enterprise", "government", UNKNOWN_SENSITIVE_CATEGORY]
CONSUMER_CATEGORIES = ["social", "entertainment", "shopping", "gaming"]
# Transition from sensitive → consumer triggers cache flush
HOT_RETENTION_EVENTS = 500
WARM_RETENTION_EVENTS = 2000
COLD_RETENTION_DAYS = 15
TRACE_RETENTION_EVENTS = 1000
GRAPH_RETENTION_DAYS = NODE_EVICTION_DAYS

# ── Gemma Model ────────────────────────────────────────────────────────────
GEMMA_MODEL_ID = "google/gemma-2b"
GEMMA_LOCAL_PATH = os.path.join(MODELS_DIR, "gemma-2b")
GEMMA_MAX_NEW_TOKENS = 128
GEMMA_DEVICE = os.getenv("DEVICE", "cpu")

# ── Dashboard ──────────────────────────────────────────────────────────────
DASHBOARD_PORT = 8501
DASHBOARD_REFRESH_SECONDS = 5

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ── Baseline Names (used as dict keys throughout) ──────────────────────────
BASELINE_LMKD = "LMKD_Reactive"
BASELINE_ART = "ART_StaticProfile"
BASELINE_LRU = "UsageStats_LRU"
BASELINE_BIXBY = "Bixby_Frequency"
BASELINE_GRAPHMIND = "GraphMind_RL"
