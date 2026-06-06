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
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# ── Device Analyzer Dataset ────────────────────────────────────────────────
DEVICE_ANALYZER_DIR = os.path.join(DATA_DIR, "device_analyzer")
DEVICE_ANALYZER_RAW_DIR = os.path.join(DEVICE_ANALYZER_DIR, "raw")
DEVICE_ANALYZER_PROCESSED_DIR = os.path.join(DEVICE_ANALYZER_DIR, "processed")
DEVICE_ANALYZER_SPLITS_DIR = os.path.join(DEVICE_ANALYZER_DIR, "splits")
DEVICE_ANALYZER_URL = "https://deviceanalyzer.cl.cam.ac.uk/"  # Requires registration

# Dataset split ratios (must sum to 1.0)
DATASET_TRAIN_RATIO = 0.80
DATASET_VAL_RATIO = 0.10
DATASET_TEST_RATIO = 0.10

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
HOT_TIER_CAPACITY  = 5             # HOT tier: top-5 apps in RAM (matches benchmark HOT_SIZE=5)
WARM_TIER_CAPACITY = 15            # WARM tier: top-15 apps pre-loaded (matches benchmark WARM_SIZE=15)
COLD_DB_PATH = os.path.join(DATA_DIR, "cold_graph.db")

# ── RL Training ────────────────────────────────────────────────────────────
PPO_TOTAL_TIMESTEPS = 200_000
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
RL_MODELS_DIR = os.path.join(MODELS_DIR, "rl_policies")

# Reward weights (v1 — used by src/rl/reward.py)
REWARD_ALPHA = 1.0    # cache hit rate weight
REWARD_BETA = 0.8     # launch speed gain weight
REWARD_GAMMA = 0.5    # thrash penalty weight
REWARD_DELTA = 0.3    # battery cost weight
REWARD_EPSILON = 0.4  # friction saved weight
REWARD_ZETA = 0.3     # prefetch false positive penalty weight

# ── Reward V2 Weights (src/rl/reward_v2.py) ────────────────────────────────
# Positive reward components
REWARD_V2_HIT_RATE_WEIGHT = 2.0       # cache hit rate bonus (primary objective)
REWARD_V2_LATENCY_SAVED_WEIGHT = 1.0  # latency reduction reward (ms saved / 100)

# Negative reward components (penalties)
REWARD_V2_BATTERY_WEIGHT = 0.5        # battery overhead penalty
REWARD_V2_FALSE_PREFETCH_WEIGHT = 0.8 # false prefetch penalty
REWARD_V2_THRASH_WEIGHT = 1.2         # cache thrashing penalty (strong)

# Normalisation denominators for v2 reward
REWARD_V2_MAX_LATENCY_SAVED_MS = 800.0   # cold_start_ms approx. ceiling
REWARD_V2_MAX_BATTERY_OVERHEAD_PCT = 5.0 # 5% drain = max battery penalty
REWARD_V2_MAX_THRASH_PER_STEP = 10       # 10 thrash events = max thrash penalty

# ── Pre-fetch Daemon ───────────────────────────────────────────────────────
PREFETCH_INTERVAL_MINUTES = 15
PREFETCH_TOP_K = 5                # number of nodes to pre-warm each cycle
BATTERY_SUPPRESS_THRESHOLD = 20  # percent — suppress aggressive pre-fetch below this

# ── Confidence-Based Prefetch (src/prefetch/confidence_prefetch.py) ─────────
# GraphMindRL_V5 configuration — validated 2026-06-06
# Benchmark: F1=0.7745, ΔF1=+0.0321 over baseline, p=0.0115, 31 users
#
# confidence = W_TRANS*transition_prob + W_RECENCY*recency + W_FREQ*freq + W_CTX*context
#
# V5 finding: frequency was the dominant underweighted signal.
# Recency was overweighted. Context term zeroed out (hurts on 2-month datasets).
PREFETCH_CONFIDENCE_THRESHOLD = 0.16    # V5: was 0.05/0.70; adaptive ±0.005 on 20-step HR
PREFETCH_CONFIDENCE_W_TRANSITION = 0.50 # unchanged — transition prob is primary
PREFETCH_CONFIDENCE_W_RECENCY = 0.10    # V5: was 0.20 — recency overweighted historical data
PREFETCH_CONFIDENCE_W_FREQUENCY = 0.40  # V5: was 0.20 — frequency captures habitual use
PREFETCH_CONFIDENCE_W_CONTEXT = 0.00    # V5: zeroed — time context adds noise on short datasets
PREFETCH_RECENCY_DECAY = 0.95           # per-step exponential decay for recency (unchanged)

# ── RL Environment V2 Observation Dimensions ──────────────────────────────
# state = [current_app_ohe(50), prev_app_ohe(50), time_bucket_norm(1), day_of_week_norm(1),
#          hot_occupancy_norm(1), warm_occupancy_norm(1), hit_history_5(5)]
# Total: 50 + 50 + 1 + 1 + 1 + 1 + 5 = 109
# NOTE: Battery deliberately excluded — not available in UbiqLog dataset.
RL_V2_APP_VOCAB_SIZE = 50          # max unique apps tracked in observation OHE
RL_V2_HIT_HISTORY_LEN = 5         # number of recent cache hit/miss steps tracked
RL_V2_OBS_DIM = (
    RL_V2_APP_VOCAB_SIZE   # current app one-hot
    + RL_V2_APP_VOCAB_SIZE # previous app one-hot
    + 1                    # time_bucket normalised (0–47 → 0.0–1.0)
    + 1                    # day_of_week normalised (0=Mon, 6=Sun → 0.0–1.0)
    + 1                    # HOT occupancy ratio
    + 1                    # WARM occupancy ratio
    + RL_V2_HIT_HISTORY_LEN  # recent hit/miss binary history
)

# RL V2 Action Space: 3 continuous actions in [-1, 1] each
# [0] hot_budget_delta: adjust HOT tier target size (-1 = shrink, +1 = grow)
# [1] warm_budget_delta: adjust WARM tier target size
# [2] prefetch_aggressiveness: scale prefetch top-k (-1 = conservative, +1 = aggressive)
RL_V2_ACTION_DIM = 3

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

# ── Sensitivity Model (src/security/sensitivity_model.py) ──────────────────
# Numeric sensitivity levels: 0=public, 1=personal, 2=financial, 3=health
SENSITIVITY_PUBLIC = 0
SENSITIVITY_PERSONAL = 1
SENSITIVITY_FINANCIAL = 2
SENSITIVITY_HEALTH = 3

# Category → numeric sensitivity mapping
CATEGORY_SENSITIVITY_MAP: dict = {
    "entertainment": SENSITIVITY_PUBLIC,
    "gaming":        SENSITIVITY_PUBLIC,
    "shopping":      SENSITIVITY_PUBLIC,
    "utility":       SENSITIVITY_PUBLIC,
    "travel":        SENSITIVITY_PUBLIC,
    "food":          SENSITIVITY_PUBLIC,
    "social":        SENSITIVITY_PERSONAL,
    "productivity":  SENSITIVITY_PERSONAL,
    "enterprise":    SENSITIVITY_PERSONAL,
    "government":    SENSITIVITY_PERSONAL,
    "financial":     SENSITIVITY_FINANCIAL,
    "health":        SENSITIVITY_HEALTH,
    UNKNOWN_SENSITIVE_CATEGORY: SENSITIVITY_FINANCIAL,  # conservative default
}

# ── Latency Model — Literature Values (Samsung Galaxy A23 / mid-range Android)
# Source: Android developer documentation + academic measurements cited in latency_model.py
LATENCY_COLD_START_MS: dict = {
    "com.instagram.android":        820.0,
    "com.whatsapp":                 650.0,
    "com.google.youtube":           980.0,
    "com.spotify.music":            740.0,
    "com.google.android.gm":        590.0,
    "com.google.android.maps":      860.0,
    "com.android.chrome":           520.0,
    "com.netflix.mediaclient":      910.0,
    "com.amazon.mShop.android":     760.0,
    "com.slack.android":            680.0,
    "com.phonepe.app":              580.0,
    "net.one97.paytm":              620.0,
    "com.samsung.health":           540.0,
    "default":                      850.0,   # fallback for unmapped apps
}

LATENCY_WARM_START_MS: dict = {
    "com.instagram.android":        210.0,
    "com.whatsapp":                 170.0,
    "com.google.youtube":           260.0,
    "com.spotify.music":            195.0,
    "com.google.android.gm":        155.0,
    "com.google.android.maps":      230.0,
    "com.android.chrome":           140.0,
    "com.netflix.mediaclient":      245.0,
    "com.amazon.mShop.android":     200.0,
    "com.slack.android":            180.0,
    "com.phonepe.app":              150.0,
    "net.one97.paytm":              165.0,
    "com.samsung.health":           145.0,
    "default":                      210.0,
}

LATENCY_HOT_START_MS: dict = {
    "com.instagram.android":        45.0,
    "com.whatsapp":                 38.0,
    "com.google.youtube":           55.0,
    "com.spotify.music":            42.0,
    "com.google.android.gm":        35.0,
    "com.google.android.maps":      50.0,
    "com.android.chrome":           32.0,
    "com.netflix.mediaclient":      58.0,
    "com.amazon.mShop.android":     44.0,
    "com.slack.android":            40.0,
    "com.phonepe.app":              34.0,
    "net.one97.paytm":              37.0,
    "com.samsung.health":           33.0,
    "default":                      45.0,
}

# Path to ADB-measured latency CSV (generated by scripts/collect_app_latency.py)
LATENCY_MEASURED_CSV_PATH = os.path.join(DATA_DIR, "measured_latency.csv")

# Latency record provenance template — all literature entries must populate these fields.
# This ensures reproducibility: 6 months later you know exactly where numbers came from.
LATENCY_PROVENANCE_TEMPLATE: dict = {
    "source": "",            # "literature" | "measured"
    "device_class": "Samsung Galaxy A23",  # target device
    "android_version": "Android 12",
    "app_version": "latest",  # version at time of measurement / citation
    "measurement_date": "",   # ISO-8601 date string, e.g. "2024-03"
    "citation": "",           # BibTeX key or URL for literature values
    "cold_ms": 0.0,
    "warm_ms": 0.0,
    "hot_ms": 0.0,
}

# ── Statistical Evaluation (src/benchmarks/statistics.py) ──────────────────
STATS_CONFIDENCE_LEVEL = 0.95          # 95% confidence intervals
STATS_MIN_SAMPLES_FOR_TEST = 5         # minimum samples to run t-test

# ── Benchmark V2 Output Paths ──────────────────────────────────────────────
BENCHMARK_V2_RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_results_v2.csv")
BENCHMARK_V2_ADVANCED_CSV = os.path.join(RESULTS_DIR, "advanced_metrics_v2.csv")
BENCHMARK_V2_STATISTICAL_CSV = os.path.join(RESULTS_DIR, "statistical_results_v2.csv")
BENCHMARK_V2_ABLATION_CSV = os.path.join(RESULTS_DIR, "ablation_results_v2.csv")

# ── Baseline Names V2 ──────────────────────────────────────────────────────
BASELINE_V2_RANDOM = "Random"
BASELINE_V2_LRU = "LRU"
BASELINE_V2_LFU = "LFU"
BASELINE_V2_MRU = "MRU"
BASELINE_V2_FREQUENCY = "Frequency"
BASELINE_V2_RECENCY_FREQUENCY = "RecencyFrequency"  # strong classical: α*recency + β*frequency
BASELINE_V2_MARKOV = "FirstOrderMarkov"
BASELINE_V2_MARKOV2 = "SecondOrderMarkov"
BASELINE_V2_GRAPH_ONLY = "GraphOnly"
BASELINE_V2_GRAPHMIND_RL = "GraphMind_RL"

# RecencyFrequency scoring weights
BASELINE_RF_ALPHA = 0.6   # recency weight (sum α+β must equal 1.0)
BASELINE_RF_BETA = 0.4    # frequency weight
BASELINE_RF_RECENCY_DECAY = 0.90  # exponential decay per access event

# ── RL V2 Action Space Levels ──────────────────────────────────────────────
# MultiDiscrete([N_HOT_LEVELS, N_WARM_LEVELS, N_CONF_LEVELS])
# Each dimension selects one option from a fixed menu.
RL_V2_HOT_CAPACITY_OPTIONS: list = [1, 5, 10, 20, 30]    # HOT tier target sizes
RL_V2_WARM_CAPACITY_OPTIONS: list = [10, 30, 50, 100, 150] # WARM tier target sizes
RL_V2_CONF_THRESHOLD_OPTIONS: list = [0.5, 0.6, 0.7, 0.8, 0.9]  # prefetch thresholds
RL_V2_N_HOT_LEVELS: int = len(RL_V2_HOT_CAPACITY_OPTIONS)   # = 5
RL_V2_N_WARM_LEVELS: int = len(RL_V2_WARM_CAPACITY_OPTIONS) # = 5
RL_V2_N_CONF_LEVELS: int = len(RL_V2_CONF_THRESHOLD_OPTIONS) # = 5

# ── Ablation Experiment Names ──────────────────────────────────────────────
ABLATION_NO_RL = "No_RL"                            # GraphOnly path
ABLATION_NO_GRAPH = "No_Graph"                       # LRU fallback (no graph structure)
ABLATION_NO_CONFIDENCE = "No_ConfidencePrefetch"     # Graph+RL, fixed top-k prefetch
ABLATION_NO_SECURITY = "No_Security"                 # Full system minus security flush
ABLATION_NO_CONTEXT = "No_Context"                   # Graph+RL, no contextual features
ABLATION_GRAPH_PLUS_CONFIDENCE = "Graph+Confidence"  # graph + confidence, no RL
ABLATION_GRAPH_CONFIDENCE_NO_RL = "Graph+Confidence+NoRL"  # isolates confidence contribution
ABLATION_GRAPH_RL_ONLY = "Graph+RL"                 # graph + RL, fixed top-k prefetch
ABLATION_FULL_SYSTEM = "Full_System"                 # Graph + RL + Confidence + Security

# Ablation comparison table (ordered for paper display)
ABLATION_ORDERED_VARIANTS: list = [
    ABLATION_NO_RL,              # baseline: graph prediction only
    ABLATION_GRAPH_PLUS_CONFIDENCE,       # + confidence scoring
    ABLATION_GRAPH_CONFIDENCE_NO_RL,     # confidence without RL
    ABLATION_GRAPH_RL_ONLY,              # RL without confidence
    ABLATION_FULL_SYSTEM,        # full system
]

# ── Gemma Model ────────────────────────────────────────────────────────────
# Set ENABLE_GEMMA = False to prove Gemma does NOT inflate benchmark results.
# All benchmark runs execute identically regardless of this flag.
ENABLE_GEMMA: bool = os.getenv("ENABLE_GEMMA", "false").lower() == "true"
GEMMA_MODEL_ID = "google/gemma-2b"
GEMMA_LOCAL_PATH = os.path.join(MODELS_DIR, "gemma-2b")
GEMMA_MAX_NEW_TOKENS = 128
GEMMA_DEVICE = os.getenv("DEVICE", "cpu")

# ── Statistical Evaluation Additions ──────────────────────────────────────
STATS_BOOTSTRAP_N_SAMPLES = 10_000    # bootstrap resampling iterations

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
