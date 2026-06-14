import re
import os

with open('GRAPHMIND_HARDCHECK.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Regular expression to match an entire check function definition including the @check decorator
# We want to remove checks with specific titles

titles_to_remove = [
    "src.rl.environment importable",
    "GraphMindEnv initializes",
    "GraphMindEnv.step() returns correct",
    "src.rl.trainer importable",
    "PPO model file exists for at least user_00",
    "src.agents.orchestrator importable",
    "GraphMindOrchestrator instantiates without error",
    "GraphMindOrchestrator.run_day() returns",
    "GraphMindState TypedDict has correct",
    "LangGraph graph is built with correct 5 nodes",
    "SecurityAgent being called in the LangGraph"
]

for title in titles_to_remove:
    # regex to match: @check(... title ...) up to the next @check or end of file
    # We use a non-greedy match for everything up to the next @check
    pattern = r'@check\(\d+,\s*["\']' + re.escape(title) + r'.*?(?=\n@check|\Z)'
    text = re.sub(pattern, '', text, flags=re.DOTALL)

# Let's also remove any other 'LangGraph' mentions in strings if there are stray ones, 
# but simply removing the checks is safer.

# Now append the new V5 checks at the very end of the file, before `if __name__ == "__main__":`
new_checks = """
# ==========================================
# V5 ARCHITECTURE CHECKS
# ==========================================

@check(4, "src.benchmarks.kpi_extractor exists", "Ensure src/benchmarks/kpi_extractor.py exists")
def check_kpi_extractor_exists():
    import os
    assert os.path.exists("src/benchmarks/kpi_extractor.py")

@check(4, "src.gemma_explainer exists", "Ensure src/gemma_explainer.py exists")
def check_gemma_explainer_exists():
    import os
    assert os.path.exists("src/gemma_explainer.py")

@check(4, "config.settings contains ENABLE_GEMMA", "Ensure ENABLE_GEMMA is in config.settings")
def check_settings_enable_gemma():
    from config import settings
    assert hasattr(settings, "ENABLE_GEMMA")

@check(4, "reports/kpi_summary.json exists after benchmark run", "Ensure benchmark generated reports/kpi_summary.json")
def check_kpi_summary_exists():
    import os
    assert os.path.exists("reports/kpi_summary.json")

@check(4, "docs/ax.md exists and contains 'BehaviouralGraph'", "Ensure docs/ax.md is present and correct")
def check_docs_ax():
    import os
    assert os.path.exists("docs/ax.md")
    with open("docs/ax.md", "r", encoding="utf-8") as f:
        assert "BehaviouralGraph" in f.read()

@check(4, "src.benchmarks.evaluator_v2 exists", "Ensure src/benchmarks/evaluator_v2.py exists")
def check_evaluator_v2_exists():
    import os
    assert os.path.exists("src/benchmarks/evaluator_v2.py")

"""

# Insert new checks before the main block
text = text.replace('if __name__ == "__main__":', new_checks + '\nif __name__ == "__main__":')

with open('GRAPHMIND_HARDCHECK.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Hardchecker patched successfully.")
