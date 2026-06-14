import re

with open('GRAPHMIND_HARDCHECK.py', 'r', encoding='utf-8') as f:
    text = f.read()

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

@check(4, "docs/ax.md exists and contains BehaviouralGraph", "Ensure docs/ax.md is present and correct")
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

text = text.replace('if __name__ == "__main__":', new_checks + '\nif __name__ == "__main__":')

to_add = '''    check_kpi_extractor_exists,
    check_gemma_explainer_exists,
    check_settings_enable_gemma,
    check_kpi_summary_exists,
    check_docs_ax,
    check_evaluator_v2_exists,
    phase_filter=args.phase,'''

text = text.replace('phase_filter=args.phase,', to_add)

with open('GRAPHMIND_HARDCHECK.py', 'w', encoding='utf-8') as f:
    f.write(text)
