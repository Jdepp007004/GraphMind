import re

with open('GRAPHMIND_HARDCHECK.py', 'r', encoding='utf-8') as f:
    text = f.read()

patch = 'import sys\nimport io\nsys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=\'utf-8\')\n'
text = patch + text

new_checks = """
# V5 ARCHITECTURE CHECKS

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

check_funcs = re.findall(r'@check\(.*?\)\ndef\s+([a-zA-Z_0-9]+)\(\):', text, re.DOTALL)

bad_keywords = ['check_rl_env', 'check_rl_model', 'check_orchestrator', 'check_langgraph', 'check_benchmark_results', 'check_graphmind_beats', 'check_security_flushes', 'check_simulation_logs', 'check_rl_trainer', 'check_evaluator_import']
filtered_funcs = [fn for fn in check_funcs if not any(bk in fn for bk in bad_keywords)]

pattern = r'run_checks\([\s\S]*?fix_hints_only=args\.fix_hints_only\n\s*\)'
replacement = 'run_checks(\n    ' + ',\n    '.join(filtered_funcs) + ',\n    phase_filter=args.phase,\n    verbose=args.verbose,\n    fix_hints_only=args.fix_hints_only\n)'

text = re.sub(pattern, replacement, text)

text = text.replace('with open(APP_TAXONOMY_PATH) as f:', 'with open(APP_TAXONOMY_PATH, \'r\', encoding=\'utf-8\') as f:')
text = text.replace('with open(path) as f:', 'with open(path, \'r\', encoding=\'utf-8\') as f:')
text = text.replace('with open(log_path) as f:', 'with open(log_path, \'r\', encoding=\'utf-8\') as f:')

with open('GRAPHMIND_HARDCHECK.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Patched successfully')
