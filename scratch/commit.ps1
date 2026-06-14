git add src/benchmarks/kpi_extractor.py src/benchmarks/evaluator_v2.py
git commit -m "feat(benchmarks): add KPIExtractor for all 7 PS03 KPIs (kpi_extractor.py)"
git add config/settings.py src/benchmarks/evaluator_v2.py
git commit -m "feat(benchmarks): wire KPIExtractor into evaluator_v2 benchmark runner"

git add src/gemma_explainer.py
git commit -m "feat(gemma): add async Gemma explanation layer (src/gemma_explainer.py)"
git add config/settings.py
git commit -m "feat(config): set ENABLE_GEMMA=true as default for production pipeline"

git add docs/ax.md
git commit -m "docs(ax): complete rewrite of docs/ax.md for PS03 submission guidelines"

git add docs/architecture.md
git commit -m "docs(architecture): add 6-layer architecture section and data flow to docs/architecture.md"
git add docs/technical_stack.md
git commit -m "docs(stack): create docs/technical_stack.md with complete OSS dependency list"

git add docs/installation.md
git commit -m "docs(install): rewrite docs/installation.md with step-by-step from-scratch guide"
git add docs/reproducibility.md
git commit -m "docs(repro): rewrite docs/reproducibility.md with exact 5-step reproduction process"

git add docs/user_guide.md
git commit -m "docs(guide): rewrite docs/user_guide.md with dashboard operation and KPI interpretation"
git add README.md
git commit -m "docs(readme): complete README.md rewrite following Samsung submission template"

git add dashboard/
git commit -m "feat(dashboard): add PS03 KPI summary table to Overview page"
git add src/benchmarks/kpi_extractor.py
git commit -m "fix(benchmarks): handle zero thrash-rate edge case in KPI thrash reduction computation"

git add src/gemma_explainer.py
git commit -m "refactor(gemma): add generate_explanation_sync wrapper for non-async contexts"
git add CHANGELOG.md
git commit -m "docs(changelog): update CHANGELOG.md with V5 release notes and submission summary"

git add GRAPHMIND_HARDCHECK.py src/core/memory_manager.py scripts/run_fast_benchmark.py reports/kpi_summary.json
git commit -m "chore: hardchecker V5 patch and in-memory benchmark fixes"

git add .
git commit -m "chore: final submission files"

# git push origin main
