import re

with open('docs/ax.md', 'r', encoding='utf-8') as f:
    txt = f.read()

txt = re.sub(r'\|\s*App Load Time Improvement\s*\|\s*≥\s*20%\s*\|\s*See `reports/kpi_summary\.json`\s*\|\s*TBD \(run benchmark\)\s*\|', 
             '| App Load Time Improvement | ≥ 20% | 42.21% | ✅ PASS |', txt)
txt = re.sub(r'\|\s*App Launch Time Improvement\s*\|\s*≥\s*10%\s*\|\s*See `reports/kpi_summary\.json`\s*\|\s*TBD \(run benchmark\)\s*\|', 
             '| App Launch Time Improvement | ≥ 10% | 45.14% | ✅ PASS |', txt)
txt = re.sub(r'\|\s*Memory Thrashing Reduction\s*\|\s*≥\s*50%\s*vs LRU\s*\|\s*See `reports/kpi_summary\.json`\s*\|\s*TBD \(run benchmark\)\s*\|', 
             '| Memory Thrashing Reduction | ≥ 50% vs LRU | 100.00% | ✅ PASS |', txt)
txt = re.sub(r'\|\s*Memory Utilisation Efficiency\s*\|\s*≥\s*30%\s*improvement\s*\|\s*See `reports/kpi_summary\.json`\s*\|\s*TBD \(run benchmark\)\s*\|', 
             '| Memory Utilisation Efficiency | ≥ 30% improvement | 0.37% | ❌ FAIL |', txt)

with open('docs/ax.md', 'w', encoding='utf-8') as f:
    f.write(txt)
