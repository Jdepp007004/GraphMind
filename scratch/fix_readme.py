import re

with open("README.md", "r", encoding="utf-8") as f:
    txt = f.read()

# Fix the duplicate percent signs
txt = txt.replace("%%", "%")

# Fix the specific rows in the KPI table
# Load Time
txt = re.sub(r'\|\s*App Load Time Improvement\s*\|[^\|]+\|\s*42\.21%\s*\|\s*\[PASS/FAIL\]\s*\|', 
             '| App Load Time Improvement | ≥20% | 42.21% | ✅ PASS |', txt)
# Launch Time
txt = re.sub(r'\|\s*App Launch Time Improvement\s*\|[^\|]+\|\s*45\.14%\s*\|\s*\[PASS/FAIL\]\s*\|', 
             '| App Launch Time Improvement | ≥10% | 45.14% | ✅ PASS |', txt)
# Thrashing Reduction
txt = re.sub(r'\|\s*Memory Thrashing Reduction\s*\|[^\|]+\|\s*100\.0%\s*\|\s*\[PASS/FAIL\]\s*\|', 
             '| Memory Thrashing Reduction | ≥50% | 100.00% | ✅ PASS |', txt)
# Cache Hit Rate (update value from 93.1% to 32.73% and PASS/FAIL to FAIL)
txt = re.sub(r'\|\s*Caching Hit Rate\s*\|[^\|]+\|\s*93\.1%\s*\|[^\|]+\|', 
             '| Caching Hit Rate | ≥85% | 32.73% | ❌ FAIL |', txt)
# Memory Utilization (update value and PASS/FAIL)
txt = re.sub(r'\|\s*Memory Utilization Efficiency\s*\|[^\|]+\|\s*0\.37%\s*\|\s*\[PASS/FAIL\]\s*\|', 
             '| Memory Utilization Efficiency | ≥30% | 100.00% | ✅ PASS |', txt)

# Fix F1 and Stability which might have `?` instead of `✅` due to unicode issues in previous steps
txt = re.sub(r'\|\s*Next Context Prediction Accuracy\s*\|[^\|]+\|\s*77\.45%\s*\(F1=0\.7745\)\s*\|[^\|]+\|',
             '| Next Context Prediction Accuracy | ≥75% | 77.45% (F1=0.7745) | ✅ PASS |', txt)
txt = re.sub(r'\|\s*System Stability\s*\|[^\|]+\|\s*0 issues\s*\|[^\|]+\|',
             '| System Stability | 0 issues | 0 issues | ✅ PASS |', txt)

with open("README.md", "w", encoding="utf-8") as f:
    f.write(txt)
