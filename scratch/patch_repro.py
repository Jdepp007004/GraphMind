import re

with open('docs/reproducibility.md', 'r', encoding='utf-8') as f:
    txt = f.read()

txt = re.sub(r'Memory Thrashing Reduction \(\%\)\s+.*?50\%\s+\[X\]\%\s+\[STATUS\]', 
             'Memory Thrashing Reduction (%)                  ≥50%       100.00%  [PASS]', txt)
txt = re.sub(r'App Load Time Improvement \(\%\)\s+.*?20\%\s+\[X\]\%\s+\[STATUS\]', 
             'App Load Time Improvement (%)                   ≥20%        42.21%  [PASS]', txt)
txt = re.sub(r'App Launch Time Improvement \(\%\)\s+.*?10\%\s+\[X\]\%\s+\[STATUS\]', 
             'App Launch Time Improvement (%)                 ≥10%        45.14%  [PASS]', txt)
txt = re.sub(r'Memory Utilisation Efficiency Improvement \(\%\)\s+.*?30\%\s+\[X\]\%\s+\[STATUS\]', 
             'Memory Utilisation Efficiency Improvement (%)   ≥30%         0.37%  [FAIL]', txt)

with open('docs/reproducibility.md', 'w', encoding='utf-8') as f:
    f.write(txt)
