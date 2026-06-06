import csv, json
import numpy as np
from collections import defaultdict

with open('results/benchmark_results_v2.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

TOP10 = {'18_F','19_F','24_F','28_F','33_F','35_F','22_M','7_F','12_M','31_F'}
FAST_POLICIES = {'Markov-1','Markov-2','GraphOnly','Graph+Confidence','GraphMindRL'}
METRICS = ['hit_rate','precision','recall','f1','latency_saved_ms','latency_saved_pct']

fast_rows = [r for r in rows if r['user_id'] in TOP10 and r['policy'] in FAST_POLICIES]

with open('results/benchmark_results_fast.csv','w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(fast_rows[0].keys()))
    w.writeheader(); w.writerows(fast_rows)

ul_cols = ['user_id','policy','hit_rate','precision','recall','f1','latency_saved_ms','latency_saved_pct']
with open('results/user_level_results.csv','w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=ul_cols)
    w.writeheader()
    for r in fast_rows:
        w.writerow({k: r[k] for k in ul_cols})

agg = defaultdict(lambda: defaultdict(list))
for r in fast_rows:
    for m in METRICS:
        agg[r['policy']][m].append(float(r[m]))

print("Fast benchmark written.")
for pol in ['GraphMindRL','Graph+Confidence','Markov-2','Markov-1','GraphOnly']:
    if pol not in agg: continue
    v = agg[pol]
    print(pol, "HR:", round(np.mean(v['hit_rate']),3), "F1:", round(np.mean(v['f1']),3), "Lat:", round(np.mean(v['latency_saved_ms']),1))
