import csv, numpy as np
from collections import defaultdict

METRICS = ['hit_rate','precision','recall','f1','latency_saved_ms','latency_saved_pct']
VARIANTS = ['GraphOnly','Graph+Confidence','Graph+RL','Full GraphMind']

rows = list(csv.DictReader(open('results/ablation_results_v2.csv', encoding='utf-8')))
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    for m in METRICS:
        agg[r['variant']][m].append(float(r[m]))

print('=== ABLATION RESULTS ===')
for v in VARIANTS:
    if v not in agg: continue
    hr = np.mean(agg[v]['hit_rate'])
    f1 = np.mean(agg[v]['f1'])
    lat = np.mean(agg[v]['latency_saved_ms'])
    print(f'{v}: HR={hr:.4f} F1={f1:.4f} Lat={lat:.1f}ms')

print()
print('=== BENCHMARK RESULTS ===')
rows2 = list(csv.DictReader(open('results/benchmark_results_fast.csv', encoding='utf-8')))
agg2 = defaultdict(lambda: defaultdict(list))
for r in rows2:
    for m in METRICS:
        agg2[r['policy']][m].append(float(r[m]))

POLICIES = ['Markov-1','Markov-2','GlobalMarkov2','GraphOnly','Graph+Confidence','GraphMindRL']
for p in POLICIES:
    if p not in agg2: continue
    hr = np.mean(agg2[p]['hit_rate'])
    f1 = np.mean(agg2[p]['f1'])
    lat = np.mean(agg2[p]['latency_saved_ms'])
    print(f'{p}: HR={hr:.4f} F1={f1:.4f} Lat={lat:.1f}ms')

print()
print('=== STAT RESULTS ===')
rows3 = list(csv.DictReader(open('results/statistical_results_fast.csv', encoding='utf-8')))
for r in rows3:
    if r['metric'] == 'f1':
        print(f"F1: {r['treatment']} vs {r['control']}: d={r['mean_improvement']} p={r['p_value']} sig={r['significant']} cohens_d={r['cohens_d']}")
