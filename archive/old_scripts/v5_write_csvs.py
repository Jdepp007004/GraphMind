import csv, numpy as np, os
from collections import defaultdict

rows = list(csv.DictReader(open('results/v5_all_experiments.csv')))
v4   = list(csv.DictReader(open('results/benchmark_results_v4.csv')))

baseline_f1 = 0.7424

# Write time_context_results.csv
time_rows = [r for r in rows if r['phase'] == 'phase3_time']
order_rows = [r for r in rows if r['phase'] == 'phase4_order']

def agg_rows(policy_rows):
    by_pol = defaultdict(list)
    for r in policy_rows:
        by_pol[r['policy']].append(r)
    out = []
    for pol, prows in sorted(by_pol.items()):
        f1s  = [float(r['f1']) for r in prows]
        hrs  = [float(r['hit_rate']) for r in prows]
        ps   = [float(r['precision']) for r in prows]
        rs   = [float(r['recall']) for r in prows]
        lats = [float(r['latency_saved_ms']) for r in prows]
        out.append({
            'policy': pol,
            'n_users': len(prows),
            'mean_f1': round(np.mean(f1s), 4),
            'std_f1':  round(np.std(f1s), 4),
            'mean_hr': round(np.mean(hrs), 4),
            'mean_precision': round(np.mean(ps), 4),
            'mean_recall': round(np.mean(rs), 4),
            'mean_lat_ms': round(np.mean(lats), 1),
            'delta_f1': round(np.mean(f1s) - baseline_f1, 4),
        })
    return sorted(out, key=lambda x: -x['mean_f1'])

def write_csv(path, rows_list):
    if not rows_list: return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_list[0].keys()))
        w.writeheader(); w.writerows(rows_list)
    print(f"Written: {path}")

write_csv('results/time_context_results.csv', agg_rows(time_rows))
write_csv('results/order_analysis_v5.csv', agg_rows(order_rows))
write_csv('results/v5_combined_results.csv', agg_rows([r for r in rows if r['phase']=='phase5_combined']))
write_csv('results/v5_graph_results.csv',   agg_rows([r for r in rows if r['phase']=='phase6_graph']))
write_csv('results/v5_rl_results.csv',      agg_rows([r for r in rows if r['phase']=='phase7_rl']))
write_csv('results/v5_decay_results.csv',   agg_rows([r for r in rows if r['phase']=='phase8_decay']))

# Master agg
all_agg = agg_rows(rows)
write_csv('results/v5_master_summary.csv', all_agg)

print("\n=== TOP POLICIES ===")
for r in all_agg[:5]:
    print(r)
