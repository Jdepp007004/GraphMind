import csv, numpy as np
rows = list(csv.DictReader(open('results/benchmark_results_v4.csv')))
policies = ['GraphMindRL','Graph+Confidence','Markov-2','Markov-1','GraphOnly',
            'GlobalMarkov2','VariableOrderMarkov','RLAdaptiveEnsemble','Random','LRU','LFU',
            'Frequency','RecencyFrequency','ClusterMarkov','ContextMarkov']
print(f"{'Policy':25s} {'F1':>7} {'HitRate':>8} {'LatMs':>8} {'N':>4}")
print("-"*55)
for p in policies:
    vals = [float(r['f1']) for r in rows if r['policy']==p]
    hr   = [float(r['hit_rate']) for r in rows if r['policy']==p]
    lat  = [float(r['latency_saved_ms']) for r in rows if r['policy']==p]
    if vals:
        print(f"{p:25s} {np.mean(vals):7.4f} {np.mean(hr):8.4f} {np.mean(lat):8.1f} {len(vals):4d}")
