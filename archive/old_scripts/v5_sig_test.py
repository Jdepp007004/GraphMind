import csv, numpy as np
from scipy import stats

v4 = list(csv.DictReader(open('results/benchmark_results_v4.csv')))
v5 = list(csv.DictReader(open('results/v5_rl_ablation.csv')))

baseline = {r['user_id']: float(r['f1']) for r in v4 if r['policy']=='GraphMindRL'}
m1       = {r['user_id']: float(r['f1']) for r in v4 if r['policy']=='Markov-1'}
m2       = {r['user_id']: float(r['f1']) for r in v4 if r['policy']=='Markov-2'}

rl_lat   = {r['user_id']: float(r['f1']) for r in v5 if r['policy']=='RL_LatencyFocus'}
rl_f1    = {r['user_id']: float(r['f1']) for r in v5 if r['policy']=='RL_F1Reward'}
rl_thr   = {r['user_id']: float(r['f1']) for r in v5 if r['policy']=='RL_Threshold'}

users = sorted(set(baseline) & set(rl_lat))
print("n users:", len(users))

def paired_t(a_dict, b_dict, users):
    a = np.array([a_dict[u] for u in users])
    b = np.array([b_dict[u] for u in users])
    stat, p = stats.ttest_rel(b, a)
    diff = b - a
    d = diff.mean() / (diff.std() + 1e-9)
    return stat, p, d

comparisons = [
    ('RL_LatencyFocus', rl_lat),
    ('RL_F1Reward', rl_f1),
    ('RL_Threshold', rl_thr),
    ('Markov-1', m1),
    ('Markov-2', m2),
]

for name, exp_dict in comparisons:
    t, p, d = paired_t(baseline, exp_dict, users)
    delta = np.mean([exp_dict[u]-baseline[u] for u in users])
    sig = 'SIG' if p < 0.05 else 'n.s.'
    print(f"{name:25s} delta={delta:+.4f} t={t:.3f} p={p:.4f} d={d:.3f} {sig}")
