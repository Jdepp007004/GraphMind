#!/usr/bin/env python3
"""
scripts/run_phase11_e.py — Phase E only (A–D already complete).

Reads best weights from v5_weight_grid.csv and best threshold from
v5_threshold_sweep.csv, then benchmarks GraphMindRL_V5.
"""

import csv, json, logging, math, os, sys
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR   = os.path.join(REPORTS_DIR, "figures")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
UBIQLOG_ROOT  = os.path.join(PROJECT_ROOT, "datasets", "ubiqlog", "UbiqLog4UCI")
LATENCY_CSV   = os.path.join(PROJECT_ROOT, "datasets", "app_launch_latency.csv")

HOT_SIZE = 5; WARM_SIZE = 15
TRAIN_RATIO = 0.80; VAL_RATIO = 0.10
MIN_YEAR = 2011; MAX_YEAR = 2016
BASELINE_F1       = 0.7424
BEST_CANDIDATE_F1 = 0.7539

SYSTEM_PREFIXES = (
    "com.android.", "com.google.android.providers",
    "com.google.android.gms", "com.google.android.gsf",
    "com.sec.android.provider", "com.samsung.android.provider",
    "com.redbend.", "android.",
)
SYSTEM_SUFFIXES = (":engine", ":client", ":daemon", ":service", ":pushservice", ":sync")


class MeasuredLatencyModel:
    _DC=2763.0;_DW=1301.0;_DH=274.0
    def __init__(self, path):
        self._cold={};self._warm={};self._hot={};self._pkg={}
        if os.path.exists(path):
            b=defaultdict(lambda:defaultdict(list))
            with open(path,encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    b[r["app_id"]][r["start_type"]].append(float(r["total_time_ms"]))
                    self._pkg[r["package_name"]]=r["app_id"]
            for aid,tiers in b.items():
                if "cold" in tiers:self._cold[aid]=float(np.mean(tiers["cold"]))
                if "warm" in tiers:self._warm[aid]=float(np.mean(tiers["warm"]))
                if "hot"  in tiers:self._hot[aid] =float(np.mean(tiers["hot"]))
    def saved(self,pkg,tier):
        k=pkg if pkg in self._cold else self._pkg.get(pkg)
        cold=self._cold.get(k,self._DC) if k else self._DC
        if tier=="hot":  return max(0.0,cold-(self._hot.get(k,self._DH)  if k else self._DH))
        if tier=="warm": return max(0.0,cold-(self._warm.get(k,self._DW) if k else self._DW))
        return 0.0


class Cache:
    def __init__(self, policy=None):
        self._hot = []
        self._warm = []
        self.policy = policy
        self.use_smart = policy is not None and hasattr(policy, "_g")
        if self.use_smart:
            # Seed persistent apps from the top 3 most frequent training apps
            self.policy._persistent_apps = [app for app, _ in sorted(self.policy._freq.items(), key=lambda x: -x[1])[:3]]

    def lookup(self, app):
        if app in self._hot:  return "hot"
        if app in self._warm: return "warm"
        return "miss"

    def eviction_score(self, app, cur_app):
        if not self.use_smart or not self.policy:
            return 0.0
        trans_prob = 0.0
        if cur_app and cur_app in self.policy._g:
            trans_prob = self.policy._g[cur_app].get(app, 0.0)
        tot = self.policy._total or 1.0
        freq_score = self.policy._freq.get(app, 0.0) / tot
        recency_score = self.policy._rec.get(app, 0.0)
        return trans_prob * 0.50 + freq_score * 0.30 + recency_score * 0.20

    def access(self, app, cur_app=None):
        evicted = []
        if app in self._hot:
            self._hot.remove(app)
            self._hot.insert(0, app)
            return []
            
        if app in self._warm:
            self._warm.remove(app)
            
        self._hot.insert(0, app)
        
        # Determine how to evict from HOT to WARM
        if self.use_smart:
            while len(self._hot) > HOT_SIZE:
                EVICTION_PROBABILITY_FLOOR = 0.05
                tot = self.policy._total or 1.0
                
                evictable = []
                for a in self._hot:
                    if a in getattr(self.policy, "_persistent_apps", []):
                        continue
                    trans_prob = 0.0
                    if cur_app and cur_app in self.policy._g:
                        trans_prob = self.policy._g[cur_app].get(a, 0.0)
                    freq_score = self.policy._freq.get(a, 0.0) / tot
                    relevance = trans_prob * 0.60 + freq_score * 0.40
                    if relevance < EVICTION_PROBABILITY_FLOOR:
                        evictable.append(a)
                        
                if not evictable:
                    # Do not evict existing HOT apps; demote newly accessed app to WARM instead
                    self._hot.remove(app)
                    self._warm.insert(0, app)
                    evicted.append(app)
                else:
                    scored = []
                    for a in evictable:
                        score = self.eviction_score(a, cur_app)
                        scored.append((score, a))
                    scored.sort(key=lambda x: x[0])
                    lowest_app = scored[0][1]
                    self._hot.remove(lowest_app)
                    self._warm.insert(0, lowest_app)
                    evicted.append(lowest_app)
        else:
            # Standard LRU eviction/demotion
            while len(self._hot) > HOT_SIZE:
                demoted = self._hot.pop()
                self._warm.insert(0, demoted)
                evicted.append(demoted) # Count HOT demotion as eviction

        # Evict from WARM to out
        if self.use_smart:
            while len(self._warm) > WARM_SIZE:
                scored = []
                for a in self._warm:
                    score = self.eviction_score(a, cur_app)
                    scored.append((score, a))
                scored.sort(key=lambda x: x[0])
                lowest_app = scored[0][1]
                self._warm.remove(lowest_app)
                evicted.append(lowest_app)
        else:
            while len(self._warm) > WARM_SIZE:
                evicted.append(self._warm.pop())
                
        return evicted

    def prefetch(self, apps, cur_app=None):
        evicted = []
        for a in apps:
            if a not in self._hot and a not in self._warm:
                self._warm.insert(0, a)
                if self.use_smart:
                    while len(self._warm) > WARM_SIZE:
                        scored = []
                        for wa in self._warm:
                            score = self.eviction_score(wa, cur_app)
                            scored.append((score, wa))
                        scored.sort(key=lambda x: x[0])
                        lowest_app = scored[0][1]
                        self._warm.remove(lowest_app)
                        evicted.append(lowest_app)
                else:
                    while len(self._warm) > WARM_SIZE:
                        evicted.append(self._warm.pop())
        return evicted

    def reset(self):
        self._hot = []
        self._warm = []


def _is_system(p):
    p=p.lower()
    for pfx in SYSTEM_PREFIXES:
        if p.startswith(pfx): return True
    for sfx in SYSTEM_SUFFIXES:
        if p.endswith(sfx): return True
    return False

def _parse_ts(s):
    from datetime import datetime
    try:
        dt=datetime.strptime(s.strip(),"%m-%d-%Y %H:%M:%S")
        return dt if MIN_YEAR<=dt.year<=MAX_YEAR else None
    except: return None

def load_user_data(user_id):
    user_dir=os.path.join(UBIQLOG_ROOT,user_id); raw=[]
    for fname in sorted(os.listdir(user_dir)):
        if not fname.endswith(".txt"): continue
        try:
            with open(os.path.join(user_dir,fname),encoding="utf-8",errors="replace") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try:
                        obj=json.loads(line)
                        if "Application" not in obj: continue
                        app=obj["Application"]
                        pkg=app.get("ProcessName","").strip()
                        if not pkg or _is_system(pkg): continue
                        dt=_parse_ts(app.get("Start",""))
                        if dt is None: continue
                        tb=dt.hour*2+(1 if dt.minute>=30 else 0)
                        wd=dt.weekday()
                        raw.append((dt,pkg,tb,wd))
                    except: pass
        except: pass
    raw.sort(key=lambda x:x[0])
    return ([r[1] for r in raw],[r[2] for r in raw],[r[3] for r in raw])


class ConfidencePolicy:
    def __init__(self,w_trans,w_rec,w_freq,threshold,budget=HOT_SIZE,name="ConfidencePolicy"):
        self.name=name; self.w_trans=w_trans; self.w_rec=w_rec; self.w_freq=w_freq
        self.threshold=threshold; self._init_thresh=threshold; self.budget=budget
        self._g={}; self._rec=defaultdict(float); self._freq=defaultdict(float)
        self._total=0.0; self._hist=deque(maxlen=20); self._last_preds=[]
    def train(self,apps,tbs=None,wds=None,**kw):
        c=defaultdict(lambda:defaultdict(int))
        for i in range(1,len(apps)): c[apps[i-1]][apps[i]]+=1
        self._g={s:dict(sorted({k:v/sum(d.values()) for k,v in d.items()}.items(),key=lambda x:-x[1])) for s,d in c.items()}
    def predict(self,cur,prev=None,tb=0,wd=0):
        if cur not in self._g: self._last_preds=[]; return []
        tot=self._total or 1.0
        cands={app:(self.w_trans*p+self.w_rec*self._rec.get(app,0.0)+self.w_freq*self._freq.get(app,0.0)/tot) for app,p in self._g[cur].items()}
        self._last_preds=sorted((a for a,c in cands.items() if c>=self.threshold),key=lambda a:-cands[a])[:self.budget]
        return self._last_preds
    def update(self,app,hit=False):
        for k in self._rec: self._rec[k]*=0.95
        self._rec[app]=1.0; self._freq[app]+=1; self._total+=1
        self._hist.append(1.0 if hit else 0.0)
        if len(self._hist)==20:
            hr=sum(self._hist)/20
            if hr<0.5:   self.threshold=max(0.05,self.threshold-0.005)
            elif hr>0.8: self.threshold=min(0.25,self.threshold+0.005)
    def reset(self):
        self._rec.clear();self._freq.clear();self._total=0.0
        self._hist.clear();self._last_preds=[];self.threshold=self._init_thresh


class LRUPolicy:
    def __init__(self, budget=HOT_SIZE):
        self.budget = budget
        self._lru = []
        self._warmup_apps = []
    def train(self, apps, **kwargs):
        self._warmup_apps = apps[-20:] if len(apps) >= 20 else apps
    def update(self, app, hit=False):
        if app in self._lru:
            self._lru.remove(app)
        self._lru.insert(0, app)
    def predict(self, cur, prev=None, tb=0, wd=0):
        return self._lru[:self.budget]
    def reset(self):
        self._lru = list(self._warmup_apps)


def evaluate_policy(policy,tr_a,va_a,ts_a,tr_t,va_t,ts_t,tr_w,va_w,ts_w,lat,uid="x"):
    policy.train(tr_a,tbs=tr_t,wds=tr_w,val_apps=va_a,val_tbs=va_t,val_wds=va_w)
    policy.reset()
    cache=Cache(policy=policy)
    for app in tr_a[-20:]: cache.access(app)
    hits=misses=tp=fp=fn=0; lat_saved=0.0; prev=None
    
    thrash_events = 0
    evictions = {}  # app -> step_index of eviction
    
    for i,cur in enumerate(ts_a):
        tb=ts_t[i] if ts_t else 0; wd=ts_w[i] if ts_w else 0
        
        # Check if cur was evicted within the last 3 events
        if cur in evictions and (i - evictions[cur] <= 3):
            thrash_events += 1
            del evictions[cur]
            
        preds=policy.predict(cur,prev=prev,tb=tb,wd=wd)
        
        evicted_prefetch = []
        if preds:
            evicted_prefetch = cache.prefetch(preds, cur_app=cur)
            
        tier=cache.lookup(cur); is_hit=tier in ("hot","warm")
        if is_hit: hits+=1;tp+=1;lat_saved+=lat.saved(cur,tier)
        else: misses+=1
        if i+1<len(ts_a):
            nxt=ts_a[i+1]
            if preds:
                if nxt in preds: tp+=1
                else: fn+=1;fp+=len(preds)
            else: fn+=1
            
        evicted_access = cache.access(cur, cur_app=cur)
        policy.update(cur,hit=is_hit); prev=cur
        
        # Record all evictions at step index i
        for app in evicted_prefetch + evicted_access:
            evictions[app] = i
            
    total=hits+misses or 1
    hr=hits/total; pr=tp/(tp+fp) if (tp+fp)>0 else 0.0
    re=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=2*pr*re/(pr+re) if (pr+re)>0 else 0.0
    return {"hit_rate":round(hr,4),"precision":round(pr,4),"recall":round(re,4),
            "f1":round(f1,4),"latency_saved_ms":round(lat_saved/total,2),
            "thrash_events": thrash_events}


def paired_t(exp_by_user,baseline_by_user):
    users=sorted(set(exp_by_user)&set(baseline_by_user))
    if len(users)<5: return float("nan"),float("nan"),float("nan")
    e=np.array([exp_by_user[u] for u in users]); b=np.array([baseline_by_user[u] for u in users])
    t,p=stats.ttest_rel(e,b); diff=e-b; d=diff.mean()/(diff.std()+1e-9)
    return float(t),float(p),float(d)


def main():
    os.makedirs(RESULTS_DIR,exist_ok=True); os.makedirs(REPORTS_DIR,exist_ok=True)
    lat=MeasuredLatencyModel(LATENCY_CSV)

    # Load best weights from Phase A
    weight_grid_path = os.path.join(RESULTS_DIR, "v5_weight_grid.csv")
    best_wt, best_wr, best_wf = 0.5, 0.1, 0.4   # known best from log
    if os.path.exists(weight_grid_path):
        rows = list(csv.DictReader(open(weight_grid_path, encoding="utf-8")))
        rows.sort(key=lambda r: -float(r["f1"]))
        if rows:
            best_wt = float(rows[0]["w_trans"])
            best_wr = float(rows[0]["w_rec"])
            best_wf = float(rows[0]["w_freq"])
            logger.info(f"Loaded best weights from Phase A: trans={best_wt} rec={best_wr} freq={best_wf}  F1={rows[0]['f1']}")

    # Load best threshold from Phase B
    thresh_path = os.path.join(RESULTS_DIR, "v5_threshold_sweep.csv")
    best_thresh = 0.16   # known best from log
    if os.path.exists(thresh_path):
        rows = list(csv.DictReader(open(thresh_path, encoding="utf-8")))
        rows.sort(key=lambda r: -float(r["f1"]))
        if rows:
            best_thresh = float(rows[0]["threshold"])
            logger.info(f"Loaded best threshold from Phase B: {best_thresh}  F1={rows[0]['f1']}")

    # Load Phase D results
    phase_d_rows = []
    mkn_path = os.path.join(RESULTS_DIR, "v5_modified_kn.csv")
    if os.path.exists(mkn_path):
        phase_d_rows = list(csv.DictReader(open(mkn_path, encoding="utf-8")))

    # Load baseline per-user F1
    baseline_by_user: Dict[str,float] = {}
    v4_path = os.path.join(RESULTS_DIR, "benchmark_results_v4.csv")
    if os.path.exists(v4_path):
        with open(v4_path,encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["policy"]=="GraphMindRL":
                    baseline_by_user[r["user_id"]]=float(r["f1"])

    with open(os.path.join(PROCESSED_DIR,"users.json"),encoding="utf-8") as f:
        usable_users=[u["user_id"] for u in json.load(f)["users"]]

    logger.info("Pre-loading user data…")
    user_cache={}
    for uid in usable_users:
        try:
            apps,tbs,wds=load_user_data(uid)
            if len(apps)>=200: user_cache[uid]=(apps,tbs,wds)
        except Exception as e: logger.warning(f"Skip {uid}: {e}")
    logger.info(f"Loaded {len(user_cache)} users")

    def split(uid):
        apps,tbs,wds=user_cache[uid]; n=len(apps)
        te,ve=int(n*TRAIN_RATIO),int(n*(TRAIN_RATIO+VAL_RATIO))
        return (apps[:te],apps[te:ve],apps[ve:],tbs[:te],tbs[te:ve],tbs[ve:],wds[:te],wds[te:ve],wds[ve:])

    # Compute total unique apps in the dataset dynamically
    unique_apps_all = set()
    for uid in usable_users:
        if uid not in user_cache: continue
        apps, _, _ = user_cache[uid]
        unique_apps_all.update(apps)
    num_unique_apps_in_dataset = len(unique_apps_all) if len(unique_apps_all) > 0 else 30
    logger.info(f"Unique apps in dataset: {num_unique_apps_in_dataset}")

    # Evaluate LRU baseline for thrashing counts
    logger.info("Evaluating LRU baseline for thrashing...")
    lru_thrash_by_user = {}
    for uid in usable_users:
        if uid not in user_cache: continue
        tr_a,va_a,ts_a,tr_t,va_t,ts_t,tr_w,va_w,ts_w=split(uid)
        if len(ts_a)<10: continue
        lru_pol = LRUPolicy(budget=HOT_SIZE)
        m = evaluate_policy(lru_pol, tr_a, va_a, ts_a, tr_t, va_t, ts_t, tr_w, va_w, ts_w, lat, uid)
        lru_thrash_by_user[uid] = m["thrash_events"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Phase E — GraphMindRL_V5")
    logger.info(f"  Best weights: trans={best_wt} rec={best_wr} freq={best_wf}")
    logger.info(f"  Best threshold: {best_thresh}")
    logger.info(f"{'='*60}")

    # Also test: best weights with BOTH best-threshold AND threshold=0.10
    # to see if threshold interacts with weights
    policies_e = [
        ("GraphMindRL_V5",        ConfidencePolicy(best_wt, best_wr, best_wf, best_thresh, name="GraphMindRL_V5")),
        ("GraphMindRL_V5_t10",    ConfidencePolicy(best_wt, best_wr, best_wf, 0.10,        name="GraphMindRL_V5_t10")),
        ("RL_LatencyFocus",       ConfidencePolicy(0.5, 0.3, 0.2, 0.10,                    name="RL_LatencyFocus")),
        ("GraphMindRL_Base",      ConfidencePolicy(0.5, 0.3, 0.2, 0.05,                    name="GraphMindRL_Base")),
    ]

    stability_issues = 0
    v5_f1 = 0.0
    v5_hit_rate = 0.0
    v5_latency_saved = 0.0
    v5_thrash_events = 0

    phase_e_rows = []
    for pol_name, policy in policies_e:
        f1_list=[]; hr_list=[]; pr_list=[]; re_list=[]; la_list=[]; by_user={}
        thrash_list=[]
        for uid in usable_users:
            if uid not in user_cache: continue
            tr_a,va_a,ts_a,tr_t,va_t,ts_t,tr_w,va_w,ts_w=split(uid)
            if len(ts_a)<10: continue
            
            try:
                # Re-instantiate to get fresh policy for each user
                if pol_name=="GraphMindRL_V5":
                    pol=ConfidencePolicy(best_wt,best_wr,best_wf,best_thresh,name=pol_name)
                elif pol_name=="GraphMindRL_V5_t10":
                    pol=ConfidencePolicy(best_wt,best_wr,best_wf,0.10,name=pol_name)
                elif pol_name=="RL_LatencyFocus":
                    pol=ConfidencePolicy(0.5,0.3,0.2,0.10,name=pol_name)
                else:
                    pol=ConfidencePolicy(0.5,0.3,0.2,0.05,name=pol_name)
                m=evaluate_policy(pol,tr_a,va_a,ts_a,tr_t,va_t,ts_t,tr_w,va_w,ts_w,lat,uid)
                f1_list.append(m["f1"]); hr_list.append(m["hit_rate"])
                pr_list.append(m["precision"]); re_list.append(m["recall"])
                la_list.append(m["latency_saved_ms"])
                thrash_list.append(m["thrash_events"])
                by_user[uid]=m["f1"]
            except Exception as e:
                logger.error(f"Error evaluating user {uid} for policy {pol_name}: {e}")
                stability_issues += 1

        if pol_name == "GraphMindRL_V5":
            v5_f1 = float(np.mean(f1_list))
            v5_hit_rate = float(np.mean(hr_list))
            v5_latency_saved = float(np.mean(la_list))
            v5_thrash_events = int(sum(thrash_list))

        t_s,p_v,d_v=paired_t(by_user,baseline_by_user)
        row={
            "policy":           pol_name,
            "f1":               round(float(np.mean(f1_list)),4),
            "std_f1":           round(float(np.std(f1_list)),4),
            "precision":        round(float(np.mean(pr_list)),4),
            "recall":           round(float(np.mean(re_list)),4),
            "hit_rate":         round(float(np.mean(hr_list)),4),
            "latency_saved_ms": round(float(np.mean(la_list)),2),
            "delta_f1_vs_baseline": round(float(np.mean(f1_list))-BASELINE_F1,4),
            "t_stat":   round(t_s,3)  if not math.isnan(t_s) else "—",
            "p_value":  round(p_v,4)  if not math.isnan(p_v) else "—",
            "cohen_d":  round(d_v,3)  if not math.isnan(d_v) else "—",
            "significant": bool(p_v<0.05) if not math.isnan(p_v) else False,
            "n_users":  len(f1_list),
        }
        phase_e_rows.append(row)
        p_s=f"{p_v:.4f}" if not math.isnan(p_v) else "—"
        d_s=f"{d_v:.3f}" if not math.isnan(d_v) else "—"
        sig="✅ SIG" if row["significant"] else "❌ n.s."
        meets="✅" if row["delta_f1_vs_baseline"]>=0.02 else "❌"
        logger.info(f"  {pol_name:30s}: F1={row['f1']:.4f}  ΔF1={row['delta_f1_vs_baseline']:+.4f}  "
                    f"HR={row['hit_rate']:.4f}  p={p_s}  d={d_s}  {sig}  ≥+0.02:{meets}")

    # Write CSV
    with open(os.path.join(RESULTS_DIR,"v5_final_comparison.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(phase_e_rows[0].keys()))
        w.writeheader(); w.writerows(phase_e_rows)
    logger.info(f"\nWritten: {os.path.join(RESULTS_DIR,'v5_final_comparison.csv')}")

    # Write decision report
    v5_row  = next((r for r in phase_e_rows if r["policy"]=="GraphMindRL_V5"),None)
    v5t10   = next((r for r in phase_e_rows if r["policy"]=="GraphMindRL_V5_t10"),None)
    lf_row  = next((r for r in phase_e_rows if r["policy"]=="RL_LatencyFocus"),None)

    # Determine best E policy
    best_e = max(phase_e_rows, key=lambda r: r["f1"])

    lines=[
        "# V5 Final Decision",
        "",
        f"**Date:** 2026-06-06",
        f"**Baseline:** GraphMindRL F1={BASELINE_F1}  HR=0.9357  Lat=2002.5ms",
        f"**Previous best:** RL_LatencyFocus F1={BEST_CANDIDATE_F1} (p=0.0003, d=0.752)",
        "",
        "---",
        "",
        "## Phase E Results",
        "",
        "| Policy | F1 | ΔF1 | HR | p | Cohen d | Sig | ≥+0.02 |",
        "|--------|-----|-----|-----|---|---------|-----|--------|",
    ]
    for r in phase_e_rows:
        p_s=f"{r['p_value']:.4f}" if isinstance(r['p_value'],float) else str(r['p_value'])
        d_s=f"{r['cohen_d']:.3f}" if isinstance(r['cohen_d'],float)  else str(r['cohen_d'])
        sig="✅" if r["significant"] else "❌"
        meets="✅" if r["delta_f1_vs_baseline"]>=0.02 else "❌"
        lines.append(f"| {r['policy']} | {r['f1']:.4f} | {r['delta_f1_vs_baseline']:+.4f} "
                     f"| {r['hit_rate']:.4f} | {p_s} | {d_s} | {sig} | {meets} |")

    lines+=[
        "",
        "---",
        "",
        "## Configuration",
        "",
        f"- **Best weights (Phase A):** trans={best_wt}  rec={best_wr}  freq={best_wf}",
        f"- **Best threshold (Phase B):** {best_thresh}",
        f"- **Modified KN:** All variants underperform baseline (excluded from V5)",
        "",
        "---",
        "",
        "## Decision",
        "",
    ]

    if best_e["significant"] and best_e["delta_f1_vs_baseline"]>=0.02:
        lines.append(f"**✅ RECOMMEND V5 PRODUCTION DEPLOYMENT**")
        lines.append(f"")
        lines.append(f"**{best_e['policy']}** achieves:")
        lines.append(f"- F1 = {best_e['f1']:.4f}  (ΔF1 = {best_e['delta_f1_vs_baseline']:+.4f})")
        p_s=f"{best_e['p_value']:.4f}" if isinstance(best_e['p_value'],float) else str(best_e['p_value'])
        d_s=f"{best_e['cohen_d']:.3f}"  if isinstance(best_e['cohen_d'],float)  else str(best_e['cohen_d'])
        lines.append(f"- p = {p_s}  Cohen d = {d_s}")
        lines.append(f"- Statistically significant, meets +0.02 threshold")
        lines.append(f"")
        lines.append(f"**Configuration to promote to production:**")
        lines.append(f"```python")
        lines.append(f"confidence = {best_wt}*trans_prob + {best_wr}*recency + {best_wf}*frequency")
        lines.append(f"threshold  = {best_thresh}  # adaptive ±0.005 based on hit rate")
        lines.append(f"budget     = {HOT_SIZE}")
        lines.append(f"```")
    elif best_e["significant"]:
        lines.append(f"**⚠️ PARTIAL SUCCESS — significant improvement but below +0.02**")
        lines.append(f"")
        lines.append(f"Best policy: **{best_e['policy']}** F1={best_e['f1']:.4f} (ΔF1={best_e['delta_f1_vs_baseline']:+.4f})")
        lines.append(f"Freeze at RL_LatencyFocus (F1={BEST_CANDIDATE_F1}). Proceed to dashboard.")
    else:
        lines.append(f"**❌ FREEZE RL_LatencyFocus. Stop research. Proceed to dashboard.**")
        lines.append(f"")
        lines.append(f"RL_LatencyFocus remains the best validated policy (F1={BEST_CANDIDATE_F1}).")

    path=os.path.join(REPORTS_DIR,"v5_final_decision.md")
    with open(path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Written: {path}")

    # Write full summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 11 COMPLETE — FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"Baseline GraphMindRL:       F1={BASELINE_F1}")
    logger.info(f"Best RL_LatencyFocus:       F1={BEST_CANDIDATE_F1}  ΔF1=+0.0116  p=0.0003")
    logger.info(f"Phase A best (weights):     trans={best_wt} rec={best_wr} freq={best_wf}")
    logger.info(f"Phase B best (threshold):   {best_thresh}")
    if v5_row:
        p_s=f"{v5_row['p_value']:.4f}" if isinstance(v5_row['p_value'],float) else str(v5_row['p_value'])
        logger.info(f"GraphMindRL_V5 (combined):  F1={v5_row['f1']:.4f}  ΔF1={v5_row['delta_f1_vs_baseline']:+.4f}  p={p_s}  d={v5_row['cohen_d']}")
    logger.info("=" * 70)

    # Compute PS03 target KPIs on the real 31-user UbiqLog dataset
    lru_thrash_total = sum(lru_thrash_by_user.values())
    if lru_thrash_total > 0:
        thrash_reduction_pct = ((lru_thrash_total - v5_thrash_events) / lru_thrash_total) * 100.0
    else:
        thrash_reduction_pct = 100.0 if v5_thrash_events == 0 else 0.0
    thrash_reduction_pct = max(0.0, round(thrash_reduction_pct, 2))

    # Note: As only one cold-start constant (_DC = 2763.0) exists in MeasuredLatencyModel, it is used for both load and launch cold-start baselines.
    cold_start_load_ms = 2763.0
    cold_start_launch_ms = 2763.0
    load_time_improvement_pct = round((v5_latency_saved / cold_start_load_ms) * 100.0, 2)
    launch_time_improvement_pct = round((v5_latency_saved / cold_start_launch_ms) * 100.0, 2)

    random_baseline = (HOT_SIZE + WARM_SIZE) / num_unique_apps_in_dataset
    if random_baseline > 0:
        memory_util_improvement_pct = ((v5_hit_rate - random_baseline) / random_baseline) * 100.0
    else:
        memory_util_improvement_pct = 0.0
    memory_util_improvement_pct = round(memory_util_improvement_pct, 2)

    kpi_pass_fail = {
        "next_context_prediction_f1": "PASS" if v5_f1 >= 0.75 else "FAIL",
        "cache_hit_rate_pct": "PASS" if (v5_hit_rate * 100.0) >= 85.0 else "FAIL",
        "thrash_reduction_pct": "PASS" if thrash_reduction_pct >= 50.0 else "FAIL",
        "load_time_improvement_pct": "PASS" if load_time_improvement_pct >= 20.0 else "FAIL",
        "launch_time_improvement_pct": "PASS" if launch_time_improvement_pct >= 10.0 else "FAIL",
        "system_stability_issues": "PASS" if stability_issues == 0 else "FAIL",
        "memory_utilization_efficiency_improvement_pct": "PASS" if memory_util_improvement_pct >= 30.0 else "FAIL"
    }

    kpi_summary_real = {
        "dataset": "UbiqLog_real_31_users",
        "next_context_prediction_f1": round(v5_f1, 4),
        "cache_hit_rate_pct": round(v5_hit_rate * 100.0, 2),
        "thrash_reduction_pct": round(thrash_reduction_pct, 2),
        "load_time_improvement_pct": round(load_time_improvement_pct, 2),
        "launch_time_improvement_pct": round(launch_time_improvement_pct, 2),
        "system_stability_issues": int(stability_issues),
        "memory_utilization_efficiency_improvement_pct": round(memory_util_improvement_pct, 2),
        "kpi_pass_fail": kpi_pass_fail
    }

    real_summary_path = os.path.join(REPORTS_DIR, "kpi_summary_real.json")
    with open(real_summary_path, "w", encoding="utf-8") as fh:
        json.dump(kpi_summary_real, fh, indent=2)
    logger.info(f"Written KPI summary of real dataset evaluation to: {real_summary_path}")

    # Print the 7-row KPI table format used by evaluator_v2.py to stdout
    print()
    print("=" * 82)
    print(f"  {'KPI':<45} {'Target':>10} {'Achieved':>12} {'Status':>8}")
    print("=" * 82)

    rows = [
        ("Next Context Prediction Accuracy (F1)",
         ">=0.75",
         f"{v5_f1:.4f}",
         kpi_pass_fail["next_context_prediction_f1"]),
        ("Cache Hit Rate (%)",
         ">=85%",
         f"{v5_hit_rate * 100.0:.2f}%",
         kpi_pass_fail["cache_hit_rate_pct"]),
        ("Memory Thrashing Reduction (%)",
         ">=50%",
         f"{thrash_reduction_pct:.2f}%",
         kpi_pass_fail["thrash_reduction_pct"]),
        ("App Load Time Improvement (%)",
         ">=20%",
         f"{load_time_improvement_pct:.2f}%",
         kpi_pass_fail["load_time_improvement_pct"]),
        ("App Launch Time Improvement (%)",
         ">=10%",
         f"{launch_time_improvement_pct:.2f}%",
         kpi_pass_fail["launch_time_improvement_pct"]),
        ("System Stability (issues)",
         "= 0",
         str(stability_issues),
         kpi_pass_fail["system_stability_issues"]),
        ("Memory Utilisation Efficiency Improvement (%)",
         ">=30%",
         f"{memory_util_improvement_pct:.2f}%",
         kpi_pass_fail["memory_utilization_efficiency_improvement_pct"]),
    ]

    for name, target, achieved, status in rows:
        status_str = f"[PASS]" if status == "PASS" else f"[FAIL]"
        print(f"  {name:<45} {target:>10} {achieved:>12}  {status_str}")

    print("=" * 82)
    n_pass = sum(1 for v in kpi_pass_fail.values() if v == "PASS")
    n_total = len(kpi_pass_fail)
    print(f"  Overall: {n_pass}/{n_total} KPIs PASS")
    print("=" * 82)


if __name__=="__main__":
    main()
