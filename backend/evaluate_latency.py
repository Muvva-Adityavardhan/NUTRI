# import time
# import numpy as np
# import torch
# from rl_agents import UCBAgent, DQNAgent, STATE_SIZE
# from data_loader import load_and_process_data

# # Load Data
# data = load_and_process_data("data/food_data.csv").get("Lunch", [])[:30]
# state = [0.4, 1.0, 0.5, 1.0]

# # Init Agents
# ucb = UCBAgent(data)
# dqn = DQNAgent(STATE_SIZE, data)

# # Measure UCB
# start = time.time()
# for _ in range(1000):
#     ucb.select_action()
# end = time.time()
# ucb_time = (end - start) / 1000 * 1000 # Convert to ms

# # Measure DQN
# # Warmup pytorch
# dqn.select_action(state)
# start = time.time()
# for _ in range(1000):
#     dqn.select_action(state)
# end = time.time()
# dqn_time = (end - start) / 1000 * 1000 # Convert to ms

# print(f"--- LATENCY RESULTS ---")
# print(f"UCB Inference Time: {ucb_time:.4f} ms")
# print(f"DQN Inference Time: {dqn_time:.4f} ms")











# ============================================================
# RESEARCH-GRADE RL NUTRITION SIM (Contexts + Cumulative + Regret + Stats)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.stats import f_oneway
import statsmodels.stats.multicomp as mc
import torch
import torch.nn as nn
import torch.optim as optim
import random

# reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("data/food_data.csv")

feat_cols = [
    "energy_kcal","carb_g","protein_g","fat_g",
    "freesugar_g","fibre_g","unit_serving_vitc_mg"
]
df[feat_cols] = df[feat_cols].fillna(0).astype(float)
food_names = df["food_name"].tolist()
data = df[feat_cols].values

# ------------------------------------------------------------
# 2. GMM CONTEXT ASSIGNMENT
# ------------------------------------------------------------
X = StandardScaler().fit_transform(data)
Xp = PCA(n_components=4).fit_transform(X)
gmm = GaussianMixture(n_components=4,random_state=42).fit(Xp)
cluster = gmm.predict(Xp)
contexts = ["C1","C2","C3","C4"]
cluster_map = {0:"C3",1:"C1",2:"C2",3:"C4"}

def rule_ctx(n):
    kcal,carb,protein,fat,sugar,fibre,vitC=n
    scores={
        "C1": carb + kcal*0.3,        # energy biased
        "C2": protein*1.2,            # protein biased
        "C3": max(0,200-kcal),        # low calorie
        "C4": max(0,35-carb)          # low carb
    }
    return max(scores,key=scores.get)

def assign(i):
    base=rule_ctx(data[i])
    cm=cluster_map[cluster[i]]
    if base in ["C2","C4"]: return base
    return cm

ctx_assign=[assign(i) for i in range(len(df))]
ctx_dict={c:[] for c in contexts}
for i in range(len(df)):
    ctx_dict[ctx_assign[i]].append({"name":food_names[i],"n":data[i]})

# ------------------------------------------------------------
# 3. REWARD (bounded + normalized + scientific)
# ------------------------------------------------------------
def rew(c,n):
    kcal,carb,p,fat,s,fibre,vitC=n
    protein_density=p/(kcal+1)
    energy_density=kcal/100
    insulin=carb - fibre - p*0.5
    if c=="C1": return energy_density - insulin*0.1
    if c=="C2": return protein_density*3 - fat*0.05
    if c=="C3": return -energy_density + fibre*0.1
    if c=="C4": return -insulin + protein_density*0.5
    return 0

def norm(x):
    x=np.array(x)
    return (x-x.mean())/(x.std()+1e-8)

def ema(x,a=0.05):
    y=[]; prev=x[0]
    for v in x:
        prev=a*v+(1-a)*prev
        y.append(prev)
    return y

# ------------------------------------------------------------
# 4. RL AGENTS
# ------------------------------------------------------------
class UCBAgent:
    def __init__(self,A):
        self.A=A; self.c=np.zeros(A); self.v=np.zeros(A)
    def select(self):
        t=sum(self.c)+1
        u=self.v+np.sqrt(2*np.log(t)/(self.c+1e-8))
        return np.argmax(u)
    def update(self,a,r):
        self.c[a]+=1
        self.v[a]+= (r-self.v[a])/self.c[a]

STATE=1
class DQN(nn.Module):
    def __init__(self,A):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(STATE,32),nn.ReLU(),nn.Linear(32,A))
    def forward(self,x): return self.net(x)

class DQNAgent:
    def __init__(self,A):
        self.A=A; self.m=DQN(A)
        self.opt=optim.Adam(self.m.parameters(),lr=0.001)
        self.eps=1.0
    def select(self,s):
        if random.random()<self.eps: return random.randint(0,self.A-1)
        with torch.no_grad():
            q=self.m(torch.tensor([[s]],dtype=torch.float32))
        return q.argmax().item()
    def update(self,s,a,r):
        s=torch.tensor([[s]],dtype=torch.float32)
        q=self.m(s)
        tgt=q.clone()
        tgt[0,a]=r
        loss=(q-tgt).pow(2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.eps=max(self.eps*0.995,0.05)

class PG(nn.Module):
    def __init__(self,A):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(STATE,32),nn.ReLU(),nn.Linear(32,A))
    def forward(self,x): return self.net(x)

class PGAgent:
    def __init__(self,A):
        self.A=A; self.m=PG(A)
        self.opt=optim.Adam(self.m.parameters(),lr=0.0005)
    def select(self,s):
        logits=self.m(torch.tensor([[s]],dtype=torch.float32))
        p=torch.softmax(logits,dim=-1)
        p=torch.clamp(p,1e-8,1-1e-8); p=p/p.sum()
        d=torch.distributions.Categorical(p)
        a=d.sample()
        self.saved=d.log_prob(a)
        return a.item()
    def update(self,r):
        loss=-self.saved*r
        self.opt.zero_grad(); loss.backward(); self.opt.step()

# ------------------------------------------------------------
# 5. RUN EXPERIMENT (multi-seed)
# ------------------------------------------------------------
def run(c,algo,eps=500,seeds=7):
    foods=ctx_dict[c]; A=len(foods)
    if A<2: return None,None,None
    all_runs=[]
    for sd in range(seeds):
        np.random.seed(sd); torch.manual_seed(sd); random.seed(sd)
        ag = UCBAgent(A) if algo=="UCB" else DQNAgent(A) if algo=="DQN" else PGAgent(A)
        r=[]
        for _ in range(eps):
            s=0.5
            a=ag.select(s) if algo!="UCB" else ag.select()
            val=rew(c,foods[a]["n"])
            if algo=="UCB": ag.update(a,val)
            elif algo=="DQN": ag.update(s,a,val)
            else: ag.update(val)
            r.append(val)
        r=norm(r); r=ema(r)
        all_runs.append(r)
    all_runs=np.array(all_runs)
    mean=np.mean(all_runs,axis=0)
    std=np.std(all_runs,axis=0)
    cum=np.cumsum(mean)
    cum_std=np.std(np.cumsum(all_runs,axis=1),axis=0)
    return mean,std,(cum,cum_std),all_runs

algos=["UCB","DQN","PG"]

# ------------------------------------------------------------
# 6. LEARNING CURVES (already validated style)
# ------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig,axs=plt.subplots(2,2,figsize=(14,10))
axs=axs.flatten()
colors={"UCB":"#2d7dd2","DQN":"#f26419","PG":"#2a9d8f"}

res={}
for i,c in enumerate(contexts):
    ax=axs[i]
    res[c]={}
    for a in algos:
        m,s,(cum,cum_s),raw = run(c,a)
        res[c][a]=dict(mean=m,std=s,cum=cum,cum_std=cum_s,raw=raw)
        x=np.arange(len(m))
        ax.plot(x,m,label=a,color=colors[a],linewidth=2.2)
        ax.fill_between(x,m-s,m+s,color=colors[a],alpha=0.2)
    ax.set_title(f"Context {c}",fontsize=14,fontweight='bold')
    ax.set_ylim(-2,2)
    ax.set_xlabel("Episode"); ax.set_ylabel("Norm Reward")
    ax.grid(alpha=0.3); ax.legend()

plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# 7. CUMULATIVE REWARD PLOTS
# ------------------------------------------------------------
fig,axs=plt.subplots(2,2,figsize=(14,10))
axs=axs.flatten()
for i,c in enumerate(contexts):
    ax=axs[i]
    for a in algos:
        cum=res[c][a]["cum"]
        cum_s=res[c][a]["cum_std"]
        x=np.arange(len(cum))
        ax.plot(x,cum,label=a,color=colors[a],linewidth=2.2)
        ax.fill_between(x,cum-cum_s,cum+cum_s,color=colors[a],alpha=0.2)
    ax.set_title(f"Cumulative Reward — {c}",fontsize=14,fontweight='bold')
    ax.grid(alpha=0.3); ax.legend()

plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# 8. REGRET CURVES
# ------------------------------------------------------------
fig,axs=plt.subplots(2,2,figsize=(14,10))
axs=axs.flatten()
for i,c in enumerate(contexts):
    ax=axs[i]
    all_cums=[res[c][a]["cum"] for a in algos]
    opt=np.max(all_cums,axis=0)  # hindsight oracle
    for a in algos:
        reg=opt - res[c][a]["cum"]
        ax.plot(reg,label=a,color=colors[a],linewidth=2.2)
    ax.set_title(f"Regret — {c}",fontsize=14,fontweight='bold')
    ax.set_xlabel("Episode"); ax.set_ylabel("Regret")
    ax.grid(alpha=0.3); ax.legend()

plt.tight_layout(); plt.show()

# # ------------------------------------------------------------
# # 9. STATISTICAL TESTS (ANOVA + Tukey)
# # ------------------------------------------------------------
# print("\n=== STATISTICAL TESTS (Final 50 episodes) ===")
# for c in contexts:
#     print(f"\nContext {c}")
#     samples=[]
#     for a in algos:
#         raw=res[c][a]["raw"]
#         last=raw[:,-50:].mean(axis=1)  # avg of last 50 episodes across seeds
#         samples.append(last)
#     F,p=f_oneway(*samples)
#     print(f"ANOVA: F={F:.4f}, p={p:.4f}")
#     if p<0.05:
#         data=[]
#         label=[]
#         for ix,a in enumerate(algos):
#             for v in samples[ix]:
#                 data.append(v); label.append(a)
#         t=mc.MultiComparison(data,label)
#         print(t.tukeyhsd())
#     else:
#         print("No significant difference (p>=0.05).")

# # ------------------------------------------------------------
# # 10. TABLES
# # ------------------------------------------------------------
# print("\n=== PERFORMANCE TABLE (Final Window Mean ± Std) ===")
# tbl=pd.DataFrame(index=algos,columns=contexts)
# for c in contexts:
#     for a in algos:
#         raw=res[c][a]["raw"]
#         last=raw[:,-50:].mean(axis=1)
#         tbl.loc[a,c]=f"{last.mean():.3f} ± {last.std():.3f}"
# print(tbl)

# print("\n=== CUMULATIVE REWARD TABLE (Final Value) ===")
# tbl2=pd.DataFrame(index=algos,columns=contexts)
# for c in contexts:
#     for a in algos:
#         cv=res[c][a]["cum"][-1]
#         tbl2.loc[a,c]=f"{cv:.2f}"
# print(tbl2)

# ============================================================
# 9. FIXED STATISTICAL ANALYSIS (Correct Tukey + Std + Effect Sizes)
# ============================================================

from scipy.stats import f_oneway
import statsmodels.stats.multicomp as mc
import itertools

def cohen_d(x,y):
    nx, ny = len(x), len(y)
    sx, sy = np.var(x,ddof=1), np.var(y,ddof=1)
    pooled = np.sqrt(((nx-1)*sx + (ny-1)*sy) / (nx+ny-2))
    return (np.mean(x)-np.mean(y)) / (pooled+1e-12)

print("\n=== FINAL STATISTICAL ANALYSIS (SEEDS=7, Last 50 Episodes) ===")

anova_rows = []
tukey_rows = []
rank_rows = []

for c in contexts:
    print(f"\n--- Context {c} ---")

    samples = {}
    for a in algos:
        raw = res[c][a]["raw"]          # (seeds, episodes)
        window = raw[:, -50:]           # last 50 episodes across seeds
        per_seed_mean = window.mean(axis=1)
        samples[a] = per_seed_mean

    # ---------- ANOVA ----------
    F, p = f_oneway(samples["UCB"], samples["DQN"], samples["PG"])

    # effect size η²
    all_vals = np.concatenate(list(samples.values()))
    grand = all_vals.mean()
    ss_total = ((all_vals-grand)**2).sum()
    ss_between = sum([len(samples[a])*(samples[a].mean()-grand)**2 for a in algos])
    eta = ss_between/(ss_total+1e-12)

    anova_rows.append([c, F, p, eta])
    print(f"ANOVA: F={F:.4f}, p={p:.4g}, η²={eta:.3f}")

    # ---------- Tukey ----------
    data=[]
    labels=[]
    for a in algos:
        for v in samples[a]:
            data.append(v)
            labels.append(a)

    comp = mc.MultiComparison(data, labels)
    t_res = comp.tukeyhsd()
    print("\nTukey HSD (Corrected Direction, g1-g2):")

    for row in t_res._results_table.data[1:]:
        g1,g2,_,p_adj,_,_,reject = row

        # corrected difference = mean(g1) - mean(g2)
        diff = samples[g1].mean() - samples[g2].mean()

        # cohen d (directional)
        d = cohen_d(samples[g1], samples[g2])

        tukey_rows.append([c,g1,g2,diff,p_adj,reject,d])

        # readable printing:
        symbol = ">" if diff>0 else "<"
        print(f"{g1} {symbol} {g2}: Δ={diff:.3f}, p={p_adj:.4g}, reject={reject}, d={d:.3f}")

    # ---------- Ranking (with ties) ----------
    means = {a: samples[a].mean() for a in algos}
    sorted_means = sorted(means.items(), key=lambda x:x[1], reverse=True)
    top_val = sorted_means[0][1]
    winners = [a for a,v in sorted_means if abs(v-top_val)<1e-6]
    losers  = [a for a,v in sorted_means if a not in winners]
    rank_rows.append([c, ", ".join(winners), ", ".join(losers)])

# ============================================================
# 10. PERFORMANCE TABLES (Correct Mean ± Std)
# ============================================================

print("\n=== PERFORMANCE TABLE (Mean ± Std, Final Window) ===")
perf_tbl=pd.DataFrame(index=algos,columns=contexts)

for c in contexts:
    for a in algos:
        raw=res[c][a]["raw"]
        window=raw[:,-50:]
        per_seed=window.mean(axis=1)
        perf_tbl.loc[a,c]=f"{per_seed.mean():.3f} ± {per_seed.std():.3f}"

print(perf_tbl)

print("\n=== CUMULATIVE REWARD TABLE (Final Value) ===")
cum_tbl=pd.DataFrame(index=algos,columns=contexts)
for c in contexts:
    for a in algos:
        cv=res[c][a]["cum"][-1]
        cum_tbl.loc[a,c]=f"{cv:.2f}"
print(cum_tbl)

print("\n=== RANK TABLE (Winner / Loser) ===")
rank_tbl=pd.DataFrame(rank_rows,columns=["Context","Winner(s)","Loser(s)"])
print(rank_tbl)

# ============================================================
# 11. EXPORTS (CSV + LaTeX)
# ============================================================

pd.DataFrame(anova_rows,columns=["Context","F","p","eta_sq"]).to_csv("anova_results.csv",index=False)
pd.DataFrame(tukey_rows,columns=["Context","Group1","Group2","Diff(g1-g2)","p_adj","Reject","Cohen_d"]).to_csv("tukey_results.csv",index=False)
perf_tbl.to_csv("performance_table.csv")
cum_tbl.to_csv("cumulative_table.csv")
rank_tbl.to_csv("rank_table.csv",index=False)

with open("performance_table.tex","w") as f: f.write(perf_tbl.to_latex())
with open("cumulative_table.tex","w") as f: f.write(cum_tbl.to_latex())
with open("rank_table.tex","w") as f: f.write(rank_tbl.to_latex(index=False))

print("\nExported CSV + LaTeX tables successfully.")
