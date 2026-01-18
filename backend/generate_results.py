# ============================================================
# RESEARCH-GRADE RL NUTRITION SIM (Fixed Statistics & Logic)
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
import os

# ------------------------------------------------------------
# 0. CONFIG & SEEDS
# ------------------------------------------------------------
# reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ------------------------------------------------------------
# 1. LOAD DATA (With Dummy Fallback for Testing)
# ------------------------------------------------------------
try:
    df = pd.read_csv("food_data.csv")
except FileNotFoundError:
    print("WARNING: 'data/food_data.csv' not found. Generating dummy data for simulation...")
    # Generate synthetic data if file missing
    data_size = 500
    df = pd.DataFrame({
        "food_name": [f"Food_{i}" for i in range(data_size)],
        "energy_kcal": np.random.uniform(50, 800, data_size),
        "carb_g": np.random.uniform(0, 100, data_size),
        "protein_g": np.random.uniform(0, 50, data_size),
        "fat_g": np.random.uniform(0, 50, data_size),
        "freesugar_g": np.random.uniform(0, 50, data_size),
        "fibre_g": np.random.uniform(0, 20, data_size),
        "unit_serving_vitc_mg": np.random.uniform(0, 100, data_size)
    })

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
gmm = GaussianMixture(n_components=4, random_state=42).fit(Xp)
cluster = gmm.predict(Xp)
contexts = ["C1","C2","C3","C4"]
cluster_map = {0:"C3", 1:"C1", 2:"C2", 3:"C4"}

def rule_ctx(n):
    kcal,carb,protein,fat,sugar,fibre,vitC = n
    scores={
        "C1": carb + kcal*0.3,        # energy biased
        "C2": protein*1.2,            # protein biased
        "C3": max(0, 200-kcal),       # low calorie
        "C4": max(0, 35-carb)         # low carb
    }
    return max(scores, key=scores.get)

def assign(i):
    base = rule_ctx(data[i])
    cm = cluster_map.get(cluster[i], "C1")
    if base in ["C2","C4"]: return base
    return cm

ctx_assign = [assign(i) for i in range(len(df))]
ctx_dict = {c:[] for c in contexts}
for i in range(len(df)):
    ctx_dict[ctx_assign[i]].append({"name":food_names[i], "n":data[i]})

# ------------------------------------------------------------
# 3. REWARD (Stochastic to fix UCB Std=0 Issue)
# ------------------------------------------------------------
def rew(c, n):
    kcal,carb,p,fat,s,fibre,vitC = n
    protein_density = p / (kcal+1)
    energy_density = kcal / 100
    insulin = carb - fibre - p*0.5
    
    base_r = 0
    if c=="C1": base_r = energy_density - insulin*0.1
    elif c=="C2": base_r = protein_density*3 - fat*0.05
    elif c=="C3": base_r = -energy_density + fibre*0.1
    elif c=="C4": base_r = -insulin + protein_density*0.5
    
    # [FIX] Added Gaussian Noise to simulate real-world variability
    # This prevents UCB from having 0.000 std dev
    noise = np.random.normal(0, 0.05) 
    return base_r + noise

def norm(x):
    x = np.array(x)
    if x.std() == 0: return x - x.mean()
    return (x - x.mean()) / (x.std() + 1e-8)

def ema(x, a=0.05):
    y = []
    prev = x[0]
    for v in x:
        prev = a*v + (1-a)*prev
        y.append(prev)
    return y

# ------------------------------------------------------------
# 4. RL AGENTS
# ------------------------------------------------------------
class UCBAgent:
    def __init__(self, A):
        self.A = A
        self.c = np.zeros(A)
        self.v = np.zeros(A) # Estimated value
        
    def select(self):
        t = sum(self.c) + 1
        # UCB1 formula
        u = self.v + np.sqrt(2 * np.log(t) / (self.c + 1e-8))
        return np.argmax(u)
        
    def update(self, a, r):
        self.c[a] += 1
        # Incremental mean update
        self.v[a] += (r - self.v[a]) / self.c[a]

STATE_DIM = 1
class DQN(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, A)
        )
    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(self, A):
        self.A = A
        self.m = DQN(A)
        self.opt = optim.Adam(self.m.parameters(), lr=0.001)
        self.eps = 1.0
        
    def select(self, s):
        if random.random() < self.eps:
            return random.randint(0, self.A-1)
        with torch.no_grad():
            q = self.m(torch.tensor([[s]], dtype=torch.float32))
        return q.argmax().item()
        
    def update(self, s, a, r):
        s_tens = torch.tensor([[s]], dtype=torch.float32)
        q = self.m(s_tens)
        tgt = q.clone()
        tgt[0, a] = r
        loss = (q - tgt).pow(2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.eps = max(self.eps*0.995, 0.05)

class PG(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, A)
        )
    def forward(self, x): return self.net(x)

class PGAgent:
    def __init__(self, A):
        self.A = A
        self.m = PG(A)
        self.opt = optim.Adam(self.m.parameters(), lr=0.0005)
        self.saved_log_prob = None
        
    def select(self, s):
        logits = self.m(torch.tensor([[s]], dtype=torch.float32))
        p = torch.softmax(logits, dim=-1)
        p = torch.clamp(p, 1e-8, 1-1e-8) # Numerical stability
        p = p / p.sum()
        d = torch.distributions.Categorical(p)
        a = d.sample()
        self.saved_log_prob = d.log_prob(a)
        return a.item()
        
    def update(self, r):
        if self.saved_log_prob is None: return
        loss = -self.saved_log_prob * r
        self.opt.zero_grad(); loss.backward(); self.opt.step()

# ------------------------------------------------------------
# 5. RUN EXPERIMENT (multi-seed)
# ------------------------------------------------------------
def run(c, algo, eps=500, seeds=7):
    foods = ctx_dict[c]
    A = len(foods)
    if A < 2: return None, None, None, None
    
    all_runs = []
    
    for sd in range(seeds):
        # Set seeds for reproducibility PER RUN
        np.random.seed(sd)
        torch.manual_seed(sd)
        random.seed(sd)
        
        ag = UCBAgent(A) if algo=="UCB" else DQNAgent(A) if algo=="DQN" else PGAgent(A)
        
        r_history = []
        for _ in range(eps):
            s = 0.5 # Constant state for bandits/simplified env
            a = ag.select(s) if algo!="UCB" else ag.select()
            
            # Get Reward
            val = rew(c, foods[a]["n"])
            
            if algo=="UCB": ag.update(a, val)
            elif algo=="DQN": ag.update(s, a, val)
            else: ag.update(val)
            
            r_history.append(val)
            
        # Normalize PER EPISODE history to make them comparable
        r_history = norm(r_history)
        
        # Save raw (normalized) data for stats
        all_runs.append(r_history)
        
    all_runs = np.array(all_runs) # Shape: (seeds, eps)
    
    # Calculate smoothed curves for plotting
    smoothed_runs = np.array([ema(run) for run in all_runs])
    mean_curve = np.mean(smoothed_runs, axis=0)
    std_curve = np.std(smoothed_runs, axis=0)
    
    # Cumulative calculation
    cum_curve = np.cumsum(mean_curve)
    cum_std = np.std(np.cumsum(all_runs, axis=1), axis=0)
    
    return mean_curve, std_curve, (cum_curve, cum_std), all_runs

algos = ["UCB", "DQN", "PG"]

# ------------------------------------------------------------
# 6. GENERATE DATA FIRST
# ------------------------------------------------------------
print("Running simulations... please wait.")
res = {}
for c in contexts:
    res[c] = {}
    for a in algos:
        m, s, (cum, cum_s), raw = run(c, a)
        res[c][a] = dict(mean=m, std=s, cum=cum, cum_std=cum_s, raw=raw)

# ------------------------------------------------------------
# 7. PLOTTING (Learning Curves)
# ------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
colors = {"UCB":"#2d7dd2", "DQN":"#f26419", "PG":"#2a9d8f"}

for i, c in enumerate(contexts):
    ax = axs[i]
    for a in algos:
        data_ = res[c][a]
        m, s = data_["mean"], data_["std"]
        x = np.arange(len(m))
        ax.plot(x, m, label=a, color=colors[a], linewidth=2.2)
        ax.fill_between(x, m-s, m+s, color=colors[a], alpha=0.2)
    ax.set_title(f"Context {c}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Normalized Reward")
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. STATISTICAL ANALYSIS (With Logic Fixes)
# ------------------------------------------------------------
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx-1)*sx + (ny-1)*sy) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / (pooled + 1e-12)

print("\n=== FINAL STATISTICAL ANALYSIS (SEEDS=7, Last 50 Episodes) ===")

anova_rows = []
tukey_rows = []
rank_rows = []

for c in contexts:
    print(f"\n--- Context {c} ---")
    
    # 1. Aggregate samples (Mean of last 50 episodes per seed)
    samples = {}
    for a in algos:
        raw = res[c][a]["raw"] # (seeds, 500)
        window = raw[:, -50:]  # Last 50 episodes
        per_seed_mean = window.mean(axis=1) # (seeds,)
        samples[a] = per_seed_mean

    # 2. ANOVA
    F, p = f_oneway(samples["UCB"], samples["DQN"], samples["PG"])
    
    # Eta-squared
    all_vals = np.concatenate(list(samples.values()))
    grand_mean = all_vals.mean()
    ss_total = ((all_vals - grand_mean)**2).sum()
    ss_between = sum([len(samples[a]) * (samples[a].mean() - grand_mean)**2 for a in algos])
    eta = ss_between / (ss_total + 1e-12)
    
    anova_rows.append([c, F, p, eta])
    print(f"ANOVA: F={F:.4f}, p={p:.4g}, η²={eta:.3f}")

    # 3. Tukey HSD
    data_list = []
    labels_list = []
    for a in algos:
        data_list.extend(samples[a])
        labels_list.extend([a]*len(samples[a]))
        
    comp = mc.MultiComparison(data_list, labels_list)
    t_res = comp.tukeyhsd()
    
    # Store significant relationships for ranking
    # Dict format: significant_diffs[(Winner, Loser)] = True
    significant_wins = {} 
    
    print("\nTukey HSD (Corrected Direction, g1-g2):")
    # Access internal data of Tukey Results
    # Row format: group1, group2, meandiff, p-adj, lower, upper, reject
    for row in t_res._results_table.data[1:]:
        g1, g2, _, p_adj, _, _, reject = row
        
        # Calculate manual diff to ensure direction (Group1 - Group2)
        diff = samples[g1].mean() - samples[g2].mean()
        d_score = cohen_d(samples[g1], samples[g2])
        
        tukey_rows.append([c, g1, g2, diff, p_adj, reject, d_score])
        
        symbol = ">" if diff > 0 else "<"
        print(f"{g1} {symbol} {g2}: Δ={diff:.3f}, p={p_adj:.4g}, reject={reject}, d={d_score:.3f}")
        
        # Record wins for ranking logic
        if reject:
            winner = g1 if diff > 0 else g2
            loser = g2 if diff > 0 else g1
            significant_wins[(winner, loser)] = True

    # 4. Smart Ranking (Handling Statistical Ties)
    # Sort by numerical mean first
    means = {a: samples[a].mean() for a in algos}
    sorted_algos = sorted(algos, key=lambda x: means[x], reverse=True)
    
    numerical_winner = sorted_algos[0]
    
    # Identify Statistical Ties for First Place
    # An algo is tied with the winner if (Winner, Algo) is NOT in significant_wins
    actual_winners = [numerical_winner]
    for candidate in sorted_algos[1:]:
        # If the numerical winner did NOT statistically beat the candidate
        if (numerical_winner, candidate) not in significant_wins:
            actual_winners.append(candidate)
            
    # Losers are everyone else
    actual_losers = [a for a in algos if a not in actual_winners]
    
    rank_rows.append([c, ", ".join(actual_winners), ", ".join(actual_losers)])

# ============================================================
# 9. OUTPUT TABLES
# ============================================================

print("\n=== PERFORMANCE TABLE (Mean ± Std, Final Window) ===")
perf_tbl = pd.DataFrame(index=algos, columns=contexts)

for c in contexts:
    for a in algos:
        raw = res[c][a]["raw"]
        window = raw[:, -50:]
        per_seed_mean = window.mean(axis=1) # Average reward per seed
        # Calculate Mean and Std ACROSS SEEDS
        m_val = per_seed_mean.mean()
        s_val = per_seed_mean.std() 
        perf_tbl.loc[a, c] = f"{m_val:.3f} ± {s_val:.3f}"

print(perf_tbl)

print("\n=== RANK TABLE (Winner / Loser) ===")
rank_tbl = pd.DataFrame(rank_rows, columns=["Context", "Winner(s)", "Loser(s)"])
print(rank_tbl)

# Export
rank_tbl.to_csv("rank_table.csv", index=False)
perf_tbl.to_csv("performance_table.csv")
print("\nAnalysis Complete.")