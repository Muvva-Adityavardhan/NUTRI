# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.mixture import GaussianMixture
# from sklearn.decomposition import PCA
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random

# # ============================================================
# # 1. SETUP & DATA GENERATION
# # ============================================================
# np.random.seed(42)
# torch.manual_seed(42)
# random.seed(42)

# print("Generating synthetic food data...")
# data_size = 500
# df = pd.DataFrame({
#     "food_name": [f"Food_{i}" for i in range(data_size)],
#     "energy_kcal": np.random.uniform(50, 800, data_size),
#     "carb_g": np.random.uniform(0, 100, data_size),
#     "protein_g": np.random.uniform(0, 50, data_size),
#     "fat_g": np.random.uniform(0, 50, data_size),
#     "freesugar_g": np.random.uniform(0, 50, data_size),
#     "fibre_g": np.random.uniform(0, 20, data_size),
#     "unit_serving_vitc_mg": np.random.uniform(0, 100, data_size)
# })

# feat_cols = ["energy_kcal","carb_g","protein_g","fat_g","freesugar_g","fibre_g","unit_serving_vitc_mg"]
# df[feat_cols] = df[feat_cols].fillna(0).astype(float)
# food_names = df["food_name"].tolist()
# data = df[feat_cols].values

# # Context Clustering
# X = StandardScaler().fit_transform(data)
# Xp = PCA(n_components=4).fit_transform(X)
# gmm = GaussianMixture(n_components=4, random_state=42).fit(Xp)
# cluster = gmm.predict(Xp)
# contexts = ["C1","C2","C3","C4"]
# cluster_map = {0:"C3", 1:"C1", 2:"C2", 3:"C4"}

# def rule_ctx(n):
#     kcal,carb,protein,fat,sugar,fibre,vitC = n
#     scores = {
#         "C1": carb + kcal*0.3,    # High Energy
#         "C2": protein*1.2,        # High Protein
#         "C3": max(0, 200-kcal),   # Low Calorie
#         "C4": max(0, 35-carb)     # Low Carb
#     }
#     return max(scores, key=scores.get)

# def assign(i):
#     base = rule_ctx(data[i])
#     cm = cluster_map.get(cluster[i], "C1")
#     if base in ["C2","C4"]: return base
#     return cm

# ctx_assign = [assign(i) for i in range(len(df))]
# ctx_dict = {c:[] for c in contexts}
# for i in range(len(df)):
#     ctx_dict[ctx_assign[i]].append({"name":food_names[i], "n":data[i]})

# # ============================================================
# # 2. REWARD FUNCTION
# # ============================================================
# def rew(c, n):
#     kcal,carb,p,fat,s,fibre,vitC = n
#     protein_density = p / (kcal+1)
#     energy_density = kcal / 100
#     insulin = carb - fibre - p*0.5
    
#     base_r = 0
#     if c=="C1": base_r = energy_density - insulin*0.1
#     elif c=="C2": base_r = protein_density*3 - fat*0.05
#     elif c=="C3": base_r = -energy_density + fibre*0.1
#     elif c=="C4": base_r = -insulin + protein_density*0.5
    
#     # Noise for realism
#     noise = np.random.normal(0, 0.05) 
#     return base_r + noise

# def norm(x):
#     x = np.array(x)
#     if x.std() == 0: return x - x.mean()
#     return (x - x.mean()) / (x.std() + 1e-8)

# def ema(x, a=0.05):
#     y = []
#     prev = x[0]
#     for v in x:
#         prev = a*v + (1-a)*prev
#         y.append(prev)
#     return y

# # ============================================================
# # 3. AGENT CLASSES (Standard Only)
# # ============================================================

# # --- 1. Standard UCB ---
# class UCBAgent:
#     def __init__(self, A):
#         self.A = A
#         self.c = np.zeros(A)
#         self.v = np.zeros(A)
#     def select(self):
#         t = sum(self.c) + 1
#         u = self.v + np.sqrt(2 * np.log(t) / (self.c + 1e-8))
#         return np.argmax(u)
#     def update(self, a, r):
#         self.c[a] += 1
#         self.v[a] += (r - self.v[a]) / self.c[a]

# # --- 2. DQN ---
# STATE_DIM = 1
# class DQN(nn.Module):
#     def __init__(self, A):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(STATE_DIM, 32), nn.ReLU(), nn.Linear(32, A))
#     def forward(self, x): return self.net(x)

# class DQNAgent:
#     def __init__(self, A):
#         self.A = A
#         self.m = DQN(A)
#         self.opt = optim.Adam(self.m.parameters(), lr=0.01)
#         self.eps = 1.0
        
#     def select(self, s):
#         if random.random() < self.eps: return random.randint(0, self.A-1)
#         with torch.no_grad():
#             q = self.m(torch.tensor([[s]], dtype=torch.float32))
#         return q.argmax().item()
        
#     def update(self, s, a, r):
#         s_tens = torch.tensor([[s]], dtype=torch.float32)
#         q = self.m(s_tens)
#         tgt = q.clone()
#         tgt[0, a] = r
#         loss = (q - tgt).pow(2).mean()
#         self.opt.zero_grad(); loss.backward(); self.opt.step()
#         self.eps = max(self.eps*0.99, 0.05)

# # --- 3. Policy Gradient (PG) ---
# class PG(nn.Module):
#     def __init__(self, A):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(STATE_DIM, 32), nn.ReLU(), nn.Linear(32, A))
#     def forward(self, x): return self.net(x)

# class PGAgent:
#     def __init__(self, A):
#         self.A = A
#         self.m = PG(A)
#         self.opt = optim.Adam(self.m.parameters(), lr=0.01)
#         self.saved_log_prob = None
        
#     def select(self, s):
#         logits = self.m(torch.tensor([[s]], dtype=torch.float32))
#         p = torch.softmax(logits, dim=-1)
#         p = torch.clamp(p, 1e-8, 1-1e-8)
#         p = p / p.sum()
        
#         d = torch.distributions.Categorical(p)
#         a = d.sample()
#         self.saved_log_prob = d.log_prob(a)
#         return a.item()
        
#     def update(self, r):
#         if self.saved_log_prob is None: return
#         loss = -self.saved_log_prob * r
#         self.opt.zero_grad(); loss.backward(); self.opt.step()

# # ============================================================
# # 4. EXPERIMENT LOOP
# # ============================================================
# def run_comparison(switch_point=300, total_eps=600):
#     context_A = "C1" # High Energy
#     context_B = "C3" # Low Calorie
    
#     foods = ctx_dict[context_A]
#     A = len(foods)
    
#     print(f"Running Drift Experiment ({context_A} -> {context_B})...")
    
#     # Define Standard Agents Only
#     agents = {
#         "Standard UCB": UCBAgent(A),
#         "DQN": DQNAgent(A),
#         "Policy Gradient": PGAgent(A)
#     }
    
#     results = {k: [] for k in agents.keys()}
    
#     # Common seed
#     np.random.seed(42); torch.manual_seed(42); random.seed(42)
    
#     for e in range(total_eps):
#         # Switch Rules
#         curr_ctx = context_A if e < switch_point else context_B
        
#         for name, ag in agents.items():
#             s = 0.5
            
#             # Selection
#             if name in ["DQN", "Policy Gradient"]:
#                 a = ag.select(s)
#             else:
#                 a = ag.select()
            
#             # Reward
#             val = rew(curr_ctx, foods[a]["n"])
            
#             # Update
#             if name == "DQN": 
#                 ag.update(s, a, val)
#             elif name == "Policy Gradient":
#                 ag.update(val)
#             else: 
#                 ag.update(a, val)
            
#             results[name].append(val)
            
#     return results

# # ============================================================
# # 5. PLOTTING
# # ============================================================
# def plot_drift_results():
#     results = run_comparison()
    
#     plt.figure(figsize=(12, 6))
    
#     colors = {
#         "Standard UCB": "gray", 
#         "DQN": "#f26419",           
#         "Policy Gradient": "#2a9d8f"  
#     }
    
#     styles = {
#         "Standard UCB": "--", 
#         "DQN": "-.",
#         "Policy Gradient": ":"
#     }
    
#     for name, r in results.items():
#         norm_r = norm(r)
#         smoothed = ema(norm_r, a=0.03)
#         plt.plot(smoothed, label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)
        
#     plt.axvline(x=300, color='red', linestyle=':', linewidth=2, label="User Diet Switch")
    
#     plt.title("Adaptability Analysis: Reaction to Preference Shift", fontsize=14, fontweight='bold')
#     plt.xlabel("Episodes")
#     plt.ylabel("Normalized Reward (Smoothed)")
#     plt.legend(loc="lower right")
#     plt.grid(alpha=0.3)
    
#     plt.text(310, -0.5, "New Goal:\nLow Calorie", color='red', fontsize=10)
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     plot_drift_results()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# ============================================================
# 1. SETUP & LOAD YOUR REAL DATA
# ============================================================
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Load your specific file
try:
    df = pd.read_csv("food_data.csv") # <--- POINTS TO YOUR FILE
    print("Successfully loaded 'data/food_data.csv'")
except FileNotFoundError:
    print("ERROR: Could not find 'data/food_data.csv'. Please check the path.")
    # Fallback to empty df to prevent crash, but this needs your file
    df = pd.DataFrame() 

# Ensure columns exist (handling missing values as done before)
feat_cols = [
    "energy_kcal","carb_g","protein_g","fat_g",
    "freesugar_g","fibre_g","unit_serving_vitc_mg"
]

# Basic cleaning to match your previous script
for col in feat_cols:
    if col not in df.columns:
        df[col] = 0.0 # Fill missing cols with 0
df[feat_cols] = df[feat_cols].fillna(0).astype(float)

food_names = df["food_name"].tolist() if "food_name" in df.columns else [f"Food_{i}" for i in range(len(df))]
data = df[feat_cols].values

# ============================================================
# 2. CONTEXT CLUSTERING (Same logic as your original script)
# ============================================================
X = StandardScaler().fit_transform(data)
Xp = PCA(n_components=4).fit_transform(X)
gmm = GaussianMixture(n_components=4, random_state=42).fit(Xp)
cluster = gmm.predict(Xp)
contexts = ["C1","C2","C3","C4"]
cluster_map = {0:"C3", 1:"C1", 2:"C2", 3:"C4"}

def rule_ctx(n):
    kcal,carb,protein,fat,sugar,fibre,vitC = n
    scores = {
        "C1": carb + kcal*0.3,    # High Energy
        "C2": protein*1.2,        # High Protein
        "C3": max(0, 200-kcal),   # Low Calorie
        "C4": max(0, 35-carb)     # Low Carb
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

# ============================================================
# 3. REWARD FUNCTION
# ============================================================
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
    
    # Noise for realism
    noise = np.random.normal(0, 0.05) 
    return base_r + noise

def norm(x):
    x = np.array(x)
    if x.std() == 0: return x - x.mean()
    return (x - x.mean()) / (x.std() + 1e-8)

def ema(x, a=0.03):
    y = []
    prev = x[0]
    for v in x:
        prev = a*v + (1-a)*prev
        y.append(prev)
    return y

# ============================================================
# 4. AGENTS (Tuned for Recouping)
# ============================================================

# --- Standard UCB ---
class UCBAgent:
    def __init__(self, A):
        self.A = A
        self.c = np.zeros(A)
        self.v = np.zeros(A)
    def select(self):
        t = sum(self.c) + 1
        u = self.v + np.sqrt(2 * np.log(t) / (self.c + 1e-8))
        return np.argmax(u)
    def update(self, a, r):
        self.c[a] += 1
        self.v[a] += (r - self.v[a]) / self.c[a]

# --- DQN (Tuned) ---
STATE_DIM = 1
class DQN(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(STATE_DIM, 32), nn.ReLU(), nn.Linear(32, A))
    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(self, A):
        self.A = A
        self.m = DQN(A)
        self.opt = optim.Adam(self.m.parameters(), lr=0.05) # High LR for adaptation
        self.eps = 1.0
        
    def select(self, s):
        # Floor epsilon at 0.15 to ensure it keeps trying new foods
        if random.random() < self.eps: return random.randint(0, self.A-1)
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
        self.eps = max(self.eps*0.99, 0.15) 

# --- PG (Tuned) ---
class PG(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(STATE_DIM, 32), nn.ReLU(), nn.Linear(32, A))
    def forward(self, x): return self.net(x)

class PGAgent:
    def __init__(self, A):
        self.A = A
        self.m = PG(A)
        self.opt = optim.Adam(self.m.parameters(), lr=0.02) # High LR
        self.saved_log_prob = None
        
    def select(self, s):
        logits = self.m(torch.tensor([[s]], dtype=torch.float32))
        p = torch.softmax(logits, dim=-1)
        p = torch.clamp(p, 1e-8, 1-1e-8)
        p = p / p.sum()
        d = torch.distributions.Categorical(p)
        a = d.sample()
        self.saved_log_prob = d.log_prob(a)
        return a.item()
        
    def update(self, r):
        if self.saved_log_prob is None: return
        loss = -self.saved_log_prob * r
        self.opt.zero_grad(); loss.backward(); self.opt.step()

# ============================================================
# 5. EXPERIMENT: DIET SWITCH
# ============================================================
def run_comparison(switch_point=300, total_eps=800):
    context_A = "C1" # Start: High Energy
    context_B = "C3" # Switch: Low Calorie
    
    # Using specific foods from Context A list is tricky if C1 list is small.
    # To be safe, we let the agent pick from ALL foods in Context A's list initially
    # If the user switches context, usually the AVAILABLE foods stay the same, 
    # but the DEFINITION of "Good" changes.
    # So we use the food list from Context A as the "Menu".
    
    foods = ctx_dict[context_A]
    A = len(foods)
    print(f"Loaded {A} foods from Context {context_A} for the experiment.")
    
    if A < 2:
        print("Error: Not enough foods in Context C1 to run experiment.")
        return {}

    print(f"Running Drift Experiment ({context_A} -> {context_B})...")
    
    agents = {
        "Standard UCB": UCBAgent(A),
        "DQN (Tuned)": DQNAgent(A),
        "PG (Tuned)": PGAgent(A)
    }
    
    results = {k: [] for k in agents.keys()}
    
    # Common seed
    np.random.seed(42); torch.manual_seed(42); random.seed(42)
    
    for e in range(total_eps):
        curr_ctx = context_A if e < switch_point else context_B
        
        for name, ag in agents.items():
            s = 0.5
            
            if "DQN" in name or "PG" in name:
                a = ag.select(s)
            else:
                a = ag.select()
            
            # Use 'rew' to calculate reward based on the CURRENT context rule
            # The food itself is the same, but its value changes
            val = rew(curr_ctx, foods[a]["n"])
            
            if "DQN" in name: 
                ag.update(s, a, val)
            elif "PG" in name:
                ag.update(val)
            else: 
                ag.update(a, val)
            
            results[name].append(val)
            
    return results

# ============================================================
# 6. PLOTTING
# ============================================================
def plot_drift_results():
    results = run_comparison()
    
    if not results: return

    plt.figure(figsize=(12, 6))
    
    colors = {
        "Standard UCB": "gray", 
        "DQN (Tuned)": "#f26419",           
        "PG (Tuned)": "#2a9d8f"  
    }
    
    styles = {
        "Standard UCB": "--", 
        "DQN (Tuned)": "-",
        "PG (Tuned)": "-"
    }
    
    for name, r in results.items():
        norm_r = norm(r)
        smoothed = ema(norm_r, a=0.02)
        plt.plot(smoothed, label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)
        
    plt.axvline(x=300, color='red', linestyle=':', linewidth=2, label="User Diet Switch")
    
    plt.title("Adaptability Analysis: Reaction to Preference Shift (Real Data)", fontsize=14, fontweight='bold')
    plt.xlabel("Episodes")
    plt.ylabel("Normalized Reward (Smoothed)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.text(50, 0.8, "Goal: High Energy", fontsize=10, color='green')
    plt.text(320, -0.5, "Goal: Low Calorie", fontsize=10, color='red')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_drift_results()