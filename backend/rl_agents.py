# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import math

# # --- Mock Meal Database (Action Space) ---
# # In a real app, you might load this from a database file.
# # We map Index -> Meal Name. The RL agent selects an Index.
# MEAL_DATABASE = [
#     "Oatmeal & Berries", "Protein Shake", "Chicken Salad", "Grilled Salmon", 
#     "Vegetable Stir Fry", "Lentil Soup", "Greek Yogurt Parfait", "Quinoa Bowl",
#     "Egg White Omelet", "Avocado Toast", "Brown Rice & Beans", "Tuna Sandwich",
#     "Paneer Tikka", "Chicken Curry", "Fruit Salad", "Cottage Cheese",
#     "Almonds & Walnuts", "Boiled Eggs", "Protein Bar", "Green Smoothie"
# ]
# ACTION_SIZE = len(MEAL_DATABASE)
# STATE_SIZE = 4  # [TimeOfDay(0-3), CalorieGoal, CurrentCalories, IsWorkoutDay(0/1)]

# # --- 1. Upper Confidence Bound (UCB) ---
# class UCBAgent:
#     def __init__(self, n_actions):
#         self.n_actions = n_actions
#         self.counts = np.zeros(n_actions)
#         self.values = np.zeros(n_actions) # Average reward

#     def select_action(self):
#         # Exploration constant
#         c = 1.5 
#         total_counts = np.sum(self.counts)
        
#         if total_counts == 0:
#             return np.random.choice(self.n_actions)

#         ucb_values = np.zeros(self.n_actions)
#         for a in range(self.n_actions):
#             if self.counts[a] == 0:
#                 return a # Force exploration of untouched items
#             exploitation = self.values[a]
#             exploration = c * np.sqrt(np.log(total_counts) / self.counts[a])
#             ucb_values[a] = exploitation + exploration
        
#         return np.argmax(ucb_values)

#     def update(self, action, reward):
#         self.counts[action] += 1
#         n = self.counts[action]
#         # Incremental average update
#         self.values[action] = ((n - 1) / n) * self.values[action] + (1 / n) * reward

# # --- 2. Deep Q-Network (DQN) ---
# class DQN(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_size, 24)
#         self.fc2 = nn.Linear(24, 24)
#         self.fc3 = nn.Linear(24, action_size)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         return self.fc3(x) # Returns Q-values for all actions

# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.model = DQN(state_size, action_size)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.criterion = nn.MSELoss()
#         self.epsilon = 1.0 # Exploration rate
#         self.epsilon_decay = 0.995
#         self.epsilon_min = 0.01

#     def select_action(self, state):
#         # Epsilon-greedy policy
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
        
#         state_tensor = torch.FloatTensor(state).unsqueeze(0) # Add batch dim
#         with torch.no_grad():
#             q_values = self.model(state_tensor)
#         return torch.argmax(q_values).item()

#     def update(self, state, action, reward, next_state, done):
#         state_t = torch.FloatTensor(state).unsqueeze(0)
#         next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
#         action_t = torch.LongTensor([action])
#         reward_t = torch.FloatTensor([reward])

#         # Current Q
#         q_values = self.model(state_t)
#         q_value = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)

#         # Target Q (Bellman Equation)
#         with torch.no_grad():
#             next_q_values = self.model(next_state_t)
#             max_next_q = next_q_values.max(1)[0]
#             target_q = reward_t + (0.99 * max_next_q * (1 - done))

#         loss = self.criterion(q_value, target_q)
        
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

# # --- 3. Policy Gradient (REINFORCE) ---
# class PolicyNetwork(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_size, 128)
#         self.fc2 = nn.Linear(128, action_size)
#         self.softmax = nn.Softmax(dim=1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         return self.softmax(self.fc2(x)) # Returns probabilities

# class PolicyGradientAgent:
#     def __init__(self, state_size, action_size):
#         self.model = PolicyNetwork(state_size, action_size)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
#         self.saved_log_probs = []
#         self.rewards = []

#     def select_action(self, state):
#         state_t = torch.FloatTensor(state).unsqueeze(0)
#         probs = self.model(state_t)
        
#         # Sample an action from the distribution
#         m = torch.distributions.Categorical(probs)
#         action = m.sample()
        
#         self.saved_log_probs.append(m.log_prob(action))
#         return action.item()

#     def store_reward(self, reward):
#         self.rewards.append(reward)

#     def update(self):
#         # Standard REINFORCE update (simplified for single-step online learning)
#         if not self.rewards: return
        
#         R = 0
#         policy_loss = []
#         returns = []
        
#         # Calculate discounted returns
#         for r in self.rewards[::-1]:
#             R = r + 0.99 * R
#             returns.insert(0, R)
            
#         returns = torch.tensor(returns)
#         # Normalize returns
#         if len(returns) > 1:
#             returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
#         for log_prob, R in zip(self.saved_log_probs, returns):
#             policy_loss.append(-log_prob * R)
            
#         self.optimizer.zero_grad()
#         if policy_loss:
#             loss = torch.stack(policy_loss).sum()
#             loss.backward()
#             self.optimizer.step()
        
#         # Clear memory
#         del self.saved_log_probs[:]
#         del self.rewards[:]










import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

STATE_SIZE = 4  # [TimeNorm, CalorieGoalNorm, CurrentCalNorm, IsWorkout]

# --- 1. Upper Confidence Bound (UCB) ---
class UCBAgent:
    def __init__(self, action_list):
        self.actions = action_list
        self.n_actions = len(action_list)
        self.counts = np.zeros(self.n_actions)
        self.values = np.zeros(self.n_actions) 

    def select_action(self):
        c = 1.5 
        total_counts = np.sum(self.counts)
        if total_counts == 0: return np.random.choice(self.n_actions)

        ucb_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            if self.counts[a] == 0: return a
            exploitation = self.values[a]
            exploration = c * np.sqrt(np.log(total_counts) / self.counts[a])
            ucb_values[a] = exploitation + exploration
        
        return np.argmax(ucb_values)

    def update(self, action_idx, reward):
        if 0 <= action_idx < self.n_actions:
            self.counts[action_idx] += 1
            n = self.counts[action_idx]
            self.values[action_idx] = ((n - 1) / n) * self.values[action_idx] + (1 / n) * reward

# --- 2. Deep Q-Network (DQN) ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64) # Increased neurons for larger datasets
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_list):
        self.actions = action_list
        self.action_size = len(action_list)
        self.model = DQN(state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0 
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state):
        # Standard DQN Update
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        action_t = torch.LongTensor([action])
        reward_t = torch.FloatTensor([reward])

        q_values = self.model(state_t)
        q_value = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.model(next_state_t)
            target_q = reward_t + (0.99 * next_q_values.max(1)[0])

        loss = self.criterion(q_value, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- 3. Policy Gradient ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.softmax(self.fc2(x))

class PolicyGradientAgent:
    def __init__(self, state_size, action_list):
        self.actions = action_list
        self.action_size = len(action_list)
        self.model = PolicyNetwork(state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.model(state_t)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self, reward):
        # Simplified one-step update
        policy_loss = []
        # In a real scenario, you'd normalize rewards over a batch
        # Here we just take the single step reward
        loss = -self.saved_log_probs[-1] * reward
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.saved_log_probs = [] # Reset