# # Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000




# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Optional, Dict
# from data_loader import load_and_process_data
# from rl_agents import UCBAgent, DQNAgent, PolicyGradientAgent, STATE_SIZE
# from typing import List

# app = FastAPI()

# # --- Global Data Store ---
# MEAL_DATA = {} # Will hold {"Breakfast": [...], "Lunch": [...]}
# USER_AGENTS: Dict[str, Dict] = {} # { "user_123": { "ucb": agent, "dqn": agent... } }

# # --- Startup Event ---
# @app.on_event("startup")
# def load_data():
#     global MEAL_DATA
#     # Ensure your CSV is at backend/data/food_data.csv
#     MEAL_DATA = load_and_process_data("data/food_data.csv")

# # --- Helper to Get/Create User Agent ---
# def get_user_agent(user_id: str, algorithm: str, meal_type: str):
#     # Create user entry if not exists
#     if user_id not in USER_AGENTS:
#         USER_AGENTS[user_id] = {}
    
#     # Key for the specific agent (e.g., "ucb_Breakfast")
#     # We need separate agents for Breakfast vs Lunch because the food lists are different!
#     agent_key = f"{algorithm}_{meal_type}"
    
#     if agent_key not in USER_AGENTS[user_id]:
#         # Initialize new agent
#         action_list = MEAL_DATA.get(meal_type, [])
#         if not action_list:
#             # Fallback if list empty
#             action_list = [{"name": "Standard Meal", "calories": 500}]

#         if algorithm == "ucb":
#             USER_AGENTS[user_id][agent_key] = UCBAgent(action_list)
#         elif algorithm == "dqn":
#             USER_AGENTS[user_id][agent_key] = DQNAgent(STATE_SIZE, action_list)
#         elif algorithm == "pg":
#             USER_AGENTS[user_id][agent_key] = PolicyGradientAgent(STATE_SIZE, action_list)
            
#     return USER_AGENTS[user_id][agent_key]

# # --- Data Models ---
# class StateInput(BaseModel):
#     user_id: str
#     time_of_day: float 
#     calorie_goal: float 
#     current_calories: float 
#     is_workout_day: float 

# class RecommendationResponse(BaseModel):
#     meal_name: str
#     calories: float
#     protein: float
#     fat: float             # <--- NEW
#     carbs: float           # <--- NEW
#     cholesterol: float     # <--- NEW
#     micros: List[str]      # <--- NEW
#     meal_id: int
#     meal_type: str
#     algorithm_used: str

# class FeedbackInput(BaseModel):
#     user_id: str
#     algorithm: str
#     meal_type: str # "Breakfast", "Lunch", etc.
#     meal_id: int
#     reward: float 
#     state: Optional[StateInput] = None
#     next_state: Optional[StateInput] = None 

# # --- API Endpoints ---

# @app.post("/recommend/{algorithm}", response_model=RecommendationResponse)
# def get_recommendation(algorithm: str, state: StateInput):
#     algo_name = algorithm.lower()
    
#     # Determine Meal Type based on Time of Day
#     # 0.0-0.25 (Morning), 0.25-0.5 (Lunch), 0.5-0.75 (Snack), 0.75-1.0 (Dinner)
#     if state.time_of_day < 0.25: meal_type = "Breakfast"
#     elif state.time_of_day < 0.50: meal_type = "Lunch"
#     elif state.time_of_day < 0.75: meal_type = "Snack"
#     else: meal_type = "Dinner"
    
#     # Get the specific agent for this User + Algo + MealType
#     agent = get_user_agent(state.user_id, algo_name, meal_type)
    
#     # Select Action
#     state_vector = [state.time_of_day, state.calorie_goal, state.current_calories, state.is_workout_day]
    
#     if algo_name == "ucb":
#         action_idx = agent.select_action()
#     else:
#         action_idx = agent.select_action(state_vector)
    
#     # Retrieve Food Details
#     food_list = agent.actions
#     # Safety check
#     if action_idx >= len(food_list): action_idx = 0
    
#     selected_food = food_list[action_idx]
    
#     return {
#         "meal_name": selected_food["name"],
#         "calories": selected_food["calories"],
#         "protein": selected_food["protein"],
#         "fat": selected_food.get("fat", 0),                 # <--- Fetch NEW Data
#         "carbs": selected_food.get("carbs", 0),             # <--- Fetch NEW Data
#         "cholesterol": selected_food.get("cholesterol", 0), # <--- Fetch NEW Data
#         "micros": selected_food.get("micros", []),          # <--- Fetch NEW Data
#         "meal_id": action_idx,
#         "meal_type": meal_type,
#         "algorithm_used": algo_name
#     }

# @app.post("/feedback")
# def submit_feedback(feedback: FeedbackInput):
#     algo_name = feedback.algorithm.lower()
#     agent = get_user_agent(feedback.user_id, algo_name, feedback.meal_type)
    
#     if algo_name == "ucb":
#         agent.update(feedback.meal_id, feedback.reward)
#         return {"status": "UCB Updated"}
        
#     if feedback.state:
#         state_vector = [
#             feedback.state.time_of_day, 
#             feedback.state.calorie_goal, 
#             feedback.state.current_calories, 
#             feedback.state.is_workout_day
#         ]
        
#         if algo_name == "dqn" and feedback.next_state:
#             next_state_vector = [
#                 feedback.next_state.time_of_day, 
#                 feedback.next_state.calorie_goal, 
#                 feedback.next_state.current_calories, 
#                 feedback.next_state.is_workout_day
#             ]
#             agent.update(state_vector, feedback.meal_id, feedback.reward, next_state_vector)
#             return {"status": "DQN Updated"}
            
#         elif algo_name == "pg":
#             agent.update(feedback.reward)
#             return {"status": "PG Updated"}
            
#     return {"status": "Feedback received"}






from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from data_loader import load_and_process_data
from rl_agents import UCBAgent, DQNAgent, PolicyGradientAgent, STATE_SIZE

app = FastAPI()

# --- Global Data Store ---
MEAL_DATA = {} 
USER_AGENTS: Dict[str, Dict] = {} 

@app.on_event("startup")
def load_data():
    global MEAL_DATA
    MEAL_DATA = load_and_process_data("data/food_data.csv")

def get_user_agent(user_id: str, algorithm: str, meal_type: str):
    if user_id not in USER_AGENTS: USER_AGENTS[user_id] = {}
    agent_key = f"{algorithm}_{meal_type}"
    if agent_key not in USER_AGENTS[user_id]:
        action_list = MEAL_DATA.get(meal_type, [])
        if not action_list: action_list = [{"name": "Standard Meal", "calories": 500}]
        if algorithm == "ucb": USER_AGENTS[user_id][agent_key] = UCBAgent(action_list)
        elif algorithm == "dqn": USER_AGENTS[user_id][agent_key] = DQNAgent(STATE_SIZE, action_list)
        elif algorithm == "pg": USER_AGENTS[user_id][agent_key] = PolicyGradientAgent(STATE_SIZE, action_list)
    return USER_AGENTS[user_id][agent_key]

class StateInput(BaseModel):
    user_id: str
    time_of_day: float 
    calorie_goal: float 
    current_calories: float 
    is_workout_day: float 

class RecommendationResponse(BaseModel):
    meal_name: str
    calories: float
    protein: float
    fat: float
    carbs: float
    cholesterol: float
    micros: List[str]
    serving_size: str    # <--- NEW FIELD
    meal_id: int
    meal_type: str
    algorithm_used: str

class FeedbackInput(BaseModel):
    user_id: str
    algorithm: str
    meal_type: str
    meal_id: int
    reward: float 
    state: Optional[StateInput] = None
    next_state: Optional[StateInput] = None 

@app.post("/recommend/{algorithm}", response_model=RecommendationResponse)
def get_recommendation(algorithm: str, state: StateInput):
    algo_name = algorithm.lower()
    if state.time_of_day < 0.25: meal_type = "Breakfast"
    elif state.time_of_day < 0.50: meal_type = "Lunch"
    elif state.time_of_day < 0.75: meal_type = "Snack"
    else: meal_type = "Dinner"
    
    agent = get_user_agent(state.user_id, algo_name, meal_type)
    state_vector = [state.time_of_day, state.calorie_goal, state.current_calories, state.is_workout_day]
    
    if algo_name == "ucb": action_idx = agent.select_action()
    else: action_idx = agent.select_action(state_vector)
    
    food_list = agent.actions
    if action_idx >= len(food_list): action_idx = 0
    selected = food_list[action_idx]
    
    return {
        "meal_name": selected["name"],
        "calories": selected["calories"],
        "protein": selected["protein"],
        "fat": selected.get("fat", 0),
        "carbs": selected.get("carbs", 0),
        "cholesterol": selected.get("cholesterol", 0),
        "micros": selected.get("micros", []),
        "serving_size": selected.get("serving_size", "1 serving"), # <--- Send to App
        "meal_id": action_idx,
        "meal_type": meal_type,
        "algorithm_used": algo_name
    }

@app.post("/feedback")
def submit_feedback(feedback: FeedbackInput):
    algo_name = feedback.algorithm.lower()
    agent = get_user_agent(feedback.user_id, algo_name, feedback.meal_type)
    
    if algo_name == "ucb":
        agent.update(feedback.meal_id, feedback.reward)
        return {"status": "UCB Updated"}
    if feedback.state:
        state_vec = [feedback.state.time_of_day, feedback.state.calorie_goal, feedback.state.current_calories, feedback.state.is_workout_day]
        if algo_name == "dqn" and feedback.next_state:
            next_vec = [feedback.next_state.time_of_day, feedback.next_state.calorie_goal, feedback.next_state.current_calories, feedback.next_state.is_workout_day]
            agent.update(state_vec, feedback.meal_id, feedback.reward, next_vec)
            return {"status": "DQN Updated"}
        elif algo_name == "pg":
            agent.update(feedback.reward)
            return {"status": "PG Updated"}
    return {"status": "Feedback received"}