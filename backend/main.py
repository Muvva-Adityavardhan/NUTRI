# # # Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional, Dict
# from data_loader import load_and_process_data
# from rl_agents import UCBAgent, DQNAgent, PolicyGradientAgent, STATE_SIZE

# app = FastAPI()

# # --- Global Data Store ---
# MEAL_DATA = {} 
# USER_AGENTS: Dict[str, Dict] = {} 

# @app.on_event("startup")
# def load_data():
#     global MEAL_DATA
#     MEAL_DATA = load_and_process_data("data/food_data.csv")

# def get_user_agent(user_id: str, algorithm: str, meal_type: str):
#     if user_id not in USER_AGENTS: USER_AGENTS[user_id] = {}
#     agent_key = f"{algorithm}_{meal_type}"
#     if agent_key not in USER_AGENTS[user_id]:
#         action_list = MEAL_DATA.get(meal_type, [])
#         if not action_list: action_list = [{"name": "Standard Meal", "calories": 500}]
#         if algorithm == "ucb": USER_AGENTS[user_id][agent_key] = UCBAgent(action_list)
#         elif algorithm == "dqn": USER_AGENTS[user_id][agent_key] = DQNAgent(STATE_SIZE, action_list)
#         elif algorithm == "pg": USER_AGENTS[user_id][agent_key] = PolicyGradientAgent(STATE_SIZE, action_list)
#     return USER_AGENTS[user_id][agent_key]

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
#     fat: float
#     carbs: float
#     cholesterol: float
#     micros: List[str]
#     serving_size: str    # <--- NEW FIELD
#     meal_id: int
#     meal_type: str
#     algorithm_used: str

# class FeedbackInput(BaseModel):
#     user_id: str
#     algorithm: str
#     meal_type: str
#     meal_id: int
#     reward: float 
#     state: Optional[StateInput] = None
#     next_state: Optional[StateInput] = None 

# @app.post("/recommend/{algorithm}", response_model=RecommendationResponse)
# def get_recommendation(algorithm: str, state: StateInput):
#     algo_name = algorithm.lower()
#     if state.time_of_day < 0.25: meal_type = "Breakfast"
#     elif state.time_of_day < 0.50: meal_type = "Lunch"
#     elif state.time_of_day < 0.75: meal_type = "Snack"
#     else: meal_type = "Dinner"
    
#     agent = get_user_agent(state.user_id, algo_name, meal_type)
#     state_vector = [state.time_of_day, state.calorie_goal, state.current_calories, state.is_workout_day]
    
#     if algo_name == "ucb": action_idx = agent.select_action()
#     else: action_idx = agent.select_action(state_vector)
    
#     food_list = agent.actions
#     if action_idx >= len(food_list): action_idx = 0
#     selected = food_list[action_idx]
    
#     return {
#         "meal_name": selected["name"],
#         "calories": selected["calories"],
#         "protein": selected["protein"],
#         "fat": selected.get("fat", 0),
#         "carbs": selected.get("carbs", 0),
#         "cholesterol": selected.get("cholesterol", 0),
#         "micros": selected.get("micros", []),
#         "serving_size": selected.get("serving_size", "1 serving"), # <--- Send to App
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
#         state_vec = [feedback.state.time_of_day, feedback.state.calorie_goal, feedback.state.current_calories, feedback.state.is_workout_day]
#         if algo_name == "dqn" and feedback.next_state:
#             next_vec = [feedback.next_state.time_of_day, feedback.next_state.calorie_goal, feedback.next_state.current_calories, feedback.next_state.is_workout_day]
#             agent.update(state_vec, feedback.meal_id, feedback.reward, next_vec)
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

# --- NEW: Helper to filter foods & get specific agent ---
def get_user_agent_and_food_list(user_id, algorithm, meal_type, is_veg_only, allergies):
    # 1. Get raw list for this meal type
    raw_list = MEAL_DATA.get(meal_type, [])
    
    # 2. Filter based on Veg preference and Allergies
    filtered_list = []
    for food in raw_list:
        # Check Veg Constraint
        if is_veg_only and not food["is_veg"]:
            continue
            
        # Check Allergies (Basic string matching)
        has_allergy = False
        if allergies:
            food_name_lower = food["name"].lower()
            for allergen in allergies:
                if allergen.lower() in food_name_lower:
                    has_allergy = True
                    break
        
        if not has_allergy:
            filtered_list.append(food)

    # Fallback if filter removes everything
    if not filtered_list:
        filtered_list = [{"name": "No matching food", "calories": 0, "protein": 0, "is_veg": True}]

    # 3. Get/Create Agent
    # We create a unique agent for "Veg" mode so it doesn't conflict with "Mixed" mode
    if user_id not in USER_AGENTS: USER_AGENTS[user_id] = {}
    
    mode_suffix = "Veg" if is_veg_only else "All"
    agent_key = f"{algorithm}_{meal_type}_{mode_suffix}"
    
    if agent_key not in USER_AGENTS[user_id]:
        if algorithm == "ucb": 
            USER_AGENTS[user_id][agent_key] = UCBAgent(filtered_list)
        elif algorithm == "dqn": 
            USER_AGENTS[user_id][agent_key] = DQNAgent(STATE_SIZE, filtered_list)
        elif algorithm == "pg": 
            USER_AGENTS[user_id][agent_key] = PolicyGradientAgent(STATE_SIZE, filtered_list)
    else:
        # Update action list in case data changed (Optional safety)
        if hasattr(USER_AGENTS[user_id][agent_key], 'actions'):
             USER_AGENTS[user_id][agent_key].actions = filtered_list
             USER_AGENTS[user_id][agent_key].action_size = len(filtered_list)
             if hasattr(USER_AGENTS[user_id][agent_key], 'n_actions'):
                 USER_AGENTS[user_id][agent_key].n_actions = len(filtered_list)

    return USER_AGENTS[user_id][agent_key], filtered_list

class StateInput(BaseModel):
    user_id: str
    time_of_day: float 
    calorie_goal: float 
    current_calories: float 
    is_workout_day: float 
    is_veg_only: bool = False       # <--- NEW
    allergies: List[str] = []       # <--- NEW

class RecommendationResponse(BaseModel):
    meal_name: str
    calories: float
    protein: float
    fat: float
    carbs: float
    cholesterol: float
    micros: List[str]
    serving_size: str
    is_veg: bool                    # <--- NEW
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
    
    # Use new helper to get filtered agent and list
    agent, food_list = get_user_agent_and_food_list(
        state.user_id, algo_name, meal_type, state.is_veg_only, state.allergies
    )
    
    state_vector = [state.time_of_day, state.calorie_goal, state.current_calories, state.is_workout_day]
    
    if algo_name == "ucb": 
        action_idx = agent.select_action()
    else: 
        action_idx = agent.select_action(state_vector)
    
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
        "serving_size": selected.get("serving_size", "1 serving"),
        "is_veg": selected.get("is_veg", True),  # <--- Return Flag
        "meal_id": action_idx,
        "meal_type": meal_type,
        "algorithm_used": algo_name
    }

@app.post("/feedback")
def submit_feedback(feedback: FeedbackInput):
    algo_name = feedback.algorithm.lower()
    
    # Must retrieve the correct filtered agent to update
    if feedback.state:
        agent, _ = get_user_agent_and_food_list(
            feedback.user_id, algo_name, feedback.meal_type,
            feedback.state.is_veg_only, feedback.state.allergies
        )
    else:
        # Fallback if state not provided (shouldn't happen with new app code)
        # We try to get the 'All' agent by default
        agent, _ = get_user_agent_and_food_list(feedback.user_id, algo_name, feedback.meal_type, False, [])

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
