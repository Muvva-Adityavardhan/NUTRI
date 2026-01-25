

# import pandas as pd
# import numpy as np
# import os

# # Keywords for categorization remains same
# KEYWORDS = {
#     "Breakfast": ["idli", "dosa", "poha", "upma", "paratha", "oats", "egg", "omelette", "toast", "sandwich", "milk", "coffee", "tea", "corn flakes", "muesli", "puri", "bhatura"],
#     "Lunch": ["rice", "roti", "chapati", "dal", "curry", "sabzi", "chicken", "fish", "mutton", "paneer", "biryani", "pulao", "salad", "curd", "yogurt", "rajma", "chole"],
#     "Dinner": ["rice", "roti", "chapati", "dal", "soup", "salad", "chicken", "fish", "paneer", "vegetable", "khichdi"],
#     "Snack": ["biscuit", "cookie", "cake", "chips", "nut", "fruit", "juice", "shake", "samosa", "pakora", "namkeen", "chocolate", "ice cream", "sweet", "tea", "coffee"]
# }

# # Mapping of Column Name -> (Display Name, Unit, Multiplier_to_Micrograms)
# # Multiplier is used to sort "best" vitamins by value (mg * 1000 = ug)
# MICRONUTRIENTS_MAP = {
#     "calcium_mg": ("Calcium", "mg", 1000),
#     "phosphorus_mg": ("Phosphorus", "mg", 1000),
#     "magnesium_mg": ("Magnesium", "mg", 1000),
#     "sodium_mg": ("Sodium", "mg", 1000),
#     "potassium_mg": ("Potassium", "mg", 1000),
#     "iron_mg": ("Iron", "mg", 1000),
#     "copper_mg": ("Copper", "mg", 1000),
#     "selenium_ug": ("Selenium", "ug", 1),
#     "chromium_mg": ("Chromium", "mg", 1000),
#     "manganese_mg": ("Manganese", "mg", 1000),
#     "molybdenum_mg": ("Molybdenum", "mg", 1000),
#     "zinc_mg": ("Zinc", "mg", 1000),
#     "vita_ug": ("Vitamin A", "ug", 1),
#     "vite_mg": ("Vitamin E", "mg", 1000),
#     "vitd2_ug": ("Vitamin D2", "ug", 1),
#     "vitd3_ug": ("Vitamin D3", "ug", 1),
#     "vitk1_ug": ("Vitamin K1", "ug", 1),
#     "vitk2_ug": ("Vitamin K2", "ug", 1),
#     "folate_ug": ("Folate", "ug", 1),
#     "vitb1_mg": ("Vitamin B1", "mg", 1000),
#     "vitb2_mg": ("Vitamin B2", "mg", 1000),
#     "vitb3_mg": ("Vitamin B3", "mg", 1000),
#     "vitb5_mg": ("Vitamin B5", "mg", 1000),
#     "vitb6_mg": ("Vitamin B6", "mg", 1000),
#     "vitb7_ug": ("Vitamin B7", "ug", 1),
#     "vitb9_ug": ("Vitamin B9", "ug", 1),
#     "vitc_mg": ("Vitamin C", "mg", 1000),
#     "carotenoids_ug": ("Carotenoids", "ug", 1)
# }

# def load_and_process_data(filepath="data/food_data.csv"):
#     if not os.path.exists(filepath):
#         print(f"Error: File not found at {filepath}")
#         return {}

#     try:
#         df = pd.read_csv(filepath)
#     except Exception as e:
#         print(f"Error reading CSV: {e}")
#         return {}

#     # Helper to safely clean numeric columns
#     def clean_numeric(val):
#         try:
#             return float(val)
#         except (ValueError, TypeError):
#             return 0.0

#     meal_database = {"Breakfast": [], "Lunch": [], "Dinner": [], "Snack": []}

#     for _, row in df.iterrows():
#         food_name = str(row.get("food_name", "Unknown"))
        
#         # 1. serving Unit
#         serving_unit = str(row.get("servings_unit", "1 serving"))
#         if serving_unit == "nan": serving_unit = "1 serving"

#         # 2. Main Macros - Prefer 'unit_serving' columns if available, else standard
#         # Calories
#         cal_val = clean_numeric(row.get("unit_serving_energy_kcal", 0))
#         if cal_val == 0: cal_val = clean_numeric(row.get("energy_kcal", 0))
        
#         # Protein
#         prot_val = clean_numeric(row.get("unit_serving_protein_g", 0))
#         if prot_val == 0: prot_val = clean_numeric(row.get("protein_g", 0))

#         # Fat
#         fat_val = clean_numeric(row.get("unit_serving_fat_g", 0))
#         if fat_val == 0: fat_val = clean_numeric(row.get("fat_g", 0))

#         # Carbs
#         carb_val = clean_numeric(row.get("unit_serving_carb_g", 0))
#         if carb_val == 0: carb_val = clean_numeric(row.get("carb_g", 0))
        
#         # Cholesterol (mg)
#         chol_val = clean_numeric(row.get("unit_serving_cholesterol_mg", 0))
#         if chol_val == 0: chol_val = clean_numeric(row.get("cholesterol_mg", 0))

#         # 3. Top 5 Micronutrients Logic
#         # We collect all valid micronutrients, normalize them to micrograms (ug) for comparison,
#         # sort them, and pick the top 5.
        
#         all_micros = []
#         for col, (disp_name, unit, multiplier) in MICRONUTRIENTS_MAP.items():
#             # Try serving unit column first (e.g., unit_serving_calcium_mg)
#             serving_col = "unit_serving_" + col
            
#             val = clean_numeric(row.get(serving_col, 0))
#             if val == 0: # Fallback to standard column
#                 val = clean_numeric(row.get(col, 0))
            
#             if val > 0:
#                 normalized_val = val * multiplier # Convert to ug for sorting
#                 # Store tuple: (normalized_val, original_val, display_name, unit)
#                 all_micros.append((normalized_val, val, disp_name, unit))
        
#         # Sort by normalized value (Descending)
#         all_micros.sort(key=lambda x: x[0], reverse=True)
        
#         # Format top 5 strings
#         top_5_micros = []
#         for _, val, name, unit in all_micros[:5]:
#             # Format: "Vitamin C: 50.5mg"
#             top_5_micros.append(f"{name}: {val:.2f}{unit}")

#         food_item = {
#             "name": food_name,
#             "calories": cal_val,
#             "protein": prot_val,
#             "fat": fat_val,
#             "carbs": carb_val,
#             "cholesterol": chol_val,
#             "micros": top_5_micros, 
#             "serving_size": serving_unit
#         }
        
#         # Categorize
#         food_name_lower = food_name.lower()
#         assigned = False
#         for meal_type, keywords in KEYWORDS.items():
#             if any(k in food_name_lower for k in keywords):
#                 meal_database[meal_type].append(food_item)
#                 assigned = True
        
#         if not assigned:
#             meal_database["Lunch"].append(food_item)
#             meal_database["Dinner"].append(food_item)

#     print(f"Loaded {len(df)} items. Extracted Nutrients & Serving Sizes.")
#     return meal_database



import pandas as pd
import numpy as np
import os

# --- NEW: Keywords to identify Non-Veg foods ---
NON_VEG_KEYWORDS = ["chicken", "mutton", "fish", "egg", "omelette", "prawn", "shrimp", "beef", "pork", "meat", "lamb", "seafood", "crab", "bacon", "sausage", "ham", "salami"]

# Keywords for categorization remains same
KEYWORDS = {
    "Breakfast": ["idli", "dosa", "poha", "upma", "paratha", "oats", "egg", "omelette", "toast", "sandwich", "milk", "coffee", "tea", "corn flakes", "muesli", "puri", "bhatura"],
    "Lunch": ["rice", "roti", "chapati", "dal", "curry", "sabzi", "chicken", "fish", "mutton", "paneer", "biryani", "pulao", "salad", "curd", "yogurt", "rajma", "chole"],
    "Dinner": ["rice", "roti", "chapati", "dal", "soup", "salad", "chicken", "fish", "paneer", "vegetable", "khichdi"],
    "Snack": ["biscuit", "cookie", "cake", "chips", "nut", "fruit", "juice", "shake", "samosa", "pakora", "namkeen", "chocolate", "ice cream", "sweet", "tea", "coffee"]
}

# Mapping of Column Name -> (Display Name, Unit, Multiplier_to_Micrograms)
MICRONUTRIENTS_MAP = {
    "calcium_mg": ("Calcium", "mg", 1000),
    "phosphorus_mg": ("Phosphorus", "mg", 1000),
    "magnesium_mg": ("Magnesium", "mg", 1000),
    "sodium_mg": ("Sodium", "mg", 1000),
    "potassium_mg": ("Potassium", "mg", 1000),
    "iron_mg": ("Iron", "mg", 1000),
    "copper_mg": ("Copper", "mg", 1000),
    "selenium_ug": ("Selenium", "ug", 1),
    "chromium_mg": ("Chromium", "mg", 1000),
    "manganese_mg": ("Manganese", "mg", 1000),
    "molybdenum_mg": ("Molybdenum", "mg", 1000),
    "zinc_mg": ("Zinc", "mg", 1000),
    "vita_ug": ("Vitamin A", "ug", 1),
    "vite_mg": ("Vitamin E", "mg", 1000),
    "vitd2_ug": ("Vitamin D2", "ug", 1),
    "vitd3_ug": ("Vitamin D3", "ug", 1),
    "vitk1_ug": ("Vitamin K1", "ug", 1),
    "vitk2_ug": ("Vitamin K2", "ug", 1),
    "folate_ug": ("Folate", "ug", 1),
    "vitb1_mg": ("Vitamin B1", "mg", 1000),
    "vitb2_mg": ("Vitamin B2", "mg", 1000),
    "vitb3_mg": ("Vitamin B3", "mg", 1000),
    "vitb5_mg": ("Vitamin B5", "mg", 1000),
    "vitb6_mg": ("Vitamin B6", "mg", 1000),
    "vitb7_ug": ("Vitamin B7", "ug", 1),
    "vitb9_ug": ("Vitamin B9", "ug", 1),
    "vitc_mg": ("Vitamin C", "mg", 1000),
    "carotenoids_ug": ("Carotenoids", "ug", 1)
}

def load_and_process_data(filepath="data/food_data.csv"):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return {}

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}

    def clean_numeric(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    meal_database = {"Breakfast": [], "Lunch": [], "Dinner": [], "Snack": []}

    for _, row in df.iterrows():
        food_name = str(row.get("food_name", "Unknown"))
        food_name_lower = food_name.lower() # Used for categorization and Veg detection
        
        # --- NEW: Veg Detection Logic ---
        is_veg = True
        for nv_word in NON_VEG_KEYWORDS:
            if nv_word in food_name_lower:
                is_veg = False
                break

        # 1. Serving Unit
        serving_unit = str(row.get("servings_unit", "1 serving"))
        if serving_unit == "nan": serving_unit = "1 serving"

        # 2. Main Macros
        cal_val = clean_numeric(row.get("unit_serving_energy_kcal", 0))
        if cal_val == 0: cal_val = clean_numeric(row.get("energy_kcal", 0))
        
        prot_val = clean_numeric(row.get("unit_serving_protein_g", 0))
        if prot_val == 0: prot_val = clean_numeric(row.get("protein_g", 0))

        fat_val = clean_numeric(row.get("unit_serving_fat_g", 0))
        if fat_val == 0: fat_val = clean_numeric(row.get("fat_g", 0))

        carb_val = clean_numeric(row.get("unit_serving_carb_g", 0))
        if carb_val == 0: carb_val = clean_numeric(row.get("carb_g", 0))
        
        chol_val = clean_numeric(row.get("unit_serving_cholesterol_mg", 0))
        if chol_val == 0: chol_val = clean_numeric(row.get("cholesterol_mg", 0))

        # 3. Top 5 Micronutrients Logic
        all_micros = []
        for col, (disp_name, unit, multiplier) in MICRONUTRIENTS_MAP.items():
            serving_col = "unit_serving_" + col
            val = clean_numeric(row.get(serving_col, 0))
            if val == 0: 
                val = clean_numeric(row.get(col, 0))
            
            if val > 0:
                normalized_val = val * multiplier 
                all_micros.append((normalized_val, val, disp_name, unit))
        
        all_micros.sort(key=lambda x: x[0], reverse=True)
        
        top_5_micros = []
        for _, val, name, unit in all_micros[:5]:
            top_5_micros.append(f"{name}: {val:.2f}{unit}")

        food_item = {
            "name": food_name,
            "calories": cal_val,
            "protein": prot_val,
            "fat": fat_val,
            "carbs": carb_val,
            "cholesterol": chol_val,
            "micros": top_5_micros, 
            "serving_size": serving_unit,
            "is_veg": is_veg  # <--- NEW FIELD ADDED
        }
        
        # Categorize
        assigned = False
        for meal_type, keywords in KEYWORDS.items():
            if any(k in food_name_lower for k in keywords):
                meal_database[meal_type].append(food_item)
                assigned = True
        
        if not assigned:
            meal_database["Lunch"].append(food_item)
            meal_database["Dinner"].append(food_item)

    print(f"Loaded {len(df)} items. Extracted Nutrients, Serving Sizes, and Veg status.")
    return meal_database
