import pandas as pd
import joblib

# Load trained model
model = joblib.load("outputs/credit_score_model.pkl")

# Get model feature names
model_columns = model.feature_names_in_

def prepare_input(new_data_dict):
    """
    Convert user input into DataFrame with the exact columns model expects.
    Missing columns will be filled with 0.
    """
    # Initialize empty DataFrame with all model columns
    input_df = pd.DataFrame(columns=model_columns)
    
    # Add new data
    input_df = pd.concat([input_df, pd.DataFrame([new_data_dict])], ignore_index=True)
    
    # Fill missing columns with 0
    input_df = input_df.fillna(0)
    
    # Reorder columns to match model
    input_df = input_df[model_columns]
    
    return input_df

if __name__ == "__main__":
    # Example new applicant data
    new_applicant = {
        'Age': 28,
        'Income': 55000,
        'Number_of_Children': 1,
        'Gender_Male': 1,                 # 0 or 1
        'Education_Master\'s Degree': 0,
        'Education_PhD': 0,
        'Marital_Status_Married': 0,
        'Home_Ownership_Owned': 1,
        # Add other dummy columns if any
    }

    # Prepare input
    new_data = prepare_input(new_applicant)

    # Predict
    prediction = model.predict(new_data)
    print("Predicted Credit Score:", prediction[0])
