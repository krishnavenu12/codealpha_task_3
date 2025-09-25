import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# -----------------------------
# 1. Load and preprocess data
# -----------------------------
def load_data(filepath):
    """
    Load CSV data, clean column names, handle missing values,
    separate target column, and one-hot encode categorical features.
    """
    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Drop missing values
    df = df.dropna()

    # Separate target
    y = df['Credit_Score']

    # One-hot encode categorical features (exclude target)
    X = pd.get_dummies(df.drop('Credit_Score', axis=1), drop_first=True)

    return X, y

# -----------------------------
# 2. Split and scale data
# -----------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# -----------------------------
# 3. Train RandomForest model
# -----------------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# -----------------------------
# 4. Evaluate model
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 5. Save model
# -----------------------------
def save_model(model, path="outputs/credit_score_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved successfully at {path}")

# -----------------------------
# Main
# -----------------------------
def main():
    # Replace this path with your CSV location
    data_path = r"C:\Users\krish\OneDrive\Desktop\CreditScoringProject\data\raw\credit_score_data.csv"

    # Load and preprocess
    X, y = load_data(data_path)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Data loaded and split successfully!")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Train
    model = train_model(X_train, y_train)
    print("Model trained successfully!")

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save
    save_model(model)

if __name__ == "__main__":
    main()
