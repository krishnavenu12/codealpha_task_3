import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets and scale numeric features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
