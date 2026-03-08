import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from scipy.sparse import hstack

# -----------------------------
# Function: load_data
# -----------------------------
def load_data(file_path):
    """
    Load expense data from a CSV file into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded data
    """
    df = pd.read_csv(file_path)

    # Ensure 'Payee' column exists (handle old CSVs with 'Merchant')
    if 'Payee' not in df.columns and 'Merchant' in df.columns:
        df = df.rename(columns={'Merchant': 'Payee'})

    return df


# -----------------------------
# Function: preprocess_data
# -----------------------------
def preprocess_data(df):
    """
    Preprocess the expense data for modeling:
        - Cleans and normalizes the 'Payee' text column
        - Converts 'Date' to datetime and extracts features
        - Creates numeric transformations of 'Amount'
        - Adds text length features

    Parameters:
        df (pd.DataFrame): Raw expense data

    Returns:
        pd.DataFrame: Preprocessed data with additional features
    """
    if 'Payee' not in df.columns:
        raise ValueError("DataFrame must have a 'Payee' column")

    df['Payee'] = df['Payee'].fillna("unknown").str.lower().str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['AmountLog'] = np.log1p(df['Amount'])
    df['PayeeLength'] = df['Payee'].str.len()

    return df


# -----------------------------
# Function: train_model
# -----------------------------
def train_model(df, save_path='../models/expense_model.pkl'):
    """
    Train a Random Forest classifier to predict expense categories.

    Parameters:
        df (pd.DataFrame): Preprocessed data
        save_path (str): File path to save the trained model

    Returns:
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
    """
    df = preprocess_data(df)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, stop_words='english')
    X_text_vec = vectorizer.fit_transform(df['Payee'])

    numeric_cols = ['Amount', 'AmountLog', 'DayOfWeek', 'Month', 'IsWeekend', 'PayeeLength']
    X_numeric = df[numeric_cols].values

    X = hstack([X_text_vec, X_numeric])
    y = df['Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    print("Model Accuracy:", model.score(X_test, y_test))

    # Ensure the folder exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump((model, vectorizer, numeric_cols), f)

    return model, vectorizer


# -----------------------------
# Function: predict_category
# -----------------------------
def predict_category(Payee_name, amount, date, model, vectorizer, numeric_cols):
    """
    Predict the expense category for a single transaction.

    Parameters:
        Payee_name (str): Name of the merchant/payee
        amount (float): Transaction amount
        date (str/datetime): Transaction date
        model: Trained classifier
        vectorizer: TF-IDF vectorizer
        numeric_cols: List of numeric features

    Returns:
        str: Predicted category
    """
    if pd.isna(Payee_name):
        Payee_name = "unknown"

    Payee_name = str(Payee_name).strip().lower()
    Payee_vec = vectorizer.transform([Payee_name])
    date = pd.to_datetime(date)
    day_of_week = date.dayofweek
    month = date.month
    is_weekend = 1 if day_of_week in [5, 6] else 0
    amount_log = np.log1p(amount)
    Payee_length = len(Payee_name)

    numeric_features = np.array([[amount, amount_log, day_of_week, month, is_weekend, Payee_length]])
    X_input = hstack([Payee_vec, numeric_features])

    return model.predict(X_input)[0]