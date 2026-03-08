import os
import pickle
import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_data, train_model, predict_category

# --------------------------
# Paths
# --------------------------
# Current file directory
BASE_DIR = os.path.dirname(__file__)

# CSV inside the app folder
DATA_PATH = os.path.join(BASE_DIR, "expense_a.csv")

# Model folder is at repo root
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "expense_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------
# Train model if missing
# --------------------------
if not os.path.exists(MODEL_PATH):
    st.warning("No trained model found. Training model now...")
    df_train = load_data(DATA_PATH)
    train_model(df_train, save_path=MODEL_PATH)
    st.success("Model trained and saved!")

# --------------------------
# Load model
# --------------------------
with open(MODEL_PATH, "rb") as f:
    model, vectorizer, numeric_cols = pickle.load(f)

st.success("Model loaded successfully!")

# --------------------------
# UI Styling
# --------------------------
st.markdown("""
<style>
html, body, .main {
    background-color: #fef9f4 !important;
    color: #333333 !important;
    font-family: 'Segoe UI', sans-serif !important;
}

div.stButton > button {
    background-color: #a3cef1 !important;
    color: black !important;
    border-radius: 10px !important;
    padding: 0.5em 1em !important;
    font-weight: 600 !important;
}

div.stButton > button:hover {
    background-color: #89b8e0 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Title
# --------------------------
st.title("Personal Expense Analyzer 💳")

# --------------------------
# Manual Prediction
# --------------------------
payee = st.text_input("Payee Name")
amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
date = st.date_input("Transaction Date", datetime.date.today())

if st.button("Predict Category"):
    category = predict_category(payee, amount, date, model, vectorizer, numeric_cols)
    st.success(f"Predicted Category: {category}")

# --------------------------
# CSV Upload
# --------------------------
st.subheader("Upload CSV of Transactions")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename common variations
    column_map = {
        "Description": "Payee",
        "Merchant": "Payee",
        "Transaction": "Payee",
        "Transaction Amount": "Amount",
        "Debit": "Amount",
        "Credit": "Amount",
        "Transaction Date": "Date"
    }

    df.rename(columns=column_map, inplace=True)

    # Validate columns
    required = ["Payee", "Amount", "Date"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error(f"CSV missing columns: {missing}")
        st.stop()

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Predict
    df["PredictedCategory"] = df.apply(
        lambda row: predict_category(
            str(row["Payee"]),
            float(row["Amount"]),
            row["Date"],
            model,
            vectorizer,
            numeric_cols
        ),
        axis=1
    )

    st.dataframe(df)

    # --------------------------
    # Analytics
    # --------------------------
    st.subheader("Expense Analytics")

    pastel = sns.color_palette("pastel")

    # Category Distribution
    st.write("### Category Distribution")
    fig1, ax1 = plt.subplots()
    df["PredictedCategory"].value_counts().plot(
        kind="bar", color=pastel, ax=ax1
    )
    ax1.set_ylabel("Transactions")
    ax1.set_xlabel("Category")
    st.pyplot(fig1)

    # Spending by Category
    st.write("### Total Spending by Category")
    category_spend = df.groupby("PredictedCategory")["Amount"].sum()

    fig2, ax2 = plt.subplots()
    category_spend.plot(kind="barh", color=pastel, ax=ax2)
    ax2.set_xlabel("Total Spending ($)")
    st.pyplot(fig2)

    # Daily Spending
    st.write("### Daily Spending Trend")

    daily_spend = df.groupby(df["Date"].dt.date)["Amount"].sum()

    fig3, ax3 = plt.subplots()
    daily_spend.plot(ax=ax3, color=pastel[0])
    ax3.set_ylabel("Spending ($)")
    ax3.set_xlabel("Date")
    st.pyplot(fig3)

    # Monthly Spending
    st.write("### Monthly Spending")

    monthly_spend = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum()

    fig4, ax4 = plt.subplots(figsize=(10,5))
    monthly_spend.sort_index().plot(
        kind="bar",
        ax=ax4,
        color=pastel
    )

    ax4.set_xlabel("Month")
    ax4.set_ylabel("Total Spending ($)")
    plt.xticks(rotation=45)

    st.pyplot(fig4)
    #latin1 (ISO-8859-1) encoding can decode any byte sequence without errors (but may misinterpret some characters),has been used as a fallback when UTF-8 decoding fails.
