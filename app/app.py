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
# Predict Single Transaction Category in an expander
# --------------------------
with st.expander("Predict Single Transaction Category"):
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

    # Convert types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])
    df["Payee"] = df["Payee"].astype(str).str.strip()

    # --------------------------
    # Predict Categories
    # --------------------------
    st.subheader("Predicted Categories")
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
    # Expense Analytics
    # --------------------------
    st.subheader("Expense Analytics")
    pastel = sns.color_palette("pastel")

    # Category Distribution
    st.write("### Category Distribution")
    fig1, ax1 = plt.subplots()
    df["PredictedCategory"].value_counts().plot(kind="bar", color=pastel, ax=ax1)
    ax1.set_ylabel("Transactions")
    ax1.set_xlabel("Category")
    st.pyplot(fig1)

    # Total Spending by Category
    st.write("### Total Spending by Category")
    category_spend = df.groupby("PredictedCategory")["Amount"].sum()
    fig2, ax2 = plt.subplots()
    category_spend.plot(kind="barh", color=pastel, ax=ax2)
    ax2.set_xlabel("Total Spending ($)")
    st.pyplot(fig2)

    # Daily Spending Trend
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
    monthly_spend.sort_index().plot(kind="bar", color=pastel, ax=ax4)
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Total Spending ($)")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    # Spending Heatmap by Weekday & Category
    st.write("### Spending Heatmap (Weekday vs Category)")
    heatmap_data = df.pivot_table(
        index=df["Date"].dt.day_name(),
        columns="PredictedCategory",
        values="Amount",
        aggfunc="sum"
    )
    fig5, ax5 = plt.subplots(figsize=(10,5))
    sns.heatmap(heatmap_data.fillna(0), annot=True, fmt=".0f", cmap="Blues", ax=ax5)
    st.pyplot(fig5)

    # --------------------------
    # Personalized Recommendations
    # --------------------------
    st.subheader("💡 Personalized Recommendations")

    # 1. Budget Suggestions
    monthly_budget = df.groupby("PredictedCategory")["Amount"].sum().to_dict()
    recommended_budget = {cat: amt * 1.2 for cat, amt in monthly_budget.items()}  # 20% buffer
    st.write("### Suggested Monthly Budgets")
    for cat, limit in recommended_budget.items():
        st.write(f"**{cat}**: ${limit:.2f} (based on current spending)")

    # Overspending Alerts
    st.write("### Overspending Alerts")
    today = datetime.date.today()
    current_month = df[df["Date"].dt.month == today.month]
    current_spend = current_month.groupby("PredictedCategory")["Amount"].sum()
    for cat, spent in current_spend.items():
        if spent > recommended_budget.get(cat, 0):
            st.warning(f"You're overspending in **{cat}**! (${spent:.2f} spent)")

    # 2. Detect Unusual Transactions
    st.write("### 🚨 Unusual Transactions")
    threshold_factor = 2
    category_avg = df.groupby("PredictedCategory")["Amount"].mean().to_dict()
    unusual = df[df.apply(lambda row: row["Amount"] > category_avg.get(row["PredictedCategory"], 0) * threshold_factor, axis=1)]
    if not unusual.empty:
        st.dataframe(unusual[["Payee", "Amount", "Date", "PredictedCategory"]])
    else:
        st.info("No unusual transactions detected.")

    # 3. Identify Recurring Payments
    st.write("### 🔁 Recurring Payments / Subscriptions")
    recurring = df.groupby("Payee").filter(lambda x: len(x) > 1)
    recurring_summary = recurring.groupby("Payee")["Amount"].mean().reset_index()
    recurring_summary.columns = ["Payee", "Average Amount"]
    if not recurring_summary.empty:
        st.dataframe(recurring_summary)
    else:
        st.info("No recurring payments detected.")

    # --------------------------
    # Download CSV
    # --------------------------
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name='predicted_expenses.csv',
        mime='text/csv'
    )