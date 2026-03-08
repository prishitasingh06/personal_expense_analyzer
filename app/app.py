import streamlit as st
import pickle
from utils import predict_category
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Pastel background and button styling via CSS
st.markdown("""
<style>
    html, body, .main {
        background-color: #fef9f4 !important;
        color: #333333 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    /* Buttons */
    div.stButton > button {
        background-color: #a3cef1 !important;
        color: black !important;
        border-radius: 10px !important;
        padding: 0.5em 1em !important;
        font-weight: 600 !important;
        transition: background-color 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #89b8e0 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
with open('../models/expense_model.pkl', 'rb') as f:
    model, vectorizer, numeric_cols = pickle.load(f)

st.title("Personal Expense Analyzer💳")

Payee = st.text_input("Payee Name")
amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
date = st.date_input("Transaction Date", datetime.date.today())

if st.button("Predict Category"):
    category = predict_category(
        Payee, amount, date,
        model, vectorizer, numeric_cols
    )
    st.success(f"Predicted Category: {category}")

# upload CSV and show predictions
uploaded_file = st.file_uploader("Upload CSV of Transactions", type=["csv"], key="upload1")

if uploaded_file:
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1')

    df['PredictedCategory'] = df.apply(lambda row:
        predict_category(
            row['Payee'],
            row['Amount'],
            row['Date'],
            model,
            vectorizer,
            numeric_cols
        ), axis=1
    )

    st.dataframe(df)

    st.subheader("📊 Expense Analytics")

    df['Date'] = pd.to_datetime(df['Date'])

    # Use pastel palette from seaborn
    pastel_palette = sns.color_palette("pastel")

    # 1. Category Distribution
    st.write("### Category Distribution")
    fig1, ax1 = plt.subplots()
    df['PredictedCategory'].value_counts().plot(kind='bar', ax=ax1, color=pastel_palette)
    ax1.set_ylabel("Number of Transactions")
    ax1.set_xlabel("Category")
    st.pyplot(fig1)

    # 2️. Total Spending by Category
    st.write("### Total Spending by Category")
    category_spend = df.groupby('PredictedCategory')['Amount'].sum().sort_values()
    fig2, ax2 = plt.subplots()
    category_spend.plot(kind='barh', ax=ax2, color=pastel_palette)
    ax2.set_xlabel("Total Spending ($)")
    ax2.set_ylabel("Category")
    st.pyplot(fig2)

    # 3️. Daily Spending Trend
    st.write("### Daily Spending Trend")
    daily_spend = df.groupby(df['Date'].dt.date)['Amount'].sum()
    fig3, ax3 = plt.subplots()
    daily_spend.plot(ax=ax3, color=pastel_palette[0])
    ax3.set_ylabel("Spending ($)")
    ax3.set_xlabel("Date")
    st.pyplot(fig3)

    # 4️4. Monthly Spending
    st.write("### Monthly Spending")

    monthly_spend = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum()
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    monthly_spend.sort_index().plot(kind='bar', ax=ax4, color=pastel_palette)
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Total Spending ($)")
    ax4.set_title("Monthly Spending")
    plt.xticks(rotation=45)
    st.pyplot(fig4)
    #latin1 (ISO-8859-1) encoding can decode any byte sequence without errors (but may misinterpret some characters),has been used as a fallback when UTF-8 decoding fails.
