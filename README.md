# Personal Expense Analyzer 💳
A personal expense analyzer with AI-powered category predictions and visual spending analytics.


I have built a Streamlit-based web application that helps users analyze and categorize their financial transactions automatically.

🔗 **Live App:** https://personalexpenseanalyzer-ygxncnzst4mc2lkpdrpcjq.streamlit.app/
🔗 **GitHub Repository:** https://github.com/prishitasingh06/personal_expense_analyzer

---

# Features
### 1. Expense Category Prediction

Users can manually enter:
* Payee name
* Transaction amount
* Transaction date

The model predicts the most likely expense category

### 2. CSV Transaction Upload

Users can upload a CSV file of transactions, and the app will:

* Automatically clean column names
* Handle different CSV encodings
* Predict categories for all transactions

Supported column variations:

* `Payee`, `Description`, `Merchant`, `Transaction`
* `Amount`, `Debit`, `Credit`
* `Date`, `Transaction Date`

### 3. Expense Analytics Dashboard

After processing transactions, the app generates visual insights:

* **Category Distribution** – number of transactions per category
* **Total Spending by Category** – horizontal bar chart
* **Daily Spending Trend** – time-based line chart
* **Monthly Spending Summary** – spending per month

These visualizations help users understand their spending patterns.

### 4. Automatic Model Training

If the trained model file is missing, the app will:

1. Load the dataset
2. Train a machine learning model
3. Save it automatically for future predictions
---

# Tech Stack

**Frontend / UI**

* Streamlit

**Data Processing**

* Pandas
* NumPy

**Machine Learning**

* Scikit-learn

**Visualization**

* Matplotlib
* Seaborn

---

# Project Structure

```
personal_expense_analyzer
│
├── app/
│   ├── app.py                # Main Streamlit application
│   ├── utils.py              # Data loading, training, prediction functions
│   └── expense_a.csv         # Training dataset
│
├── models/
│   └── expense_model.pkl     # Saved trained model
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

# Installation (Steps to run locally)

### 1. Clone the repository

```
git clone https://github.com/prishitasingh06/personal_expense_analyzer.git
cd personal_expense_analyzer
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```
streamlit run app/app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

# CSV Format Example

Example input file:

```
Payee,Amount,Date
Starbucks,5.50,2024-01-01
Uber,18.20,2024-01-02
Amazon,45.99,2024-01-03
```

The app will automatically predict the **expense category** for each transaction.

---

# Deployment
The application has been deployed using Streamlit Community Cloud.


Live version:
https://personalexpenseanalyzer-ygxncnzst4mc2lkpdrpcjq.streamlit.app/
