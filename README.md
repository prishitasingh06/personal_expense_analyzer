# Personal Expense Analyzer 💳
A personal expense analyzer with AI-powered category predictions and visual spending analytics.


I have built a Streamlit-based web application that helps users analyze and categorize their financial transactions automatically.

🔗 **Live App:** https://personalexpenseanalyzer-ygxncnzst4mc2lkpdrpcjq.streamlit.app/

🔗 **GitHub Repository:** https://github.com/prishitasingh06/personal_expense_analyzer

---
# ML Methodology

This project follows a structured machine learning workflow to automatically classify financial transactions into expense categories. I have tried to explain it as thoroughly as possible-

### 1. Data Preparation

The training dataset contains labeled financial transactions with the following fields:

* Payee (merchant name)
* Transaction amount
* Transaction date
* Category (target label)

Data preprocessing includes:

* Cleaning and normalizing merchant names
* Converting dates to datetime format
* Extracting day of week and month
* Handling missing values
* Log-transforming the transaction amount to reduce skewness

### 2. Feature Engineering

The model uses a combination of text features and numeric features which I have described below-

**Text Features**

* Merchant names are transformed using **TF-IDF vectorization**
* Uses **unigrams and bigrams**
* Maximum 3000 text features
* Removes English stop words

**Numeric Features**

* Transaction amount
* Log-transformed amount
* Day of week
* Month
* Weekend indicator
* Length of merchant name

The text and numeric features are combined into a single feature matrix.

### 3. Model Training

A **Random Forest Classifier** is used for category prediction because it:

* Handles mixed feature types effectively
* Is robust to noise
* Works well for tabular datasets

Model parameters:

* 200 trees
* Maximum depth of 20
* Fixed random seed for reproducibility

The dataset is split into **training (80%) and testing (20%) sets** to evaluate performance.

### 4. Prediction Pipeline

For each new transaction:

1. Merchant name is vectorized using the trained TF-IDF model.
2. Numeric features are generated from the transaction amount and date.
3. Features are combined into a single vector.
4. The trained Random Forest model predicts the expense category.

### 5. Model Persistence

The trained model and vectorizer are saved using **Pickle**, enabling the application to load the model instantly during deployment without retraining.
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
