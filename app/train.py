from utils import load_data, train_model

# dataset filename
data_path = "expense_a.csv"
df = load_data(data_path)
print(df.columns)
train_model(df)
print("Model trained and saved successfully!")