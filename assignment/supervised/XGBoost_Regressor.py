# ==========================================
# E-COMMERCE PURCHASE AMOUNT PREDICTION
# MODEL 1: XGBOOST REGRESSOR
# ==========================================

# 🔹 Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# 🔹 Step 2: Load Dataset
df = pd.read_csv("online_shoppers_intention.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# 🔹 Step 3: Basic Preprocessing

# Convert Revenue column to numeric (True/False → 1/0)
df['Revenue'] = df['Revenue'].astype(int)

# Convert categorical columns using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

print("Shape After Encoding:", df.shape)


# 🔹 Step 4: Define Features & Target for Regression

# Target = PageValues (purchase amount proxy)
y = df['PageValues']

# Features = All other columns except PageValues
X = df.drop(['PageValues'], axis=1)


# 🔹 Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)


# 🔹 Step 6: Initialize XGBoost Regressor
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)


# 🔹 Step 7: Train Model
xgb.fit(X_train, y_train)


# 🔹 Step 8: Make Predictions
y_pred = xgb.predict(X_test)


# 🔹 Step 9: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)