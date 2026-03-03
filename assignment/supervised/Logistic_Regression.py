# ==========================================
# E-COMMERCE PURCHASE PREDICTION
# MODEL 2: LOGISTIC REGRESSION (L1)
# ==========================================

# 🔹 Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 🔹 Step 2: Load Dataset
df = pd.read_csv("online_shoppers_intention.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# 🔹 Step 3: Basic Preprocessing

# Convert Revenue column (True/False → 1/0)
df['Revenue'] = df['Revenue'].astype(int)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

print("Shape After Encoding:", df.shape)


# 🔹 Step 4: Define Features & Target

# Target = Revenue (0 or 1)
y = df['Revenue']

# Features = All other columns except Revenue
X = df.drop(['Revenue'], axis=1)


# 🔹 Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)


# 🔹 Step 6: Feature Scaling (Important for Logistic Regression)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 🔹 Step 7: Initialize Logistic Regression with L1 Regularization
log_reg = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    random_state=42
)


# 🔹 Step 8: Train Model
log_reg.fit(X_train, y_train)


# 🔹 Step 9: Make Predictions
y_pred = log_reg.predict(X_test)


# 🔹 Step 10: Evaluate Model

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))