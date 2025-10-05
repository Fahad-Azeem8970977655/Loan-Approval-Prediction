# Loan Approval Prediction - script.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ----------------------------
# Step 1: Load Dataset
# ----------------------------
df = pd.read_csv("loan.csv")  # change filename to match your dataset
print("Dataset shape:", df.shape)
print(df.head())

# ----------------------------
# Step 2: Data Cleaning
# ----------------------------
# Handle missing values
df = df.fillna(method="ffill")  # simple forward fill

# Encode categorical variables
label_enc = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = label_enc.fit_transform(df[col].astype(str))

# ----------------------------
# Step 3: Features & Target
# ----------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Step 4: Handle Imbalanced Data (SMOTE)
# ----------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())

# ----------------------------
# Step 5: Train Models
# ----------------------------
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_res, y_train_res)
y_pred_lr = log_reg.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train_res, y_train_res)
y_pred_dt = dt.predict(X_test)

# ----------------------------
# Step 6: Evaluation
# ----------------------------
print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, y_pred_lr))

print("\n--- Decision Tree Report ---")
print(classification_report(y_test, y_pred_dt))

# Confusion Matrix Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Logistic Regression - Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Decision Tree - Confusion Matrix")

plt.show()
