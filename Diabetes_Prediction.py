# Import libraries for data manipulation
import pandas as pd
import numpy as np

# Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


# ----------------------------------------
# 1. Load Dataset
# ----------------------------------------

# Read the dataset from CSV file
df = pd.read_csv("diabetes.csv")


# ----------------------------------------
# 2. Handle Invalid Zero Values
# ----------------------------------------

# Some medical measurements cannot logically be zero
# Replace those zeros with NaN so they can be treated as missing values

cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)


# ----------------------------------------
# 3. Fill Missing Values
# ----------------------------------------

# Use Median Imputation to fill missing values
# Median is robust to outliers and works well with medical data

imputer = SimpleImputer(strategy="median")
df[cols_with_zero] = imputer.fit_transform(df[cols_with_zero])


# ----------------------------------------
# 4. Split Features and Target
# ----------------------------------------

# X contains all input features
# y contains the target variable (Outcome)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ----------------------------------------
# 5. Train-Test Split
# ----------------------------------------

# Split dataset into training and testing sets
# 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ----------------------------------------
# 6. Feature Scaling
# ----------------------------------------

# Standardize the data so all features have similar scale

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ========================================
# MODEL 1: LOGISTIC REGRESSION
# ========================================

# ----------------------------------------
# 7. Train Logistic Regression Model
# ----------------------------------------

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)


# ----------------------------------------
# 8. Predictions
# ----------------------------------------

# Predict class labels
y_pred = log_model.predict(X_test)

# Predict probabilities (used for ROC curve)
y_prob = log_model.predict_proba(X_test)[:,1]


# ----------------------------------------
# 9. Model Evaluation
# ----------------------------------------

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_prob))


# ----------------------------------------
# 10. Confusion Matrix (Logistic Regression)
# ----------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("images/confusion_matrix_logistic.png", dpi=300, bbox_inches="tight")
plt.show()


# ----------------------------------------
# 11. ROC Curve (Logistic Regression)
# ----------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0,1], [0,1], linestyle="--")

plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.savefig("images/roc_curve_logistic.png", dpi=300, bbox_inches="tight")
plt.show()


# ========================================
# MODEL 2: RANDOM FOREST
# ========================================

# ----------------------------------------
# 12. Train Random Forest Model
# ----------------------------------------

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)


# ----------------------------------------
# 13. Predictions
# ----------------------------------------

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]


# ----------------------------------------
# 14. Model Evaluation
# ----------------------------------------

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, y_prob_rf))


# ----------------------------------------
# 15. Confusion Matrix (Random Forest)
# ----------------------------------------

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("images/confusion_matrix_random_forest.png", dpi=300, bbox_inches="tight")
plt.show()


# ----------------------------------------
# 16. ROC Curve (Random Forest)
# ----------------------------------------

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(6,5))
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0,1], [0,1], linestyle="--")

plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.savefig("images/roc_curve_random_forest.png", dpi=300, bbox_inches="tight")
plt.show()


# ----------------------------------------
# 17. Model Comparison
# ----------------------------------------

print("\nModel Comparison")
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))