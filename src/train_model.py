# src/train_model.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src folder
CSV_FILE = r"C:\Users\Peddyreddy hema\Desktop\student_placement_prediction\notebook\final_transformed_data.csv"
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "placement_model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

# -------------------------------
# Load Dataset
# -------------------------------
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found! Check your file path.")

df = pd.read_csv(CSV_FILE)
print("Dataset loaded successfully! Rows:", len(df))

# -------------------------------
# Feature Columns & Target
# -------------------------------
FEATURE_COLS = [
    "cgpa","coding_skill","communication_skill","aptitude_skill","problem_solving",
    "projects_count","internship_count","internship_company_level",
    "certification_count","certification_company_level"
]

TARGET_COL = "placement_status"  # 1 = Placed, 0 = Not Placed

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Scale Features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Train Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# -------------------------------
# Evaluate Model
# -------------------------------
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
print(f"Testing Accuracy: {test_acc*100:.2f}%\n")
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# -------------------------------
# Predict Probabilities for Test Set
# -------------------------------
df_test = X_test.copy()
df_test["actual_label"] = y_test
df_test["predicted_label"] = y_test_pred
df_test["placement_probability"] = model.predict_proba(X_test_scaled)[:,1]*100
df_test["placement_probability"] = df_test["placement_probability"].round(2)

print("\n--- Sample Predictions ---")
print(df_test.head())

# -------------------------------
# Save Model & Scaler
# -------------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"\nModel and scaler saved in {ARTIFACTS_DIR}")
