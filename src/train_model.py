# ------------------------------
# IMPORT LIBRARIES
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Disable GUI backend for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

import xgboost as xgb

# ------------------------------
# CONFIG
# ------------------------------
DATA_PATH = "notebook/final_transformed_data.csv"
OUTPUT_DIR = "notebook/ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# STEP 1: LOAD DATA
# ------------------------------
df = pd.read_csv(DATA_PATH)

# ------------------------------
# STEP 2: DATA PREPARATION
# ------------------------------
# Separate features and target
X = df.drop("placement_status", axis=1)
y = df["placement_status"]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# STEP 3: BASELINE MODEL TRAINING
# ------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

baseline_results = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    baseline_results.append([name, round(acc,3), round(roc,3), round(fit_time,2)])
    print(f"{name} -> Accuracy: {acc:.3f}, ROC-AUC: {roc:.3f}, Fit Time: {fit_time:.2f}s")

# Save baseline results
baseline_df = pd.DataFrame(baseline_results, columns=["Model", "Accuracy", "ROC-AUC", "Fit Time"])
baseline_df.to_csv(os.path.join(OUTPUT_DIR,"baseline_model_comparison.csv"), index=False)
print("\nBaseline model comparison saved ✅\n")
print(baseline_df)

# ------------------------------
# STEP 4: ADVANCED MODELS (Random Forest + XGBoost)
# ------------------------------
advanced_models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5,
                                 random_state=42, use_label_encoder=False, eval_metric='logloss')
}

advanced_results = []

for name, model in advanced_models.items():
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    advanced_results.append([name, round(acc,3), round(roc,3), round(fit_time,2)])
    print(f"{name} -> Accuracy: {acc:.3f}, ROC-AUC: {roc:.3f}, Fit Time: {fit_time:.2f}s")

# Save advanced model comparison
adv_df = pd.DataFrame(advanced_results, columns=["Model", "Accuracy", "ROC-AUC", "Fit Time"])
adv_df.to_csv(os.path.join(OUTPUT_DIR,"advanced_model_comparison.csv"), index=False)
print("\nAdvanced model comparison saved ✅")
print(adv_df)

# ------------------------------
# STEP 5: FEATURE IMPORTANCE
# ------------------------------
# Random Forest Feature Importance
rf_model = advanced_models["Random Forest"]
rf_importances = rf_model.feature_importances_
rf_feat_df = pd.DataFrame({"Feature": X.columns, "Importance": rf_importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=rf_feat_df)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"rf_feature_importance.png"))
plt.close()
print("Random Forest Feature Importance saved ✅")

# XGBoost Feature Importance
xgb_model = advanced_models["XGBoost"]
xgb_importances = xgb_model.feature_importances_
xgb_feat_df = pd.DataFrame({"Feature": X.columns, "Importance": xgb_importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=xgb_feat_df)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"xgb_feature_importance.png"))
plt.close()
print("XGBoost Feature Importance saved ✅")

# ------------------------------
# STEP 6: LEARNING CURVE (Random Forest)
# ------------------------------
train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X, y, cv=5, scoring="roc_auc", train_sizes=np.linspace(0.1,1.0,5)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color='red', label="Validation score")
plt.title("Learning Curve (Random Forest)")
plt.xlabel("Training Size")
plt.ylabel("ROC-AUC Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"rf_learning_curve.png"))
plt.close()
print("Random Forest Learning Curve saved ✅")

# ------------------------------
# STEP 7: CONFUSION MATRIX + THRESHOLD TUNING (XGBoost)
# ------------------------------
y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]
threshold = 0.4
y_pred_thresh = (y_prob_xgb >= threshold).astype(int)

cm = confusion_matrix(y_test, y_pred_thresh)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Placed","Placed"])
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix (Threshold=0.4)")
plt.savefig(os.path.join(OUTPUT_DIR,"xgb_confusion_matrix.png"))
plt.close()
print("XGBoost Confusion Matrix saved ✅")

# ------------------------------
# STEP 8: CROSS-VALIDATION (Random Forest)
# ------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring="roc_auc")
print(f"Random Forest CV ROC-AUC: {cv_scores.mean():.3f}")

# ------------------------------
# STEP 9: HYPERPARAMETER TUNING (RandomizedSearchCV - Optional)
# ------------------------------
param_grid = {
    "n_estimators": [100,200,300],
    "max_depth": [None,5,10],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4]
}

rf_random = RandomizedSearchCV(
    rf_model, param_distributions=param_grid, n_iter=10,
    cv=3, scoring="roc_auc", random_state=42
)
rf_random.fit(X_train, y_train)
print("Best Random Forest Hyperparameters:", rf_random.best_params_)

# ------------------------------
# ALL ARTIFACTS SAVED
# ------------------------------
print(f"\nAll ML outputs saved in: {OUTPUT_DIR} ✅")
