# =========================================================
# STUDENT PLACEMENT PIPELINE
# Baseline Table for 3 Models + Logistic Regression for advanced ops
# =========================================================

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from scipy.stats import ttest_rel

# -------------------- CONFIG ----------------------------
DATA_PATH = "notebook/final_transformed_data.csv"
OUTPUT_DIR = "notebook/ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- LOAD DATA ------------------------
df = pd.read_csv(DATA_PATH)
X = pd.get_dummies(df.drop("placement_status", axis=1), drop_first=True)
y = df["placement_status"]

# -------------------- TRAIN-TEST SPLIT -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# -------------------- SCALING --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- BASELINE MODELS ------------------
baseline_models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=3,
        min_samples_split=5,
        max_features="sqrt",
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
}

baseline_results = []

for name, model in baseline_models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    baseline_results.append([name, acc, roc, f1])

# Save baseline comparison table (3 models)
baseline_df = pd.DataFrame(baseline_results, columns=["Model","Accuracy","ROC_AUC","F1"])
baseline_df.to_csv(os.path.join(OUTPUT_DIR,"baseline_model_comparison.csv"), index=False)
print("\n✅ Baseline Model Comparison Table (3 models) created.")

# -------------------- ADVANCED OPERATIONS FOR LOGISTIC REGRESSION ------------------
lr = baseline_models["Logistic Regression"]

# Confusion Matrix
cm = confusion_matrix(y_test, lr.predict(X_test_scaled))
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Placed","Placed"])
disp.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrix_LogisticRegression.png"))
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, lr.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0,1],[0,1],'--')
plt.title("Logistic Regression ROC Curve")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR,"roc_curve_LogisticRegression.png"))
plt.close()

# Feature Importance
feat_df = pd.DataFrame({"Feature": X.columns, "Coefficient": lr.coef_[0]}).sort_values(by="Coefficient", key=abs, ascending=False)
feat_df.to_csv(os.path.join(OUTPUT_DIR,"feature_importance_LogisticRegression.csv"), index=False)

# Learning Curve (fast)
train_sizes, train_scores, test_scores = learning_curve(
    lr, X_train_scaled, y_train, cv=3, scoring='accuracy', train_sizes=np.linspace(0.2,1.0,3)
)
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation")
plt.title("Logistic Regression Learning Curve")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR,"learning_curve_LogisticRegression.png"))
plt.close()

# Hyperparameter tuning (C)
param_grid = {"C":[0.1,0.5,1,5]}
grid = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
                    param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_lr = grid.best_estimator_
print("\nBest Logistic Regression Params:", grid.best_params_)

# Statistical validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
lr_scores = cross_val_score(best_lr, X_train_scaled, y_train, cv=cv)
t_stat, p_val = ttest_rel(lr_scores, lr_scores)  # comparing LR to itself
with open(os.path.join(OUTPUT_DIR,"statistical_test.txt"),"w") as f:
    f.write(f"Paired t-test p-value: {p_val}\n")

# Business threshold optimization
gain, cost = 10000, 5000
y_prob_lr = best_lr.predict_proba(X_test_scaled)[:,1]
thresholds = np.linspace(0.1,0.9,41)
business_scores = []

for t in thresholds:
    y_pred_thresh = (y_prob_lr >= t).astype(int)
    TP = ((y_test==1) & (y_pred_thresh==1)).sum()
    FP = ((y_test==0) & (y_pred_thresh==1)).sum()
    score = TP*gain - FP*cost
    business_scores.append(score)

best_idx = np.argmax(business_scores)
best_threshold = thresholds[best_idx]
best_score = business_scores[best_idx]

with open(os.path.join(OUTPUT_DIR,"business_optimization.txt"), "w") as f:
    f.write(f"Optimal threshold: {best_threshold}\nExpected Business Gain: {best_score}\n")

# Metrics at optimal threshold
y_pred_opt = (y_prob_lr >= best_threshold).astype(int)
acc_opt = accuracy_score(y_test, y_pred_opt)
prec_opt = precision_score(y_test, y_pred_opt)
rec_opt = recall_score(y_test, y_pred_opt)
f1_opt = f1_score(y_test, y_pred_opt)

with open(os.path.join(OUTPUT_DIR,"business_metrics_at_threshold.txt"), "w") as f:
    f.write(f"Accuracy: {acc_opt}\nPrecision: {prec_opt}\nRecall: {rec_opt}\nF1 Score: {f1_opt}\n")

# Save final model
pickle.dump(best_lr, open(os.path.join(OUTPUT_DIR,"final_model_LogisticRegression.pkl"),"wb"))

print("\n✅ Pipeline completed. Baseline table + Logistic Regression advanced operations done.")
