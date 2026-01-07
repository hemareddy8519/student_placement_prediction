# =========================================================
# STUDENT PLACEMENT PIPELINE – FINAL SAFE VERSION
# =========================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, GridSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)

# -------------------- CONFIG ----------------------------
DATA_PATH = "notebook/final_transformed_data.csv"
OUTPUT_DIR = "notebook/ml_outputs"
TARGET = "placement_status"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- LOAD DATA -------------------------
df = pd.read_csv(DATA_PATH)
X = pd.get_dummies(df.drop(TARGET, axis=1), drop_first=True)
y = df[TARGET]

# -------------------- TRAIN-TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# -------------------- SCALING ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- MODELS ----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000
    ),
    "Decision Tree": DecisionTreeClassifier(
    max_depth=3,                 # VERY IMPORTANT
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )
}

# -------------------- PARAMETER GRIDS -------------------
param_grids = {
    "Logistic Regression": {"C": [0.5, 1]},
    "Decision Tree": {"max_depth": [5, 6]},
    "Random Forest": {"max_depth": [6, 8]}
}

results = []

# -------------------- TRAIN + TUNE ----------------------
for name, model in models.items():
    print(f"\n=== Training {name} ===")

    X_tr = X_train_scaled if name == "Logistic Regression" else X_train
    X_te = X_test_scaled if name == "Logistic Regression" else X_test

    grid = GridSearchCV(
        model,
        param_grids[name],
        cv=3,
        scoring="f1",
        n_jobs=1      # 🔥 IMPORTANT (prevents memory crash)
    )
    grid.fit(X_tr, y_train)
    print("Best Parameters:", grid.best_params_)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_te)
    y_prob = best_model.predict_proba(X_te)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Estimator": best_model
    })

# -------------------- PRINT COMPARISON -------------------
comparison_df = pd.DataFrame(results).drop(columns=["Estimator"])
print("\n=== MODEL COMPARISON ===")
print(comparison_df.to_string(index=False))

# -------------------- CONFUSION MATRIX + ROC -------------
plt.figure(figsize=(8,6))
for r in results:
    model = r["Estimator"]
    name = r["Model"]
    X_eval = X_test_scaled if name == "Logistic Regression" else X_test

    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:,1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Placed","Placed"])
    disp.plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{name.replace(' ','')}.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=name)

plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_comparison.png"))
plt.close()

# -------------------- LEARNING CURVE ---------------------
for r in results:
    model = r["Estimator"]
    name = r["Model"]
    X_tr = X_train_scaled if name == "Logistic Regression" else X_train

    train_sizes, train_scores, test_scores = learning_curve(
        model, X_tr, y_train, cv=3, scoring="accuracy"
    )

    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"lc_{name.replace(' ','')}.png"))
    plt.close()

# -------------------- BUSINESS THRESHOLD -----------------
gain, cost = 10000, 5000
best_lr = [r for r in results if r["Model"]=="Logistic Regression"][0]["Estimator"]

y_prob_lr = best_lr.predict_proba(X_test_scaled)[:,1]
thresholds = np.linspace(0.1, 0.9, 41)

scores = []
for t in thresholds:
    y_t = (y_prob_lr >= t).astype(int)
    TP = ((y_test==1) & (y_t==1)).sum()
    FP = ((y_test==0) & (y_t==1)).sum()
    scores.append(TP*gain - FP*cost)

best_t = thresholds[np.argmax(scores)]

# Business score at default threshold 0.5
y_default = (y_prob_lr >= 0.5).astype(int)
TP_d = ((y_test==1) & (y_default==1)).sum()
FP_d = ((y_test==0) & (y_default==1)).sum()
default_score = TP_d*gain - FP_d*cost

print(f"Business Score @0.5 Threshold: {default_score}")
print(f"Business Score @Optimal Threshold ({best_t}): {max(scores)}")

# -------------------- BEST MODEL -------------------------
# Ignore Decision Tree for final deployment
final_candidates = [
    r for r in results if r["Model"] != "Decision Tree"
]

best_model = max(
    final_candidates,
    key=lambda x: (x["ROC_AUC"], x["F1"])
)



pickle.dump(
    best_model["Estimator"],
    open(os.path.join(OUTPUT_DIR, "final_model.pkl"), "wb")
)

print(f"\n✅ Best Model Selected: {best_model['Model']}")
print(f"✅ Optimal Business Threshold: {best_t}")
# -------------------- FEATURE IMPORTANCE -----------------
if best_model["Model"] == "Random Forest":
    importances = best_model["Estimator"].feature_importances_
    feature_importance = pd.Series(
        importances, index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

    plt.figure(figsize=(8,5))
    feature_importance.head(10).plot(kind="bar")
    plt.title("Top 10 Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_rf.png"))
    plt.close()
# -------------------- FEATURE IMPORTANCE (LOGISTIC REGRESSION) -----------------
if best_model["Model"] == "Logistic Regression":

    coef = best_model["Estimator"].coef_[0]

    feature_importance = pd.Series(
        coef, index=X.columns
    ).sort_values(key=abs, ascending=False)

    print("\nTop 10 Important Features (Logistic Regression):")
    print(feature_importance.head(10))

    plt.figure(figsize=(8,5))
    feature_importance.head(10).plot(kind="bar")
    plt.title("Top 10 Feature Importance - Logistic Regression")
    plt.ylabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_logistic.png"))
    plt.close()
