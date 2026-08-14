"""
CODSOFT - TASK 3: CUSTOMER CHURN PREDICTION
==============================================
Dataset: Bank Customer Churn Modelling (Kaggle)
File: Churn_Modelling.csv
Target column: Exited (1 = customer churned, 0 = customer stayed)

Approach:
 1. Load Churn_Modelling.csv
 2. Drop identifying columns (RowNumber, CustomerId, Surname) - not predictive
 3. Encode categorical features (Geography, Gender)
 4. Scale numeric features (CreditScore, Age, Balance, EstimatedSalary, etc.)
 5. Train 3 classifiers: Logistic Regression, Random Forest, Gradient Boosting
 6. Evaluate with Accuracy, Precision, Recall, F1-score, ROC-AUC
 7. Save best model + plots + feature importance
"""

import os
import pickle
import numpy as np
import pandas as pd

# Always look for files in the same folder as this script,
# regardless of where the terminal's working directory happens to be.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_PATH = "Churn_Modelling.csv"
TARGET_COL = "Exited"
RANDOM_STATE = 42


def main():
    # -----------------------------------------------------------
    # STEP 1: LOAD DATA
    # -----------------------------------------------------------
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} customers with {df.shape[1]} columns")

    # -----------------------------------------------------------
    # STEP 2: DROP NON-PREDICTIVE IDENTIFIER COLUMNS
    # -----------------------------------------------------------
    print("\nCleaning data...")
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # -----------------------------------------------------------
    # STEP 3: EXPLORE CHURN DISTRIBUTION
    # -----------------------------------------------------------
    churn_counts = df[TARGET_COL].value_counts()
    churn_pct = (churn_counts.get(1, 0) / len(df)) * 100
    print(f"\nChurned customers: {churn_counts.get(1, 0)} ({churn_pct:.1f}%)")
    print(f"Retained customers: {churn_counts.get(0, 0)} ({100 - churn_pct:.1f}%)")

    plt.figure(figsize=(5, 4))
    df[TARGET_COL].value_counts().plot(kind="bar", color=["#4C72B0", "#C44E52"])
    plt.xticks([0, 1], ["Retained", "Churned"], rotation=0)
    plt.ylabel("Count")
    plt.title("Churn Distribution")
    plt.tight_layout()
    plt.savefig("churn_distribution.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------
    # STEP 4: ENCODE CATEGORICAL FEATURES
    # -----------------------------------------------------------
    print("\nEncoding categorical features...")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # -----------------------------------------------------------
    # STEP 5: SCALE NUMERIC FEATURES
    # -----------------------------------------------------------
    print("Scaling numeric features...")
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    num_cols = [c for c in num_cols if c in df.columns]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # -----------------------------------------------------------
    # STEP 6: TRAIN / TEST SPLIT
    # -----------------------------------------------------------
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # -----------------------------------------------------------
    # STEP 7: TRAIN MULTIPLE MODELS
    # -----------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    results = {}
    trained_models = {}
    roc_data = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
        trained_models[name] = model
        roc_data[name] = roc_curve(y_test, probs)

        print(f"{name} -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
              f"Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    # -----------------------------------------------------------
    # STEP 8: PICK BEST MODEL (by F1-score)
    # -----------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = trained_models[best_name]
    print(f"\nBest model (by F1-score): {best_name}")

    best_preds = best_model.predict(X_test)
    print("\nClassification Report (best model):")
    print(classification_report(y_test, best_preds, target_names=["Retained", "Churned"]))

    # -----------------------------------------------------------
    # STEP 9: SAVE COMPARISON CHART
    # -----------------------------------------------------------
    metrics_df = pd.DataFrame(results).T
    metrics_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(
        kind="bar", figsize=(10, 5), colormap="Set2"
    )
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------
    # STEP 10: CONFUSION MATRIX FOR BEST MODEL
    # -----------------------------------------------------------
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------
    # STEP 11: ROC CURVES
    # -----------------------------------------------------------
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, _) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------
    # STEP 12: FEATURE IMPORTANCE (for tree-based best model)
    # -----------------------------------------------------------
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False).head(10)
        plt.figure(figsize=(8, 5))
        importances.sort_values().plot(kind="barh", color="#55A868")
        plt.title(f"Top 10 Feature Importances - {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        plt.close()
        print("\nTop 5 most important features:")
        print(importances.head(5))

    # -----------------------------------------------------------
    # STEP 13: SAVE MODEL + SCALER + ENCODERS
    # -----------------------------------------------------------
    with open("churn_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    print("\nSaved churn_model.pkl, scaler.pkl, and encoders.pkl")

    return best_model, results


if __name__ == "__main__":
    best_model, results = main()
    print("\nDone. Charts saved: churn_distribution.png, model_comparison.png, "
          "confusion_matrix.png, roc_curves.png, feature_importance.png")
