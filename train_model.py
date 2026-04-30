"""
train_model.py
==============

Trains three classification models (Logistic Regression, Random Forest, XGBoost)
on the Kaggle Loan Prediction dataset, evaluates them, and saves the best one
as a single pickle file (`model/loan_model.pkl`) along with the preprocessing
pipeline.

Usage
-----
    python train_model.py

The script expects `data/loan_train.csv` to exist. If it doesn't, a synthetic
dataset (with the same schema as the Kaggle one) is generated automatically so
the project still works end-to-end out of the box.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from utils.preprocessing import (CATEGORICAL_COLS, NUMERIC_COLS,
                                 clean_dataframe, engineer_features)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "loan_train.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.pkl")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Synthetic data generator (used only if the real CSV isn't available)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(n: int = 1000) -> pd.DataFrame:
    """
    Build a synthetic loan dataset that mirrors the Kaggle schema.
    Approval is driven by realistic factors (credit history, income vs loan
    amount, education) plus some noise so the models have something to learn.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    df = pd.DataFrame({
        "Gender":            rng.choice(["Male", "Female"], n, p=[0.8, 0.2]),
        "Married":           rng.choice(["Yes", "No"], n, p=[0.65, 0.35]),
        "Dependents":        rng.choice(["0", "1", "2", "3+"], n,
                                        p=[0.55, 0.18, 0.17, 0.10]),
        "Education":         rng.choice(["Graduate", "Not Graduate"], n,
                                        p=[0.78, 0.22]),
        "Self_Employed":     rng.choice(["Yes", "No"], n, p=[0.15, 0.85]),
        "ApplicantIncome":   rng.integers(1500, 20000, n),
        "CoapplicantIncome": rng.choice(
            [0, 0, 0] + list(range(1000, 8000, 500)), n
        ),
        "LoanAmount":        rng.integers(50, 600, n),
        "Loan_Amount_Term":  rng.choice([120, 180, 240, 300, 360, 480], n,
                                        p=[0.05, 0.05, 0.10, 0.10, 0.65, 0.05]),
        "Credit_History":    rng.choice([1.0, 0.0], n, p=[0.85, 0.15]),
        "Property_Area":     rng.choice(["Urban", "Semiurban", "Rural"], n,
                                        p=[0.35, 0.40, 0.25]),
    })

    # Inject some missing values like the original dataset
    for col, frac in [("Gender", 0.02), ("Married", 0.01),
                      ("Self_Employed", 0.05), ("LoanAmount", 0.03),
                      ("Credit_History", 0.08)]:
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Build a realistic target signal
    income       = df["ApplicantIncome"] + df["CoapplicantIncome"].fillna(0)
    loan_amt     = df["LoanAmount"].fillna(df["LoanAmount"].median()) * 1000
    debt_ratio   = loan_amt / income.replace(0, np.nan)
    credit       = df["Credit_History"].fillna(1.0)

    score = (
        2.5 * credit
        - 0.6 * (debt_ratio > 4).astype(float)
        + 0.4 * (df["Education"] == "Graduate").astype(float)
        + 0.2 * (df["Property_Area"] == "Semiurban").astype(float)
        + rng.normal(0, 0.7, n)
    )
    df["Loan_Status"] = np.where(score > 1.5, "Y", "N")
    return df


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the training CSV, falling back to synthetic data if absent."""
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Loading dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print(f"[WARN] {DATA_PATH} not found. Generating synthetic dataset.")
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_dataset(n=2000)
    df.to_csv(DATA_PATH, index=False)
    print(f"[INFO] Synthetic dataset saved to {DATA_PATH}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer that scales numerics and one-hot encodes categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Print a full evaluation report and return metrics as a dict."""
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds)
    rec   = recall_score(y_test, preds)
    cm    = confusion_matrix(y_test, preds)

    print(f"\n========== {name} ==========")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))

    return {"name": name, "accuracy": acc, "precision": prec,
            "recall": rec, "confusion_matrix": cm.tolist(), "model": model}


def main() -> None:
    # 1. Load + clean data --------------------------------------------------
    df = load_data()
    df = clean_dataframe(df)
    df = engineer_features(df)

    # Encode target
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    df = df.dropna(subset=["Loan_Status"])

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df["Loan_Status"].astype(int)

    # 2. Train/test split ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    preprocessor = build_preprocessor()

    # 3. Define candidate models -------------------------------------------
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    # 4. Train + evaluate each model ---------------------------------------
    results = []
    for name, clf in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                   ("classifier",   clf)])
        pipeline.fit(X_train, y_train)
        results.append(evaluate_model(name, pipeline, X_test, y_test))

    # 5. Pick the best model (by accuracy) ---------------------------------
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n[INFO] Best model: {best['name']} "
          f"(accuracy = {best['accuracy']:.4f})")

    # 6. Save model + metadata ---------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "model":         best["model"],
        "best_name":     best["name"],
        "metrics":       {k: v for k, v in best.items() if k != "model"},
        "feature_cols":  feature_cols,
        "all_results":   [{k: v for k, v in r.items() if k != "model"}
                          for r in results],
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"[INFO] Saved best model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
