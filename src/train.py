"""Train and compare species classifiers on the Palmer Penguins dataset.

Pipeline:
  1. Load data/penguins.csv, drop rows with any missing values (report count).
  2. One-hot encode island and sex; target = species.
  3. Stratified 80/20 train/test split (random_state=42).
  4. 5-fold stratified CV on the train set comparing:
       - StandardScaler + LogisticRegression(max_iter=1000)
       - StandardScaler + RandomForestClassifier(random_state=42)
  5. Evaluate both on the test set (accuracy, macro F1, classification report).
  6. Pick the better test-accuracy model; save its confusion matrix to
     assets/confusion_matrix.png; print top-3 random-forest feature importances.

All reported numbers are measured on the real data.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "penguins.csv"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"Rows before cleaning: {n_before}")
    print(f"Rows dropped (any missing value): {n_dropped}")
    print(f"Rows after cleaning: {len(df)}")

    X = pd.get_dummies(df.drop(columns=["species"]), columns=["island", "sex"])
    y = df["species"]
    print(f"\nFeature matrix: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"Features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    print("Test set class distribution:")
    print(y_test.value_counts().sort_index().to_string())

    models = {
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()),
             ("clf", LogisticRegression(max_iter=1000))]
        ),
        "RandomForest": Pipeline(
            [("scaler", StandardScaler()),
             ("clf", RandomForestClassifier(random_state=42))]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n5-fold stratified CV accuracy on train set:")
    cv_results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = scores
        print(f"  {name}: mean={scores.mean():.4f} std={scores.std():.4f}")

    print("\nTest set evaluation:")
    test_results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        test_results[name] = (acc, preds)
        print(f"\n--- {name} ---")
        print(f"Test accuracy: {acc:.4f}")
        print(f"Test macro F1: {macro_f1:.4f}")
        print("Classification report:")
        print(classification_report(y_test, preds))

    best_name = max(test_results, key=lambda k: test_results[k][0])
    best_acc, best_preds = test_results[best_name]
    print(f"\nBest model on test accuracy: {best_name} ({best_acc:.4f})")

    # Confusion matrix for the best model
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_preds, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion matrix ({best_name}, test set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(ASSETS / "confusion_matrix.png")
    plt.close(fig)
    print(f"Saved assets/confusion_matrix.png")

    # Top-3 random-forest feature importances
    rf_pipe = models["RandomForest"]
    importances = rf_pipe.named_steps["clf"].feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print("\nTop-3 RandomForest feature importances:")
    for feat, imp in feat_imp.head(3).items():
        print(f"  {feat}: {imp:.4f}")


if __name__ == "__main__":
    main()
