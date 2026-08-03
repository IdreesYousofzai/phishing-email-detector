"""
train.py
Loads the labelled email dataset, cleans the text, vectorises it with
TF-IDF, then trains and compares three classifiers:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Random Forest

The best model (by F1-score) is saved to models/ along with the TF-IDF
vectoriser, ready for the Flask app to load.
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import clean_text, combine_subject_body

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "phishing_email_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)



def load_data():
    df = pd.read_csv(DATA_PATH)
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("").astype(str)

    print(f"Loaded {len(df)} emails")
    print(df["label"].value_counts().rename({0: "legitimate", 1: "phishing"}))

    df["combined_text"] = df.apply(
        lambda r: combine_subject_body(r["subject"], r["body"]), axis=1
    )
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # drop anything that cleaned down to nothing
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    return df



def evaluate(name, model, X_test, y_test):
    start = time.time()
    y_pred = model.predict(X_test)
    elapsed = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Prediction time for {len(y_test)} emails: {elapsed*1000:.1f} ms")
    print("Confusion matrix (rows=actual, cols=predicted) [legit, phishing]:")
    print(cm)

    return {
        "name": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist(),
    }


def main():
    df = load_data()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print(f"\nTrain set: {len(X_train_text)} | Test set: {len(X_test_text)}")

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    results = []

    nb = MultinomialNB(alpha=0.3)
    nb.fit(X_train, y_train)
    results.append(evaluate("Naive Bayes", nb, X_test, y_test))

    logreg = LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")
    logreg.fit(X_train, y_train)
    results.append(evaluate("Logistic Regression", logreg, X_test, y_test))

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    # --- comparison table ---
    print("\n=== Model comparison ===")
    header = f"{'Model':<22}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}{'F1':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<22}{r['accuracy']:<10}{r['precision']:<11}{r['recall']:<9}{r['f1']:<8}")

    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest model by F1-score: {best['name']}")

    model_map = {"Naive Bayes": nb, "Logistic Regression": logreg, "Random Forest": rf}
    best_model = model_map[best["name"]]

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
    joblib.dump(model_map, MODELS_DIR / "all_models.pkl")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best["name"]}, f, indent=2)

    print(f"\nSaved best model ('{best['name']}') and vectoriser to {MODELS_DIR}")


if __name__ == "__main__":
    main()
