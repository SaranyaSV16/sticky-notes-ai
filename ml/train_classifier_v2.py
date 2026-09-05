# ml/train_classifier_v2.py
# Trains an IMPROVED classifier on the expanded dataset_v2.csv.
# Saves as separate _v2 files so the original v1 model stays untouched
# and usable until we've confirmed v2 is actually better.

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from scipy.sparse import hstack, csr_matrix

from features import extract_features

DATASET_PATH = "dataset_v2.csv"
MODELS_DIR = os.path.join("..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_feature_matrix(sentences):
    all_features = [extract_features(s) for s in sentences]
    return pd.DataFrame(all_features)


def main():
    print("[INFO] Loading dataset_v2.csv...")
    df = pd.read_csv(DATASET_PATH)
    print(f"[INFO] Loaded {len(df)} labeled sentences.")
    print(df["label"].value_counts())

    sentences = df["sentence"].tolist()
    labels = df["label"].tolist()

    print("\n[INFO] Extracting hand-crafted features...")
    feature_df = build_feature_matrix(sentences)
    handcrafted_matrix = csr_matrix(feature_df.values.astype(float))

    print("[INFO] Building TF-IDF word features...")
    vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(sentences)

    X = hstack([handcrafted_matrix, tfidf_matrix])
    y = labels

    # Same random_state=42 as v1, for a fair apples-to-apples split methodology
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Training on {X_train.shape[0]} sentences, testing on {X_test.shape[0]} sentences.")

    print("[INFO] Training Logistic Regression classifier (v2)...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "=" * 50)
    print("V2 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label='Relevant'):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred, pos_label='Relevant'):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, pos_label='Relevant'):.3f}")

    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    labels_order = ["Relevant", "Irrelevant"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    print(f"                 Predicted Relevant   Predicted Irrelevant")
    print(f"Actual Relevant        {cm[0][0]:>5}                {cm[0][1]:>5}")
    print(f"Actual Irrelevant      {cm[1][0]:>5}                {cm[1][1]:>5}")

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred))

    print("\n" + "=" * 50)
    print("TOP FEATURES DRIVING 'RELEVANT' PREDICTIONS (v2)")
    print("=" * 50)
    feature_names = list(feature_df.columns) + list(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    top_positive_idx = np.argsort(coefficients)[-15:][::-1]
    top_negative_idx = np.argsort(coefficients)[:15]

    print("\nPush toward RELEVANT:")
    for idx in top_positive_idx:
        print(f"  {feature_names[idx]:30s}  weight={coefficients[idx]:.3f}")

    print("\nPush toward IRRELEVANT:")
    for idx in top_negative_idx:
        print(f"  {feature_names[idx]:30s}  weight={coefficients[idx]:.3f}")

    # Save as v2-suffixed files - does NOT touch the live v1 model
    joblib.dump(model, os.path.join(MODELS_DIR, "classifier_v2.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer_v2.joblib"))
    joblib.dump(list(feature_df.columns), os.path.join(MODELS_DIR, "feature_columns_v2.joblib"))

    print(f"\n[INFO] v2 model saved to {MODELS_DIR}/classifier_v2.joblib")
    print("[INFO] The LIVE app still uses the v1 model - nothing has changed yet.")


if __name__ == "__main__":
    main()
    