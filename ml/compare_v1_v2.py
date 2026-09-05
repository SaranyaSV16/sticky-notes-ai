# ml/compare_v1_v2.py
# Loads BOTH the old (v1) and new (v2) models and runs them side-by-side
# on the exact sentences that were misclassified during live testing.
# This proves whether the dataset expansion actually fixed the real bug,
# not just improved abstract metrics.

import os
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from features import extract_features

MODELS_DIR = os.path.join("..", "models")


def load_model(suffix):
    model = joblib.load(os.path.join(MODELS_DIR, f"classifier{suffix}.joblib"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_vectorizer{suffix}.joblib"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, f"feature_columns{suffix}.joblib"))
    return model, vectorizer, feature_columns


def classify_with(sentence, model, vectorizer, feature_columns):
    feats = extract_features(sentence)
    feat_df = pd.DataFrame([feats]).reindex(columns=feature_columns, fill_value=0)
    handcrafted_matrix = csr_matrix(feat_df.values.astype(float))
    tfidf_matrix = vectorizer.transform([sentence])
    X = hstack([handcrafted_matrix, tfidf_matrix])
    label = model.predict(X)[0]
    confidence = float(max(model.predict_proba(X)[0]))
    return label, confidence


if __name__ == "__main__":
    v1_model, v1_vec, v1_cols = load_model("_v1_backup")
    v2_model, v2_vec, v2_cols = load_model("_v2")

    # These are the EXACT sentences that were misclassified during live testing
    test_cases = [
        ("Honestly, I am so tired today and I barely slept last night, but we need to submit the final project by Friday, and Rahul will complete the results section.", None),
        ("Honestly, I am so tired today and I barely slept last night,", "Irrelevant"),
        ("the full of the cafeteria was terrible today.", "Irrelevant"),
        ("I really hate doing documentation and honestly I don't feel like working today", "Irrelevant"),
        ("we should schedule the project presentation from Monday at 10 am.", "Relevant"),
        ("We need to submit the final report by Friday and Rahul will complete the results section.", "Relevant"),
    ]

    print(f"{'EXPECTED':12s} {'V1 RESULT':22s} {'V2 RESULT':22s}  SENTENCE")
    print("-" * 110)

    for sentence, expected in test_cases:
        v1_label, v1_conf = classify_with(sentence, v1_model, v1_vec, v1_cols)
        v2_label, v2_conf = classify_with(sentence, v2_model, v2_vec, v2_cols)

        v1_str = f"{v1_label} ({v1_conf:.0%})"
        v2_str = f"{v2_label} ({v2_conf:.0%})"
        expected_str = expected if expected else "-"

        print(f"{expected_str:12s} {v1_str:22s} {v2_str:22s}  {sentence[:60]}...")
        