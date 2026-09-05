# ml/classify.py
# Loads the trained classifier + TF-IDF vectorizer, and provides a function
# to classify new sentences as Relevant / Irrelevant with a confidence score.
#
# IMPORTANT: uses the exact same feature extraction as train_classifier.py
# to ensure consistency between training and prediction.

import os
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from features import extract_features

# --- Load trained artifacts once, when this module is first imported ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

print("[INFO] Loading trained classifier and vectorizer...")
model = joblib.load(os.path.join(MODELS_DIR, "classifier.joblib"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
print("[INFO] Classifier loaded and ready.")


def classify_sentence(sentence):
    """
    Classifies a single sentence.
    Returns (label, confidence) where label is "Relevant" or "Irrelevant"
    and confidence is a float between 0 and 1.
    """
    # Step 1: hand-crafted features (same as training)
    feats = extract_features(sentence)
    feat_df = pd.DataFrame([feats])
    # Ensure columns are in the exact same order as during training
    feat_df = feat_df.reindex(columns=feature_columns, fill_value=0)
    handcrafted_matrix = csr_matrix(feat_df.values.astype(float))

    # Step 2: TF-IDF features (using the SAME fitted vectorizer from training)
    tfidf_matrix = vectorizer.transform([sentence])

    # Step 3: combine, exactly like training
    X = hstack([handcrafted_matrix, tfidf_matrix])

    # Step 4: predict
    label = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))

    return label, confidence


def classify_sentences(sentences):
    """
    Classifies a list of sentence strings.
    Returns a list of dicts: [{"text": ..., "label": ..., "confidence": ...}, ...]
    """
    results = []
    for sentence in sentences:
        label, confidence = classify_sentence(sentence)
        results.append({
            "text": sentence,
            "label": label,
            "confidence": round(confidence, 3)
        })
    return results


if __name__ == "__main__":
    # Quick manual test
    test_sentences = [
        "We need to submit the report by Friday.",
        "Did you watch the match yesterday?",
        "The model is not performing well, we need to retrain it.",
        "I am very happy because I bought a new phone."
    ]
    for s in test_sentences:
        label, conf = classify_sentence(s)
        print(f"[{label:10s}] ({conf:.2f}) {s}") 