# ml/features.py
# Extracts features from a sentence to help the classifier decide
# Relevant vs Irrelevant. Combines hand-crafted linguistic features
# with spaCy's NLP analysis.

import spacy

# Load spaCy's English model once (reused for every sentence)
nlp = spacy.load("en_core_web_sm")

# --- Keyword lists based on the patterns faculty described ---

ACTION_PHRASES = [
    "i will", "i'll", "we will", "we'll", "we need to", "we have to",
    "let's", "lets", "you should", "we should", "we decided",
    "we agreed", "please", "can you", "could you", "make sure",
    "need to", "have to", "going to", "plan to"
]

MODAL_VERBS = [
    "will", "shall", "should", "must", "need", "have to", "going to"
]

IRRELEVANCE_PHRASES = [
    "i don't like", "i dont like", "i hate", "i'm tired", "im tired",
    "haha", "lol", "funny", "movie", "match", "game", "weekend",
    "how are you", "did you watch", "did you see", "by the way",
    "anyway", "yeah", "yep", "cool", "thanks for", "appreciate you"
]

DEADLINE_WORDS = [
    "deadline", "by friday", "by monday", "by tuesday", "by wednesday",
    "by thursday", "by saturday", "by sunday", "today", "tomorrow",
    "this week", "next week", "end of day", "end of week",
    "this quarter", "next month"
]


def extract_features(sentence):
    """
    Takes a single sentence (string) and returns a dictionary of
    numeric/boolean features describing it.
    """
    text_lower = sentence.lower()
    doc = nlp(sentence)

    features = {}

    # --- 1. Lexical / keyword features ---
    features["has_action_phrase"] = int(any(p in text_lower for p in ACTION_PHRASES))
    features["has_modal_verb"] = int(any(m in text_lower for m in MODAL_VERBS))
    features["has_irrelevance_phrase"] = int(any(p in text_lower for p in IRRELEVANCE_PHRASES))
    features["has_deadline_word"] = int(any(d in text_lower for d in DEADLINE_WORDS))

    # --- 2. Structural features ---
    features["sentence_length"] = len(doc)
    features["is_question"] = int(sentence.strip().endswith("?"))
    features["starts_with_i"] = int(text_lower.strip().startswith("i "))

    # --- 3. spaCy-based linguistic features ---
    pos_tags = [token.pos_ for token in doc]
    features["num_verbs"] = pos_tags.count("VERB") + pos_tags.count("AUX")
    features["num_nouns"] = pos_tags.count("NOUN") + pos_tags.count("PROPN")
    features["has_imperative"] = int(len(doc) > 0 and doc[0].pos_ == "VERB")

    # --- 4. Named entity features (dates, people, orgs) ---
    ent_labels = [ent.label_ for ent in doc.ents]
    features["has_date_entity"] = int("DATE" in ent_labels)
    features["has_time_entity"] = int("TIME" in ent_labels)
    features["has_person_entity"] = int("PERSON" in ent_labels)
    features["has_org_entity"] = int("ORG" in ent_labels)
    features["num_entities"] = len(ent_labels)

    return features


if __name__ == "__main__":
    # Quick manual test
    test_sentences = [
        "We need to submit the report by Friday.",
        "Did you watch the match yesterday?",
        "The model is not performing well, we need to retrain it.",
        "I am very happy because I bought a new phone."
    ]

    for s in test_sentences:
        print(f"\nSentence: {s}")
        feats = extract_features(s)
        for k, v in feats.items():
            print(f"  {k}: {v}")