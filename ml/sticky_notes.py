# ml/sticky_notes.py
# Converts a Relevant sentence into a structured sticky note using
# spaCy named entity recognition + rule-based pattern matching.
# No external LLM API used - fully local and explainable.

import re
import spacy

nlp = spacy.load("en_core_web_sm")

# --- Patterns that indicate WHO is responsible ---
# Captures things like "Rahul will...", "Abhishek can you...", "I'll..."
RESPONSIBILITY_PATTERNS = [
    re.compile(r"\b([A-Z][a-z]+)\s+will\b"),           # "Rahul will..."
    re.compile(r"\b([A-Z][a-z]+),?\s+can you\b", re.I),  # "Abhishek, can you..."
    re.compile(r"\bI['’]ll\b"),                          # "I'll..." -> speaker themself
    re.compile(r"\bI will\b"),
]

# --- Category keyword patterns ---
# NOTE: Order matters - more specific/important signals are checked FIRST,
# so a sentence like "the model is not performing well, we need to fix it"
# is correctly tagged Issue/Problem rather than the more generic Task.
CATEGORY_RULES = [
    ("Issue/Problem",  ["not performing", "not working", "broken", "bug", "crashed",
                         "issue", "problem", "behind schedule", "missed the"]),
    ("Deadline",      ["deadline", "by friday", "by monday", "by tuesday", "by wednesday",
                        "by thursday", "by saturday", "by sunday", "due", "end of week",
                        "end of day", "this week", "next week"]),
    ("Decision",       ["we decided", "we agreed", "decided to", "agreed to"]),
    ("Meeting/Schedule",["meeting is", "scheduled for", "presentation is", "call with"]),
    ("Task",           ["need to", "have to", "let's", "lets", "please", "can you",
                         "could you", "make sure", "should", " will "]),
]


def extract_responsible_person(sentence):
    """
    Tries to find who is responsible for this item.
    Returns a name string, "Speaker" (for I/I'll), or None if unclear.
    """
    for pattern in RESPONSIBILITY_PATTERNS:
        match = pattern.search(sentence)
        if match:
            groups = match.groups()
            if groups and groups[0]:
                return groups[0]  # a proper name was captured
            return "Speaker"  # matched "I'll" / "I will" with no captured name

    # Fallback: use spaCy to find any PERSON entity in the sentence
    doc = nlp(sentence)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text

    return None


def extract_date_time(sentence):
    """
    Uses spaCy NER to pull out DATE and TIME entities.
    Returns (date_str, time_str) - either can be None.
    """
    doc = nlp(sentence)
    date_str = None
    time_str = None

    for ent in doc.ents:
        if ent.label_ == "DATE" and date_str is None:
            date_str = ent.text
        elif ent.label_ == "TIME" and time_str is None:
            time_str = ent.text

    return date_str, time_str


def classify_category(sentence):
    """
    Assigns a category label to the sentence based on keyword rules.
    Checked in order - first match wins.
    """
    text_lower = sentence.lower()
    for category, keywords in CATEGORY_RULES:
        if any(kw in text_lower for kw in keywords):
            return category
    return "General"


def generate_title(sentence, category):
    """
    Builds a short title for the sticky note.
    Strategy: strip filler words/phrases from the start, take the core
    subject, and truncate to a reasonable length.
    """
    text = sentence.strip().rstrip(".")

    # Remove common leading filler phrases
    fillers = [
        "so ", "okay ", "ok ", "yeah ", "well ", "let's ", "lets ",
        "we need to ", "we have to ", "please ", "i'll ", "i will ",
        "can you ", "could you ", "we decided to ", "we agreed to "
    ]
    text_lower = text.lower()
    for filler in fillers:
        if text_lower.startswith(filler):
            text = text[len(filler):]
            text_lower = text.lower()

    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    # Truncate to keep it "sticky-note" concise
    max_len = 60
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."

    return text if text else category


def generate_sticky_note(sentence, confidence=None):
    """
    Main function: takes a Relevant sentence and produces a structured
    sticky note dictionary.
    """
    category = classify_category(sentence)
    responsible = extract_responsible_person(sentence)
    date_str, time_str = extract_date_time(sentence)

    # Fallback: if no keyword rule matched (category is still "General")
    # but we found a responsible person, this is almost certainly a task
    # assignment (e.g. "Rahul will prepare the conclusion section").
    if category == "General" and responsible is not None:
        category = "Task"

    title = generate_title(sentence, category)

    note = {
        "title": title,
        "category": category,
        "responsible": responsible,
        "date": date_str,
        "time": time_str,
        "original_sentence": sentence,
        "confidence": confidence
    }

    return note

if __name__ == "__main__":
    # Quick manual test
    test_sentences = [
        "Rahul will prepare the conclusion section.",
        "We need to submit the report by Friday.",
        "The presentation is on Monday at 10 AM.",
        "Abhishek, can you send the report to Sonali by end of week?",
        "The model is not performing well, we need to retrain it.",
        "We decided to use Python for the backend.",
    ]

    for s in test_sentences:
        note = generate_sticky_note(s)
        print(f"\n📌 {note['title']}")
        print(f"   Category: {note['category']}")
        if note['responsible']:
            print(f"   Responsible: {note['responsible']}")
        if note['date']:
            print(f"   Date: {note['date']}")
        if note['time']:
            print(f"   Time: {note['time']}")