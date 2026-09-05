import os
import re
import sys
import json
from database import init_db, save_meeting, save_sticky_notes, update_note_status, get_all_meetings, get_meeting_by_id
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from faster_whisper import WhisperModel
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

sys.path.append(os.path.join(os.path.dirname(__file__), "ml"))
from classify import classify_sentence
from sticky_notes import generate_sticky_note

app = Flask(__name__)

init_db()

AUDIO_FOLDER = os.path.join("data", "audio")
os.makedirs(AUDIO_FOLDER, exist_ok=True)

print("[INFO] Loading Whisper model (base)... this happens once at startup.")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("[INFO] Whisper model loaded and ready.")

# -----------------------------------------------------------------------
# Custom sentence tokenizer that knows about common abbreviations
# (a.m., p.m., Mr., Dr., etc.) so it doesn't wrongly glue sentences
# together after them.
# -----------------------------------------------------------------------
punkt_params = PunktParameters()
punkt_params.abbrev_types = set([
    "a.m", "p.m", "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "inc", "ltd", "vs", "etc", "e.g", "i.e"
])
sentence_tokenizer = PunktSentenceTokenizer(punkt_params)

# -----------------------------------------------------------------------
# Splits a sentence further on contrastive conjunctions ("but", "however",
# "although") when BOTH sides look like a substantial independent clause.
# This catches compound sentences that mix relevant and irrelevant content,
# e.g. "I hate documentation, but we should schedule the presentation."
# -----------------------------------------------------------------------
# Conjunctions that require BOTH sides to be substantial before splitting
# (avoids over-splitting short phrases like "small but important")
CLAUSE_SPLIT_PATTERN = re.compile(r"\s+\b(but|however|although|though)\b\s+", re.IGNORECASE)

# Topic-switch phrases - "and" before the phrase is optional and consumed
# as PART of the same match, so it's removed cleanly instead of being left
# behind as an orphan fragment (e.g. "And by the way" -> both removed together).
TOPIC_SWITCH_PATTERN = re.compile(
    r"\s*(?:and\s+|oh\s+and\s+|yeah\s+and\s+)?"
    r"(?:by the way|anyway|speaking of which|on a related note|"
    r"i forgot to (?:tell you|mention)|actually)\b[\s,]*",
    re.IGNORECASE
)

MIN_CLAUSE_WORDS = 3


def split_into_clauses(sentence):
    # Step 1: split unconditionally on topic-switch phrases.
    # Non-capturing groups mean re.split discards the matched separator
    # text entirely - no leftover fragments.
    topic_parts = [p.strip() for p in TOPIC_SWITCH_PATTERN.split(sentence) if p.strip()]

    # Step 2: within each topic-separated piece, also split on conjunctions
    final_clauses = []
    for part in topic_parts:
        conj_parts = CLAUSE_SPLIT_PATTERN.split(part)

        if len(conj_parts) == 1:
            final_clauses.append(part)
            continue

        current = conj_parts[0]
        i = 1
        while i < len(conj_parts):
            conjunction = conj_parts[i]
            next_text = conj_parts[i + 1] if (i + 1) < len(conj_parts) else ""

            current_words = len(current.split())
            next_words = len(next_text.split())

            if current_words >= MIN_CLAUSE_WORDS and next_words >= MIN_CLAUSE_WORDS:
                final_clauses.append(current.strip())
                current = next_text
            else:
                current = current + " " + conjunction + " " + next_text

            i += 2

        final_clauses.append(current.strip())

    return [c.strip() for c in final_clauses if c.strip()]

def transcribe_and_split(audio_path):
    """
    Transcribes the audio file, splits the FULL transcript into proper
    sentences (abbreviation-aware), then further splits compound sentences
    on contrastive conjunctions so mixed-relevance content gets separated.

    Returns a list of dicts: { "text": ..., "start": ..., "end": ... }
    """
    segments, info = whisper_model.transcribe(audio_path, beam_size=5)

    full_text = ""
    char_time_map = []

    for segment in segments:
        segment_text = segment.text.strip()
        if not segment_text:
            continue

        char_time_map.append({
            "char_start": len(full_text),
            "char_end": len(full_text) + len(segment_text),
            "time_start": segment.start,
            "time_end": segment.end
        })

        full_text += segment_text + " "

    full_text = full_text.strip()

    if not full_text:
        return []

    raw_sentences = sentence_tokenizer.tokenize(full_text)

    # Further split each sentence into clauses on "but"/"however"/etc.
    final_texts = []
    for raw_sentence in raw_sentences:
        clauses = split_into_clauses(raw_sentence)
        final_texts.extend(clauses)

    sentences = []
    search_pos = 0

    for text_piece in final_texts:
        text_piece = text_piece.strip()
        if not text_piece:
            continue

        idx = full_text.find(text_piece, search_pos)
        if idx == -1:
            idx = search_pos

        piece_char_start = idx
        piece_char_end = idx + len(text_piece)
        search_pos = piece_char_end

        start_time = None
        end_time = None
        for entry in char_time_map:
            overlaps = (piece_char_start < entry["char_end"]) and (piece_char_end > entry["char_start"])
            if overlaps:
                if start_time is None:
                    start_time = entry["time_start"]
                end_time = entry["time_end"]

        sentences.append({
            "text": text_piece,
            "start": round(start_time, 2) if start_time is not None else None,
            "end": round(end_time, 2) if end_time is not None else None
        })

    return sentences


def classify_transcript(sentences):
    for sentence_obj in sentences:
        label, confidence = classify_sentence(sentence_obj["text"])
        sentence_obj["label"] = label
        sentence_obj["confidence"] = round(confidence, 3)
    return sentences


def generate_sticky_notes_for_transcript(sentences):
    """
    Takes the classified sentence list and generates a sticky note
    for every sentence labeled Relevant. Irrelevant sentences are skipped.
    """
    notes = []
    for sentence_obj in sentences:
        if sentence_obj["label"] == "Relevant":
            note = generate_sticky_note(
                sentence_obj["text"],
                confidence=sentence_obj["confidence"]
            )
            note["start"] = sentence_obj["start"]
            note["end"] = sentence_obj["end"]
            notes.append(note)
    return notes


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]
    meeting_name = request.form.get("meeting_name", "untitled_meeting")

    safe_name = "".join(c for c in meeting_name if c.isalnum() or c in (" ", "_", "-")).strip()
    safe_name = safe_name.replace(" ", "_") or "untitled_meeting"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.webm"
    filepath = os.path.join(AUDIO_FOLDER, filename)

    audio_file.save(filepath)
    print(f"[INFO] Saved audio file: {filepath}")

    print("[INFO] Transcribing audio, please wait...")
    try:
        sentences = transcribe_and_split(filepath)
        print(f"[INFO] Transcription complete. {len(sentences)} sentences found.")
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    print("[INFO] Classifying sentences...")
    try:
        sentences = classify_transcript(sentences)
        relevant_count = sum(1 for s in sentences if s["label"] == "Relevant")
        print(f"[INFO] Classification complete. {relevant_count}/{len(sentences)} marked Relevant.")
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500

        print("[INFO] Generating sticky notes...")
    try:
        sticky_notes = generate_sticky_notes_for_transcript(sentences)
        print(f"[INFO] Generated {len(sticky_notes)} sticky notes.")
    except Exception as e:
        print(f"[ERROR] Sticky note generation failed: {e}")
        return jsonify({"error": f"Sticky note generation failed: {str(e)}"}), 500

    # --- Save everything to the database ---
    print("[INFO] Saving meeting to database...")
    try:
        transcript_json = json.dumps(sentences)
        meeting_id = save_meeting(meeting_name, filepath, transcript_json)
        sticky_notes = save_sticky_notes(meeting_id, sticky_notes)
        print(f"[INFO] Meeting saved with ID {meeting_id}, {len(sticky_notes)} sticky notes persisted.")
    except Exception as e:
        print(f"[ERROR] Database save failed: {e}")
        return jsonify({"error": f"Database save failed: {str(e)}"}), 500

    return jsonify({
        "message": "Audio processed successfully",
        "meeting_id": meeting_id,
        "filename": filename,
        "filepath": filepath,
        "sentences": sentences,
        "sticky_notes": sticky_notes
    })

@app.route("/note_status/<int:note_id>", methods=["POST"])
def note_status(note_id):
    """
    Updates a sticky note's status - called when the user clicks
    Keep (-> approved) or Remove (-> removed) in the UI.
    """
    data = request.get_json()
    new_status = data.get("status")

    if new_status not in ("approved", "removed"):
        return jsonify({"error": "Invalid status"}), 400

    update_note_status(note_id, new_status)
    return jsonify({"message": "Status updated", "note_id": note_id, "status": new_status})


@app.route("/meetings", methods=["GET"])
def list_meetings():
    """Returns all past meetings for the history view."""
    meetings = get_all_meetings()
    return jsonify({"meetings": meetings})


@app.route("/meetings/<int:meeting_id>", methods=["GET"])
def meeting_detail(meeting_id):
    """Returns full details (transcript + sticky notes) for one meeting."""
    meeting = get_meeting_by_id(meeting_id)
    if meeting is None:
        return jsonify({"error": "Meeting not found"}), 404
    return jsonify(meeting)

if __name__ == "__main__":
    app.run(debug=True, port=5000)