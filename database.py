# database.py
# Sets up the SQLite database and provides helper functions for
# saving/retrieving meetings and sticky notes.

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join("data", "meetings.db")


def get_connection():
    """
    Returns a connection to the SQLite database.
    row_factory lets us access columns by name (like a dictionary)
    instead of by numeric index - much easier to work with.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates the meetings and sticky_notes tables if they don't
    already exist. Safe to call every time the app starts.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            audio_path TEXT,
            transcript TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sticky_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id INTEGER NOT NULL,
            title TEXT,
            category TEXT,
            responsible TEXT,
            date TEXT,
            time TEXT,
            original_sentence TEXT,
            confidence REAL,
            status TEXT DEFAULT 'pending',
            created_at TEXT NOT NULL,
            FOREIGN KEY (meeting_id) REFERENCES meetings (id)
        )
    """)

    conn.commit()
    conn.close()
    print("[INFO] Database initialized (meetings.db)")


def save_meeting(meeting_name, audio_path, transcript_json):
    """
    Saves a new meeting record. transcript_json should be a JSON string
    of the full sentence-by-sentence transcript.
    Returns the new meeting's ID.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO meetings (meeting_name, created_at, audio_path, transcript)
        VALUES (?, ?, ?, ?)
    """, (meeting_name, datetime.now().isoformat(), audio_path, transcript_json))

    meeting_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return meeting_id


def save_sticky_notes(meeting_id, notes):
    """
    Saves a list of sticky note dicts for the given meeting.
    Returns the list of notes with their new database IDs attached.
    """
    conn = get_connection()
    cursor = conn.cursor()

    saved_notes = []
    for note in notes:
        cursor.execute("""
            INSERT INTO sticky_notes
                (meeting_id, title, category, responsible, date, time,
                 original_sentence, confidence, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            meeting_id,
            note.get("title"),
            note.get("category"),
            note.get("responsible"),
            note.get("date"),
            note.get("time"),
            note.get("original_sentence"),
            note.get("confidence"),
            datetime.now().isoformat()
        ))
        note_id = cursor.lastrowid
        note_with_id = dict(note)
        note_with_id["id"] = note_id
        note_with_id["status"] = "pending"
        saved_notes.append(note_with_id)

    conn.commit()
    conn.close()
    return saved_notes


def update_note_status(note_id, new_status):
    """
    Updates a sticky note's status to 'approved' or 'removed'.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE sticky_notes SET status = ? WHERE id = ?
    """, (new_status, note_id))

    conn.commit()
    conn.close()


def get_all_meetings():
    """
    Returns a list of all meetings, most recent first.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, meeting_name, created_at, audio_path
        FROM meetings
        ORDER BY created_at DESC
    """)

    meetings = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return meetings


def get_meeting_by_id(meeting_id):
    """
    Returns full details for one meeting, including its sticky notes
    (excluding any notes with status 'removed').
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
    meeting_row = cursor.fetchone()

    if meeting_row is None:
        conn.close()
        return None

    meeting = dict(meeting_row)

    cursor.execute("""
        SELECT * FROM sticky_notes
        WHERE meeting_id = ? AND status != 'removed'
        ORDER BY created_at ASC
    """, (meeting_id,))

    meeting["sticky_notes"] = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return meeting