// recorder.js
// Handles microphone access, recording, timer display,
// uploading audio to Flask, and displaying sticky notes + transcript.

let mediaRecorder;
let audioChunks = [];
let timerInterval;
let secondsElapsed = 0;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const timerEl = document.getElementById("timer");
const statusText = document.getElementById("statusText");
const meetingNameInput = document.getElementById("meetingName");
const resultArea = document.getElementById("resultArea");
const resultText = document.getElementById("resultText");
const stickyNotesArea = document.getElementById("stickyNotesArea");
const stickyNotesGrid = document.getElementById("stickyNotesGrid");
const transcriptArea = document.getElementById("transcriptArea");
const transcriptList = document.getElementById("transcriptList");

function formatTime(totalSeconds) {
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function startTimer() {
  secondsElapsed = 0;
  timerEl.textContent = "00:00";
  timerInterval = setInterval(() => {
    secondsElapsed++;
    timerEl.textContent = formatTime(secondsElapsed);
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
}

function renderStickyNotes(notes) {
  stickyNotesGrid.innerHTML = "";

  if (!notes || notes.length === 0) {
    stickyNotesGrid.innerHTML = "<p>No sticky notes generated from this recording.</p>";
    return;
  }

  notes.forEach((note) => {
    const card = document.createElement("div");
    card.className = "sticky-card";
    card.dataset.noteId = note.id;

    let detailsHtml = "";
    if (note.responsible) {
      detailsHtml += `<div class="sticky-detail"><strong>Responsible:</strong> ${note.responsible}</div>`;
    }
    if (note.date) {
      detailsHtml += `<div class="sticky-detail"><strong>Date:</strong> ${note.date}</div>`;
    }
    if (note.time) {
      detailsHtml += `<div class="sticky-detail"><strong>Time:</strong> ${note.time}</div>`;
    }

    const confidencePct = Math.round(note.confidence * 100);
    const isUncertain = confidencePct < 70;

    card.innerHTML = `
      <div class="sticky-category">${note.category}</div>
      <div class="sticky-title">📌 ${note.title}</div>
      ${detailsHtml}
      ${isUncertain ? `<div class="sticky-uncertain">⚠️ Possible important content (${confidencePct}%)</div>` : ""}
      <div class="sticky-actions">
        <button class="keep-btn">Keep</button>
        <button class="remove-btn">Remove</button>
      </div>
    `;
    stickyNotesGrid.appendChild(card);
  });

  document.querySelectorAll(".remove-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const card = e.target.closest(".sticky-card");
      const noteId = card.dataset.noteId;

      try {
        await fetch(`/note_status/${noteId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ status: "removed" })
        });
        card.remove();
      } catch (err) {
        console.error("Failed to remove note:", err);
        alert("Could not remove note. Please try again.");
      }
    });
  });

  document.querySelectorAll(".keep-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const card = e.target.closest(".sticky-card");
      const noteId = card.dataset.noteId;

      try {
        await fetch(`/note_status/${noteId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ status: "approved" })
        });
        card.querySelector(".sticky-uncertain")?.remove();
        card.classList.add("kept");
        card.querySelector(".sticky-actions").innerHTML = "<span class='kept-label'>✅ Kept</span>";
      } catch (err) {
        console.error("Failed to update note:", err);
        alert("Could not update note. Please try again.");
      }
    });
  });
}

function renderTranscript(sentences) {
  transcriptList.innerHTML = "";

  if (!sentences || sentences.length === 0) {
    transcriptList.innerHTML = "<p>No speech detected.</p>";
    return;
  }

  sentences.forEach((s) => {
    const row = document.createElement("div");
    const isRelevant = s.label === "Relevant";
    row.className = "transcript-row " + (isRelevant ? "relevant" : "irrelevant");

    const confidencePct = Math.round(s.confidence * 100);

    row.innerHTML = `
      <div class="transcript-meta">
        <span class="transcript-time">[${s.start}s - ${s.end}s]</span>
        <span class="transcript-label ${isRelevant ? 'label-relevant' : 'label-irrelevant'}">
          ${s.label} (${confidencePct}%)
        </span>
      </div>
      <span class="transcript-text">${s.text}</span>
    `;
    transcriptList.appendChild(row);
  });
}

startBtn.addEventListener("click", async () => {
  const meetingName = meetingNameInput.value.trim();
  if (!meetingName) {
    alert("Please enter a meeting name first.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      stopTimer();
      statusText.textContent = "Processing... (this can take 10-30 seconds)";

      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });

      const formData = new FormData();
      formData.append("audio", audioBlob, "recording.webm");
      formData.append("meeting_name", meetingName);

      try {
        const response = await fetch("/upload_audio", {
          method: "POST",
          body: formData
        });
        const data = await response.json();

        if (data.error) {
          statusText.textContent = "Error";
          alert("Error: " + data.error);
          return;
        }

        statusText.textContent = "Idle";

        resultArea.style.display = "block";
        resultText.textContent = `Saved as: ${data.filename}`;

        stickyNotesArea.style.display = "block";
        renderStickyNotes(data.sticky_notes);

        transcriptArea.style.display = "block";
        renderTranscript(data.sentences);

      } catch (err) {
        statusText.textContent = "Upload failed";
        console.error(err);
      }

      stream.getTracks().forEach((track) => track.stop());
    };

    mediaRecorder.start();
    startTimer();

    statusText.textContent = "Recording...";
    startBtn.disabled = true;
    stopBtn.disabled = false;

  } catch (err) {
    alert("Could not access microphone. Please allow microphone permissions.");
    console.error(err);
  }
});

stopBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
});