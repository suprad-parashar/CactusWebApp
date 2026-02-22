(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const micBtn = $("#micBtn");
  const micRing = $("#micRing");
  const micHint = $("#micHint");
  const statusArea = $("#statusArea");
  const transcriptArea = $("#transcriptArea");
  const resultsArea = $("#resultsArea");
  const textInput = $("#textInput");
  const sendBtn = $("#sendBtn");

  let ws = null;
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let audioCtx = null;
  let analyser = null;
  let silenceTimer = null;
  const SILENCE_THRESHOLD = 0.01;
  const SILENCE_DURATION = 1500;
  const MIN_RECORD_MS = 600;

  // ── WebSocket ───────────────────────────────────────────

  function connectWS() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onopen = () => console.log("[ws] connected");
    ws.onclose = () => {
      console.log("[ws] disconnected, reconnecting...");
      setTimeout(connectWS, 2000);
    };
    ws.onerror = (e) => console.error("[ws] error", e);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    };
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case "status":
        showStatus(msg.message);
        break;
      case "transcript":
        showTranscript(msg.text, msg.latency_ms);
        break;
      case "result":
        clearStatus();
        renderResults(msg);
        break;
      case "error":
        clearStatus();
        showError(msg.message);
        break;
    }
  }

  // ── Status ──────────────────────────────────────────────

  function showStatus(message) {
    statusArea.innerHTML = `
      <div class="status-pill">
        <span class="status-dot"></span>
        <span>${message}</span>
      </div>`;
  }

  function clearStatus() {
    statusArea.innerHTML = "";
  }

  function showError(message) {
    resultsArea.innerHTML = `<div class="no-result">${message}</div>`;
  }

  // ── Transcript ──────────────────────────────────────────

  function showTranscript(text, latencyMs) {
    transcriptArea.innerHTML = `
      <p class="transcript-text">
        <span class="quote">&ldquo;</span>${escapeHtml(text)}<span class="quote">&rdquo;</span>
      </p>`;
  }

  // ── Results ─────────────────────────────────────────────

  const CARD_CONFIG = {
    get_weather:      { icon: "🌤", label: "Weather",  cls: "weather" },
    play_music:       { icon: "🎵", label: "Music",    cls: "music" },
    set_alarm:        { icon: "⏰", label: "Alarm",    cls: "alarm" },
    set_timer:        { icon: "⏱",  label: "Timer",    cls: "timer" },
    create_reminder:  { icon: "📌", label: "Reminder", cls: "reminder" },
    send_message:     { icon: "💬", label: "Message",  cls: "message" },
    search_contacts:  { icon: "👤", label: "Contacts", cls: "contacts" },
  };

  function renderResults(msg) {
    resultsArea.innerHTML = "";

    if (!msg.executions || msg.executions.length === 0) {
      resultsArea.innerHTML = `<div class="no-result">No actions matched your request. Try again?</div>`;
      renderLatencyBar(msg.latency);
      return;
    }

    msg.executions.forEach((exec, i) => {
      const card = document.createElement("div");
      card.className = "result-card";
      card.style.animationDelay = `${i * 80}ms`;

      const config = CARD_CONFIG[exec.function_name] || { icon: "⚡", label: exec.function_name, cls: "unknown" };

      card.innerHTML = `
        <div class="card-header">
          <div class="card-icon ${config.cls}">${config.icon}</div>
          <span class="card-title">${config.label}</span>
        </div>
        <div class="card-body">
          ${renderCardBody(exec)}
        </div>
        <div class="card-footer">
          <span class="card-badge ${exec.success ? "success" : "error"}">${exec.success ? "Done" : "Failed"}</span>
          <span class="card-badge">${exec.execution_time_ms}ms</span>
        </div>`;

      resultsArea.appendChild(card);
    });

    renderLatencyBar(msg.latency);
  }

  function renderCardBody(exec) {
    const data = exec.data || {};

    switch (exec.function_name) {
      case "get_weather":
        return `
          <p class="card-summary">${escapeHtml(exec.summary)}</p>
          <div class="weather-grid">
            <div class="weather-stat">
              <span class="weather-stat-label">Feels Like</span>
              <span class="weather-stat-value">${data.feels_like_f || "?"}°F</span>
            </div>
            <div class="weather-stat">
              <span class="weather-stat-label">Humidity</span>
              <span class="weather-stat-value">${data.humidity || "?"}</span>
            </div>
            <div class="weather-stat">
              <span class="weather-stat-label">Wind</span>
              <span class="weather-stat-value">${data.wind || "?"}</span>
            </div>
            <div class="weather-stat">
              <span class="weather-stat-label">Condition</span>
              <span class="weather-stat-value">${data.condition || "?"}</span>
            </div>
          </div>`;

      case "play_music":
        return `
          <p class="card-summary">${escapeHtml(exec.summary)}</p>
          ${data.url ? `<a href="${data.url}" target="_blank" class="music-link">&#9654; Open YouTube</a>` : ""}`;

      case "search_contacts":
        if (data.results && data.results.length > 0) {
          const contacts = data.results.map((c) => `
            <div class="contact-item">
              <div class="contact-avatar">${(c.name || "?")[0]}</div>
              <div class="contact-info">
                <span class="contact-name">${escapeHtml(c.name)}</span>
                <span class="contact-detail">${escapeHtml(c.phone || c.email || "")}</span>
              </div>
            </div>`).join("");
          return `<p class="card-summary">${escapeHtml(exec.summary)}</p><div class="contact-list">${contacts}</div>`;
        }
        return `<p class="card-summary">${escapeHtml(exec.summary)}</p>`;

      default:
        return `<p class="card-summary">${escapeHtml(exec.summary)}</p>`;
    }
  }

  function renderLatencyBar(latency) {
    if (!latency) return;
    const bar = document.createElement("div");
    bar.className = "latency-bar";

    const items = [];
    if (latency.transcribe_ms > 0) {
      items.push(`<div class="latency-item"><span class="latency-dot transcribe"></span>Transcribe ${latency.transcribe_ms}ms</div>`);
    }
    items.push(`<div class="latency-item"><span class="latency-dot route"></span>Route ${latency.route_ms}ms</div>`);
    items.push(`<div class="latency-item"><span class="latency-dot execute"></span>Execute ${latency.execute_ms}ms</div>`);
    items.push(`<div class="latency-item"><span class="latency-dot total"></span>Total ${latency.total_ms}ms</div>`);

    bar.innerHTML = items.join("");
    resultsArea.appendChild(bar);
  }

  // ── Mic Recording ───────────────────────────────────────

  async function toggleRecording() {
    if (isRecording) {
      stopRecording();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      const recordStart = Date.now();

      mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        stopSilenceDetection();
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunks, { type: "audio/webm" });
        if (Date.now() - recordStart > MIN_RECORD_MS) {
          sendAudio(blob);
        }
      };

      mediaRecorder.start();
      isRecording = true;
      micRing.classList.add("recording");
      micHint.textContent = "Listening...";
      micHint.classList.add("active");

      resultsArea.innerHTML = "";
      transcriptArea.innerHTML = "";
      clearStatus();

      startSilenceDetection(stream, recordStart);
    } catch (err) {
      console.error("Mic error:", err);
      showError("Microphone access denied. Please allow mic permissions.");
    }
  }

  function startSilenceDetection(stream, recordStart) {
    audioCtx = new AudioContext();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);

    const data = new Float32Array(analyser.fftSize);
    let silentSince = null;

    function check() {
      if (!isRecording) return;
      analyser.getFloatTimeDomainData(data);
      let rms = 0;
      for (let i = 0; i < data.length; i++) rms += data[i] * data[i];
      rms = Math.sqrt(rms / data.length);

      if (rms < SILENCE_THRESHOLD) {
        if (!silentSince) silentSince = Date.now();
        else if (Date.now() - silentSince > SILENCE_DURATION && Date.now() - recordStart > MIN_RECORD_MS) {
          stopRecording();
          return;
        }
      } else {
        silentSince = null;
      }
      silenceTimer = requestAnimationFrame(check);
    }
    check();
  }

  function stopSilenceDetection() {
    if (silenceTimer) { cancelAnimationFrame(silenceTimer); silenceTimer = null; }
    if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
    isRecording = false;
    micRing.classList.remove("recording");
    micHint.textContent = "Tap to speak";
    micHint.classList.remove("active");
  }

  function sendAudio(blob) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      showError("Not connected to server. Retrying...");
      connectWS();
      return;
    }
    showStatus("Sending audio...");
    blob.arrayBuffer().then((buf) => ws.send(buf));
  }

  // ── Text Input ──────────────────────────────────────────

  function sendTextQuery() {
    const text = textInput.value.trim();
    if (!text) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      showError("Not connected to server.");
      return;
    }

    resultsArea.innerHTML = "";
    transcriptArea.innerHTML = "";
    clearStatus();

    ws.send(JSON.stringify({ type: "text_query", text }));
    textInput.value = "";
    showStatus("Processing...");
  }

  // ── Event Listeners ─────────────────────────────────────

  micBtn.addEventListener("click", (e) => {
    e.preventDefault();
    toggleRecording();
  });

  sendBtn.addEventListener("click", sendTextQuery);
  textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendTextQuery();
  });

  // ── Helpers ─────────────────────────────────────────────

  function escapeHtml(str) {
    if (!str) return "";
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }

  // ── Init ────────────────────────────────────────────────

  function clearAll() {
    resultsArea.innerHTML = "";
    transcriptArea.innerHTML = "";
    statusArea.innerHTML = "";
    textInput.value = "";
  }

  clearAll();
  connectWS();

  window.addEventListener("pageshow", (e) => {
    if (e.persisted) clearAll();
  });
})();
