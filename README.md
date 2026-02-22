# Cactus Voice

A voice-to-action web application that runs **entirely on-device** using [Cactus](https://github.com/cactus-compute/cactus) for speech transcription and [FunctionGemma](https://huggingface.co/google/functiongemma-270m-it) for intelligent function routing. Speak naturally, and actions happen instantly — no cloud APIs needed for processing.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [Supported Actions](#supported-actions)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## Overview

Cactus Voice turns natural language — spoken or typed — into real actions on your machine. It uses a fully on-device pipeline: audio is transcribed by Whisper (via Cactus), queries are routed by a hybrid FunctionGemma + deterministic parser, and matched functions are executed in real time. The entire routing layer runs locally with sub-millisecond latency.

### Key Features

- **On-device transcription**: Whisper Small model running locally via Cactus — no audio leaves your machine
- **Hybrid routing**: FunctionGemma neural model combined with a deterministic parser for fast, reliable intent matching
- **Real-time WebSocket communication**: Live status updates as audio is transcribed, routed, and executed
- **7 built-in actions**: Weather, music, alarms, timers, reminders, messages, and contacts
- **Voice and text input**: Tap-to-speak microphone with silence detection, or type commands directly
- **Latency transparency**: Full breakdown of transcription, routing, and execution times displayed in the UI
- **Modern dark UI**: Animated microphone button, staggered result cards, and per-action icons

## Tech Stack

| Layer | Technology |
| --- | --- |
| **Backend** | Python 3.11+ with FastAPI |
| **Frontend** | Vanilla JavaScript (ES6), HTML5, CSS3 |
| **Real-time** | WebSockets |
| **Transcription** | Cactus (Whisper Small) — on-device |
| **Routing** | FunctionGemma 270M + deterministic parser — on-device |
| **Audio** | MediaRecorder API (browser), ffmpeg (conversion) |
| **External APIs** | wttr.in (weather), YouTube (music search) |

## Project Structure

```
CactusWebApp/
├── server.py              # FastAPI server, WebSocket handler, pipeline orchestration
├── executors.py           # Function executors (weather, music, alarms, etc.)
├── requirements.txt       # Python dependencies
├── TEST_SUMMARY.md        # Test results documentation
├── test_results.html      # HTML test report
└── static/
    ├── index.html         # Main HTML page
    ├── app.js             # Frontend JavaScript (WebSocket client, UI logic)
    └── style.css          # Dark theme styling and animations
```

### External Dependencies (sibling directories)

The application expects the following directories alongside the project root:

```
Parent Directory/
├── CactusWebApp/                  # This project
├── functiongemma-hackathon/       # FunctionGemma hybrid routing algorithm
└── cactus/                        # Cactus transcription library
    ├── python/src/                # Cactus Python bindings
    └── weights/                   # Model weights
        ├── whisper-small/         # Whisper transcription model
        └── functiongemma-270m-it/ # FunctionGemma routing model
```

## Setup Instructions

### Prerequisites

- **Python 3.11 or higher** — [Download Python](https://www.python.org/downloads/)
- **ffmpeg** installed and available in PATH — [Download ffmpeg](https://ffmpeg.org/download.html)
- **macOS** (required for native notifications via `osascript`; other platforms can run without notification support)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd CactusWebApp
```

### Step 2: Set Up Sibling Dependencies

Ensure the Cactus library and FunctionGemma code exist in sibling directories as shown in the [project structure](#external-dependencies-sibling-directories) above. Download the model weights into the `cactus/weights/` directory:

- **Whisper Small** → `../cactus/weights/whisper-small/`
- **FunctionGemma 270M-IT** → `../cactus/weights/functiongemma-270m-it/`

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start the Server

```bash
python server.py
```

The server will:
1. Load the Whisper model from `../cactus/weights/whisper-small`
2. Load the FunctionGemma model from `../cactus/weights/functiongemma-270m-it`
3. Start the FastAPI server on `http://localhost:8000`

### Step 5: Access the Application

Open **http://localhost:8000** in your browser and grant microphone permissions when prompted.

## How It Works

### Pipeline Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Voice / Text Input                     │
│  (Browser microphone with silence detection, or text)    │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Step 1: Audio Conversion                                │
│  - Browser audio → 16kHz mono WAV via ffmpeg             │
│  - (Skipped for text input)                              │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Step 2: On-Device Transcription                         │
│  - Whisper Small via Cactus                              │
│  - Runs entirely on-device, no cloud calls               │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Step 3: Hybrid Routing                                  │
│  - FunctionGemma neural model (primary)                  │
│  - Deterministic parser (fallback + speed boost)         │
│  - Matches query to one of 7 tool definitions            │
│  - Latency: 0.1–0.5ms                                   │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Step 4: Function Execution                              │
│  - Concurrent execution via thread pool (4 workers)      │
│  - Real actions: API calls, notifications, URL opens     │
│  - Results returned to browser via WebSocket              │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Step 5: Result Display                                  │
│  - Animated result cards with per-action icons           │
│  - Full latency breakdown (transcribe / route / execute) │
│  - Success/failure badges and execution times            │
└──────────────────────────────────────────────────────────┘
```

### Real-time WebSocket Communication

The frontend and backend communicate over a persistent WebSocket connection. Status updates stream to the UI at every stage of the pipeline:

| Event | Direction | Description |
| --- | --- | --- |
| `text_query` | Client → Server | User submits a typed command |
| Binary audio | Client → Server | Recorded microphone audio |
| `status` | Server → Client | Processing stage updates (transcribing, routing, executing) |
| `transcript` | Server → Client | Transcription result with latency |
| `result` | Server → Client | Final output with function calls, executions, and full latency breakdown |
| `error` | Server → Client | Error message if any step fails |

## Supported Actions

| Action | Function | Description | Example Command |
| --- | --- | --- | --- |
| 🌤️ **Weather** | `get_weather` | Current weather for a location via wttr.in | *"What's the weather in Paris?"* |
| 🎵 **Music** | `play_music` | Search and open a song on YouTube | *"Play Bohemian Rhapsody"* |
| ⏰ **Alarm** | `set_alarm` | Set an alarm with macOS notification | *"Set an alarm for 7 AM"* |
| ⏱️ **Timer** | `set_timer` | Start a countdown timer | *"Set a timer for 5 minutes"* |
| 📌 **Reminder** | `create_reminder` | Create a reminder with title and time | *"Remind me to call the dentist at 3 PM"* |
| 💬 **Message** | `send_message` | Send a message to a contact (simulated) | *"Send a message to Alice saying hello"* |
| 👤 **Contacts** | `search_contacts` | Search contacts by name | *"Find Bob in my contacts"* |

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Serves the main application page |
| `/static/*` | GET | Serves static assets (CSS, JS) |

### WebSocket Endpoint

| Endpoint | Description |
| --- | --- |
| `/ws` | Main WebSocket for voice/text queries and real-time results |

**Client → Server message formats:**

```json
{ "type": "text_query", "text": "What's the weather in Tokyo?" }
```

Or send raw binary audio data for voice input.

**Server → Client result format:**

```json
{
  "type": "result",
  "transcript": "What's the weather in Tokyo?",
  "function_calls": [{ "name": "get_weather", "arguments": { "location": "Tokyo" } }],
  "executions": [{ "success": true, "summary": "Clear sky, 68°F..." }],
  "latency": {
    "transcribe_ms": 0,
    "route_ms": 0.1,
    "execute_ms": 3200.5,
    "total_ms": 3201.2
  },
  "source": "on-device"
}
```

## Performance

All routing is performed on-device with no cloud dependency:

| Metric | Value |
| --- | --- |
| **Routing latency** | 0.1–0.5 ms |
| **Contact search** | < 1 ms |
| **Music / Alarm / Timer / Reminder / Message** | 0–175 ms |
| **Weather** | ~3–4 s (external API) |
| **Max audio size** | 10 MB |
| **Thread pool workers** | 4 |

## Troubleshooting

### Server Issues

**Problem**: `ModuleNotFoundError: No module named 'cactus'`
- **Solution**: Ensure the `../cactus/python/src/` directory exists and contains the Cactus Python bindings. The server adds this to `sys.path` at startup.

**Problem**: `ModuleNotFoundError: No module named 'main'`
- **Solution**: Ensure the `../functiongemma-hackathon/` directory exists with the FunctionGemma hybrid algorithm code.

**Problem**: `ffmpeg failed` or audio conversion errors
- **Solution**: Install ffmpeg and ensure it is available in your system PATH. Verify with `ffmpeg -version`.

**Problem**: `Connection refused` when opening `http://localhost:8000`
- **Solution**: Check that the server started successfully and both models loaded without errors. Look for `Ready at http://localhost:8000` in the console output.

### Frontend Issues

**Problem**: Microphone not working
- **Solution**: Ensure you are accessing the app over `localhost` (not a raw IP), and grant microphone permissions when the browser prompts.

**Problem**: No results appearing
- **Solution**: Open the browser console (F12) and check for WebSocket connection errors. Verify the backend is running on port 8000.

### Platform Notes

- **macOS**: Full functionality including native notifications via `osascript`
- **Windows / Linux**: All features work except native notifications (alarms, timers, and reminders will execute but won't display OS-level alerts)

## License

MIT License
