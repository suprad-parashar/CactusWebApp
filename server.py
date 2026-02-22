#!/usr/bin/env python3
"""
Voice-to-Action Server

FastAPI backend that:
1. Accepts audio from the browser via WebSocket
2. Transcribes on-device using cactus_transcribe (Whisper)
3. Routes through the hybrid algorithm (FunctionGemma + deterministic parser)
4. Executes real function calls (weather, YouTube, notifications, etc.)
5. Returns results with full latency breakdown
"""

import asyncio
import atexit
import concurrent.futures
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HACKATHON_DIR = PROJECT_ROOT / "functiongemma-hackathon"
CACTUS_DIR = PROJECT_ROOT / "cactus"
WEIGHTS_DIR = CACTUS_DIR / "weights"

sys.path.insert(0, str(CACTUS_DIR / "python" / "src"))
sys.path.insert(0, str(HACKATHON_DIR))

from cactus import cactus_init, cactus_transcribe, cactus_destroy, cactus_reset
import main as _main_module
_main_module.functiongemma_path = str(WEIGHTS_DIR / "functiongemma-270m-it")
from main import generate_hybrid
from executors import execute_function_call

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    },
    {
        "name": "play_music",
        "description": "Play a song or playlist on YouTube",
        "parameters": {
            "type": "object",
            "properties": {
                "song": {"type": "string", "description": "Song or artist name"}
            },
            "required": ["song"],
        },
    },
    {
        "name": "set_alarm",
        "description": "Set an alarm for a specific time",
        "parameters": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer", "description": "Hour (0-23)"},
                "minute": {"type": "integer", "description": "Minute (0-59)"},
            },
            "required": ["hour", "minute"],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Timer duration in minutes"}
            },
            "required": ["minutes"],
        },
    },
    {
        "name": "create_reminder",
        "description": "Create a reminder with a title and time",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Reminder title"},
                "time": {"type": "string", "description": "When to remind"},
            },
            "required": ["title", "time"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to a contact",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Contact name"},
                "message": {"type": "string", "description": "Message content"},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "search_contacts",
        "description": "Search contacts by name",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
]

_whisper_model = None
_executor_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        whisper_path = str(WEIGHTS_DIR / "whisper-small")
        print(f"[init] Loading Whisper model from {whisper_path}...")
        _whisper_model = cactus_init(whisper_path)
        print("[init] Whisper model loaded.")
    return _whisper_model


@atexit.register
def _cleanup_whisper():
    global _whisper_model
    if _whisper_model is not None:
        try:
            cactus_destroy(_whisper_model)
        except Exception:
            pass
        _whisper_model = None
    _executor_pool.shutdown(wait=False)


def transcribe_audio(audio_path: str) -> tuple[str, float]:
    """Transcribe a WAV file. Returns (text, latency_ms)."""
    model = get_whisper()
    cactus_reset(model)
    prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

    start = time.time()
    raw = cactus_transcribe(model, audio_path, prompt=prompt)
    latency = (time.time() - start) * 1000

    try:
        result = json.loads(raw)
        text = (result.get("response") or "").strip()
    except (json.JSONDecodeError, TypeError, AttributeError):
        text = (raw or "").strip()

    return text, round(latency, 1)


def route_query(text: str) -> tuple[list, float, str]:
    """Route a text query through the full hybrid algorithm."""
    start = time.time()
    messages = [{"role": "user", "content": text}]
    result = generate_hybrid(messages, TOOLS)
    latency = (time.time() - start) * 1000
    calls = result.get("function_calls", [])
    source = result.get("source", "unknown")
    return calls, round(latency, 1), source


def execute_calls(function_calls: list) -> tuple[list, float]:
    """Execute all function calls concurrently. Returns (results, latency_ms)."""
    start = time.time()

    if len(function_calls) == 1:
        c = function_calls[0]
        results = [execute_function_call(c.get("name", ""), c.get("arguments", {}))]
    else:
        futures = [
            _executor_pool.submit(execute_function_call, c.get("name", ""), c.get("arguments", {}))
            for c in function_calls
        ]
        results = [f.result(timeout=15) for f in futures]

    latency = (time.time() - start) * 1000
    return results, round(latency, 1)


def convert_audio(input_path: str, output_path: str):
    """Convert browser audio to 16kHz mono WAV using ffmpeg. Raises on failure."""
    import subprocess
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path],
        capture_output=True, timeout=10,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")[:200]
        raise RuntimeError(f"ffmpeg failed (code {proc.returncode}): {stderr}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
        raise RuntimeError("ffmpeg produced empty output")


app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive()

            if "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "text_query":
                    await handle_text_query(ws, msg.get("text", ""))
                    continue

            if "bytes" in data:
                audio_bytes = data["bytes"]
                if len(audio_bytes) > MAX_AUDIO_BYTES:
                    await ws.send_json({"type": "error", "message": "Audio too large (10MB limit)"})
                    continue
                await handle_audio(ws, audio_bytes)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def handle_audio(ws: WebSocket, audio_bytes: bytes):
    """Full pipeline: audio -> transcribe -> route -> execute."""
    pipeline_start = time.time()

    await ws.send_json({"type": "status", "stage": "transcribing", "message": "Transcribing audio..."})

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    wav_path = audio_path + ".wav"
    try:
        await asyncio.to_thread(convert_audio, audio_path, wav_path)
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"Audio conversion failed: {e}"})
        return
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass

    try:
        text, transcribe_ms = await asyncio.to_thread(transcribe_audio, wav_path)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    if not text:
        await ws.send_json({"type": "error", "message": "Could not transcribe audio"})
        return

    await ws.send_json({
        "type": "transcript",
        "text": text,
        "latency_ms": transcribe_ms,
    })

    await _route_and_execute(ws, text, pipeline_start, transcribe_ms)


async def handle_text_query(ws: WebSocket, text: str):
    """Pipeline for typed text: route -> execute."""
    pipeline_start = time.time()
    await ws.send_json({"type": "transcript", "text": text, "latency_ms": 0})
    await _route_and_execute(ws, text, pipeline_start, 0)


async def _route_and_execute(ws: WebSocket, text: str, pipeline_start: float, transcribe_ms: float):
    """Shared routing + execution pipeline."""
    await ws.send_json({"type": "status", "stage": "routing", "message": "Routing query..."})

    function_calls, route_ms, source = await asyncio.to_thread(route_query, text)

    if not function_calls:
        await ws.send_json({
            "type": "result",
            "transcript": text,
            "function_calls": [],
            "executions": [],
            "latency": {
                "transcribe_ms": transcribe_ms,
                "route_ms": route_ms,
                "execute_ms": 0,
                "total_ms": round((time.time() - pipeline_start) * 1000, 1),
            },
            "source": source,
        })
        return

    await ws.send_json({"type": "status", "stage": "executing", "message": "Executing actions..."})

    executions, execute_ms = await asyncio.to_thread(execute_calls, function_calls)

    total_ms = round((time.time() - pipeline_start) * 1000, 1)

    await ws.send_json({
        "type": "result",
        "transcript": text,
        "function_calls": function_calls,
        "executions": executions,
        "latency": {
            "transcribe_ms": transcribe_ms,
            "route_ms": route_ms,
            "execute_ms": execute_ms,
            "total_ms": total_ms,
        },
        "source": source,
    })


if __name__ == "__main__":
    import uvicorn
    from main import _get_model
    sys.stdout.reconfigure(line_buffering=True)
    print("\n  Cactus Voice - Voice-to-Action Server")
    print("  Loading models...\n")
    get_whisper()
    print("[init] Loading FunctionGemma...")
    _get_model()
    print("[init] FunctionGemma loaded.")
    print("\n  Ready at http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
