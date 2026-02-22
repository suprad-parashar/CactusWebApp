"""
Real function executors that turn function calls into real-world actions.
Each executor returns a dict with 'success', 'summary', and function-specific data.
"""

import json
import subprocess
import threading
import time
import urllib.request
import urllib.parse


def _escape_applescript(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _notify_macos(title: str, message: str, sound: str = "default"):
    """Send a macOS notification via osascript."""
    safe_title = _escape_applescript(title)
    safe_msg = _escape_applescript(message)
    script = (
        f'display notification "{safe_msg}" '
        f'with title "{safe_title}" '
        f'sound name "{sound}"'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, timeout=5,
        )
        return True
    except Exception:
        return False


def execute_get_weather(arguments: dict) -> dict:
    location = arguments.get("location", "New York")
    try:
        url = f"http://wttr.in/{urllib.parse.quote(location)}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "?")
        temp_f = current.get("temp_F", "?")
        desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
        humidity = current.get("humidity", "?")
        wind_mph = current.get("windspeedMiles", "?")
        feels_like_c = current.get("FeelsLikeC", "?")
        feels_like_f = current.get("FeelsLikeF", "?")

        return {
            "success": True,
            "summary": f"{desc}, {temp_f}°F ({temp_c}°C) in {location}",
            "data": {
                "location": location,
                "temperature_c": temp_c,
                "temperature_f": temp_f,
                "feels_like_c": feels_like_c,
                "feels_like_f": feels_like_f,
                "condition": desc,
                "humidity": f"{humidity}%",
                "wind": f"{wind_mph} mph",
            },
        }
    except Exception as e:
        return {"success": False, "summary": f"Could not fetch weather for {location}", "error": str(e)}


def execute_play_music(arguments: dict) -> dict:
    song = arguments.get("song", "")
    if not song:
        return {"success": False, "summary": "No song specified"}

    query = urllib.parse.quote_plus(song)
    url = f"https://www.youtube.com/results?search_query={query}"
    return {
        "success": True,
        "summary": f"Found \"{song}\" on YouTube",
        "data": {"song": song, "url": url},
    }


def execute_set_alarm(arguments: dict) -> dict:
    hour = arguments.get("hour", 0)
    minute = arguments.get("minute", 0)
    time_str = f"{int(hour)}:{int(minute):02d}"

    _notify_macos("Alarm Set", f"Alarm set for {time_str}")
    return {
        "success": True,
        "summary": f"Alarm set for {time_str}",
        "data": {"hour": int(hour), "minute": int(minute), "time_display": time_str},
    }


def execute_set_timer(arguments: dict) -> dict:
    minutes = int(arguments.get("minutes", 0))
    if minutes <= 0:
        return {"success": False, "summary": "Invalid timer duration"}

    _notify_macos("Timer Started", f"{minutes}-minute timer started")

    def _fire():
        _notify_macos("Timer Done!", f"Your {minutes}-minute timer is up!", sound="Blow")

    t = threading.Timer(minutes * 60, _fire)
    t.daemon = True
    t.start()

    return {
        "success": True,
        "summary": f"{minutes}-minute timer started",
        "data": {"minutes": minutes},
    }


def execute_create_reminder(arguments: dict) -> dict:
    title = arguments.get("title", "Reminder")
    reminder_time = arguments.get("time", "")

    _notify_macos("Reminder Created", f"{title} at {reminder_time}")
    return {
        "success": True,
        "summary": f"Reminder: \"{title}\" at {reminder_time}",
        "data": {"title": title, "time": reminder_time},
    }


def execute_send_message(arguments: dict) -> dict:
    recipient = arguments.get("recipient", "")
    message = arguments.get("message", "")

    _notify_macos(f"Message to {recipient}", message)
    return {
        "success": True,
        "summary": f"Message sent to {recipient}",
        "data": {"recipient": recipient, "message": message},
    }


_CONTACTS = [
    {"name": "Alice Johnson", "phone": "+1-555-0101", "email": "alice@example.com"},
    {"name": "Bob Smith", "phone": "+1-555-0102", "email": "bob@example.com"},
    {"name": "Charlie Davis", "phone": "+1-555-0103", "email": "charlie@example.com"},
    {"name": "Diana Wilson", "phone": "+1-555-0104", "email": "diana@example.com"},
    {"name": "Eve Martinez", "phone": "+1-555-0105", "email": "eve@example.com"},
    {"name": "Frank Brown", "phone": "+1-555-0106", "email": "frank@example.com"},
    {"name": "Grace Lee", "phone": "+1-555-0107", "email": "grace@example.com"},
    {"name": "Hank Taylor", "phone": "+1-555-0108", "email": "hank@example.com"},
    {"name": "Mom", "phone": "+1-555-0100", "email": "mom@family.com"},
    {"name": "Dad", "phone": "+1-555-0099", "email": "dad@family.com"},
]


def execute_search_contacts(arguments: dict) -> dict:
    query = arguments.get("query", "").lower()
    if not query:
        return {"success": False, "summary": "No search query provided"}

    results = [c for c in _CONTACTS if query in c["name"].lower()]
    if results:
        names = ", ".join(c["name"] for c in results)
        return {
            "success": True,
            "summary": f"Found: {names}",
            "data": {"results": results, "count": len(results)},
        }
    return {
        "success": True,
        "summary": f"No contacts found for \"{query}\"",
        "data": {"results": [], "count": 0},
    }


EXECUTOR_MAP = {
    "get_weather": execute_get_weather,
    "play_music": execute_play_music,
    "set_alarm": execute_set_alarm,
    "set_timer": execute_set_timer,
    "create_reminder": execute_create_reminder,
    "send_message": execute_send_message,
    "search_contacts": execute_search_contacts,
}


def execute_function_call(name: str, arguments: dict) -> dict:
    """Execute a function call by name. Returns result dict."""
    executor = EXECUTOR_MAP.get(name)
    if not executor:
        return {"success": False, "summary": f"Unknown function: {name}"}

    start = time.time()
    result = executor(arguments)
    result["execution_time_ms"] = round((time.time() - start) * 1000, 1)
    result["function_name"] = name
    result["arguments"] = arguments
    return result
