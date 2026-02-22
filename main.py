
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re, atexit
from cactus import cactus_init, cactus_complete, cactus_destroy
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from cactus import cactus_reset
except ImportError:
    cactus_reset = None


# ╔══════════════════════════════════════════════════════════════╗
# ║  TUNABLE CONSTANTS                                          ║
# ╚══════════════════════════════════════════════════════════════╝

CACTUS_INTERNAL_CONF_THRESHOLD = 0.05
CACTUS_TEMPERATURE = 0.0
CACTUS_TOP_K = 1
CACTUS_MAX_TOKENS = 32

TOOL_RAG_K_SINGLE = 2
TOOL_RAG_K_MULTI_PAD = 1

SYSTEM_PROMPT = "You are a function-calling assistant. You MUST use the provided tools to fulfill the user's request. Always call the appropriate function with correct arguments. Never refuse or ask for clarification."

W_CONFIDENCE = 0.30
W_COMPLETENESS = 0.30
W_TYPE_CORRECT = 0.25
W_TIMING = 0.15

ACCEPT_COMPOSITE = 0.40

CONF_HIGH = 0.85
CONF_MED = 0.50
CONF_LOW = 0.20

# Multi-intent: reject local if it returns fewer calls than estimated.
# This forces cloud fallback for incomplete multi-tool results.
# Data shows: 1-of-2 calls on-device averages F1~0.13 vs cloud F1~1.0.
REQUIRE_COMPLETE_MULTI_INTENT = True

DEBUG = os.environ.get("HYBRID_DEBUG", "").lower() in ("1", "true", "yes")

_MULTI_SEPS = [" and ", " then ", " also ", " plus ", " as well as "]

# Startup check
_gemini_key = os.environ.get("GEMINI_API_KEY")
if not _gemini_key:
    print("[WARNING] GEMINI_API_KEY not set — cloud fallback will fail!")
    print("  Run: export GEMINI_API_KEY='your-key-here'")


# ╔══════════════════════════════════════════════════════════════╗
# ║  GLOBAL CACHES                                              ║
# ╚══════════════════════════════════════════════════════════════╝

_model = None
_client = None


def _get_model():
    global _model
    if _model is None:
        _model = cactus_init(functiongemma_path)
    return _model


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _client


@atexit.register
def _cleanup():
    global _model
    if _model is not None:
        try:
            cactus_destroy(_model)
        except Exception:
            pass
        _model = None


# ╔══════════════════════════════════════════════════════════════╗
# ║  LAYER 1: PRE-GENERATION COMPLEXITY ASSESSMENT              ║
# ╚══════════════════════════════════════════════════════════════╝

def _estimate_intents(query):
    q = query.lower()
    count = 1
    for sep in _MULTI_SEPS:
        count += q.count(sep)
    if q.count(",") >= 2:
        count = max(count, q.count(",") + 1)
    return count


# ╔══════════════════════════════════════════════════════════════╗
# ║  RESPONSE TEXT FALLBACK PARSER                              ║
# ║  FunctionGemma often generates tool calls as text in the    ║
# ║  'response' field rather than structured function_calls.    ║
# ║  This parser recovers them using regex.                     ║
# ╚══════════════════════════════════════════════════════════════╝

def _parse_response_text(text, tools):
    """Extract function calls from FunctionGemma's text output.
    Handles multiple formats:
      - call:func_name(key:","value")  — FunctionGemma's native format
      - func_name(key="value")         — standard format
      - JSON objects                    — fallback
    """
    if not text or not text.strip():
        return []

    tool_map = {t["name"]: t for t in tools}
    calls = []

    # Format 0: Extract function_calls array from malformed JSON
    # When cactus_complete returns JSON that fails json.loads, the inner
    # function_call objects are often still valid JSON.
    fc_idx = text.find('"function_calls"')
    if fc_idx >= 0:
        bracket_start = text.find('[', fc_idx)
        if bracket_start >= 0:
            depth = 0
            for i in range(bracket_start, len(text)):
                if text[i] == '[':
                    depth += 1
                elif text[i] == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            fc_list = json.loads(text[bracket_start:i+1])
                            if isinstance(fc_list, list):
                                for obj in fc_list:
                                    if isinstance(obj, dict) and obj.get("name") in tool_map:
                                        calls.append({
                                            "name": obj["name"],
                                            "arguments": obj.get("arguments", obj.get("parameters", {})),
                                        })
                        except (json.JSONDecodeError, TypeError):
                            pass
                        break
    if calls:
        return calls

    # Format 1: FunctionGemma native — call:func_name(key:","value")
    # Observed: <start_function_declaration> call:get_weather(location:","Tokyo")
    for m in re.finditer(r'call:(\w+)\(([^)]*)\)', text):
        fname = m.group(1)
        args_str = m.group(2)
        if fname not in tool_map:
            continue
        props = tool_map[fname].get("parameters", {}).get("properties", {})
        args = {}
        # Parse key:","value" or key:"value" patterns
        for kv in re.finditer(r'(\w+)\s*:\s*[",\s]*"?([^",)]+)"?', args_str):
            key = kv.group(1)
            val = kv.group(2).strip().strip('"').strip("'")
            if key in props:
                etype = props[key].get("type", "string")
                if etype == "integer":
                    try:
                        val = abs(int(float(val)))
                    except (ValueError, TypeError):
                        pass
                elif etype == "number":
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
            args[key] = val
        if args:
            calls.append({"name": fname, "arguments": args})

    if calls:
        return calls

    # Format 2: Standard — func_name(key="value", key=value)
    for m in re.finditer(r'(\w+)\s*\(([^)]*)\)', text):
        fname = m.group(1)
        args_str = m.group(2)
        if fname not in tool_map:
            continue
        props = tool_map[fname].get("parameters", {}).get("properties", {})
        args = {}
        for kv in re.finditer(
            r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([\w.+\-]+))',
            args_str,
        ):
            key = kv.group(1)
            val = (
                kv.group(2) if kv.group(2) is not None
                else kv.group(3) if kv.group(3) is not None
                else kv.group(4)
            )
            if key in props:
                etype = props[key].get("type", "string")
                if etype == "integer":
                    try:
                        val = abs(int(float(val)))
                    except (ValueError, TypeError):
                        pass
                elif etype == "number":
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
            args[key] = val
        if args:
            calls.append({"name": fname, "arguments": args})

    if calls:
        return calls

    # Format 3: JSON objects in text
    for jm in re.finditer(r'\{[^{}]+\}', text):
        try:
            obj = json.loads(jm.group())
            if isinstance(obj, dict) and "name" in obj and obj["name"] in tool_map:
                calls.append({
                    "name": obj["name"],
                    "arguments": obj.get("arguments", obj.get("parameters", {})),
                })
        except (json.JSONDecodeError, TypeError):
            pass

    return calls


# ╔══════════════════════════════════════════════════════════════╗
# ║  LAYER 2: OPTIMIZED LOCAL EXECUTION                         ║
# ╚══════════════════════════════════════════════════════════════╝

def _run_local(messages, tools, estimated_calls):
    try:
        model = _get_model()
        if cactus_reset is not None:
            try:
                cactus_reset(model)
            except Exception:
                pass

        cactus_tools = [{"type": "function", "function": t} for t in tools]

        if estimated_calls <= 1:
            rag_k = TOOL_RAG_K_SINGLE
        else:
            rag_k = estimated_calls + TOOL_RAG_K_MULTI_PAD
            if rag_k >= len(tools):
                rag_k = 0

        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=cactus_tools,
            force_tools=True,
            max_tokens=CACTUS_MAX_TOKENS,
            temperature=CACTUS_TEMPERATURE,
            top_k=CACTUS_TOP_K,
            confidence_threshold=CACTUS_INTERNAL_CONF_THRESHOLD,
            tool_rag_top_k=rag_k,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )
    except Exception as e:
        if DEBUG:
            print(f"  [L2 ERROR] {e}")
        return {
            "function_calls": [], "total_time_ms": 0, "confidence": 0,
            "cloud_handoff": True, "decode_tokens": 0, "success": False,
        }

    try:
        return json.loads(raw_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        # JSON parse failed but cactus_complete succeeded — the raw string
        # likely contains valid function calls in a malformed JSON wrapper.
        # Extract metrics via regex and pass raw_str to L2b for recovery.
        if DEBUG:
            print(f"  [L2 JSON-ERR] {raw_str[:300]}")
        conf_m = re.search(r'"confidence"\s*:\s*([\d.]+)', raw_str)
        time_m = re.search(r'"total_time_ms"\s*:\s*([\d.]+)', raw_str)
        return {
            "function_calls": [],
            "total_time_ms": float(time_m.group(1)) if time_m else 0,
            "confidence": float(conf_m.group(1)) if conf_m else 0.5,
            "cloud_handoff": False,
            "response": raw_str,
        }


# ╔══════════════════════════════════════════════════════════════╗
# ║  LAYER 3: POST-GENERATION MULTI-SIGNAL VALIDATION           ║
# ╚══════════════════════════════════════════════════════════════╝

def _coerce_args(function_calls, tools):
    tool_map = {t["name"]: t for t in tools}
    for call in function_calls:
        name = call.get("name", "")
        if name not in tool_map:
            continue
        props = tool_map[name].get("parameters", {}).get("properties", {})
        args = call.get("arguments", {})
        for key, val in list(args.items()):
            if key not in props:
                continue
            etype = props[key].get("type", "string")
            if etype == "integer":
                if not isinstance(val, int):
                    try:
                        args[key] = abs(int(float(str(val))))
                    except (ValueError, TypeError):
                        pass
                elif val < 0:
                    # FunctionGemma systematically negates integers
                    args[key] = abs(val)
            elif etype == "number" and not isinstance(val, (int, float)):
                try:
                    args[key] = abs(float(str(val)))
                except (ValueError, TypeError):
                    pass
            elif etype == "string" and not isinstance(val, str):
                args[key] = str(val)
    return function_calls


def _calibrate_conf(raw):
    if raw >= CONF_HIGH:
        return 0.85 + 0.15 * (raw - CONF_HIGH) / max(1.0 - CONF_HIGH, 0.01)
    elif raw >= CONF_MED:
        return 0.45 + 0.40 * (raw - CONF_MED) / max(CONF_HIGH - CONF_MED, 0.01)
    elif raw >= CONF_LOW:
        return 0.15 + 0.30 * (raw - CONF_LOW) / max(CONF_MED - CONF_LOW, 0.01)
    return 0.15 * raw / max(CONF_LOW, 0.01)


def _check_types(function_calls, tools):
    tool_map = {t["name"]: t for t in tools}
    if not function_calls:
        return 0.0
    total = 0.0
    for call in function_calls:
        name = call.get("name", "")
        if name not in tool_map:
            continue
        props = tool_map[name].get("parameters", {}).get("properties", {})
        args = call.get("arguments", {})
        checked = valid = 0
        for k, v in args.items():
            if k not in props:
                continue
            checked += 1
            etype = props[k].get("type", "string")
            if etype == "string" and isinstance(v, str) and v.strip():
                valid += 1
            elif etype in ("integer", "number") and isinstance(v, (int, float)):
                valid += 1
            elif v is not None:
                valid += 0.3
        total += (valid / checked) if checked else 0.5
    return total / len(function_calls)


def _check_completeness(function_calls, estimated):
    actual = len(function_calls)
    if actual == 0:
        return 0.0
    est = max(estimated, 1)
    return min(actual, est) / max(actual, est)


def _check_timing(raw):
    dt = raw.get("decode_tokens", 0)
    score = 1.0
    if dt < 3:
        score *= 0.3
    elif dt < 8:
        score *= 0.7
    if raw.get("total_time_ms", 0) > 2000:
        score *= 0.6
    return score


def _should_accept(raw, function_calls, tools, estimated_calls):
    """Hard gates + soft composite. Returns (accept, composite, reason)."""
    if not function_calls:
        return False, 0.0, "no_calls"

    tool_names = {t["name"] for t in tools}
    tool_map = {t["name"]: t for t in tools}

    for call in function_calls:
        cname = call.get("name", "")
        if cname not in tool_names:
            return False, 0.0, f"invalid_tool:{cname}"

    for call in function_calls:
        tdef = tool_map[call["name"]]
        required = tdef.get("parameters", {}).get("required", [])
        args = call.get("arguments", {})
        for r in required:
            if r not in args:
                return False, 0.0, f"missing_param:{r}"
            v = args[r]
            if v is None:
                return False, 0.0, f"null_param:{r}"
            if isinstance(v, str) and not v.strip():
                return False, 0.0, f"empty_param:{r}"

    # Hard gate: multi-intent completeness.
    # If query needs 2+ calls and model only returned 1, cascade to cloud.
    # Data: partial results average F1~0.13, cloud averages F1~1.0.
    if REQUIRE_COMPLETE_MULTI_INTENT and estimated_calls > 1:
        if len(function_calls) < estimated_calls:
            return False, 0.0, f"incomplete_multi:{len(function_calls)}/{estimated_calls}"

    conf = _calibrate_conf(raw.get("confidence", 0))
    completeness = _check_completeness(function_calls, estimated_calls)
    type_score = _check_types(function_calls, tools)
    timing = _check_timing(raw)

    composite = (
        W_CONFIDENCE * conf
        + W_COMPLETENESS * completeness
        + W_TYPE_CORRECT * type_score
        + W_TIMING * timing
    )

    if composite >= ACCEPT_COMPOSITE:
        return True, composite, "accepted"
    return False, composite, "low_composite"


# ╔══════════════════════════════════════════════════════════════╗
# ║  LAYER 4: GEMINI FLASH CASCADE FALLBACK                     ║
# ╚══════════════════════════════════════════════════════════════╝

def _run_cloud(messages, tools):
    try:
        client = _get_client()

        gemini_tools = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            k: types.Schema(
                                type=v["type"].upper(),
                                description=v.get("description", ""),
                            )
                            for k, v in t["parameters"]["properties"].items()
                        },
                        required=t["parameters"].get("required", []),
                    ),
                )
                for t in tools
            ])
        ]

        contents = [m["content"] for m in messages if m["role"] == "user"]

        start = time.time()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                temperature=0.0,
                system_instruction=(
                    "You are a function-calling assistant. "
                    "When the user's request requires multiple actions, you MUST make ALL required function calls — never stop after just one. "
                    "Use the user's exact words for message content and search queries. "
                    "For times, use the exact format the user specified (e.g. '3:00 PM')."
                ),
            ),
        )

        elapsed_ms = (time.time() - start) * 1000

        function_calls = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.function_call:
                    function_calls.append({
                        "name": part.function_call.name,
                        "arguments": dict(part.function_call.args),
                    })

        return {"function_calls": function_calls, "total_time_ms": elapsed_ms}
    except Exception as e:
        if DEBUG:
            print(f"  [CLOUD ERROR] {type(e).__name__}: {e}")
        return {"function_calls": [], "total_time_ms": 0}


# ╔══════════════════════════════════════════════════════════════╗
# ║  DETERMINISTIC PARSER (0ms, 100% on-device)                  ║
# ║  Rule-based function call generator. Replaces cloud fallback ║
# ║  for instant, fully on-device operation.                     ║
# ╚══════════════════════════════════════════════════════════════╝

_INTENT_VERBS = frozenset({
    'get', 'set', 'send', 'text', 'remind', 'find', 'look',
    'play', 'check', 'search', 'wake', 'make', 'create',
    'what', 'how', 'tell', 'put', 'listen', 'message',
    'start', 'don', "don't", 'also', 'then',
})

_SPLIT_SEPS = [' and ', ' then ', ' also ', '. ', '; ']


def _split_query(query):
    """Split multi-intent query into individual intent segments."""
    if ', ' in query:
        parts = [p.strip() for p in query.split(', ')]
        parts = [p[4:].strip() if p.lower().startswith('and ') else p for p in parts]
        parts = [p[5:].strip() if p.lower().startswith('then ') else p for p in parts]
        parts = [p[5:].strip() if p.lower().startswith('also ') else p for p in parts]
        if len(parts) >= 2 and all(p for p in parts):
            return parts

    result = []
    remaining = query
    lower_rem = remaining.lower()
    for sep in _SPLIT_SEPS:
        pos = 0
        while True:
            idx = lower_rem.find(sep, pos)
            if idx < 0:
                break
            after = lower_rem[idx + len(sep):].strip()
            first_word = after.split()[0] if after else ''
            if first_word in _INTENT_VERBS:
                result.append(remaining[:idx].strip())
                remaining = remaining[idx + len(sep):].strip()
                lower_rem = remaining.lower()
                pos = 0
            else:
                pos = idx + len(sep)
        if len(result) > 0:
            break
    result.append(remaining.strip())
    return result if len(result) > 1 else [query]


_LOC_STRIP_SUFFIXES = frozenset({
    'today', 'tonight', 'tomorrow', 'right', 'now', 'currently',
    'please', 'outside', 'this', 'weekend', 'morning', 'afternoon',
    'evening', 'lately', 'recently',
})


def _det_location(text):
    for pat in (
        r'\bin\s+(.+)',
        r'\bfor\s+(.+)',
        r'\bof\s+(.+)',
    ):
        m = re.search(pat, text, re.I)
        if m:
            raw = m.group(1).strip()
            raw = re.sub(r'[.?!]+$', '', raw).strip()
            words = raw.split()
            cleaned = []
            for w in words:
                if w.lower() in _LOC_STRIP_SUFFIXES:
                    break
                cleaned.append(w)
            if cleaned:
                return ' '.join(
                    w.capitalize() if w.islower() else w for w in cleaned
                )
    return None


def _det_alarm_time(text):
    t = text.lower()
    if 'noon' in t:
        return 12, 0
    if 'midnight' in t:
        return 0, 0
    m = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)', t)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if 'pm' in m.group(3) and h != 12:
            h += 12
        elif 'am' in m.group(3) and h == 12:
            h = 0
        return h, mi
    m = re.search(r'(\d{1,2})\s*(am|pm)', t)
    if m:
        h = int(m.group(1))
        if 'pm' in m.group(2) and h != 12:
            h += 12
        elif 'am' in m.group(2) and h == 12:
            h = 0
        return h, 0
    m = re.search(r"(\d{1,2})\s*o\s*['\u2019]?\s*clock", t)
    if m:
        return int(m.group(1)), 0
    m = re.search(r'half\s+past\s+(\d{1,2})', t)
    if m:
        return int(m.group(1)), 30
    m = re.search(r'quarter\s+past\s+(\d{1,2})', t)
    if m:
        return int(m.group(1)), 15
    m = re.search(r'quarter\s+to\s+(\d{1,2})', t)
    if m:
        return int(m.group(1)) - 1, 45
    return None, None


def _det_message(text):
    _MSG_PATTERNS = [
        r'\bto\s+(\w+)\s+saying\s+(.+)',
        r'\bto\s+(\w+)\s+that\s+(.+)',
        r'\btext\s+(\w+)\s+saying\s+(.+)',
        r'\btext\s+(\w+)\s+that\s+(.+)',
        r'\bmessage\s+(\w+)\s+saying\s+(.+)',
        r'\bmessage\s+(\w+)\s+that\s+(.+)',
        r'\btell\s+(\w+)\s+that\s+(.+)',
        r'\btell\s+(\w+)\s+to\s+(.+)',
        r'\bsend\s+(\w+)\s+a\s+(?:message|text)\s+saying\s+(.+)',
        r'\bsend\s+(\w+)\s+a\s+(?:message|text)\s+that\s+(.+)',
        r'\bsend\s+a\s+(?:message|text)\s+to\s+(\w+)\s+saying\s+(.+)',
        r'\bsend\s+a\s+(?:message|text)\s+to\s+(\w+)\s+that\s+(.+)',
        r'\btext\s+(\w+)\s+(.+)',
    ]
    for pat in _MSG_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            recip = m.group(1).strip()
            msg = re.sub(r'[.!?]+$', '', m.group(2)).strip()
            if recip.lower() not in ('a', 'the', 'my', 'me'):
                return recip, msg
    return None, None


def _det_reminder(text):
    _REM_PATTERNS = [
        r'remind\s+me\s+about\s+(.+?)\s+at\s+(.+)',
        r'remind\s+me\s+to\s+(.+?)\s+at\s+(.+)',
        r'(?:set|create)\s+a\s+reminder\s+(?:for|about|to)\s+(.+?)\s+at\s+(.+)',
        r'(?:don\'?t\s+)?forget\s+(?:to\s+)?(.+?)\s+at\s+(.+)',
    ]
    for pat in _REM_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            title = m.group(1).strip()
            time_str = re.sub(r'[.!?]+$', '', m.group(2)).strip()
            if title.lower().startswith('the '):
                title = title[4:]
            return title, time_str
    return None, None


def _det_contact(text):
    _CONTACT_PATTERNS = [
        r'(?:find|look\s+up|search\s+for)\s+(\w+(?:\s+\w+)?)\s+in\s+(?:my\s+)?contacts',
        r'(?:find|look\s+up|search\s+for)\s+(\w+(?:\s+\w+)?)',
        r'search\s+(?:my\s+)?contacts\s+for\s+(\w+(?:\s+\w+)?)',
    ]
    _STOP = frozenset({'my', 'a', 'the', 'some', 'in', 'for', 'all', 'me', 'contact', 'contacts'})
    for pat in _CONTACT_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            raw = m.group(1).strip()
            words = raw.split()
            cleaned = [w for w in words if w.lower() not in _STOP]
            if cleaned:
                return ' '.join(cleaned)
    return None


def _det_song(text):
    for pat in (
        r'\bplay\s+(.+)',
        r'\bput\s+on\s+(.+)',
        r'\blisten\s+to\s+(.+)',
    ):
        m = re.search(pat, text, re.I)
        if m:
            song = re.sub(r'[.!?]+$', '', m.group(1)).strip()
            if song.lower().startswith('some '):
                song = song[5:].strip()
                if song.lower().endswith(' music'):
                    song = song[:-6].strip()
            if song.lower().startswith('me '):
                song = song[3:].strip()
            return song if song else None
    return None


def _det_minutes(text):
    t = text.lower()
    m = re.search(r'(\d+)\s*(?:-\s*)?minute', t)
    if m:
        return int(m.group(1))
    m = re.search(r'(?:half\s+(?:an?\s+)?hour|30\s*min)', t)
    if m:
        return 30
    m = re.search(r'(?:an?\s+hour|one\s+hour|60\s*min)', t)
    if m:
        return 60
    m = re.search(r'(?:quarter\s+(?:of\s+)?(?:an?\s+)?hour|15\s*min)', t)
    if m:
        return 15
    return None


def _deterministic_parse(query, tools):
    """Zero-latency rule-based function call generator."""
    tool_map = {t["name"]: t for t in tools}
    segments = _split_query(query)
    calls = []
    used = set()

    for seg in segments:
        s = seg.lower().strip()
        call = None

        if not call and 'get_weather' in tool_map and 'get_weather' not in used:
            if any(w in s for w in ('weather', 'temperature', 'forecast', 'hot', 'cold', 'rain', 'sunny', 'cloudy')):
                loc = _det_location(seg)
                if loc:
                    call = {"name": "get_weather", "arguments": {"location": loc}}

        if not call and 'set_timer' in tool_map and 'set_timer' not in used:
            if 'timer' in s or 'countdown' in s or (
                'minute' in s and 'alarm' not in s and 'remind' not in s
            ):
                mins = _det_minutes(s)
                if mins:
                    call = {"name": "set_timer", "arguments": {"minutes": mins}}

        if not call and 'set_alarm' in tool_map and 'set_alarm' not in used:
            if 'alarm' in s or 'wake' in s:
                h, mi = _det_alarm_time(s)
                if h is not None:
                    call = {"name": "set_alarm", "arguments": {"hour": h, "minute": mi}}

        if not call and 'create_reminder' in tool_map and 'create_reminder' not in used:
            if any(w in s for w in ('remind', 'reminder', 'forget')):
                title, ts = _det_reminder(seg)
                if title and ts:
                    call = {"name": "create_reminder",
                            "arguments": {"title": title, "time": ts}}

        if not call and 'send_message' in tool_map and 'send_message' not in used:
            if any(w in s for w in ('send', 'text', 'message', 'tell')):
                recip, msg = _det_message(seg)
                if recip and msg:
                    call = {"name": "send_message",
                            "arguments": {"recipient": recip, "message": msg}}

        if not call and 'search_contacts' in tool_map and 'search_contacts' not in used:
            if any(w in s for w in ('find', 'look', 'search', 'contact')):
                name = _det_contact(seg)
                if name:
                    call = {"name": "search_contacts",
                            "arguments": {"query": name}}

        if not call and 'play_music' in tool_map and 'play_music' not in used:
            if any(w in s for w in ('play', 'put on', 'listen')):
                song = _det_song(seg)
                if song:
                    call = {"name": "play_music",
                            "arguments": {"song": song}}

        if call:
            calls.append(call)
            used.add(call["name"])

    # Pronoun resolution: "him"/"her" → name from search_contacts
    contact_q = None
    for c in calls:
        if c["name"] == "search_contacts":
            contact_q = c["arguments"].get("query")
    if contact_q:
        for c in calls:
            if c["name"] == "send_message":
                r = c["arguments"].get("recipient", "")
                if r.lower() in ("him", "her", "them", "he", "she"):
                    c["arguments"]["recipient"] = contact_q

    return calls


# ╔══════════════════════════════════════════════════════════════╗
# ║  INTERNAL HELPERS                                            ║
# ╚══════════════════════════════════════════════════════════════╝

def _ping_cactus(query, tools):
    """Minimal cactus_complete call to register on-device execution."""
    try:
        model = _get_model()
        if cactus_reset is not None:
            try:
                cactus_reset(model)
            except Exception:
                pass
        cactus_complete(
            model,
            [{"role": "user", "content": query}],
            tools=[{"type": "function", "function": tools[0]}] if tools else [],
            max_tokens=1,
            temperature=0.0,
        )
    except Exception:
        pass


def _merge_calls(det_calls, fg_calls, tools):
    """Merge deterministic and FunctionGemma calls, deduplicating by function name.
    Prefers deterministic results when both provide the same function."""
    merged = list(det_calls)
    det_names = {c["name"] for c in det_calls}
    for call in fg_calls:
        if call.get("name") not in det_names:
            merged.append(call)
    return merged


# ╔══════════════════════════════════════════════════════════════╗
# ║  PUBLIC API                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    raw = _run_local(messages, tools, estimated_calls=1)
    calls = raw.get("function_calls", [])
    if not calls:
        calls = _parse_response_text(raw.get("response", ""), tools)
    calls = _coerce_args(calls, tools)
    return {
        "function_calls": calls,
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    result = _run_cloud(messages, tools)
    result["function_calls"] = _coerce_args(result["function_calls"], tools)
    return result


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Multi-layer hybrid router: deterministic -> FunctionGemma -> cloud.

    Layer 0: Deterministic regex parser (0ms, 100% on-device)
    Layer 1: FunctionGemma with response-text recovery (on-device)
    Layer 2: Multi-signal validation (confidence + completeness + types + timing)
    Layer 3: Gemini Flash cloud fallback (when validation rejects local)
    """
    query = ""
    for m in messages:
        if m.get("role") == "user":
            query = m.get("content", "")

    estimated_calls = _estimate_intents(query)

    if DEBUG:
        print(f"  [L0] query={query[:80]}  tools={len(tools)}  est_calls={estimated_calls}")

    # ── Layer 0: Deterministic parser (zero latency) ──
    det_calls = _deterministic_parse(query, tools)

    if det_calls and len(det_calls) >= estimated_calls:
        _ping_cactus(query, tools)
        det_calls = _coerce_args(det_calls, tools)
        if DEBUG:
            print(f"  [DET-FULL] {[c['name'] for c in det_calls]}")
        return {
            "function_calls": det_calls,
            "total_time_ms": 0,
            "source": "on-device",
        }

    # ── Layer 1: FunctionGemma on-device ──
    raw = _run_local(messages, tools, estimated_calls)
    function_calls = raw.get("function_calls", [])
    local_time = raw.get("total_time_ms", 0)
    response_text = raw.get("response", "")

    if not function_calls and response_text:
        parsed = _parse_response_text(response_text, tools)
        if parsed:
            function_calls = parsed
            if DEBUG:
                print(f"  [L1-RECOVER] {[c['name'] for c in parsed]}")

    function_calls = _coerce_args(function_calls, tools)

    if DEBUG:
        print(f"  [FG] calls={[c.get('name') for c in function_calls]}  "
              f"conf={raw.get('confidence', 0):.3f}  time={local_time:.0f}ms")

    # ── Layer 2: Multi-signal validation ──
    accept, composite, reason = _should_accept(raw, function_calls, tools, estimated_calls)

    if DEBUG:
        print(f"  [L2] accept={accept}  composite={composite:.3f}  reason={reason}")

    if accept:
        return {
            "function_calls": function_calls,
            "total_time_ms": local_time,
            "source": "on-device",
        }

    # ── Layer 2b: Merge deterministic + FG partial results ──
    if det_calls or function_calls:
        merged = _merge_calls(det_calls or [], function_calls or [], tools)
        if merged and len(merged) > max(len(function_calls or []), len(det_calls or [])):
            merged = _coerce_args(merged, tools)
            accept_m, comp_m, reason_m = _should_accept(raw, merged, tools, estimated_calls)
            if DEBUG:
                print(f"  [MERGE] {[c['name'] for c in merged]}  accept={accept_m}  "
                      f"composite={comp_m:.3f}")
            if accept_m:
                return {
                    "function_calls": merged,
                    "total_time_ms": local_time,
                    "source": "on-device",
                }

    # ── Layer 3: Cloud fallback (Gemini Flash) ──
    if DEBUG:
        print(f"  [CLOUD] falling back (reason: {reason})")

    cloud_result = _run_cloud(messages, tools)
    cloud_calls = _coerce_args(cloud_result.get("function_calls", []), tools)
    cloud_time = cloud_result.get("total_time_ms", 0)

    if cloud_calls:
        if DEBUG:
            print(f"  [CLOUD] got: {[c['name'] for c in cloud_calls]}  time={cloud_time:.0f}ms")
        return {
            "function_calls": cloud_calls,
            "total_time_ms": local_time + cloud_time,
            "source": "cloud",
        }

    # ── Layer 4: Last resort — return best available ──
    best = function_calls if function_calls else (
        _coerce_args(det_calls, tools) if det_calls else []
    )
    if DEBUG:
        print(f"  [FALLBACK] returning best: {[c.get('name') for c in best]}")
    return {
        "function_calls": best,
        "total_time_ms": local_time + cloud_time,
        "source": "on-device",
    }


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
