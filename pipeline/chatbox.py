"""
Chatbox — a sleek browser chat UI over the logic-puzzle REPL.

A nicer front-end for the Phase-2 free-text REPL (logic/repl.py): instead of the terminal, it
opens a Claude-style chat window in your browser. One puzzle is loaded at a time; you ask
plain-English questions about it and each question is mapped (independently) to a whynot/can/forces
query and answered by the existing Z3 + verbalization path. The left panel shows the puzzle, its
variables, the extracted JSON, and the Z3-derived answer (like open_debug_viewer.py).

This is a thin server: a stdlib http.server that serves chatbox.template.html and a small JSON API.
It reuses the REPL/pipeline functions verbatim — nothing downstream changes. Needs Ollama running
with: the base model (repl.FREETEXT_MODEL = qwen3:8b) for domain classification + free-text
question mapping, and the fine-tuned extractor (EXTRACTION_MODEL = SFT_Extraction_Qwen3_0.6b-v4)
for turning a pasted puzzle into structured JSON.

Usage:
    python logic/chatbox.py          # opens http://localhost:8765 in your default browser

Endpoints (POST, JSON):
    /api/load_random   pick a random stored puzzle (gold extraction) and build a session
    /api/extract       classify + extract a pasted puzzle, then build a session
    /api/ask           map one question against the loaded puzzle and answer it
"""

import json
import os
import re
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

import repl
import explanation
from pipeline import classify_domains, extract_logic_problem, z3_solve, format_constraint
from validators import build_hybrid_schema
from explanation_debug import fetch_run, _loads, pick_random_run_id, DEFAULT_DB

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "chatbox.template.html")

# Pasted-puzzle extraction uses the fine-tuned extractor (minimal FT prompt, raw JSON under a
# grammar), not the generic base model. The base model still classifies domains and answers
# free-text questions (repl.FREETEXT_MODEL). Mirror run.py's MODEL_SETS naming.
EXTRACTION_MODEL = "SFT_Extraction_Qwen3_0.6b-v4"
EXTRACTION_FINETUNED = True
CLASSIFIER_MODEL = repl.FREETEXT_MODEL  # qwen3:8b

# Live thinking trace. The mapping runs as ONE grammar-constrained call whose schema has a
# `reasoning` field FIRST — the model reasons, then emits the answer fields as the continuation of
# that same generation (so the trace faithfully explains the selected answer, and the answer fields
# stay enum-locked so validity is unchanged). We stream that call and peel out the reasoning field.
# Set False to skip the trace and use the plain constrained repl.parse_freetext.
SHOW_THINKING = True
OLLAMA_URL = "http://localhost:11434/api/generate"
THINKING_SYSTEM_SUFFIX = (
    "\n\nFirst use the \"reasoning\" field to briefly think through which variable, value, and "
    "query_type the question maps to; then fill query_type, variable, and value."
)

# The currently loaded puzzle. Single-user local tool, so one global session is enough; the lock
# serializes the heavy work (Z3 and Ollama aren't thread-safe) and guards SESSION mutation.
SESSION: dict = {}
LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Readable extraction rendering (YAML-ish, like open_debug_viewer)
# ─────────────────────────────────────────────────────────────────────────────

def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _yaml_lines(obj, indent: int) -> list[str]:
    """Recursive YAML-ish renderer. dict keys with None values are skipped (e.g. absent
    evidence_text) to keep nested constraints clean, matching the debug viewer."""
    pad = " " * indent
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:")
                out += _yaml_lines(v, indent + 2)
            elif isinstance(v, list):
                out.append(f"{pad}{k}: []")
            elif isinstance(v, dict):
                out.append(f"{pad}{k}: {{}}")
            else:
                out.append(f"{pad}{k}: {_scalar(v)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and item:
                sub = _yaml_lines(item, indent + 2)
                out.append(pad + "- " + sub[0].lstrip())   # first field on the "- " line
                out += sub[1:]
            elif isinstance(item, (dict, list)):
                out.append(f"{pad}- " + ("[]" if isinstance(item, list) else "{}"))
            else:
                out.append(f"{pad}- {_scalar(item)}")
    return out


def format_extraction_readable(d: dict) -> str:
    """Render the extracted LogicProblem dict the way open_debug_viewer shows it:
    UPPERCASE section headers (with a count for lists) + indented YAML bodies."""
    order = ["entities", "constraints", "questions", "num_groups", "num_slots"]
    keys = [k for k in order if k in d] + [k for k in d if k not in order]
    blocks = []
    for k in keys:
        v = d[k]
        if isinstance(v, list):
            header = f"{k.upper()}  ({len(v)})"
            if not v:
                body = "  (none)"
            elif all(not isinstance(x, (dict, list)) for x in v):
                body = "\n".join(f"  - {_scalar(x)}" for x in v)          # entities etc.
            else:
                body = "\n\n".join("\n".join(_yaml_lines([x], 0)) for x in v)  # constraints/questions
        elif isinstance(v, dict):
            header = k.upper()
            body = "\n".join(_yaml_lines(v, 2)) or "  {}"
        else:
            header = k.upper()
            body = f"  {_scalar(v)}"
        blocks.append(header + "\n" + body)
    return "\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Session building (reuses repl / pipeline / explanation)
# ─────────────────────────────────────────────────────────────────────────────

def _variables_payload(ctx, lp) -> dict:
    """Variable names (exact Z3 tokens) + domain sizes, for the UI's variable chips."""
    return {
        "names": sorted(ctx.zvars),
        "num_slots": getattr(lp, "num_slots", None),
        "num_groups": getattr(lp, "num_groups", None),
    }


def _answer_payload(lp) -> dict:
    """Derive the answer by Z3-solving the extraction (same idea as run.py.derive_ground_truth).

    Returns the first question's answer choices each tagged verified/not, or the UNSAT core, so the
    insights panel can show 'the answer' the way open_debug_viewer does.
    """
    res = z3_solve(lp)
    status = res.get("status")
    out = {"status": status, "choices": [], "unsat_core": res.get("unsat_core", []) or []}
    if status != "sat":
        return out
    qresults = res.get("question_results") or []
    verified_map = qresults[0] if qresults else {}
    questions = getattr(lp, "questions", None) or []
    if questions:
        for ch in questions[0].answer_choices:
            constraints = " AND ".join(format_constraint(c) for c in ch.constraints) or "(no constraints)"
            out["choices"].append({
                "label": ch.label,
                "type": ch.type,
                "constraints": constraints,
                "verified": bool(verified_map.get(ch.label, False)),
            })
    return out


def build_session(lp, problem_text: str, active_domains, extracted_dict: dict, source: str,
                  run_id=None) -> dict:
    """Prepare a query context for `lp`, stash it in SESSION, and return the UI payload."""
    ctx, span_index = explanation.prepare_query_context(lp, problem_text)
    SESSION.clear()
    SESSION.update({
        "run_id": run_id,
        "lp": lp,
        "problem_text": problem_text,
        "ctx": ctx,
        "span_index": span_index,
        "labels": ctx.cid_label,
        "q_index": 0,
        "active_domains": active_domains,
        "source": source,
    })
    return {
        "ok": True,
        "run_id": run_id,
        "source": source,
        "problem_text": problem_text,
        "active_domains": active_domains,
        "variables": _variables_payload(ctx, lp),
        "extracted_json": format_extraction_readable(extracted_dict),
        "answer": _answer_payload(lp),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Request handlers (each returns a JSON-able dict; never raises to the socket)
# ─────────────────────────────────────────────────────────────────────────────

def handle_load_random() -> dict:
    with LOCK:
        try:
            run_id = pick_random_run_id(DEFAULT_DB)
            row = fetch_run(DEFAULT_DB, run_id)
        except SystemExit as e:  # the DB helpers sys.exit on missing DB / empty table
            return {"error": str(e)}
        active_domains = _loads(row["active_domains"])
        gold = _loads(row["expected_json"])
        if gold is None:
            return {"error": f"run {run_id} has no gold extraction"}
        try:
            lp = build_hybrid_schema(active_domains)(**gold)
            return build_session(lp, row["problem_text"], active_domains, gold, "dataset", run_id)
        except Exception as e:  # noqa: BLE001 — surface, don't crash the server
            return {"error": f"{type(e).__name__}: {e}"}


def handle_extract(data: dict) -> dict:
    text = (data.get("problem_text") or "").strip()
    if not text:
        return {"error": "paste a puzzle first"}
    with LOCK:
        try:
            active_domains = classify_domains(text, model=CLASSIFIER_MODEL)
            LogicProblem = build_hybrid_schema(active_domains)
            extracted, _unmatched, _attempts, raw = extract_logic_problem(
                text, active_domains, LogicProblem,
                model=EXTRACTION_MODEL, finetuned=EXTRACTION_FINETUNED,
            )
        except RuntimeError as e:  # Ollama unreachable / timeout
            return {"error": f"LLM error: {e}"}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
        if extracted is None:
            return {"error": "extraction failed (model did not produce valid JSON)", "raw": raw}
        try:
            return build_session(extracted, text, active_domains, extracted.model_dump(),
                                 "pasted", None)
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}


def _reasoning_schema(ctx) -> dict:
    """Mapping schema with a `reasoning` string FIRST, so the model reasons then answers in one
    grammar-constrained generation. The answer fields carry the same enums as repl._freetext_schema,
    so structural validity is identical to the non-thinking path."""
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "query_type": {"type": "string", "enum": list(repl.QUERY_TYPES) + ["cannot_answer"]},
            "variable": {"type": "string", "enum": sorted(ctx.zvars)},
            "value": {"type": ["string", "null"]},
        },
        "required": ["reasoning", "query_type", "variable", "value"],
    }


def _parse_mapping(raw: str, ctx, lp):
    """Validate the finished mapping JSON like repl.parse_freetext. Returns (var,val,qtype) or
    repl.CANNOT_ANSWER on any failure. The extra `reasoning` field is ignored here."""
    try:
        cleaned = re.sub(r"^```(json)?\s*|```$", "", raw.strip(), flags=re.M).strip()
        obj = json.loads(cleaned)
        qtype, var = obj["query_type"], obj["variable"]
        if qtype not in repl.QUERY_TYPES or var not in ctx.zvars:
            return repl.CANNOT_ANSWER
        if qtype == "forces":
            return var, None, qtype
        rawval = obj.get("value")
        if rawval is None:
            return repl.CANNOT_ANSWER
        return var, repl.parse_value(var, str(rawval), lp), qtype
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return repl.CANNOT_ANSWER


_REASONING_OPEN = re.compile(r'"reasoning"\s*:\s*"')


def _find_str_end(buf: str, start: int) -> int:
    """Index of the unescaped closing quote of a JSON string that begins at `start`, or -1."""
    i, esc = start, False
    while i < len(buf):
        c = buf[i]
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            return i
        i += 1
    return -1


def _decode_partial_jsonstr(content: str):
    """json-decode JSON-string CONTENT (no surrounding quotes) that may end mid-escape.
    Trims a dangling escape so it parses; returns the decoded text or None."""
    s = content
    if (len(s) - len(s.rstrip("\\"))) % 2 == 1:   # odd trailing backslashes -> incomplete escape
        s = s[:-1]
    m = re.search(r"\\u[0-9a-fA-F]{0,3}$", s)      # incomplete \uXXXX at the end
    if m:
        s = s[:m.start()]
    try:
        return json.loads('"' + s + '"')
    except ValueError:
        return None


def stream_ask(question: str, ctx, lp, emit_thinking):
    """ONE streamed, grammar-constrained call. Peels the `reasoning` field out of the token stream
    (calling emit_thinking with decoded deltas) and returns the validated mapping
    (var,val,qtype)|repl.CANNOT_ANSWER. Raises RuntimeError if Ollama is unreachable."""
    body = {
        "model": repl.FREETEXT_MODEL,
        "system": repl.FREETEXT_REPL_SYSTEM + THINKING_SYSTEM_SUFFIX,
        "prompt": repl.build_freetext_repl_prompt(SESSION["problem_text"], repl.list_variables(ctx, lp), question),
        "stream": True, "think": False, "format": _reasoning_schema(ctx),
        "options": {"temperature": 0, "top_k": 1, "num_ctx": 4096, "num_predict": 1024},
    }
    buf, r_start, emitted, closed = "", None, 0, False
    try:
        resp = requests.post(OLLAMA_URL, json=body, timeout=180, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            buf += chunk.get("response", "")
            if not closed:
                if r_start is None:
                    m = _REASONING_OPEN.search(buf)
                    if m:
                        r_start = m.end()
                if r_start is not None:
                    end = _find_str_end(buf, r_start)
                    decoded = _decode_partial_jsonstr(buf[r_start:end] if end != -1 else buf[r_start:])
                    if decoded is not None and len(decoded) > emitted:
                        emit_thinking(decoded[emitted:])
                        emitted = len(decoded)
                    if end != -1:
                        closed = True
            if chunk.get("done"):
                break
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    return _parse_mapping(buf, ctx, lp)


ROUTES = {
    "/api/load_random": lambda data: handle_load_random(),
    "/api/extract": handle_extract,
}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP plumbing
# ─────────────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def do_GET(self):  # noqa: N802 — http.server naming
        if self.path == "/" or self.path.startswith("/?"):
            try:
                with open(TEMPLATE, encoding="utf-8") as f:
                    body = f.read().encode("utf-8")
            except OSError as e:
                self.send_error(500, f"template missing: {e}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        if self.path == "/api/ask":
            self._handle_ask_stream()
            return
        route = ROUTES.get(self.path)
        if route is None:
            self.send_error(404)
            return
        try:
            self._send_json(route(self._read_json()))
        except Exception as e:  # noqa: BLE001 — always answer with JSON the UI can show
            self._send_json({"error": f"{type(e).__name__}: {e}"})

    def _sse(self, obj: dict) -> None:
        """Write one Server-Sent Event; raise BrokenPipeError if the client has gone."""
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _handle_ask_stream(self):
        """Stream one question over SSE. mode=="template" uses the deterministic REPL grammar
        (repl.parse_command, no LLM); mode=="llm" (default) uses the reasoning-streamed mapping.
        Either way the answer comes from the same query path, so validity is unchanged."""
        data = self._read_json()
        mode = (data.get("mode") or "llm").lower()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            q = (data.get("question") or "").strip()
            if not q:
                self._sse({"error": "empty question"}); self._sse({"done": True}); return
            with LOCK:
                if not SESSION:
                    self._sse({"error": "load a problem first"}); self._sse({"done": True}); return
                ctx, lp = SESSION["ctx"], SESSION["lp"]
                try:
                    if mode == "template":
                        parsed = repl.parse_command(q, ctx, lp)      # deterministic grammar, no LLM
                        if parsed is None:
                            self._sse({"error": "Not a command. Use: whynot <var> <val>, "
                                                "can <var> <val>, or forces <var>."})
                            self._sse({"done": True}); return
                    elif SHOW_THINKING:
                        parsed = stream_ask(q, ctx, lp, emit_thinking=lambda d: self._sse({"thinking_delta": d}))
                    else:
                        parsed = repl.parse_freetext(q, ctx, lp)
                except ValueError as e:            # malformed structured command
                    self._sse({"error": str(e)}); self._sse({"done": True}); return
                except RuntimeError as e:          # Ollama unreachable / timeout
                    self._sse({"error": f"LLM error: {e}"}); self._sse({"done": True}); return
                if parsed is repl.CANNOT_ANSWER:
                    self._sse({"cannot_answer": True}); self._sse({"done": True}); return
                var, val, qtype = parsed
                try:
                    prose, mcs = repl.run_query(
                        ctx, SESSION["q_index"], SESSION["span_index"], SESSION["labels"], var, val, qtype
                    )
                except Exception as e:  # noqa: BLE001 — surface engine errors as a chat bubble
                    self._sse({"error": f"{type(e).__name__}: {e}"}); self._sse({"done": True}); return
                payload = {"prose": prose, "mcs": mcs}
                if mode != "template":   # in template mode the user typed the structured form already
                    payload["interpreted"] = repl.format_structured(var, val, qtype)
                self._sse(payload)
                self._sse({"done": True})
        except (BrokenPipeError, ConnectionError, OSError):
            return  # client closed the stream — stop quietly

    def log_message(self, *args):  # silence the default per-request stderr logging
        pass


def _find_port(start: int = 8765, tries: int = 25) -> int:
    """First port at/after `start` that nothing is listening on."""
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def main():
    if not os.path.isfile(TEMPLATE):
        sys.exit(f"Template not found: {TEMPLATE}")
    port = _find_port()
    url = f"http://localhost:{port}/"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    print(f"NS Chatbox serving at {url}  (Ctrl-C to stop)")
    print(f"Model: {repl.FREETEXT_MODEL}   DB: {os.path.normpath(DEFAULT_DB)}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
