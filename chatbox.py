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
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

# The currently loaded puzzle. Single-user local tool, so one global session is enough; the lock
# serializes the heavy work (Z3 and Ollama aren't thread-safe) and guards SESSION mutation.
SESSION: dict = {}
LOCK = threading.Lock()


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
        "extracted_json": json.dumps(extracted_dict, indent=2, ensure_ascii=False),
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


def handle_ask(data: dict) -> dict:
    q = (data.get("question") or "").strip()
    if not q:
        return {"error": "empty question"}
    with LOCK:
        if not SESSION:
            return {"error": "load a problem first"}
        ctx, lp = SESSION["ctx"], SESSION["lp"]
        try:
            # Always route through the LLM — no raw-command / first-keyword shortcut.
            parsed = repl.parse_freetext(q, ctx, lp)
        except RuntimeError as e:     # Ollama unreachable / timeout
            return {"error": f"LLM error: {e}"}
        if parsed is repl.CANNOT_ANSWER:
            return {"cannot_answer": True}
        var, val, qtype = parsed
        try:
            prose, mcs = repl.run_query(
                ctx, SESSION["q_index"], SESSION["span_index"], SESSION["labels"], var, val, qtype
            )
        except Exception as e:  # noqa: BLE001 — surface engine errors as a chat bubble
            return {"error": f"{type(e).__name__}: {e}"}
        return {"interpreted": repl.format_structured(var, val, qtype), "prose": prose, "mcs": mcs}


ROUTES = {
    "/api/load_random": lambda data: handle_load_random(),
    "/api/extract": handle_extract,
    "/api/ask": handle_ask,
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
        route = ROUTES.get(self.path)
        if route is None:
            self.send_error(404)
            return
        try:
            self._send_json(route(self._read_json()))
        except Exception as e:  # noqa: BLE001 — always answer with JSON the UI can show
            self._send_json({"error": f"{type(e).__name__}: {e}"})

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
