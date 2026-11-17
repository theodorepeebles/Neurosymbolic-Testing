"""
Replay demo server: an animated page that replays the logic pipeline on 10 test puzzles.

A separate app from chatbox.py. It serves one page (replay.template.html) and the trace for
each puzzle as JSON. It never calls a model: the models' outputs for the 10 puzzles were saved
once by save_model_outputs.py, and everything after them (rules, source spans, Z3 solve,
answer-choice proofs) is rebuilt here by replay_trace.build_trace.

Usage:
    python logic/replay/replay_server.py                # http://127.0.0.1:8770, opens a browser
    python logic/replay/replay_server.py --no-browser

HOST and PORT env vars override the address (Cloud Run sets PORT; deploy.sh sets HOST=0.0.0.0).
Deploy with: bash logic/deploy.sh

Endpoints (GET):
    /                  the page, with the example list and the first trace embedded
    /api/examples      the 10 examples, for the picker
    /api/trace/<n>     the full trace for example n (1-10)
    /plots/<name>.png  a benchmark graph from data/plots, for the "Full results" tab
    /healthz           "ok"
"""

import gzip
import json
import os
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
TEMPLATE = os.path.join(HERE, "replay.template.html")
TEST_SET_PATH = os.path.join(DATA_DIR, "sft_test.jsonl")
SAVED_OUTPUTS_PATH = os.path.join(DATA_DIR, "demo_model_outputs.json")
# The benchmark graphs shown on the page's "Full results" tab (made by pipeline/plot_results.py).
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
PLOT_FILES = ({f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")}
              if os.path.isdir(PLOTS_DIR) else set())

from demo_examples import DEMO_PROBLEM_IDS
from replay_trace import build_trace

# THE switch for where model outputs come from.
#   "saved": data/demo_model_outputs.json, written once by save_model_outputs.py. No model calls.
#   "live":  not built yet. It would call the classifier and extractor here and hand their
#            output to the same build_trace.
MODEL_OUTPUTS = os.environ.get("REPLAY_MODEL_OUTPUTS", "saved")
TRACE_SOURCE = {"saved": "replay", "live": "live"}

# Z3 is not thread-safe, so traces are built one at a time and kept.
LOCK = threading.Lock()
_traces: dict[int, dict] = {}


def _load_json_field(value):
    return json.loads(value) if isinstance(value, str) else value


def load_test_rows() -> dict:
    """The demo puzzles' rows from sft_test.jsonl: problem text, gold domains, gold extraction."""
    wanted, rows = set(DEMO_PROBLEM_IDS), {}
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("run_id") in wanted:
                    rows[row["run_id"]] = row
    missing = wanted - set(rows)
    if missing:
        raise SystemExit(f"sft_test.jsonl is missing demo puzzles: {sorted(missing)}")
    return rows


def load_saved_outputs() -> dict:
    if not os.path.isfile(SAVED_OUTPUTS_PATH):
        raise SystemExit(f"{os.path.normpath(SAVED_OUTPUTS_PATH)} not found. "
                         f"Run save_model_outputs.py once to create it.")
    with open(SAVED_OUTPUTS_PATH, encoding="utf-8") as f:
        saved = {ex["problem_id"]: ex for ex in json.load(f)["examples"]}
    missing = set(DEMO_PROBLEM_IDS) - set(saved)
    if missing:
        raise SystemExit(f"demo_model_outputs.json is missing demo puzzles: {sorted(missing)}")
    return saved


TEST_ROWS = load_test_rows()
SAVED_OUTPUTS = load_saved_outputs() if MODEL_OUTPUTS == "saved" else {}


def get_model_outputs(problem_id: str, problem_text: str) -> dict:
    """What the two models produced for one puzzle:
    {"predicted_domains", "extraction_raw", "classifier": {meta}, "extraction": {meta}}."""
    if MODEL_OUTPUTS == "saved":
        return SAVED_OUTPUTS[problem_id]
    if MODEL_OUTPUTS == "live":
        raise NotImplementedError("live extraction is not built yet")
    raise ValueError(f"unknown REPLAY_MODEL_OUTPUTS: {MODEL_OUTPUTS!r}")


def trace_for(n: int) -> dict:
    """The trace for example n (1-based), built on first request and then reused."""
    with LOCK:
        if n not in _traces:
            problem_id = DEMO_PROBLEM_IDS[n - 1]
            row = TEST_ROWS[problem_id]
            outputs = get_model_outputs(problem_id, row["problem_text"])
            _traces[n] = build_trace(
                row["problem_text"],
                outputs["predicted_domains"],
                outputs["extraction_raw"],
                source=TRACE_SOURCE[MODEL_OUTPUTS],
                problem_id=problem_id,
                model_meta={"classifier": outputs.get("classifier"),
                            "extraction": outputs.get("extraction")},
                gold_domains=_load_json_field(row["active_domains"]),
                gold_extraction=_load_json_field(row["extracted_json"]),
            )
        return _traces[n]


def examples_summary() -> list[dict]:
    """One line per example for the picker. Builds every trace, so it also checks at startup
    that all 10 can be served."""
    out = []
    for n in range(1, len(DEMO_PROBLEM_IDS) + 1):
        trace = trace_for(n)
        stages = trace["stages"]
        out.append({
            "n": n,
            "problem_id": trace["problem_id"],
            "domains": stages["classification"]["gold_domains"],
            "num_rules": len(stages["extraction"]["rules"]),
            "num_choices": len(stages["extraction"]["choices"]),
            "correct": stages["choices"]["correct"],
        })
    return out


def render_page() -> bytes:
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    boot = {"examples": examples_summary(), "first": trace_for(1)}
    # "</" can't appear inside a <script> block; JSON allows the escaped form.
    boot_json = json.dumps(boot, ensure_ascii=False).replace("</", "<\\/")
    return html.replace("/*__BOOT__*/null", boot_json).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "ReplayDemo"

    def _send(self, body: bytes, content_type: str, code: int = 200, cache: str = "no-cache"):
        if (len(body) > 1024 and not content_type.startswith("image/")
                and "gzip" in (self.headers.get("Accept-Encoding") or "")):
            body = gzip.compress(body, compresslevel=6)
            encoding = "gzip"
        else:
            encoding = None
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'unsafe-inline'; "
                         "style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; "
                         "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj, code: int = 200, cache: str = "no-cache"):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", code, cache)

    def do_GET(self):  # noqa: N802 - http.server naming
        path = urlsplit(self.path).path
        try:
            if path == "/":
                self._send(render_page(), "text/html; charset=utf-8")
            elif path == "/api/examples":
                self._send_json(examples_summary())
            elif path.startswith("/api/trace/"):
                tail = path[len("/api/trace/"):]
                if tail.isdigit() and 1 <= int(tail) <= len(DEMO_PROBLEM_IDS):
                    self._send_json(trace_for(int(tail)))
                else:
                    self._send_json({"error": "no such example"}, 404)
            elif path.startswith("/plots/"):
                name = path[len("/plots/"):]
                if name in PLOT_FILES:   # only the file names found at startup, so no path tricks
                    with open(os.path.join(PLOTS_DIR, name), "rb") as f:
                        self._send(f.read(), "image/png", cache="public, max-age=3600")
                else:
                    self._send_json({"error": "not found"}, 404)
            elif path == "/healthz":
                self._send(b"ok", "text/plain; charset=utf-8")
            else:
                self._send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionError):
            pass   # the client went away
        except Exception as e:  # noqa: BLE001 - answer with JSON, keep serving
            print(f"error serving {path}: {type(e).__name__}: {e}", file=sys.stderr)
            try:
                self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)
            except OSError:
                pass

    do_HEAD = do_GET  # noqa: N815

    def log_message(self, *args):   # no per-request logging
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows, address reuse lets a second server bind a port that an old one still holds,
    # and the old one keeps answering. Turned off there so a second start fails loudly.
    allow_reuse_address = os.name != "nt"


def _find_port(start: int = 8770, tries: int = 25) -> int:
    """First port at/after `start` that nothing is listening on."""
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def main():
    if not os.path.isfile(TEMPLATE):
        sys.exit(f"Template not found: {TEMPLATE}")
    # Local default is 127.0.0.1; Cloud Run sets PORT and deploy.sh sets HOST=0.0.0.0.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ["PORT"]) if os.environ.get("PORT") else _find_port()

    examples = examples_summary()   # build all 10 traces now, so a broken data file fails here
    wrong = [str(e["n"]) for e in examples if not e["correct"]]
    print(f"{len(examples)} examples ready (model outputs: {MODEL_OUTPUTS}; "
          f"answered wrong: {', '.join(wrong) or 'none'})")

    try:
        server = Server((host, port), Handler)
    except OSError as e:
        sys.exit(f"Could not listen on {host}:{port} ({e}). Is another copy already running?")
    url = f"http://localhost:{port}/"
    print(f"Replay demo serving at {url}  (host {host}, Ctrl-C to stop)")
    if host == "127.0.0.1" and "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
