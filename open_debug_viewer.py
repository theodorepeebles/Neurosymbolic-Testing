"""
Open the NS extraction failure debug viewer.

Embeds *every* profiles_*.json in debug_reports/ (newest-first) into a standalone HTML
generated from debug_viewer.template.html, then opens it in the default browser. No web
server is started — the generated file is fully self-contained and offline.

Usage:
    python open_debug_viewer.py

Run analyze.py first to produce a report. Drop additional profiles_*.json onto the viewer
window to load reports from elsewhere.
"""

import glob
import json
import os
import sys
import webbrowser
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(HERE, "debug_reports")
TEMPLATE = os.path.join(HERE, "debug_viewer.template.html")
PLACEHOLDER = "__EMBEDDED_REPORTS__"
OUTPUT = os.path.join(REPORTS_DIR, "debug_viewer.html")


def _escape_for_script(text: str) -> str:
    """Make a JSON string safe to embed inside a <script type="application/json"> block.

    Escaping '<' (and '>' & '&') ensures a literal '</script>' in untrusted model output
    cannot break out of the tag. U+2028/U+2029 are escaped because, although valid in JSON,
    they are line terminators in HTML/JS contexts.
    """
    return (text.replace("&", "\\u0026")
                .replace("<", "\\u003c")
                .replace(">", "\\u003e")
                .replace(chr(0x2028), "\\u2028")
                .replace(chr(0x2029), "\\u2029"))


def load_reports():
    """Parse all profiles_*.json, attach filename, return newest-first."""
    out = []
    for path in glob.glob(os.path.join(REPORTS_DIR, "profiles_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  ! skipping unreadable report {os.path.basename(path)}: {e}")
            continue
        if not isinstance(obj, dict) or obj.get("kind") != "ns_failure_report":
            print(f"  ! skipping {os.path.basename(path)}: not an ns_failure_report")
            continue
        obj["filename"] = os.path.basename(path)
        out.append(obj)
    out.sort(key=lambda r: (str(r.get("generated_at") or ""), r["filename"]), reverse=True)
    return out


def main():
    if not os.path.isfile(TEMPLATE):
        sys.exit(f"Template not found: {TEMPLATE}")
    if not os.path.isdir(REPORTS_DIR):
        sys.exit(f"No reports directory yet ({REPORTS_DIR}). Run analyze.py first.")

    reports = load_reports()
    if not reports:
        sys.exit(f"No profiles_*.json reports found in {REPORTS_DIR}. Run analyze.py first.")

    payload = _escape_for_script(json.dumps(reports, ensure_ascii=False, separators=(",", ":")))

    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    if PLACEHOLDER not in html:
        sys.exit(f"Template is missing the {PLACEHOLDER} placeholder.")
    html = html.replace(PLACEHOLDER, payload)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Embedded {len(reports)} report(s); newest: {reports[0]['filename']}")
    print(f"Wrote {OUTPUT}")
    webbrowser.open(Path(OUTPUT).resolve().as_uri())


if __name__ == "__main__":
    main()
