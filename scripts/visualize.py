"""Launch the interactive web visualizer for any mission run.

Usage:
  python scripts/visualize.py runs/mission_easy
  python scripts/visualize.py runs/mission_decoys --browser
  python scripts/visualize.py runs/mission_decoys --serve --port 8000
"""
import argparse
import http.server
import json
import socketserver
import sys
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from walle.html_viz import HTML_TEMPLATE, _default  # noqa: E402


def generate_html_from_run_dir(run_dir: Path) -> Path:
    out_html = run_dir / "visualizer.html"
    if out_html.exists():
        return out_html

    report_path = run_dir / "report.json"
    trace_path = run_dir / "trace.jsonl"
    belief_path = run_dir / "belief.npz"

    if not report_path.exists() or not trace_path.exists():
        sys.exit(f"Error: {run_dir} must contain report.json and trace.jsonl")

    rep = json.loads(report_path.read_text())
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]

    snaps = []
    if belief_path.exists():
        npz = np.load(belief_path)
        snaps = [[round(float(v), 4) for v in s.ravel()] for s in npz["belief"]]

    path = [r["before"]["pos"] for r in trace]
    if trace:
        path.append(trace[-1]["observation"]["pos"])
    strategies = [r["strategy"] for r in trace]
    entropy = [round(float(r["belief"]["entropy"]), 4) for r in trace]
    battery = [round(float(r["before"]["battery"]), 2) for r in trace]

    # Ground truth from report
    eval_only = rep.get("evaluation_only", {})
    ground_truth = {
        "eve_pos": eval_only.get("eve_position"),
        "eve_status": eval_only.get("eve_status"),
        "decoys": eval_only.get("decoy_signal_sources", []),
        "walls": [],
    }

    # Dimensions from first trace or 21x21 default
    w, h = 21, 21
    if "resources" in rep and "stats" in rep["resources"]:
        pass

    payload = {
        "report": rep,
        "trace": trace,
        "grid_width": w,
        "grid_height": h,
        "stations": [rep["evidence"]["initial_signal"]["origin"]],
        "snapshots": snaps,
        "path": path,
        "strategies": strategies,
        "entropy": entropy,
        "battery": battery,
        "ground_truth": ground_truth,
    }

    html = HTML_TEMPLATE.replace("DATA_PAYLOAD_PLACEHOLDER", json.dumps(payload, default=_default))
    out_html.write_text(html, encoding="utf-8")
    print(f"Generated {out_html}")
    return out_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", default="runs/mission_easy", nargs="?", help="Mission output directory")
    parser.add_argument("--browser", "-b", action="store_true", default=True, help="Open in default browser")
    parser.add_argument("--no-browser", action="store_false", dest="browser", help="Do not open browser")
    parser.add_argument("--serve", "-s", action="store_true", help="Start local web server")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Web server port")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"Directory not found: {run_dir}")

    html_file = generate_html_from_run_dir(run_dir)

    if args.serve:
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(run_dir), **kw)

        url = f"http://localhost:{args.port}/visualizer.html"
        print(f"\nServing {run_dir} at: {url}")
        if args.browser:
            webbrowser.open(url)
        with socketserver.TCPServer(("", args.port), Handler) as httpd:
            try:
                print("Press Ctrl+C to stop.")
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")
    else:
        abs_uri = html_file.resolve().as_uri()
        print(f"\nVisualizer ready at: {abs_uri}")
        if args.browser:
            webbrowser.open(abs_uri)


if __name__ == "__main__":
    main()
