#!/usr/bin/env python3
"""Oculiq regression suite — para birimi disiplini (Spec v1.0 §10.4).

Runs the engine on fixed reference clips and fails if any metric drifts out of
its expected range. Run before merging ANY engine change:

    .venv/bin/python tools/regress.py            # all cases
    .venv/bin/python tools/regress.py --case fx1 # one case

Setup: put clips under tools/regress_data/ and describe them in
tools/regress_data/cases.json:

[
  {
    "name": "fx1-near-entrance",
    "clip": "fx1.mp4",
    "zones": [{"id": 0, "label": "Poster", "type": "billboard",
               "x": 0.1, "y": 0.2, "w": 0.2, "h": 0.3},
              {"id": 1, "label": "Door", "type": "line",
               "x": 0, "y": 0, "w": 0.01, "h": 0.01,
               "line": [[0.4, 0.9], [0.6, 0.7]]}],
    "sample_fps": 10, "max_seconds": 30,
    "expect": {
      "traffic": [4, 6],
      "zones.0.attention_rate": [30, 70],
      "zones.0.impressions": [2, 4],
      "lines.0.enters": [2, 4]
    }
  }
]

Expectation keys are dotted paths into the report; list index by position for
"zones"/"lines". Ranges are inclusive [lo, hi]."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tools" / "regress_data"
sys.path.insert(0, str(ROOT))


def dig(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="run a single case by name prefix")
    a = ap.parse_args()

    cases_f = DATA / "cases.json"
    if not cases_f.exists():
        sys.exit(f"no cases yet — create {cases_f} (see module docstring). "
                 "This suite is the currency discipline: add the 3 reference "
                 "clips before the first pilot ships.")
    cases = json.loads(cases_f.read_text())
    if a.case:
        cases = [c for c in cases if c["name"].startswith(a.case)]
    if not cases:
        sys.exit("no matching case")

    from server.engine import AttentionEngine
    eng = AttentionEngine()
    failures = 0
    for c in cases:
        clip = DATA / c["clip"]
        print(f"\n=== {c['name']} ({clip.name}) ===")
        job = {"progress": 0, "status": "processing"}   # engine'in beklediği kova
        report = eng.process_video(
            clip, c["zones"], job, cost_map={},
            sample_fps=c.get("sample_fps", 10),
            max_seconds=c.get("max_seconds"),
            crowd_mode=c.get("crowd_mode", "auto"),
            demographics=False, face_blur=True)
        for path, (lo, hi) in c["expect"].items():
            try:
                v = dig(report, path)
            except Exception:
                print(f"  FAIL {path}: missing in report")
                failures += 1
                continue
            ok = isinstance(v, (int, float)) and lo <= v <= hi
            print(f"  {'ok  ' if ok else 'FAIL'} {path} = {v}  (expect {lo}–{hi})")
            failures += 0 if ok else 1

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
