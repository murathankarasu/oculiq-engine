#!/usr/bin/env python3
"""Oculiq ground-truth evaluation (Spec v1.0).

Compares an engine report against human labels from tools/labeler.html and
produces the published error-margin table (docs/ACCURACY.md).

Usage:
    python3 tools/evaluate.py --report jobs/<id>/report.json --labels labels.json
    python3 tools/evaluate.py --report r1.json --labels l1.json \
                              --report r2.json --labels l2.json --write

Multiple --report/--labels pairs are pooled (report N pairs with labels N).
--write appends/refreshes docs/ACCURACY.md; without it, results print only.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_pair(report_path, labels_path):
    rep = json.loads(Path(report_path).read_text())
    lab = json.loads(Path(labels_path).read_text())
    min_dwell = (rep.get("sim") or {}).get("min_dwell", 0.4)
    dwell = (rep.get("sim") or {}).get("dwell", {})
    rows = []
    for L in lab.get("labels", []):
        eng_dwell = float(dwell.get(str(L["pid"]), {}).get(str(L["zid"]), 0.0))
        rows.append({
            "human_looked": bool(L["looked"]),
            "engine_looked": eng_dwell >= min_dwell,
            "human_dwell": L.get("dwell"),
            "engine_dwell": eng_dwell,
        })
    traffic = (rep.get("traffic"), lab.get("human_traffic"))
    return rows, traffic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", required=True)
    ap.add_argument("--labels", action="append", required=True)
    ap.add_argument("--write", action="store_true",
                    help="write docs/ACCURACY.md (otherwise print only)")
    a = ap.parse_args()
    if len(a.report) != len(a.labels):
        sys.exit("each --report needs a matching --labels")

    rows, traffics = [], []
    for r, l in zip(a.report, a.labels):
        rr, tt = load_pair(r, l)
        rows += rr
        if tt[1]:
            traffics.append(tt)

    if not rows:
        sys.exit("no labeled (person, zone) pairs found")

    tp = sum(1 for r in rows if r["engine_looked"] and r["human_looked"])
    fp = sum(1 for r in rows if r["engine_looked"] and not r["human_looked"])
    fn = sum(1 for r in rows if not r["engine_looked"] and r["human_looked"])
    tn = sum(1 for r in rows if not r["engine_looked"] and not r["human_looked"])
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None

    dpairs = [(r["human_dwell"], r["engine_dwell"]) for r in rows
              if r["human_dwell"] is not None and r["human_looked"] and r["engine_looked"]]
    dwell_mae = (sum(abs(h - e) for h, e in dpairs) / len(dpairs)) if dpairs else None

    tr_errs = [abs(e - h) / h * 100 for e, h in traffics if h]
    traffic_mape = sum(tr_errs) / len(tr_errs) if tr_errs else None

    fmt = lambda v, s="": ("—" if v is None else f"{v:.1%}" if s == "%" else f"{v:.2f}{s}")
    lines = [
        f"# Oculiq Accuracy — measured against human ground truth",
        "",
        f"*Spec v1.0 · evaluated {date.today().isoformat()} · "
        f"{len(rows)} labeled (person × zone) pairs across {len(a.report)} clip(s)*",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Look detection — precision | {fmt(prec, '%')} |",
        f"| Look detection — recall | {fmt(rec, '%')} |",
        f"| Look detection — F1 | {fmt(f1, '%')} |",
        f"| Confusion (TP/FP/FN/TN) | {tp}/{fp}/{fn}/{tn} |",
        f"| Dwell MAE (agreed lookers) | {fmt(dwell_mae, ' s')} |",
        f"| Traffic count MAPE | {fmt(traffic_mape / 100 if traffic_mape is not None else None, '%')} |",
        "",
        "Method: blind human labeling on annotated footage (tools/labeler.html), "
        "engine answer hidden during labeling. 'Looked' threshold = Spec v1.0 "
        "min_dwell. These figures ship with every audit report; when the engine "
        "changes, this table is regenerated before release (tools/regress.py).",
    ]
    out = "\n".join(lines)
    print(out)
    if a.write:
        p = ROOT / "docs" / "ACCURACY.md"
        p.write_text(out + "\n")
        print(f"\nwritten → {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
