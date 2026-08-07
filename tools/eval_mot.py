"""Public-dataset evaluation — how well do we count and track people?

Runs the engine over a sequence that ships MOT-format ground truth and reports
detection and counting accuracy. This answers the one question our own
regression cannot: when tiled scanning reads 28 people where single-pass read
18, which number was closer to the truth?

WHAT THIS CAN AND CANNOT VALIDATE
  can:    person detection (recall/precision), per-frame count error, and the
          unique-person count that is the denominator of every rate we publish.
  cannot: attention. No public pedestrian dataset labels "this person was
          looking at that surface", which is the metric we actually sell. Gaze
          accuracy still requires our own labelled footage (tools/labeler.html).

LICENCE DISCIPLINE (read before using any result)
  Research datasets (MOT17/MOT20, CrowdHuman, MERL Shopping and friends) are
  typically research-only / non-commercial. Numbers produced here are for
  ENGINE TUNING ONLY. They must not appear in ACCURACY.md, the audit report,
  the site, or investor material — a company selling audit trust cannot build
  its evidence file on a licence it is breaching. The published accuracy figure
  has to come from footage we have permission to use.

USAGE
  python tools/eval_mot.py --seq /path/to/MOT20-01 [--max-frames 300]

  Expects the standard layout:  <seq>/img1/000001.jpg ...  and  <seq>/gt/gt.txt
  (frame, id, x, y, w, h, conf, class, visibility)
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_gt(gt_file, min_vis=0.25):
    """MOT gt.txt -> {frame: [(id, x, y, w, h)]}. Heavily occluded boxes and
    non-pedestrian classes are skipped: we do not claim to count a person who
    is 90% hidden, so scoring against them would be measuring the wrong thing."""
    by_frame = defaultdict(list)
    ids = set()
    for line in Path(gt_file).read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        f, i = int(p[0]), int(p[1])
        x, y, w, h = (float(v) for v in p[2:6])
        conf = float(p[6]) if len(p) > 6 else 1.0
        cls = int(float(p[7])) if len(p) > 7 else 1
        vis = float(p[8]) if len(p) > 8 else 1.0
        if conf == 0 or cls != 1 or vis < min_vis:
            continue
        by_frame[f].append((i, x, y, w, h))
        ids.add(i)
    return by_frame, ids


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="MOT sequence dir (img1/ + gt/gt.txt)")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--stride", type=int, default=3, help="evaluate every Nth frame")
    ap.add_argument("--iou", type=float, default=0.4)
    a = ap.parse_args()

    import cv2
    from server.engine import AttentionEngine

    seq = Path(a.seq)
    gt_by_frame, gt_ids = load_gt(seq / "gt" / "gt.txt")
    frames = sorted((seq / "img1").glob("*.jpg"))[:a.max_frames]
    if not frames:
        sys.exit(f"no frames under {seq / 'img1'}")

    eng = AttentionEngine()
    from server.engine import SimpleTracker
    tracker = SimpleTracker()
    first = cv2.imread(str(frames[0]))
    H, W = first.shape[:2]

    tp = fp = fn = 0
    count_err = []
    our_ids = set()
    tiled = False
    for fp_img in frames[::a.stride]:
        idx = int(fp_img.stem)
        gt = gt_by_frame.get(idx, [])
        frame = cv2.imread(str(fp_img))
        if frame is None:
            continue
        tiled = eng._should_tile(len(gt), tiled, "auto", W, H)
        dets = eng._detect_frame(frame, tiled, tracker)
        for d in dets:
            our_ids.add(d["id"])
        used = set()
        for d in dets:
            best, bi = 0.0, None
            for k, (gid, *box) in enumerate(gt):
                if k in used:
                    continue
                v = iou(d["box"], tuple(box))
                if v > best:
                    best, bi = v, k
            if best >= a.iou:
                tp += 1
                used.add(bi)
            else:
                fp += 1
        fn += len(gt) - len(used)
        count_err.append(len(dets) - len(gt))

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    mae = sum(abs(e) for e in count_err) / max(len(count_err), 1)
    bias = sum(count_err) / max(len(count_err), 1)

    print(f"\nsequence      : {seq.name}  ({len(frames[::a.stride])} frames scored)")
    print(f"detection     : precision {prec*100:.1f}%  recall {rec*100:.1f}%  F1 {f1*100:.1f}%")
    tag = "over-counting" if bias > 0.05 else "under-counting" if bias < -0.05 else "balanced"
    print(f"per-frame count: MAE {mae:.2f}   bias {bias:+.2f} ({tag})")
    print(f"unique people : ours {len(our_ids)}  vs ground truth {len(gt_ids)}  "
          f"({(len(our_ids) - len(gt_ids)) / max(len(gt_ids), 1) * 100:+.0f}%)")
    print("\nreminder: research-dataset numbers are for tuning only — they do not "
          "go into ACCURACY.md or any customer-facing material (see module docstring).")


if __name__ == "__main__":
    main()
