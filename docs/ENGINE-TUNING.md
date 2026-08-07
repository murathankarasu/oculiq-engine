# Engine tuning notes — public-dataset runs

**This file is not ACCURACY.md and must never be used as one.** Everything here
comes from research-licensed datasets (non-commercial). It exists to steer
engineering decisions, not to support a claim to a customer, on the site, or in
investor material. The published accuracy figure has to come from footage we
have permission to use.

---

## MOT20-01 — counting and detection in a dense crowd

- Source: `Lekim89/MOT20` mirror on HuggingFace, licence **CC BY-NC-SA 3.0**
- Scored: 12 frames (every 10th of the first 120), IoU ≥ 0.4, ground-truth
  boxes below 25% visibility excluded
- Scene: outdoor, very dense pedestrian flow — far harder than a shop interior

| Metric | Result |
|---|---|
| Detection precision | 78.4% |
| Detection recall | 82.3% |
| F1 | 80.3% |
| Per-frame count MAE | 2.75 people |
| Per-frame count bias | **+1.58 (over-counts per frame)** |
| Unique identities, on the scored frames | 56 vs 55 — **+2%** |

### What this tells us

- **Per frame we produce slightly too many boxes** (+1.58). In a dense crowd
  some detections are duplicates or partial bodies. Recall 82% means we still
  miss roughly one person in six at this density.
- **Identity count is close to correct** (+2%). The extra per-frame boxes are
  mostly short-lived, and the ghost filter removes them before they become
  people.

### A correction we made to this page

The first version of this run reported "−14%, under-counts" and concluded that
the widestore regression's higher figure (28 unique people under tiled scanning
vs 18 single-pass) was therefore the more accurate one.

**That was wrong, and the error was in our own tool.** It compared the
identities we found on 12 sampled frames against every identity in the full
120-frame ground truth — a comparison that manufactures an apparent shortfall.
Scored fairly, the count is +2%.

So the widestore question stands open: this dataset gives no evidence either
way on whether 18 or 28 is closer. The regression band remains wide because we
do not know, not because we measured that we should.

### Can we raise recall? Measured, and the answer is no — not by tuning

Recall of 82% means one person in six is missed at this density, so we swept the
parameters that plausibly control it:

| Setting | Precision | Recall | F1 |
|---|---|---|---|
| conf 0.25, imgsz 960, tile 800 *(current)* | 78.4% | 82.3% | 80.3% |
| conf 0.15, imgsz 960, tile 800 | 78.4% | 82.3% | 80.3% |
| conf 0.25, imgsz 1280, tile 800 | 78.5% | 82.5% | 80.5% |
| conf 0.15, imgsz 1280, tile 640 | 78.5% | 82.5% | 80.5% |

Nothing moves. Lowering the detection threshold changes nothing at all;
doubling resolution buys 0.2 points. **The bottleneck is not the settings**, so
no parameter change was made — a config churn that buys 0.2 points on one street
clip is noise dressed as progress.

### What we actually miss

| | Detected | Missed |
|---|---|---|
| Median visibility | 0.80 | **0.46** |
| Median box height | 192 px | 122 px |

- **57% of the misses are heavily occluded** (under 50% visible).
- **0% are too small** — size is not the problem at all.

So the missing sixth is mostly people standing behind other people. That matters
less than the raw number suggests: a person whose body is half hidden has no
readable head or shoulder direction either, so they could not contribute an
attention measurement even if we detected them. The honest reading is that we
are near the practical ceiling for this density, not that we have a fixable
detector gap. A retail interior — sparser than MOT20 — should sit better.

### Limits of this run, stated plainly

1. **12 scored frames is a small sample.** Treat the percentages as indicative,
   not as a measurement.
2. **MOT20-01 is a street scene**, not retail. Occlusion, density and camera
   geometry all differ from a shop; the numbers do not transfer directly.
3. **This says nothing about attention.** No public pedestrian dataset labels
   "this person looked at that surface", which is the metric we sell. Counting
   accuracy is the denominator of our rates, not the claim itself.

### Reproduce

```
python tools/eval_mot.py --seq /path/to/MOT20-01 --max-frames 120 --stride 10
```
