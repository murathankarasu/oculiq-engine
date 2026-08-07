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
| Unique people over the clip | 56 vs 65 ground truth — **−14% (under-counts)** |

### What this tells us

The two count errors point in opposite directions, and that is the useful part:

- **Per frame we produce slightly too many boxes** (+1.58). In a dense crowd
  some detections are duplicates or partial bodies.
- **Across the clip we end up with too few distinct identities** (−14%). People
  who leave and re-enter, or who are occluded for a while, are being merged or
  lost rather than counted as they should be.

This bears directly on the open question from the widestore regression, where
tiled scanning read 28 unique people against 18 for single-pass. The engine's
bias on dense footage is to **under**-count unique people, so the higher figure
is more likely to be the closer one — the wider regression band was the right
call, and the concern that tiling merely inflates identities is not supported
here.

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
