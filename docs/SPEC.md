# Oculiq Measurement Spec — v1.0

**Status:** frozen. Any change to a definition, threshold or formula below is a
spec revision (v1.1, v2.0 …) and must be noted in reports. Reports carry
`"spec": "1.0"`.

This document is the public definition of every number Oculiq reports.
The engine implements it in `server/engine.py`; where a value is configurable,
the **spec value is the default** and any deviation must be disclosed in the
report's methodology section.

---

## 1. What is measured

Oculiq measures **head-orientation attention**, not eye movement, and is never
marketed as eye-tracking. Facial keypoints (nose vs eyes/ears; shoulder
fallback) yield a per-person viewing direction; a **look** is registered when
that direction intersects a declared surface within the attention cone.

Every measurement carries:

- a **signal grade** — `head` (facial landmarks), `body` (shoulder/nose
  fallback), `away` (no usable signal), reported as share of attentive time
  (`signal_share`);
- a **confidence interval** — all rates ship with a Wilson 95% CI (§6);
- **replayable evidence** — the longest look intervals per surface link to
  exact timestamps in the annotated footage.

## 2. Core definitions

| Term | Definition |
|---|---|
| **Person / traffic** | A tracked identity seen in ≥ 3 sampled frames (`min_sightings = 3`). Single-frame ghosts never count. |
| **Look (raw)** | Head direction intersects the surface under the attention-cone test, signal ∈ {head, body}. |
| **Look (smoothed)** | Temporal majority vote over the last 3 samples (`smooth_n = 3`) — single-frame flicker is discarded. |
| **Impression** | A person whose cumulative dwell on a surface ≥ **0.4 s** (`min_dwell`). |
| **Attentive second** | One second of smoothed looking time, accumulated with real frame timestamps (VFR-safe). |
| **Dwell** | Cumulative attentive seconds of one person on one surface. |
| **Engaged / Deep** | Impression with dwell ≥ **1.0 s** / ≥ **3.0 s**. |
| **Attention rate** | impressions ÷ traffic, with Wilson 95% CI. |
| **Time-to-first-look (TTFL)** | Mean of (first look timestamp − first seen timestamp) over lookers. |
| **Glances per looker** | Mean number of distinct look episodes per looker. |
| **Stopping power** | Mean speed drop while looking: `max(0, 1 − v_look/v_all) × 100` (speed in body-heights/sec). |
| **Enter / Exit (line)** | A tracked person's foot point crossing a declared line, with hysteresis (§5). Direction of the line's arrow = **in**. |
| **Capture rate** | enters ÷ traffic seen in frame (share of passers-by that entered). |
| **Reach** | A wrist keypoint inside a `shelf` surface, keypoint confidence ≥ 0.35, in ≥ 3 consecutive samples. Reported only when keypoint quality permits; otherwise the report states "no reach signal". |

## 3. Attention cone

- Base half-angle: **35°** (`cone_deg`), auto-tuned per person by signal
  noise and the angular width of the surface (`auto_cone = on`).
- 3D mode: when the scene reconstruction passes the reliability gate, the
  test is a true ray-vs-3D-surface intersection; otherwise a 2.5D azimuth
  test with perspective foreshortening `k(y)` from auto-calibration.
- The what-if simulator may recompute metrics at other cone angles
  (15–60°); such figures are simulations and are labeled as such.

## 4. Sampling & timing

- Default sampling: **10 fps** (`sample_fps`), stride from source fps.
- All durations accumulate with **real frame timestamps** (VFR-safe), frame
  gaps clamped to [0.01 s, 0.5 s].
- Detection: YOLO11-pose-x (single pass, `imgsz 960`) or hybrid crowd mode
  (RTMO + 2×2 overlapping tiles, IoS-NMS merge) on ≥1080p or ≥10 people.
- 3D depth: Depth Anything V2 Metric, **Indoor** variant by default (retail
  scenes are indoor); `OCULIQ_DEPTH_MODEL=outdoor` for DOOH/open-air. Runs once
  per job; gated by the reliability check (§3) — falls back to 2.5D if unreliable.

## 5. Line crossing (entrance counting)

- The line is defined by two points; **in = the side the arrow points to**
  (left normal of p1→p2).
- Side state per person confirms only beyond a hysteresis band of
  `max(0.15 × person height, 2% of line length)` from the line.
- A crossing counts when a confirmed side flips **and** the foot point's
  projection lies within the segment (±10% overhang).
- Crossings by identities later removed as ghosts are removed with them.
- Lines are measured in video mode only (a still image has no crossings).

## 6. Statistics

- **Wilson 95% CI** (z = 1.96) on attention rate and impressions.
- Aggregates use means unless stated; dwell histogram buckets:
  <1, 1–2, 2–3, 3–5, ≥5 s.

## 7. Composite & commercial metrics

- **AQS (Attention Quality Score, 0–100)** — video:
  `100 × (0.30·rate + 0.30·min(avg_dwell/5, 1) + 0.20·deep_share + 0.20·min(stopping_power/50, 1))`
  Still image: `100 × (0.6·rate + 0.4·(impressions > 0))`.
- **Reach CPM** = cost ÷ (impressions/1000).
- **Attention CPM** = cost ÷ (attentive_seconds/1000) — the cost of one
  thousand actually-watched seconds.

## 8. Surface types

`billboard`, `screen`, `window` (attention surfaces) · `shelf` (attention +
reach) · `display` (attention surface, floor stand) · `line` (entrance
counting only) · `staff` (exclusion area — persons spending ≥ 30% of their
visible time or ≥ 60 s inside are excluded from all metrics; the report
discloses the excluded count).

## 9. Privacy invariants

On-device processing only; no facial recognition, no identification, no
re-identification across sessions. Faces auto-blurred in all rendered output.
Audience estimates (if enabled) are aggregate-only with k-anonymity (k ≥ 5).

## 10. Honesty rules

1. Numbers below reliability thresholds are withheld, not extrapolated.
2. Signal mix and confidence intervals are always reported.
3. Simulated (what-if) figures are never mixed with measured figures.
4. The published error margins (docs/ACCURACY.md) travel with every audit
   report referencing this spec version.
