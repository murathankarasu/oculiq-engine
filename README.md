# Oculiq — Attention Intelligence (local B2B SaaS MVP)

Turns standard footage (CCTV, phone, drone) into **measured human attention** for ad zones.
No eye-tracking hardware, no cloud: the model runs **entirely on this machine**.

## Architecture

```
web/          Frontend (English, matches oculiq.studio) — upload → zones → analyze → report
server/       Local Python backend (FastAPI)
  engine.py   YOLO11-pose x + ByteTrack + orientation cone + metrics + overlay render
  main.py     API: /api/analyze, SSE progress/preview, report, annotated video/image
models/       yolo11x-pose.pt (113 MB, auto-managed)
jobs/         Per-analysis working dirs (input + annotated output)
run.sh        One command: venv + deps (first run) + server on :8123
```

## Run

```bash
./run.sh          # → http://localhost:8123
```

First run creates `.venv` and installs deps (ultralytics/torch — a few minutes, one time).
Everything after that is offline.

## Flow

1. **Upload** — drop a video (MP4/MOV) or photo (JPG/PNG).
2. **Ad zones** — drag a diagonal per ad surface; popover asks label, type, cost/day.
   One-time per fixed camera. Multiple zones = A/B comparison.
3. **Analyze** — backend processes with live preview stream (SSE), progress ring,
   live counters. Annotated video is rendered server-side (zones, corner-tick person
   boxes, gaze arrows, heatmap, HUD).
4. **Report** — advanced, animated:
   - Engagement funnel: traffic → impressions → engaged ≥1s → deep ≥3s
   - **Attention rate**, attentive seconds, avg/max dwell
   - **Time-to-first-look**, **glances per looker**, **stopping power** (% slowdown
     while looking) — metrics competitors don't ship
   - **AQS** (Attention Quality Score, 0–100 composite) per zone
   - Attention-over-time chart, dwell distribution histogram
   - **Dual CPM**: reach CPM + attention CPM (cost ÷ attentive-seconds/1000)
   - Zone comparison with AQS winner
   - Exports: annotated MP4/JPG, JSON, CSV

## Crowd mode (tiled multi-scan)

For dense CCTV scenes: the frame is scanned as a full-frame pass **plus 2×2
overlapping tiles** (far/small people are seen at higher resolution), duplicates are
merged with IoS-NMS (a person caught by two tiles collapses to one), a lightweight
IoU+centroid tracker keeps identities across frames, and the "looking" state is
smoothed by temporal majority vote. Traffic requires ≥2 sightings — no phantom counts.
The report adds `avg_concurrency` + a density timeline.

Enable via the **Crowded scene** checkbox (zones step) or leave on `auto`
(activates on ≥1080p footage or when ≥10 people appear).

## Unique software layer (competitors don't ship these)

- **What-if simulator** — every gaze ray is recorded during analysis (`report.sim.rays`);
  in the report you drag a virtual placement (or draw a new one) and impressions /
  attention rate / attentive seconds / TTFL recompute **instantly in the browser**
  from real measured rays — no re-processing. "Move the ad 2m right: +18% attention."
- **Auditable evidence** — each zone lists its longest look intervals as clickable
  chips; clicking seeks the annotated video to that exact moment. Every reported
  number is backed by replayable footage.
- **Confidence intervals** — attention rate ships with a Wilson 95% CI
  (`attention_rate_ci`, `impressions_ci`) and the signal mix (head/partial share) is
  reported. Honest ranges instead of fake precision.
- **Cone calibration slider** — in the simulator, tune the attention-cone half angle
  (15–60°) per camera and watch every metric recompute live from the recorded rays.
- **Branded PDF report** — one click generates a black, Inter-set, "Oculiq." branded
  PDF (`/api/jobs/{id}/pdf`): KPIs, annotated heatmap frame, per-zone funnel, metric
  cards with CI, timeline + dwell histogram, AQS rings, dual CPM, zone comparison.
  The artifact a media owner can send straight to an advertiser.
- **Workspace + portfolio benchmark** — every analysis is persisted
  (`jobs/{id}/report.json`, survives restarts), the History screen reopens past
  reports, and each zone gets a "portfolio rank #N of M zones" badge across all
  your analyzed locations — the beginning of a Nielsen-style normative dataset.

## Engine signals (honest by design)

- Person detection + persistent IDs: YOLO11-pose (x) + ByteTrack — counts people even
  facing away (true traffic).
- Attention: **orientation-based** — facial keypoints (nose vs eyes/ears) give a 2D head
  direction; cone-intersection with the zone decides "looking". Per-measurement signal
  (`head`/`partial`/`away`) and confidence are stored and reported. Never marketed as
  eye-tracking.
- Speed per person (body-heights/sec) powers stopping-power.

## Tuning

`server/engine.py` — `cone_deg` (35), `min_dwell` (0.4s), `min_mag` (frontal threshold),
`imgsz` (960), `sample_fps` (API form field, default 10).
