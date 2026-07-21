"""Oculiq local server — FastAPI. Serves the web UI + analysis API.

All inference runs on this machine (YOLO11-pose). Nothing is uploaded anywhere.
"""
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "jobs"
JOBS_DIR.mkdir(exist_ok=True)


def _load_env():
    """ROOT/.env dosyasını yükle (API anahtarları). Mevcut env değerleri ezilmez."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env()

app = FastAPI(title="Oculiq")
jobs: dict[str, dict] = {}
_engine = None
_engine_lock = threading.Lock()
_workers: dict[str, "object"] = {}   # camera_id -> StreamWorker (canlı mod)


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from server.engine import AttentionEngine
            _engine = AttentionEngine()
        return _engine


def job_log(job, msg, level="info"):
    """Analiz ekranındaki canlı işlem günlüğü (SSE ile akar)."""
    job.setdefault("log", []).append(
        {"t": time.strftime("%H:%M:%S"), "lv": level, "m": msg})


def _run(job_id: str, path: Path, zones: list, cost_map: dict, is_video: bool,
         sample_fps: int, crowd_mode: str, demographics: bool, face_blur: bool):
    job = jobs[job_id]
    job["id"] = job_id
    t_start = time.time()
    try:
        job["status"] = "loading-model"
        job_log(job, "Job accepted — input saved, preparing engine")
        job_log(job, "Loading detection model (YOLO11x-pose + ByteTrack)…")
        eng = get_engine()
        job_log(job, "Detection model ready", "ok")
        job["status"] = "processing"
        if is_video:
            report = eng.process_video(path, zones, job, cost_map,
                                       sample_fps=sample_fps, crowd_mode=crowd_mode,
                                       demographics=demographics, face_blur=face_blur)
        else:
            report = eng.process_image(path, zones, job, cost_map, crowd_mode=crowd_mode,
                                       demographics=demographics, face_blur=face_blur)
        job["report"] = report
        job["progress"] = 100
        job["status"] = "done"
        _save(job_id)
        job_log(job, f"Done in {time.time() - t_start:.1f}s — report saved", "ok")
    except Exception as e:  # surfaced to UI
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        job_log(job, f"FAILED: {type(e).__name__}: {e}", "err")


def _save(job_id: str):
    """Analizi diske kalıcılaştır — workspace/geçmiş bunun üzerinden çalışır."""
    job = jobs[job_id]
    d = JOBS_DIR / job_id
    (d / "report.json").write_text(json.dumps(job["report"]))
    rep = job["report"]
    meta = {
        "job_id": job_id, "created": job["created"], "still": rep["still"],
        "duration": rep["duration"], "traffic": rep["traffic"],
        "scan_mode": rep.get("scan_mode", ""),
        "zones": [{"label": z["label"], "aqs": z["aqs"],
                   "attention_rate": z["attention_rate"],
                   "attentive_seconds": z["attentive_seconds"]} for z in rep["zones"]],
    }
    (d / "meta.json").write_text(json.dumps(meta))


def _get_job(job_id: str):
    """Bellekte yoksa diskten yükle (server yeniden başlasa da raporlar açılır)."""
    if job_id in jobs:
        return jobs[job_id]
    d = JOBS_DIR / job_id
    if not (d / "report.json").exists():
        return None
    job = {"status": "done", "progress": 100, "preview": None, "live": None,
           "created": 0, "report": json.loads((d / "report.json").read_text())}
    for pat, key in (("*.annotated.mp4", "out_video"), ("*.annotated.jpg", "out_image"),
                     ("*.sim.jpg", "sim_frame"), ("*.scene.jpg", "scene_view")):
        f = next(iter(d.glob(pat)), None)
        if f:
            job[key] = str(f)
    if "sim_frame" not in job:  # foto: arka plan = orijinal girdi
        f = next((p for p in d.glob("input.*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
        if f:
            job["sim_frame"] = str(f)
    ins = d / "insights.json"
    if ins.exists():
        job["insights"] = json.loads(ins.read_text())
    jobs[job_id] = job
    return job


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), zones: str = Form(...),
                  costs: str = Form("{}"), sample_fps: int = Form(10),
                  crowd_mode: str = Form("auto"), demographics: str = Form("off"),
                  face_blur: str = Form("on")):
    zs = json.loads(zones)
    if not zs:
        raise HTTPException(400, "at least one zone required")
    job_id = uuid.uuid4().hex[:12]
    d = JOBS_DIR / job_id
    d.mkdir()
    suffix = Path(file.filename or "media").suffix or ".bin"
    path = d / f"input{suffix}"
    path.write_bytes(await file.read())
    is_video = (file.content_type or "").startswith("video")

    jobs[job_id] = {"status": "queued", "progress": 0, "preview": None,
                    "live": None, "report": None, "created": time.time(), "log": []}
    threading.Thread(target=_run,
                     args=(job_id, path, zs, json.loads(costs), is_video, sample_fps,
                           crowd_mode, demographics == "on", face_blur != "off"),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/events")
async def events(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404)

    def gen():
        last_preview = None
        while True:
            job = jobs[job_id]
            msg = {"status": job["status"], "progress": job["progress"], "live": job["live"],
                   "log": job.get("log", [])}
            if job["preview"] and job["preview"] != last_preview:
                msg["frame"] = job["preview"]
                last_preview = job["preview"]
            yield f"data: {json.dumps(msg)}\n\n"
            if job["status"] in ("done", "error"):
                if job["status"] == "error":
                    yield f'data: {json.dumps({"status": "error", "error": job.get("error")})}\n\n'
                return
            time.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/config")
async def config():
    prov = os.environ.get("OCULIQ_LLM") or (
        "openai" if os.environ.get("OPENAI_API_KEY")
        else "gemini" if os.environ.get("GEMINI_API_KEY") else "local")
    model = os.environ.get("OCULIQ_LLM_MODEL") or (
        "gpt-4o-mini" if prov == "openai"
        else "gemini-2.0-flash" if prov == "gemini" else "rule-based summary")
    return {"llm": prov, "model": model}


@app.get("/api/storage")
async def storage():
    total, count = 0, 0
    for d in JOBS_DIR.iterdir():
        if d.is_dir():
            count += 1
            total += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return {"analyses": count, "bytes": total}


@app.delete("/api/jobs")
async def delete_all_jobs():
    n = 0
    for d in list(JOBS_DIR.iterdir()):
        if d.is_dir():
            shutil.rmtree(d)
            n += 1
    jobs.clear()
    return {"deleted": n}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    d = JOBS_DIR / job_id
    if not d.exists():
        raise HTTPException(404)
    shutil.rmtree(d)
    jobs.pop(job_id, None)
    return {"ok": True}


@app.get("/api/history")
async def history():
    out = []
    for d in JOBS_DIR.iterdir():
        m = d / "meta.json"
        if m.exists():
            try:
                out.append(json.loads(m.read_text()))
            except Exception:
                pass
    out.sort(key=lambda x: x.get("created", 0), reverse=True)
    return out[:100]


@app.get("/api/jobs/{job_id}/report")
async def report(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404)
    if job["status"] == "error":
        return JSONResponse({"error": job.get("error")}, status_code=500)
    if not job["report"]:
        raise HTTPException(409, "not ready")
    return job["report"]


@app.get("/api/jobs/{job_id}/video")
async def video(job_id: str):
    job = _get_job(job_id)
    if not job or not job.get("out_video"):
        raise HTTPException(404)
    return FileResponse(job["out_video"], filename="oculiq-annotated.mp4")


@app.get("/api/jobs/{job_id}/scene")
async def scene_view(job_id: str):
    job = _get_job(job_id)
    if not job or not job.get("scene_view"):
        raise HTTPException(404)
    return FileResponse(job["scene_view"])


@app.get("/api/jobs/{job_id}/frame")
async def sim_frame(job_id: str):
    job = _get_job(job_id)
    if not job or not job.get("sim_frame"):
        raise HTTPException(404)
    return FileResponse(job["sim_frame"])


@app.post("/api/jobs/{job_id}/insights")
async def make_insights(job_id: str):
    job = _get_job(job_id)
    if not job or not job.get("report"):
        raise HTTPException(404)
    from server.insights import generate
    res = generate(job["report"])
    d = JOBS_DIR / job_id
    (d / "insights.json").write_text(json.dumps(res))
    (d / "report.pdf").unlink(missing_ok=True)  # PDF yeniden üretilsin (insights sayfası)
    job["insights"] = res
    return res


@app.get("/api/jobs/{job_id}/insights")
async def get_insights(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404)
    if not job.get("insights"):
        raise HTTPException(404)
    return job["insights"]


@app.get("/api/jobs/{job_id}/pdf")
async def pdf(job_id: str):
    job = _get_job(job_id)
    if not job or not job.get("report"):
        raise HTTPException(404)
    out = JOBS_DIR / job_id / "report.pdf"
    if not out.exists():
        from server.pdf import build_pdf
        build_pdf(job, str(out))
    return FileResponse(out, filename="oculiq-attention-report.pdf",
                        media_type="application/pdf")


@app.get("/api/jobs/{job_id}/image")
async def image(job_id: str):
    job = _get_job(job_id)
    if not job or not job.get("out_image"):
        raise HTTPException(404)
    return FileResponse(job["out_image"], filename="oculiq-annotated.jpg")


@app.post("/api/jobs/{job_id}/cancel")
async def cancel(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    job["cancel"] = True
    return {"ok": True}


# ---------------- live mode (Faz 2): cameras + timeseries ----------------
@app.get("/api/cameras")
async def cameras_list():
    from server import stream
    cams = stream.load_cameras()
    for c in cams:
        w = _workers.get(c["id"])
        c["status"] = w.status if w and w.is_alive() else "stopped"
    return cams


@app.post("/api/cameras")
async def cameras_save(payload: dict):
    """Kamera ekle/güncelle: {id?, name, url, zones, sample_fps?, loop?}"""
    from server import stream
    cams = stream.load_cameras()
    cam_id = payload.get("id") or uuid.uuid4().hex[:8]
    payload["id"] = cam_id
    payload.setdefault("zones", [])
    cams = [c for c in cams if c["id"] != cam_id] + [payload]
    stream.save_cameras(cams)
    return {"id": cam_id}


@app.delete("/api/cameras/{cam_id}")
async def cameras_delete(cam_id: str):
    from server import stream
    w = _workers.pop(cam_id, None)
    if w:
        w.stop()
    stream.save_cameras([c for c in stream.load_cameras() if c["id"] != cam_id])
    return {"ok": True}


@app.post("/api/cameras/{cam_id}/start")
async def camera_start(cam_id: str):
    from server import stream
    cam = next((c for c in stream.load_cameras() if c["id"] == cam_id), None)
    if not cam:
        raise HTTPException(404)
    if not cam.get("zones"):
        raise HTTPException(400, "configure zones first")
    old = _workers.get(cam_id)
    if old and old.is_alive():
        return {"ok": True, "status": old.status}
    w = stream.StreamWorker(get_engine(), cam)
    _workers[cam_id] = w
    w.start()
    return {"ok": True, "status": "starting"}


@app.post("/api/cameras/{cam_id}/stop")
async def camera_stop(cam_id: str):
    w = _workers.get(cam_id)
    if w:
        w.stop()
    return {"ok": True}


@app.get("/api/cameras/{cam_id}/frame")
async def camera_frame(cam_id: str):
    """Zone çizimi için tek kare (bellekten ya da kaynaktan; diske yazılmaz)."""
    import cv2
    from server import stream
    w = _workers.get(cam_id)
    frame = w.last_frame if w and w.last_frame is not None else None
    if frame is None:
        cam = next((c for c in stream.load_cameras() if c["id"] == cam_id), None)
        if not cam:
            raise HTTPException(404)
        url = cam["url"]
        cap = cv2.VideoCapture(int(url) if str(url).isdigit()
                               else stream.resolve_source(url))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(502, "source not reachable")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    from fastapi.responses import Response
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/cameras/{cam_id}/live_frame")
async def camera_live_frame(cam_id: str):
    """Canlı izleme: anotasyonlu + yüz-bulanık son kare (yalnızca bellekten).
    Kamera çalışmıyorsa ya da henüz kare yoksa 404 — çağıran görüntü gösterir."""
    from fastapi.responses import Response
    w = _workers.get(cam_id)
    if not w or not w.is_alive() or w.preview_jpg is None:
        raise HTTPException(404, "no live frame — start the camera")
    return Response(content=w.preview_jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/live/{cam_id}")
async def live_counters(cam_id: str):
    w = _workers.get(cam_id)
    if not w or not w.is_alive():
        return {"status": "stopped"}
    out = dict(w.live) if w.live else {}
    out["status"] = w.status
    out["error"] = w.error
    return out


@app.get("/api/timeseries")
async def timeseries(camera: str, zone: str = None, since: int = None, until: int = None):
    from server import stream
    return stream.query_timeseries(camera, zone, since, until)


@app.get("/api/dataset/stats")
async def dataset_stats():
    """Birikmiş dikkat-olay veri seti — retail benchmark + model tohumu göstergesi."""
    from server import dataset
    return dataset.stats()


@app.middleware("http")
async def no_static_cache(request, call_next):
    """UI dosyaları güncellenince tarayıcı bayat CSS/JS kullanmasın."""
    resp = await call_next(request)
    if request.url.path.endswith((".css", ".js", ".html")) or request.url.path == "/":
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
