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


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from server.engine import AttentionEngine
            _engine = AttentionEngine()
        return _engine


def _run(job_id: str, path: Path, zones: list, cost_map: dict, is_video: bool,
         sample_fps: int, crowd_mode: str, demographics: bool, face_blur: bool):
    job = jobs[job_id]
    try:
        job["status"] = "loading-model"
        eng = get_engine()
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
    except Exception as e:  # surfaced to UI
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"


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
                    "live": None, "report": None, "created": time.time()}
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
            msg = {"status": job["status"], "progress": job["progress"], "live": job["live"]}
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


@app.middleware("http")
async def no_static_cache(request, call_next):
    """UI dosyaları güncellenince tarayıcı bayat CSS/JS kullanmasın."""
    resp = await call_next(request)
    if request.url.path.endswith((".css", ".js", ".html")) or request.url.path == "/":
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
