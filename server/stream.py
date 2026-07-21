"""Oculiq live measurement — continuous stream worker (Faz 2).

Sources: rtsp:// / http(s):// stream URLs, integer webcam index, or a local
file path with loop=true (fake-live mode for testing without a camera).

Privacy contract (Spec v1.0 §9): frames are processed in memory and DISCARDED.
Nothing is recorded; only aggregate counters are persisted (hourly rows in
SQLite). Because there is no footage, live metrics carry no evidence chips —
reports say so instead of pretending.

Measurement: the exact same per-frame core as batch (AttentionEngine
._step_frame) — Spec v1.0 behavior lives in one place. Live mode runs 2.5D
(no scene3d build) and does not record what-if rays.
"""
import json
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
DB = DATA / "metrics.db"
CAMS = DATA / "cameras.json"

_infer_lock = threading.Lock()   # tek model, çok worker: sıralı çıkarım


def resolve_source(url):
    """YouTube canlı yayın URL'lerini oynatılabilir HLS'e çevir (yt-dlp).
    Çözümlenen URL saatlik dolar — worker her (yeniden) bağlanışta çağırır.
    İç test aracı: içerik kaydedilmez, kare işlenir ve atılır (Spec §9)."""
    u = str(url)
    if "youtube.com" in u or "youtu.be" in u:
        try:
            import yt_dlp
            opts = {"quiet": True, "no_warnings": True,
                    "format": "best[height<=1080]",
                    "extractor_args": {"youtube": {"player_client": ["android"]}}}
            with yt_dlp.YoutubeDL(opts) as y:
                info = y.extract_info(u, download=False)
                return info.get("url") or u
        except Exception:
            return u
    return u


# ---------------- storage ----------------
def _db():
    DATA.mkdir(exist_ok=True)
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS agg_hourly(
        camera_id TEXT NOT NULL,
        zone_id   TEXT NOT NULL,   -- zone id, line id ya da '_cam' (kamera toplamı)
        hour_ts   INTEGER NOT NULL, -- saat başlangıcı (epoch sn)
        traffic       INTEGER DEFAULT 0,
        impressions   INTEGER DEFAULT 0,
        attentive_sec REAL    DEFAULT 0,
        enters        INTEGER DEFAULT 0,
        exits         INTEGER DEFAULT 0,
        reaches       INTEGER DEFAULT 0,
        samples       INTEGER DEFAULT 0,
        PRIMARY KEY (camera_id, zone_id, hour_ts))""")
    return con


def load_cameras():
    if CAMS.exists():
        return json.loads(CAMS.read_text())
    return []


def save_cameras(cams):
    DATA.mkdir(exist_ok=True)
    CAMS.write_text(json.dumps(cams, indent=1))


def query_timeseries(camera_id, zone_id=None, since=None, until=None):
    con = _db()
    q = "SELECT zone_id, hour_ts, traffic, impressions, attentive_sec, enters, exits, reaches, samples FROM agg_hourly WHERE camera_id=?"
    args = [camera_id]
    if zone_id is not None:
        q += " AND zone_id=?"
        args.append(str(zone_id))
    if since is not None:
        q += " AND hour_ts>=?"
        args.append(int(since))
    if until is not None:
        q += " AND hour_ts<?"
        args.append(int(until))
    rows = con.execute(q + " ORDER BY hour_ts", args).fetchall()
    con.close()
    return [{"zone_id": r[0], "hour_ts": r[1], "traffic": r[2], "impressions": r[3],
             "attentive_sec": round(r[4], 1), "enters": r[5], "exits": r[6],
             "reaches": r[7], "samples": r[8]} for r in rows]


# ---------------- worker ----------------
class StreamWorker(threading.Thread):
    FLUSH_SEC = 60          # agregat upsert aralığı

    def __init__(self, engine, cam):
        """cam: {id, name, url, zones, costs?, sample_fps?, loop?}"""
        super().__init__(daemon=True, name=f"stream-{cam['id']}")
        self.eng = engine
        self.cam = cam
        self.stop_flag = threading.Event()
        self.status = "starting"
        self.error = None
        self.live = {}
        self.last_frame = None       # SADECE bellekte: zone çizimi için ham kare (blursuz)
        self.preview_jpg = None      # canlı izleme: anotasyonlu + yüz-bulanık JPEG (bellekte)
        self.started_ts = time.time()

    # -- kaynak --
    def _open(self):
        url = self.cam["url"]
        if isinstance(url, str) and url.isdigit():
            return cv2.VideoCapture(int(url))
        return cv2.VideoCapture(resolve_source(url))

    def stop(self):
        self.stop_flag.set()

    # -- pencere durumu (saatlik) --
    def _fresh_window(self, W, H, zs_att, z_lines, z_staff, z_shelf):
        from server.engine import KCalibrator, LineCounter, SimpleTracker
        self.tracker = SimpleTracker(max_gone=int(2.0 / max(self._dt_target, 0.05)))
        # kameraya özel perspektif kalibratörü — batch işlerin _cal'ı ile karışmasın
        self.my_cal = KCalibrator(H, fallback=self.eng.persp_k)
        self.line_counters = [LineCounter(z["id"], z["line_px"][0], z["line_px"][1])
                              for z in z_lines]
        self.st = {"persons": {}, "heat": np.zeros((H // 4, W // 4), np.float32),
                   "timeline": defaultdict(lambda: defaultdict(float)),
                   "foot_samples": [], "rays": [],
                   "line_counters": self.line_counters,
                   "z_staff": z_staff, "z_shelf": z_shelf, "zs_att": zs_att,
                   "W": W, "H": H, "scene": None, "scene_ok": False, "zquads": {},
                   "gaze3d_n": 0, "gaze_total": 0, "wrist_samples": 0,
                   "record_rays": False}
        self.win_samples = 0

    def _aggregate(self):
        """Pencere durumundan (persons + sayaçlar) agregat satırları üret.
        Ghost + staff filtreleri Spec v1.0 ile aynı."""
        eng = self.eng
        persons = {k: v for k, v in self.st["persons"].items()
                   if v["frames"] >= eng.min_sightings}
        if self.st["z_staff"]:
            persons = {k: v for k, v in persons.items()
                       if not (v.get("staff_sec", 0) >= 60
                               or v.get("staff_sec", 0) >= 0.3 * max(v.get("seen_sec", 0), 1e-6))}
        valid = set(persons)
        rows = {"_cam": {"traffic": len(persons), "impressions": 0, "attentive_sec": 0.0,
                         "enters": 0, "exits": 0, "reaches": 0, "samples": self.win_samples}}
        for z in self.st["zs_att"]:
            zid = z["id"]
            dwells = [p["dwell"][zid] for p in persons.values()
                      if p["dwell"][zid] >= eng.min_dwell]
            att = sum(p["dwell"][zid] for p in persons.values())
            reaches = sum(len(p.get("reach_events", {}).get(zid, []))
                          for p in persons.values())
            rows[str(zid)] = {"traffic": len(persons), "impressions": len(dwells),
                              "attentive_sec": round(att, 1), "enters": 0, "exits": 0,
                              "reaches": reaches, "samples": self.win_samples}
        for lc in self.line_counters:
            ins, outs, _ = lc.counts(valid)
            rows[str(lc.zid)] = {"traffic": len(persons), "impressions": 0,
                                 "attentive_sec": 0.0, "enters": ins, "exits": outs,
                                 "reaches": 0, "samples": self.win_samples}
        return rows

    def _filtered_persons(self):
        eng = self.eng
        persons = {k: v for k, v in self.st["persons"].items()
                   if v["frames"] >= eng.min_sightings}
        if self.st["z_staff"]:
            persons = {k: v for k, v in persons.items()
                       if not (v.get("staff_sec", 0) >= 60
                               or v.get("staff_sec", 0) >= 0.3 * max(v.get("seen_sec", 0), 1e-6))}
        return persons

    def _record_window_dataset(self):
        """Pencere kapanırken (saat devri / durma) dikkat epizotlarını veri setine yaz.
        Retail benchmark + model tohumu — kimliksiz (server/dataset.py)."""
        try:
            from server import dataset
            persons = self._filtered_persons()
            report = {"spec": "1.0", "zones": []}   # canlıda 3D yok: yüzey bağlamı boş
            eps = self.eng._episodes(persons, self.st["zs_att"], report)
            dataset.record(self.cam["id"], "live", "1.0", eps)
        except Exception:
            pass

    def _flush(self, hour_ts):
        rows = self._aggregate()
        con = _db()
        with con:
            for zid, r in rows.items():
                con.execute("""INSERT INTO agg_hourly
                    (camera_id, zone_id, hour_ts, traffic, impressions, attentive_sec,
                     enters, exits, reaches, samples)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(camera_id, zone_id, hour_ts) DO UPDATE SET
                     traffic=excluded.traffic, impressions=excluded.impressions,
                     attentive_sec=excluded.attentive_sec, enters=excluded.enters,
                     exits=excluded.exits, reaches=excluded.reaches,
                     samples=excluded.samples""",
                    (self.cam["id"], zid, hour_ts, r["traffic"], r["impressions"],
                     r["attentive_sec"], r["enters"], r["exits"], r["reaches"],
                     r["samples"]))
        con.close()

    # -- ana döngü --
    def run(self):
        cam = self.cam
        sample_fps = float(cam.get("sample_fps", 5))
        self._dt_target = 1.0 / max(sample_fps, 0.5)
        loop = bool(cam.get("loop"))
        backoff = 2

        while not self.stop_flag.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.status = "reconnecting"
                self.error = "source not reachable"
                time.sleep(min(backoff, 60))
                backoff = min(backoff * 2, 60)
                continue
            backoff = 2
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
            zs = self.eng._prep_zones(cam["zones"], W, H)
            zs_att, z_lines, z_staff = self.eng._split_zones(zs)
            z_shelf = [z for z in zs_att if z["type"] == "shelf"]
            self._zs_full = zs        # canlı çizim: tüm zone'lar (yüzey + çizgi + staff)
            self._fresh_window(W, H, zs_att, z_lines, z_staff, z_shelf)

            self.status = "live"
            self.error = None
            cur_hour = int(time.time() // 3600) * 3600
            last_flush = time.time()
            last_sample = 0.0
            last_preview = 0.0
            prev_t = None
            fi = 0

            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    if loop:                          # sahte-canlı: dosyayı başa sar
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break                             # gerçek kaynak koptu -> reconnect
                fi += 1
                now = time.time()
                if loop:                              # dosya gerçek-zaman hızında aksın
                    time.sleep(max(0.0, 1.0 / src_fps - 0.002))
                if now - last_sample < self._dt_target:
                    continue
                last_sample = now
                t = now
                dtf = self._dt_target if prev_t is None else min(max(t - prev_t, 0.01), 2.0)
                prev_t = t

                with _infer_lock:
                    # kilit içinde kalibratör takası: canlı örnekler batch işin
                    # kalibrasyonunu zehirlemesin (bilinen sınır: kilitsiz batch
                    # çıkarımıyla kısa çakışma penceresi kalır — MVP kabulü)
                    prev_cal = self.eng._cal
                    self.eng._cal = self.my_cal
                    try:
                        dets = self.eng._detect_frame(frame, False, self.tracker)
                    finally:
                        self.eng._cal = prev_cal
                b = int(t // 2) * 2
                self.st["scene"], self.st["scene_ok"], self.st["zquads"] = None, False, {}
                self.eng._step_frame(self.st, dets, t, dtf, b)
                self.win_samples += 1
                self.last_frame = frame               # bellekte tek kare; diske yazılmaz

                live_persons = {k: v for k, v in self.st["persons"].items()
                                if v["frames"] >= self.eng.min_sightings}
                self.live = self.eng._live(live_persons, self.st["zs_att"], t,
                                           self.line_counters)
                self.live["status"] = "live"
                self.live["hour_ts"] = cur_hour

                # canlı izleme karesi: ~1.2s'de bir anotasyonlu + yüz-bulanık (bellekte)
                if now - last_preview >= 1.2:
                    try:
                        lc = {c.zid: c.counts()[:2] for c in self.line_counters}
                        annotated = self.eng._draw(
                            frame, dets, self._zs_full, self.st["heat"], t,
                            len(live_persons), tiled=False, blur=True, line_counts=lc)
                        ok2, buf = cv2.imencode(".jpg", annotated,
                                                [cv2.IMWRITE_JPEG_QUALITY, 72])
                        if ok2:
                            self.preview_jpg = buf.tobytes()
                    except Exception:
                        pass
                    last_preview = now

                if now - last_flush >= self.FLUSH_SEC:
                    self._flush(cur_hour)
                    last_flush = now
                new_hour = int(now // 3600) * 3600
                if new_hour != cur_hour:              # saat devri: kapat + sıfırla
                    self._flush(cur_hour)
                    self._record_window_dataset()      # pencere epizotları -> veri seti
                    cur_hour = new_hour
                    self._fresh_window(W, H, zs_att, z_lines, z_staff, z_shelf)
                    prev_t = None

            self._flush(cur_hour)
            self._record_window_dataset()              # kaynak koptu/durdu: son pencere
            cap.release()
            if not loop and not self.stop_flag.is_set():
                self.status = "reconnecting"
                time.sleep(min(backoff, 60))

        self.status = "stopped"
