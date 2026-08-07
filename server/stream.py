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


def hourly_profile(camera_id, days=7):
    """SAAT PROFILI: gunun hangi saatinde yakalama/dikkat cokuyor?

    Gunluk toplam "kac kisi gecti"yi soyler ama NEREDE kaybedildigini gizler.
    Saatlik kirilim genelde personel ya da vitrin sorununu gosterir ve magazanin
    hemen aksiyona cevirebilecegi tek zaman-bazli kesittir.

    Az orneklem KONUSMAZ: bir saat diliminde 20'den az kisi gorulmusse o saat
    icin oran uretilmez (yuzeysel bir "%0 yakalama" yanlis aksiyon dogurur).
    """
    con = _db()
    since = int(time.time()) - days * 86400
    rows = con.execute(
        "SELECT hour_ts, zone_id, traffic, impressions, attentive_sec, enters "
        "FROM agg_hourly WHERE camera_id=? AND hour_ts>=?", (camera_id, since)).fetchall()
    con.close()
    buckets = {}
    for hour_ts, zone_id, traffic, imp, att, enters in rows:
        h = time.localtime(hour_ts).tm_hour
        b = buckets.setdefault(h, {"hour": h, "traffic": 0, "impressions": 0,
                                   "attentive_sec": 0.0, "enters": 0, "hours": 0})
        if zone_id == "_cam":
            b["traffic"] += traffic
            b["hours"] += 1
        else:
            b["impressions"] += imp
            b["attentive_sec"] += att
            b["enters"] += enters
    out = []
    for h in sorted(buckets):
        b = buckets[h]
        row = {"hour": h, "traffic": b["traffic"], "samples_hours": b["hours"],
               "attentive_sec": round(b["attentive_sec"], 1)}
        if b["traffic"] >= 20:          # durustluk esigi: az veriyle oran verme
            row["capture_rate"] = round(b["enters"] / b["traffic"] * 100, 1)
            row["attention_rate"] = round(b["impressions"] / b["traffic"] * 100, 1)
        else:
            row["note"] = "too few people this hour for a rate"
        out.append(row)
    # en zayif saat: yalnizca oran uretilebilen saatler arasinda
    rated = [r for r in out if "capture_rate" in r]
    weakest = min(rated, key=lambda r: r["capture_rate"]) if len(rated) >= 3 else None
    return {"hours": out, "days": days,
            "weakest_hour": weakest["hour"] if weakest else None,
            "weakest_capture_rate": weakest["capture_rate"] if weakest else None}


# ---------------- worker ----------------
class StreamWorker(threading.Thread):
    FLUSH_SEC = 60          # agregat upsert aralığı
    STALL_SEC = 20          # bu kadar süre kare gelmezse kaynak donmuş sayılır
    PRUNE_AT = 400          # pencerede bu kadar kişi birikince budama başlar
    PRUNE_IDLE_SEC = 90     # bu süredir görülmeyen ve hiç bakmamış kayıt düşer

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
        # --- akis sagligi (AVM pilotu): "live" demek yetmez, VERI AKIYOR MU? ---
        self.last_frame_ts = 0.0     # kaynaktan en son kare okunma zamani
        self.last_sample_ts = 0.0    # en son ISLENEN (olculen) kare zamani
        self.frames_read = 0
        self.samples_done = 0
        self.reconnects = 0
        self.stalls = 0
        self.pruned_total = 0

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
                   "record_rays": False,
                   # yogunluk kovalari 2 sn'lik: 900 kova ≈ son 30 dk. Canlida
                   # sinirsiz birakmak gunler suren calismada bellegi sisirirdi.
                   "density_keep": 900}
        self.win_samples = 0
        if not hasattr(self, "_tiled"):
            self._tiled, self._last_dets = False, []
        if not hasattr(self, "_scene"):
            self._scene, self._zquads = None, {}
            self._scene_tried, self._scene_state = False, None

    def _aggregate(self):
        """Pencere durumundan (persons + sayaçlar) agregat satırları üret.
        Dikiş + ghost + staff filtreleri Spec v1.0 / batch ile aynı."""
        eng = self.eng
        persons, valid = self._filtered_persons()
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
        """Batch ile aynı boru: dikiş -> ghost filtresi -> personel hariç tutma.
        -> (persons, valid_pids)  — valid_pids çizgi geçişleri için (dikilen
        eski parça pid'leri de kalıcı kimliğe sayılır)."""
        eng = self.eng
        persons, aliases = eng._stitch_tracks(dict(self.st["persons"]))
        persons = {k: v for k, v in persons.items()
                   if v["frames"] >= eng.min_sightings}
        if self.st["z_staff"]:
            persons = {k: v for k, v in persons.items()
                       if not (v.get("staff_sec", 0) >= 60
                               or v.get("staff_sec", 0) >= 0.3 * max(v.get("seen_sec", 0), 1e-6))}
        valid = set(persons) | {a for a, c in aliases.items() if c in persons}
        return persons, valid

    def _prune_window(self, now):
        """Uzun calisma dayanikliligi: saatlik pencere icinde de bellek buyur.

        AVM girisinde saatte binlerce kisi gecer; her biri dwell/interval/hiz
        listesi tutar. Olcume KATKISI BITMIS kayitlar (uzun suredir gorulmeyen
        ve hicbir yuzeye bakmamis olanlar) duserulur — agregat sayilari
        korunur, cunku dusen kayitlarin dwell'i zaten sifirdir. Bakmis olanlar
        pencere sonuna kadar tutulur (rapor/agregat icin gerekli)."""
        st = self.st
        persons = st["persons"]
        if len(persons) < self.PRUNE_AT:
            return 0
        cut = now - self.PRUNE_IDLE_SEC
        drop = []
        for pid, p in persons.items():
            if p.get("last_t", 0) > cut:
                continue                      # hala sahnede
            if any(v > 0 for v in p["dwell"].values()):
                continue                      # olcume katki verdi: tut
            if p.get("reach_events"):
                continue
            drop.append(pid)
        for pid in drop:
            persons.pop(pid, None)
            st.get("dir_ema", {}).pop(pid, None)
        # yon-yumusatma tablosu ve cizgi sayaci durumu da temizlenir
        ema = st.get("dir_ema", {})
        for pid in list(ema):
            if pid not in persons:
                ema.pop(pid, None)
        for lc in self.line_counters:
            for pid in list(lc.side):
                if pid not in persons:
                    lc.side.pop(pid, None)
                    lc.last_evt.pop(pid, None)
        self.pruned_total += len(drop)
        return len(drop)

    def live_report(self):
        """Canli pencerenin TAM raporu — video moduyla ayni yapi.

        Ayni _report() cagrilir, dolayisiyla AQS, TTFL, glances, stopping power,
        dwell histogrami, guven araliklari ve (paket acikken) ziyaret funnel'i
        canlida da uretilir. Fark yalnizca dogasindan gelenler: canlida kayit
        tutulmadigi icin kanit klipleri ve what-if isinlari YOKTUR — bu, rapora
        acikca yazilir, sessizce bos birakilmaz."""
        eng = self.eng
        persons, valid = self._filtered_persons()
        st = self.st
        elapsed = max(time.time() - self.started_ts, 0.1)
        rep = eng._report(persons, st["zs_att"], st["timeline"],
                          duration=elapsed, peak=st.get("peak", 0),
                          cost_map=self.cam.get("costs") or {},
                          still=False, elapsed=elapsed, sim=None)
        # t0 kova ızgarasına yuvarlanır; ham started_ts ile ilk kova eksi çıkıyordu
        eng._apply_density(rep, st, t0=int(self.started_ts // 2) * 2)
        eng._apply_benchmark(rep, st["zs_att"])
        # Kanit zaman damgalari: canlida saat unix epoch olarak birikir; raporda
        # pencere basina gore saniyeye cevrilir — video raporuyla ayni birim.
        for zr in rep.get("zones", []):
            for ev in zr.get("evidence", []) or []:
                if ev.get("start", 0) > 1e6:
                    ev["start"] = round(ev["start"] - self.started_ts, 1)
        rep["mode"] = "live"
        rep["scan_mode"] = ("live tiled multi-scan (crowd)"
                            if getattr(self, "_tiled", False) else "live single-pass")
        rep["window_started"] = int(self.started_ts)
        if self._scene_state:
            rep["scene3d"] = self._scene_state
        # 3D yuzey olculeri — video raporuyla AYNI kod yolu (eng._apply_surface_3d);
        # burada bir kopya tutulmasi iki tarafin sessizce ayrilmasina yol acmisti.
        if self._scene is not None and self._zquads:
            eng._apply_surface_3d(rep, st["zs_att"], persons, self._scene,
                                  quads=self._zquads)
        # cizgiler + capture rate (video ile ayni alan adlari)
        if self.line_counters:
            traffic_n = len(persons)
            lines_out = []
            for z, lc in zip([z for z in self._zs_full if z["type"] == "line"],
                             self.line_counters):
                ins, outs, ev = lc.counts(valid)
                lines_out.append({
                    "id": z["id"], "label": z["label"], "line": z.get("line_norm"),
                    "enters": ins, "exits": outs,
                    "capture_rate": round(ins / traffic_n * 100, 1) if traffic_n else 0.0,
                    "events": [{"t": round(e[0] - self.started_ts, 1), "pid": e[1],
                                "dir": e[2]} for e in ev[:200]],
                })
            rep["lines"] = lines_out
            rep["capture_rate"] = lines_out[0]["capture_rate"] if lines_out else None
            if (self.cam.get("modules") or {}).get("visits"):
                rep["visits"] = {"enabled": True,
                                 "lines": eng._visits(self.line_counters,
                                                      [z for z in self._zs_full
                                                       if z["type"] == "line"],
                                                      persons, valid, st["zs_att"])}
            else:
                rep["visits"] = {"enabled": False,
                                 "note": "Visit analytics is a separate package."}
        mh = st.get("mh", {})
        det_n = max(mh.get("det", 0), 1)
        rep["measurement_health"] = {
            "detections": mh.get("det", 0),
            "direction_share": round(mh.get("dir", 0) / det_n * 100, 1),
            "signal_mix": {k: round(v / det_n * 100, 1) for k, v in mh.get("sig", {}).items()},
            "avg_det_conf": round(mh.get("conf_sum", 0.0) / det_n, 2),
            "tracks_seen": len(st["persons"]),
            "tracks_stitched": 0,
            "ghosts_dropped": max(len(st["persons"]) - len(persons), 0),
            "gaze3d_pct": round(st["gaze3d_n"] / max(st["gaze_total"], 1) * 100, 1),
        }
        rep["live_limits"] = ("No footage is recorded in live mode, so evidence clips "
                              "and the what-if simulator are unavailable; every other "
                              "metric matches a video analysis.")
        return rep

    def _record_window_dataset(self):
        """Pencere kapanırken (saat devri / durma) dikkat epizotlarını veri setine yaz.
        Retail benchmark + model tohumu — kimliksiz (server/dataset.py)."""
        try:
            from server import dataset
            persons, _ = self._filtered_persons()
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
        """Dis kabuk: beklenmedik bir hata thread'i OLDURMEZ.

        Eskiden run() icindeki herhangi bir istisna daemon thread'i sessizce
        sonlandiriyordu; kullanici arayuzde yalnizca "stopped" goruyor, gunlukte
        iz kalmiyordu. Artik hata yakalanir, kullaniciya gosterilir ve worker
        artan bekleme ile kendini yeniden dener (kalici hatada durur)."""
        fails = 0
        while not self.stop_flag.is_set():
            try:
                self._run_once()
                return                      # temiz cikis (stop istendi)
            except Exception as e:
                fails += 1
                self.status = "error"
                self.error = f"{type(e).__name__}: {e}"
                try:
                    import traceback, sys
                    print(f"[stream {self.cam.get('id')}] worker crashed "
                          f"({fails}): {self.error}", file=sys.stderr)
                    traceback.print_exc()
                except Exception:
                    pass
                if fails >= 5 or self.stop_flag.is_set():
                    self.status = "failed"  # kalici hata: sessizce olme, soyle
                    return
                self.stop_flag.wait(min(5 * fails, 30))
        self.status = "stopped"

    def _run_once(self):
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

            self.last_frame_ts = time.time()      # baglanti kuruldu: sayaci baslat
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    if loop:                          # sahte-canlı: dosyayı başa sar
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break                             # gerçek kaynak koptu -> reconnect
                fi += 1
                now = time.time()
                # DONMA TESPITI: cap.read() True donse bile kaynak bayat kare
                # verebilir (RTSP/HLS donmasi). Kare gelmiyorsa yeniden baglan —
                # sessizce "live" gorunup olcum uretmemek en tehlikeli hata.
                if now - self.last_frame_ts > self.STALL_SEC:
                    self.stalls += 1
                    self.status = "stalled"
                    self.error = f"no frames for {int(now - self.last_frame_ts)}s — reconnecting"
                    break
                self.last_frame_ts = now
                self.frames_read += 1
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
                        # KALABALIK TARAMASI canlida da calisir: AVM'de kalabalik
                        # kuraldir. Karar dinamiktir (8 kisiyle acilir, 5'in
                        # altinda kapanir) — sabit False oldugu icin canli mod
                        # uzak/kucuk insanlari hic taramiyordu.
                        self._tiled = self.eng._should_tile(
                            len(getattr(self, "_last_dets", []) or []), self._tiled,
                            cam.get("crowd_mode", "auto"), W, H)
                        dets = self.eng._detect_frame(frame, self._tiled, self.tracker)
                        self._last_dets = dets
                    finally:
                        self.eng._cal = prev_cal
                b = int(t // 2) * 2
                # CANLI 3D: kamera sabit oldugundan sahne bir kez kurulur ve
                # pencere boyunca kullanilir — boylece canli mod, video moduyla
                # ayni veriyi uretir (3D bakis, gercek yuzey olcusu, mesafe).
                if (self._scene is None and not self._scene_tried
                        and len(self.st["foot_samples"]) >= 12):
                    self._scene_tried = True
                    try:
                        from server.scene3d import SceneModel
                        sc = SceneModel().build(frame, self.st["foot_samples"],
                                                scene_type=cam.get("scene_type"))
                        if sc.enabled and sc.reliable():
                            self._scene = sc
                            self._zquads = {z["id"]: sc.zone_quad(z)
                                            for z in self.st["zs_att"]}
                            for q in self._zquads.values():
                                if q:
                                    q["_mesh"], q["_nseg"] = sc.zone_mesh(q)
                            sc._grid = sc.grid_segments()
                        self._scene_state = sc.state()
                    except Exception as e:
                        self._scene_state = {"enabled": False,
                                             "note": f"{type(e).__name__}: {e}"}
                self.st["scene"] = self._scene
                self.st["scene_ok"] = self._scene is not None
                self.st["zquads"] = self._zquads
                self.eng._step_frame(self.st, dets, t, dtf, b)
                self.win_samples += 1
                self.samples_done += 1
                self.last_sample_ts = now
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
                            len(live_persons), tiled=False, blur=True, line_counts=lc,
                            scene=self._scene, zquads=self._zquads)
                        ok2, buf = cv2.imencode(".jpg", annotated,
                                                [cv2.IMWRITE_JPEG_QUALITY, 72])
                        if ok2:
                            self.preview_jpg = buf.tobytes()
                    except Exception:
                        pass
                    last_preview = now

                if now - last_flush >= self.FLUSH_SEC:
                    self._flush(cur_hour)
                    self._prune_window(now)     # pencere içi bellek budaması
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
                self.reconnects += 1
                time.sleep(min(backoff, 60))

        self.status = "stopped"
