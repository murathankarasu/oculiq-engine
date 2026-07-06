"""Oculiq attention engine — local, best-in-class.

YOLO11-pose (person detection + 17 COCO keypoints) + ByteTrack (persistent IDs)
-> head-orientation from facial keypoints (primary) with body fallback
-> attention-cone test against manually drawn ad zones
-> funnel + advanced metrics: attention rate, time-to-first-look, glance count,
   stopping power, dwell distribution, attention timeline, AQS, dual CPM.

Crowd mode (tiled multi-scan):
  the frame is split into a full-frame pass + 2x2 overlapping tiles; each tile is
  scanned separately so far/small people are seen at higher resolution, results are
  merged with IoU-NMS (the same person found in two tiles collapses to one), a
  lightweight IoU+centroid tracker keeps identities, and temporal majority-voting
  smooths the "looking" state. Traffic requires >=2 sightings (no phantom counts).

All processing on-device. Nothing leaves the machine.
"""
import base64
import json
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


def _jlog(job, msg, level="info"):
    from server.main import job_log
    try:
        job_log(job, msg, level)
    except Exception:
        pass

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12

GREEN = (143, 196, 37)
GRAY = (150, 144, 138)
WHITE = (240, 240, 240)
DARK = (20, 18, 16)

ZONE_COLORS = [(117, 158, 29), (221, 138, 55), (48, 90, 216), (221, 119, 127), (23, 117, 186)]


def _device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _ang_between(a, b):
    na = math.hypot(a[0], a[1]) or 1e-9
    nb = math.hypot(b[0], b[1]) or 1e-9
    c = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    return math.acos(max(-1.0, min(1.0, c)))


def _wilson(k, n, z=1.96):
    """Wilson %95 güven aralığı — oran için (attention rate ±)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


def _ios(a, b):
    """Intersection over smaller area — parça sınırında kesilmiş kısmi kutu,
    tam kutunun içinde kaldığında da kopya sayılır (IoU bunu kaçırır)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller > 0 else 0.0


class SimpleTracker:
    """Greedy IoU + centroid tracker for tiled (crowd) mode."""

    def __init__(self, iou_thr=0.25, dist_frac=0.7, max_gone=12):
        self.iou_thr = iou_thr
        self.dist_frac = dist_frac
        self.max_gone = max_gone
        self.next_id = 0
        self.tracks = {}  # id -> {box, gone}

    def update(self, raws):
        ids = [None] * len(raws)
        used = set()
        # greedy best-IoU matching
        cands = []
        for tid, tr in self.tracks.items():
            for i, r in enumerate(raws):
                v = _iou(tr["box"], r["box"])
                if v >= self.iou_thr:
                    cands.append((v, tid, i))
        for v, tid, i in sorted(cands, reverse=True):
            if tid in used or ids[i] is not None:
                continue
            ids[i] = tid
            used.add(tid)
        # centroid fallback for unmatched
        for i, r in enumerate(raws):
            if ids[i] is not None:
                continue
            bx, by, bw, bh = r["box"]
            cx, cy = bx + bw / 2, by + bh / 2
            best, bd = None, self.dist_frac * max(bh, 1)
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                tx, ty, tw, th = tr["box"]
                d = math.hypot(tx + tw / 2 - cx, ty + th / 2 - cy)
                if d < bd:
                    bd, best = d, tid
            if best is not None:
                ids[i] = best
                used.add(best)
        # new tracks
        for i in range(len(raws)):
            if ids[i] is None:
                ids[i] = self.next_id
                self.next_id += 1
        # state update
        for i, r in enumerate(raws):
            self.tracks[ids[i]] = {"box": r["box"], "gone": 0}
        for tid in list(self.tracks):
            if tid not in ids:
                self.tracks[tid]["gone"] += 1
                if self.tracks[tid]["gone"] > self.max_gone:
                    del self.tracks[tid]
        return ids


class KCalibrator:
    """Otomatik perspektif kalibrasyonu — LLM'siz, saf geometri.

    Sabit kamerada insanlar görüntüde yukarı çıktıkça küçülür. Tespit kutularından
    boy(y) = a*y + b doğrusu artımlı en-küçük-kareler ile oturtulur; boyun sıfıra
    indiği y = UFUK çizgisidir. Pinhole geometrisinde derinliğin görüntü-y'ye
    kısalma oranı k(y) ufuktan uzaklıkla orantılıdır: k(y) = k_bot * (y-yh)/(H-yh).
    Yeterli örnek yoksa ya da fit güvenilmezse global varsayılana düşülür."""

    def __init__(self, H, fallback=0.45, k_bot=0.75):
        self.H = H
        self.fallback = fallback
        self.k_bot = k_bot
        self.n = 0
        self.sy = self.sh = self.syy = self.syh = 0.0
        self.yh = None

    def add(self, foot_y, h):
        if h < 8:
            return
        self.n += 1
        self.sy += foot_y
        self.sh += h
        self.syy += foot_y * foot_y
        self.syh += foot_y * h
        if self.n >= 8 and self.n % 4 == 0:
            self._fit()

    def _fit(self):
        d = self.n * self.syy - self.sy * self.sy
        if abs(d) < 1e-6:
            return
        a = (self.n * self.syh - self.sy * self.sh) / d
        if a <= 1e-6:                 # perspektif gradyanı yok/ters -> güvenme
            return
        b = (self.sh - a * self.sy) / self.n
        yh = -b / a
        if yh > self.H * 0.9:         # "ufuk" karenin dibinde çıktıysa saçma
            return
        self.yh = yh

    def k(self, y):
        if self.yh is None:
            return self.fallback
        t = (y - self.yh) / max(self.H - self.yh, 1e-3)
        return min(0.95, max(0.12, self.k_bot * t))

    def state(self):
        return {"auto": self.yh is not None,
                "horizon_y": round(self.yh, 1) if self.yh is not None else None,
                "k_bottom": self.k_bot, "samples": self.n}


class AttentionEngine:
    def __init__(self, model_name="yolo11x-pose.pt", cone_deg=35.0, min_dwell=0.4,
                 kp_conf=0.35, min_mag=0.10, imgsz=960, tile_imgsz=896,
                 tile_px=800, det_conf=0.25,
                 tile_overlap=0.2, merge_iou=0.5, smooth_n=3, min_sightings=3):
        from ultralytics import YOLO
        MODELS_DIR.mkdir(exist_ok=True)
        path = MODELS_DIR / model_name
        self.model = YOLO(str(path) if path.exists() else model_name)
        try:
            src = Path(model_name)
            if src.exists() and not path.exists():
                src.rename(path)
                self.model = YOLO(str(path))
        except Exception:
            pass
        self.device = _device()
        self.cone_deg = cone_deg
        self.min_dwell = min_dwell
        self.kp_conf = kp_conf
        self.min_mag = min_mag
        self.imgsz = imgsz
        self.tile_imgsz = tile_imgsz
        self.tile_px = tile_px          # hedef parça boyutu -> adaptif ızgara
        self.det_conf = det_conf        # düşük eşik: uzaktaki küçük insanlar
        self.tile_overlap = tile_overlap
        self.merge_iou = merge_iou
        self.smooth_n = smooth_n          # "looking" majority window (samples)
        self.min_sightings = min_sightings  # frames required to count as traffic
        self.persp_k = 0.45  # perspektif kısaltma VARSAYILANI (oto-kalibrasyon devralır)
        self._cal = None     # aktif işin KCalibrator'ı (process_* başında kurulur)
        self.auto_cone = True  # koni otomatiği: kişisel gürültü + bölge açısal genişliği
        self._demo_err = None
        self.use_rtmo = os.environ.get("OCULIQ_RTMO", "on") != "off"
        self._rtmo = None          # kalabalık modu: RTMO tek-geçiş (yüklenemezse tiled YOLO)
        self._rtmo_failed = False

    # ---------- orientation ----------
    # Sinyal zinciri (yakından uzağa):
    #   1. yüz landmark'ları (kulak/göz + burun)      -> "head"  conf .85/.6
    #   2. burun + omuzlar (uzak, öne dönük)          -> "body"  conf .5
    #   3. burun YOK + omuzlar (kameraya SIRTI dönük) -> omuz-normali, kameradan
    #      uzağa bakan taraf -> "body" conf .5 — billboard'a bakanlar tam bu grupta!
    def head_dir(self, kp, kc, k):
        has = lambda i: kc[i] >= self.kp_conf
        nose_v = kc[NOSE] >= 0.3
        le, re_, lea, rea = has(L_EYE), has(R_EYE), has(L_EAR), has(R_EAR)
        lsh, rsh = kc[L_SH] >= 0.3, kc[R_SH] >= 0.3

        # 1) yüz temelli (yakın/orta mesafe)
        anchors = [kp[i] for i in (L_EAR, R_EAR) if has(i)]
        if not anchors:
            anchors = [kp[i] for i in (L_EYE, R_EYE) if has(i)]
        if anchors and nose_v:
            ax = sum(p[0] for p in anchors) / len(anchors)
            ay = sum(p[1] for p in anchors) / len(anchors)
            if lea and rea:
                scale = math.dist(kp[L_EAR], kp[R_EAR])
            elif le and re_:
                scale = math.dist(kp[L_EYE], kp[R_EYE]) * 1.6
            else:
                scale = 1.0
            scale = max(scale, 1e-4)
            dx = (kp[NOSE][0] - ax) / scale
            dy = (kp[NOSE][1] - ay) / scale + 0.10
            if lea != rea and abs(dx) < 0.35:
                dx = math.copysign(0.5, dx if abs(dx) > 0.02 else (1 if rea else -1) * -1)
            mag = math.hypot(dx, dy)
            conf = 0.85 if (lea and rea) or (le and re_) else 0.6
            if mag < self.min_mag:
                return 0.0, 0.0, conf, "frontal"
            return dx / mag, dy / mag, conf, "head"

        # 2) burun + omuzlar (uzak mesafe, KAMERAYA doğru dönük) — zemin azimutu:
        #    yanal bileşen burun ofsetinden, derinlik bileşeni kameraya doğru (+).
        if nose_v and lsh and rsh:
            mx = (kp[L_SH][0] + kp[R_SH][0]) / 2
            scale = max(math.dist(kp[L_SH], kp[R_SH]), 1e-4)
            lat = max(-1.0, min(1.0, (kp[NOSE][0] - mx) / scale * 1.6))
            depth = math.sqrt(max(0.0, 1.0 - lat * lat))   # + = kameraya doğru
            return self._az_to_img(lat, depth, 0.5, "body", k)

        # 3) kameraya SIRTI dönük: omuz çizgisini zemin düzlemine aç (y /= k),
        #    normalini al, kameradan uzağa bakan tarafı seç. Kişi z'ye (derinliğe)
        #    bakıyor — görüntüye geri yansıtırken derinlik k ile kısalır, ok göğe değil
        #    sahnenin içine doğru gösterir.
        if not nose_v and lsh and rsh:
            sx = kp[R_SH][0] - kp[L_SH][0]
            sy = (kp[R_SH][1] - kp[L_SH][1]) / k
            n = math.hypot(sx, sy)
            if n > 1e-4:
                ax, ay = -sy / n, sx / n
                if ay > 0:               # kameraya bakan normal -> ters çevir (uzağa)
                    ax, ay = -ax, -ay
                return self._az_to_img(ax, ay, 0.5, "body", k)
            return None, None, 0.55, "away"

        if nose_v or le or re_:
            return None, None, 0.4, "partial"
        return None, None, 0.55, "away"

    def _az_to_img(self, ax, ay, conf, sig, k):
        """Zemin azimutu -> görüntü düzlemi vektörü (derinlik k ile kısalır).
        Bilerek normalize edilmez: saf derinliğe bakan ok görüntüde KISA çizilir
        ('sahnenin içine bakıyor'), yana bakış uzun — göğü gösteren dev ok biter."""
        return ax, ay * k, conf, sig

    def looks_at(self, det, z):
        """Koni testi — OTOMATİK koni:
        koni(kişi, bölge) = kişisel sinyal gürültüsü (boyuta göre, det['cone'])
                          + bölgenin o kişiden görünen AÇISAL YARI GENİŞLİĞİ.
        Yakındaki büyük billboard geniş açı kaplar (kenarına bakan da bakıyordur);
        uzaktaki küçük bölge dar açı ister. Elle eşik yok.
        Body sinyali zemin-azimut uzayında karşılaştırılır (kişiye özel k)."""
        dx, dy = det["dx"], det["dy"]
        if dx is None or (dx == 0 and dy == 0):
            return False
        zc = z["center"]
        zx, zy, zw, zh = z["rect"]
        cx, cy = det["c"]
        k = det.get("k", self.persp_k) if det["sig"] == "body" else 1.0
        vx, vy = zc[0] - cx, (zc[1] - cy) / k
        ddx, ddy = dx, dy / k

        if self.auto_cone:
            # bölgenin yatay uçlarına vektörler -> kapsanan açı / 2 (üst sınır 20°)
            v1 = (zx - cx, vy)
            v2 = (zx + zw - cx, vy)
            half = math.degrees(_ang_between(v1, v2)) / 2.0
            cone = det.get("cone", 20.0) + min(half, 20.0)
        else:
            cone = self.cone_deg * (0.7 if det["sig"] == "body" else 1.0)

        ang = math.degrees(_ang_between((ddx, ddy), (vx, vy)))
        return ang <= cone

    # ---------- detection ----------
    def _raw_from_res(self, res, ox=0.0, oy=0.0):
        """ultralytics result -> raw dicts, offset back to full-frame coords."""
        raws = []
        if res.boxes is None or len(res.boxes) == 0:
            return raws
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kps = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
        kcs = (res.keypoints.conf.cpu().numpy()
               if res.keypoints is not None and res.keypoints.conf is not None else None)
        for i, (b, cf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = b
            kp = kc = None
            if kps is not None and kcs is not None:
                kp = kps[i].copy()
                kp[:, 0] += ox
                kp[:, 1] += oy
                kc = kcs[i]
            raws.append({"box": (float(x1 + ox), float(y1 + oy), float(x2 - x1), float(y2 - y1)),
                         "conf": float(cf), "kp": kp, "kc": kc})
        return raws

    def _detect_tiled(self, frame):
        """Full-frame pass + adaptive overlapping tile grid, merged with IoS-NMS."""
        H, W = frame.shape[:2]
        raws = self._raw_from_res(
            self.model(frame, verbose=False, device=self.device, imgsz=self.imgsz,
                       conf=self.det_conf, classes=[0])[0])

        cols = min(3, max(2, math.ceil(W / self.tile_px)))
        rows = min(3, max(2, math.ceil(H / self.tile_px)))
        tw, th = W / cols, H / rows
        mx, my = tw * self.tile_overlap, th * self.tile_overlap
        for r in range(rows):
            for c in range(cols):
                x1 = max(0, int(c * tw - mx))
                y1 = max(0, int(r * th - my))
                x2 = min(W, int((c + 1) * tw + mx))
                y2 = min(H, int((r + 1) * th + my))
                tile = frame[y1:y2, x1:x2]
                res = self.model(tile, verbose=False, device=self.device,
                                 imgsz=self.tile_imgsz, conf=self.det_conf, classes=[0])[0]
                raws += self._raw_from_res(res, ox=x1, oy=y1)

        # NMS merge: keep highest-conf, drop overlaps (same person from 2+ tiles).
        # IoS: kısmi (parça sınırında kesilmiş) kutular da kopya olarak yakalanır.
        raws.sort(key=lambda r: r["conf"], reverse=True)
        kept = []
        for r in raws:
            if all(_ios(r["box"], k["box"]) < self.merge_iou for k in kept):
                kept.append(r)
        return kept

    def _build_det(self, raw, pid):
        x, y, w, h = raw["box"]
        # oto-kalibrasyon: her tespit örneğe katılır, k kişinin konumuna göre alınır
        if self._cal is not None:
            self._cal.add(y + h, h)
        k = self._cal.k(y + h) if self._cal is not None else self.persp_k
        d = {"id": int(pid), "box": (int(x), int(y), int(w), int(h)), "h": h, "k": round(k, 3),
             "dx": None, "dy": None, "conf": 0.5, "sig": "body", "zone": None, "cone": 20.0}
        if raw["kp"] is not None:
            kp, kc = raw["kp"], raw["kc"]
            dx, dy, conf, sig = self.head_dir(kp, kc, k)
            d.update(dx=dx, dy=dy, conf=conf, sig=sig)
            d["c"] = ((float(kp[NOSE][0]), float(kp[NOSE][1])) if kc[NOSE] >= self.kp_conf
                      else (x + w / 2, y + h * 0.15))
        else:
            d["c"] = (x + w / 2, y + h * 0.15)
        # kişisel koni: sinyal gürültüsü kişi küçüldükçe artar (oto-kalibrasyonun parçası)
        hh = max(float(h), 20.0)
        d["cone"] = round(min(16.0, 8.0 + 350.0 / hh) if d["sig"] == "head"
                          else min(20.0, 11.0 + 350.0 / hh), 1)
        # tam-boy bayrağı: 3D kalibrasyon yalnızca tam görünen kişilerden beslenir
        # (kesik/oturan/şemsiyeli gövdeler boy örneklemini zehirliyordu)
        d["fb_ok"] = False
        if raw["kp"] is not None:
            kc_ = raw["kc"]
            d["fb_ok"] = bool((kc_[15] >= 0.3 or kc_[16] >= 0.3)
                              and (kc_[0] >= 0.3 or kc_[1] >= 0.3 or kc_[2] >= 0.3))
        # yüz kutusu (audience insights için): görünür yüz keypoint'lerinden
        if raw["kp"] is not None:
            kp, kc = raw["kp"], raw["kc"]
            pts = [kp[i] for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if kc[i] >= 0.25]
            if len(pts) >= 3:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                span = max(max(p[0] for p in pts) - min(p[0] for p in pts),
                           max(p[1] for p in pts) - min(p[1] for p in pts), h * 0.18)
                half = span * 1.1
                d["face_box"] = (cx - half, cy - half, half * 2, half * 2.2)
        return d

    def _detect_frame(self, frame, tiled, tracker):
        if tiled:
            if self.use_rtmo and not self._rtmo_failed:
                try:
                    if self._rtmo is None:
                        from server.pose_rtmo import RtmoDetector
                        self._rtmo = RtmoDetector()
                    raws = self._rtmo.detect(frame)
                    ids = tracker.update(raws)
                    return [self._build_det(r, i) for r, i in zip(raws, ids)]
                except Exception:
                    self._rtmo_failed = True   # sessizce parçalı YOLO'ya dön
            raws = self._detect_tiled(frame)
            ids = tracker.update(raws)
        else:
            res = self.model.track(frame, persist=True, verbose=False, device=self.device,
                                   imgsz=self.imgsz, conf=self.det_conf, classes=[0])[0]
            raws = self._raw_from_res(res)
            if res.boxes is not None and len(res.boxes) and res.boxes.id is not None:
                ids = res.boxes.id.cpu().numpy().astype(int)
            else:
                ids = list(range(len(raws)))
        return [self._build_det(r, i) for r, i in zip(raws, ids)]

    def _use_tiles(self, W, H, mode):
        if mode == "on":
            return True
        if mode == "off":
            return False
        return W * H >= 1920 * 1080  # auto: hi-res CCTV -> tiled

    # ---------- video ----------
    def process_video(self, path, zones, job, cost_map=None, sample_fps=10,
                      max_seconds=None, crowd_mode="auto", demographics=False,
                      face_blur=True):
        cap = cv2.VideoCapture(str(path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = n_frames / src_fps if src_fps else 0
        if max_seconds:
            duration = min(duration, max_seconds)
        stride = max(1, round(src_fps / sample_fps))
        dt = stride / src_fps

        _jlog(job, f"Video {W}x{H} · {duration:.1f}s @ {src_fps:.1f}fps — sampling {sample_fps}fps, real frame timestamps (VFR-safe)")
        tiled = self._use_tiles(W, H, crowd_mode)
        _jlog(job, f"Scan mode: {'tiled multi-scan (crowd)' if tiled else 'single-pass'}")
        if tiled and self.use_rtmo and not self._rtmo_failed:
            try:
                if self._rtmo is None:
                    from server.pose_rtmo import RtmoDetector
                    _jlog(job, "Loading RTMO crowd-pose engine (one-stage, Apache-2.0)…")
                    self._rtmo = RtmoDetector()
                _jlog(job, "Crowd engine: RTMO one-stage — single pass replaces 7-pass tiling", "ok")
            except Exception as e:
                self._rtmo_failed = True
                _jlog(job, f"RTMO unavailable ({e}) — falling back to tiled YOLO", "warn")
        if face_blur:
            _jlog(job, "GDPR: face blurring active on all outputs")
        tracker = SimpleTracker(max_gone=int(2.0 / dt))  # ~2s memory
        self._cal = KCalibrator(H, fallback=self.persp_k)  # oto perspektif kalibrasyonu
        foot_samples = []  # scene3d icin: (foot_u, foot_v, h_px)
        sim_frame_img = None
        scene = None       # sahne 3D modeli (video ortasinda kurulur, overlay'e girer)
        zquads = {}
        zs = self._prep_zones(zones, W, H)
        persons = {}
        heat = np.zeros((H // 4, W // 4), np.float32)
        timeline = defaultdict(lambda: defaultdict(float))
        density = defaultdict(lambda: [0, 0])  # bucket -> [sum, frames]
        peak = 0
        rays = []               # what-if simülasyonu için kayıtlı bakış ışınları
        sim_frame_saved = False
        gaze3d_n = 0            # kaç bakış örneği gerçek 3D testten geçti
        gaze_total = 0
        scene_ok = False        # güven kapısı: 3D yalnızca güvenilirse kararlara girer
        demo_note = None
        if demographics:        # sınıflandırıcıyı BAŞTA yükle (sessiz gecikme/çökme olmasın)
            try:
                from server import demographics as demo
                job["status"] = "loading-model"
                _jlog(job, "Loading audience classifiers (gender+age — first run downloads ~700MB)…")
                demo.preload()
                _jlog(job, "Audience classifiers ready", "ok")
                job["status"] = "processing"
            except Exception as e:
                demographics = False
                demo_note = f"gender model unavailable: {type(e).__name__}: {e}"
                _jlog(job, f"Audience classifiers FAILED: {e}", "err")


        out_path = Path(path).with_suffix(".annotated.mp4")
        out_fps = 1.0 / dt
        writer = self._writer(out_path, W, H, out_fps)
        written = 0             # VFR telafisi: çıktı, gerçek zamana hizalanır
        t_first = None
        prev_t = None

        fi = 0
        t0 = time.time()
        first_checked = False
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # GERÇEK zaman damgası (VFR kaynaklarda nominal sayaç kayar!)
            tm = cap.get(cv2.CAP_PROP_POS_MSEC)
            t_real = tm / 1000.0 if tm and tm > 0 else fi / src_fps
            if fi % stride:
                fi += 1
                continue
            t = t_real
            if max_seconds and t > max_seconds:
                break
            fi += 1
            if t_first is None:
                t_first = t
            # metrik birikimi gerçek kare aralığıyla (dwell saniyeleri doğru olsun)
            dtf = dt if prev_t is None else min(max(t - prev_t, 0.01), 0.5)
            prev_t = t

            dets = self._detect_frame(frame, tiled, tracker)

            # auto-upgrade: crowded scene detected -> switch to tiled multi-scan
            if not first_checked:
                first_checked = True
                if not tiled and crowd_mode == "auto" and len(dets) >= 10:
                    tiled = True
                    _jlog(job, f"{len(dets)} people in first frame — auto-switching to tiled multi-scan")
                    dets = self._detect_frame(frame, tiled, tracker)

            peak = max(peak, len(dets))
            b = int(t // 2) * 2
            density[b][0] += len(dets)
            density[b][1] += 1

            # what-if arka planı: video ~%25'indeyken temiz (overlay'siz) bir kare sakla
            if not sim_frame_saved and t >= duration * 0.25:
                sf = Path(path).with_suffix(".sim.jpg")
                sim_save = frame.copy()
                if face_blur:   # GDPR: what-if arka planinda da yuz kalmasin
                    for d in dets:
                        fb = d.get("face_box")
                        if not fb:
                            continue
                        x0 = max(0, int(fb[0])); y0 = max(0, int(fb[1]))
                        x1 = min(W, int(fb[0] + fb[2])); y1 = min(H, int(fb[1] + fb[3]))
                        if x1 - x0 < 4 or y1 - y0 < 4:
                            continue
                        roi = sim_save[y0:y1, x0:x1]
                        sm = cv2.resize(roi, (max(2, (x1 - x0) // 10), max(2, (y1 - y0) // 10)))
                        sim_save[y0:y1, x0:x1] = cv2.resize(sm, (x1 - x0, y1 - y0),
                                                            interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(sf), sim_save)
                job["sim_frame"] = str(sf)
                sim_frame_img = frame.copy()
                sim_frame_saved = True

            # 3D sahneyi canlı kur: yeterli insan örneği birikince (bir kez).
            # NOT: bölge testlerinden ÖNCE — bu karenin kararları 3D geometriyle verilsin.
            if scene is None and sim_frame_saved and len(foot_samples) >= 12:
                try:
                    from server.scene3d import SceneModel
                    job["status"] = "scene-3d"
                    _jlog(job, "Reconstructing 3D scene (metric depth model)…")
                    scene = SceneModel().build(sim_frame_img, foot_samples)
                    scene_ok = scene.reliable()   # GÜVEN KAPISI: düşükse 3D devre dışı
                    if scene.enabled:
                        hm = scene.height_mean
                        _jlog(job,
                              f"3D calibration: {len(foot_samples)} full-body samples, "
                              f"{getattr(scene, 'inliers', 0)} inliers"
                              + (f" → height {hm:.2f}m ± {scene.height_std:.2f}" if hm else "")
                              + f" → {scene.confidence:.0f}% — "
                              + ("ACTIVE (3D gaze + 3D zones)" if scene_ok
                                 else "too low, falling back to 2.5D (honest mode)"),
                              "ok" if scene_ok else "warn")
                    else:
                        _jlog(job, f"3D scene unavailable: {scene.note}", "warn")
                    if scene.enabled:
                        view = scene.render_view()   # LiDAR-tarzı rekonstrüksiyon (tanı)
                        if view is not None:
                            vp = Path(path).with_suffix(".scene.jpg")
                            cv2.imwrite(str(vp), view)
                            job["scene_view"] = str(vp)
                    if scene_ok:
                        scene._grid = scene.grid_segments()
                        zquads = {z["id"]: scene.zone_quad(z) for z in zs}
                        for zz, q in zip(zs, zquads.values()):   # yüzey mesh'i bir kez hesapla
                            if q:
                                q["_mesh"], q["_nseg"] = scene.zone_mesh(q)
                                _jlog(job, f"{zz['label']}: placed on 3D surface — "
                                           f"{q['w_m']}x{q['h_m']}m @ {q['depth_m']}m"
                                           + (f", tilt {q['tilt_deg']:.0f}°" if q.get('tilt_deg') is not None else ""), "ok")
                    job["status"] = "processing"
                except Exception:
                    scene = None

            # 3D bakış: kafa konumu + dünya-uzayı yön (yalnızca güven kapısı AÇIKKEN)
            if scene_ok:
                for d in dets:
                    if d["dx"] is None or (d["dx"] == 0 and d["dy"] == 0):
                        continue
                    bx, by, bw, bh = d["box"]
                    h3 = scene.head_pos(bx + bw / 2.0, by + float(bh))
                    if h3 is None:
                        continue
                    # sağlama: kafa 3D'si, kişinin gerçek baş pikseline geri düşmeli
                    pr = scene.project(h3)
                    if (pr is None or abs(pr[0] - (bx + bw / 2.0)) > bh * 1.2
                            or abs(pr[1] - by) > bh * 1.2):
                        continue   # geometri bu kişi için tutarsız -> 2.5D kal
                    g3 = scene.gaze_dir3d(d["dx"], d["dy"], d.get("k", self.persp_k), d["sig"])
                    if g3 is not None:
                        d["head3"], d["dir3"] = h3, g3

            for d in dets:
                p = persons.setdefault(d["id"], {
                    "first_t": t, "frames": 0, "dwell": defaultdict(float),
                    "episodes": defaultdict(int), "looking": defaultdict(bool),
                    "hist": defaultdict(lambda: deque(maxlen=self.smooth_n)),
                    "intervals": defaultdict(list),
                    "first_look": {}, "pos": None, "v": [], "v_look": [],
                    "sig_sec": defaultdict(float),
                })
                p["frames"] += 1
                if p["pos"] is not None:
                    v = math.dist(d["c"], p["pos"]) / dtf / max(d["h"], 1)
                    p["v"].append(v)
                    d["v"] = v
                p["pos"] = d["c"]
                # scene3d örnekleri + kişinin ortalama ayak noktası
                bx, by, bw, bh = d["box"]
                foot = (bx + bw / 2.0, by + float(bh))
                if (len(foot_samples) < 400 and d.get("fb_ok")
                        and foot[1] < H - 4 and bh >= 40):
                    foot_samples.append((foot[0], foot[1], float(bh)))
                fs = p.setdefault("foot_sum", [0.0, 0.0, 0])
                fs[0] += foot[0]; fs[1] += foot[1]; fs[2] += 1

                # bakış ışını kaydı: yön taşıyan her ölçüm (frontal (0,0) hariç)
                if (d["dx"] is not None and (d["dx"] or d["dy"]) and len(rays) < 80000):
                    ray = [round(t, 2), d["id"], int(d["c"][0]), int(d["c"][1]),
                           round(d["dx"], 3), round(d["dy"], 3),
                           round(d.get("v", 0), 2),
                           1 if d["sig"] == "body" else 0,
                           d["k"], d["cone"]]
                    if "head3" in d:  # 3D alanlar: kafa konumu + dünya yönü
                        ray += [round(float(v), 2) for v in d["head3"]]
                        ray += [round(float(v), 3) for v in d["dir3"]]
                    rays.append(ray)

                for z in zs:
                    q = zquads.get(z["id"])
                    if "head3" in d and q:   # Faz 2: gerçek 3D ışın-yüzey testi
                        raw_look = (d["sig"] in ("head", "body")
                                    and scene.looks_at_3d(d["head3"], d["dir3"], q,
                                                          d.get("cone", 14.0), d["sig"]))
                        gaze3d_n += 1
                    else:                    # sahne yokken 2.5D azimut testi
                        raw_look = (d["sig"] in ("head", "body")
                                    and self.looks_at(d, z))
                    gaze_total += 1
                    p["hist"][z["id"]].append(raw_look)
                    hist = p["hist"][z["id"]]
                    # temporal majority vote — "oyun mantığı": tek karelik titremeyi ele
                    look = (sum(hist) * 2 > len(hist)) if len(hist) >= 2 else raw_look
                    if look:
                        p["dwell"][z["id"]] += dtf
                        p["sig_sec"][d["sig"]] += dtf
                        timeline[b][z["id"]] += dtf
                        if not p["looking"][z["id"]]:
                            p["episodes"][z["id"]] += 1
                            p["first_look"].setdefault(z["id"], t)
                            p["intervals"][z["id"]].append([t, t + dtf])
                        else:
                            p["intervals"][z["id"]][-1][1] = t + dtf
                        if "v" in d:
                            p["v_look"].append(d["v"])
                        d["zone"] = z["id"]
                        # yumuşatma, yönsüz karede de "bakıyor" diyebilir — ısı haritası yön ister
                        if d["dx"] is not None and (d["dx"] or d["dy"]):
                            self._heat_add(heat, d, z, W, H)
                    p["looking"][z["id"]] = look

            if demographics:
                ok_demo = self._gender_pass(frame, dets, persons)
                if not ok_demo:
                    demographics = False
                    demo_note = self._demo_err or "classifier unavailable"


            annotated = self._draw(frame, dets, zs, heat, t, len(persons), tiled,
                                   scene=scene if scene_ok else None,
                                   zquads=zquads, blur=face_blur)
            want = max(written + 1, int(round((t - t_first) * out_fps)) + 1)
            for _ in range(min(4, want - written)):
                writer.write(annotated)
                written += 1

            if fi % (stride * 3) < stride:
                job["preview"] = self._jpeg_b64(annotated, 960)
            new_pct = min(99, int(t / max(duration, 0.1) * 100))
            for ms in (25, 50, 75):
                if job["progress"] < ms <= new_pct:
                    _jlog(job, f"{ms}% — {len(persons)} people tracked so far")
            job["progress"] = new_pct
            job["live"] = self._live(persons, zs, t)
            if job.get("cancel"):
                break

        cap.release()
        writer.release()
        _jlog(job, "Annotated video rendered — building report…")
        job["out_video"] = str(out_path)
        if not sim_frame_saved and "preview" in job:  # çok kısa video: eldeki kareyi kullan
            sf = Path(path).with_suffix(".sim.jpg")
            cv2.imwrite(str(sf), frame if frame is not None else np.zeros((H, W, 3), np.uint8))
            job["sim_frame"] = str(sf)

        # stability filter: ghosts seen in a single sampled frame don't count
        persons = {k: v for k, v in persons.items() if v["frames"] >= self.min_sightings}
        rays = [r for r in rays if r[1] in persons]  # hayaletlerin ışınları da elenir
        sim = {"w": W, "h": H, "dt": round(dt, 4), "cone_deg": self.cone_deg,
               "min_dwell": self.min_dwell, "k": self.persp_k, "auto_cone": self.auto_cone,
               "persons": {str(k): round(v["first_t"], 2) for k, v in persons.items()},
               "rays": rays}
        if scene_ok and scene.up() is not None:
            grid = scene.depth_grid()
            if grid:
                sim["s3"] = {"f": round(scene.f, 1), "cx": scene.cx, "cy": scene.cy,
                             "up": [round(float(v), 4) for v in scene.up()],
                             "grid": grid}
        report = self._report(persons, zs, timeline, duration, peak, cost_map or {},
                              still=False, elapsed=time.time() - t0, sim=sim)
        report["scan_mode"] = "tiled multi-scan (crowd)" if tiled else "single-pass"
        report["calibration"] = self._cal.state()
        self._attach_scene3d(report, sim_frame_img if sim_frame_img is not None else frame,
                             foot_samples, persons, zs, job, prebuilt=scene)
        if gaze_total and isinstance(report.get("scene3d"), dict):
            report["scene3d"]["gaze3d_pct"] = round(gaze3d_n / gaze_total * 100, 1)
        if demographics and persons:
            from server import demographics as demo
            report["audience"] = demo.aggregate(persons, zs, self.min_dwell)
        elif demo_note:
            report["audience"] = {"enabled": False, "note": demo_note}
        if density:
            report["density_timeline"] = [
                {"t": b, "avg": round(s / max(f, 1), 1)} for b, (s, f) in sorted(density.items())]
            report["avg_concurrency"] = round(
                sum(s for s, _ in density.values()) / max(sum(f for _, f in density.values()), 1), 1)
        return report

    # ---------- image ----------
    def process_image(self, path, zones, job, cost_map=None, crowd_mode="auto",
                      demographics=False, face_blur=True):
        frame = cv2.imread(str(path))
        H, W = frame.shape[:2]
        tiled = self._use_tiles(W, H, crowd_mode)
        self._cal = KCalibrator(H, fallback=self.persp_k)
        zs = self._prep_zones(zones, W, H)

        if tiled:
            raws = self._detect_tiled(frame)
        else:
            res = self.model(frame, verbose=False, device=self.device,
                             imgsz=self.imgsz, classes=[0])[0]
            raws = self._raw_from_res(res)
            if crowd_mode == "auto" and len(raws) >= 10:
                tiled = True
                raws = self._detect_tiled(frame)
        dets = [self._build_det(r, i) for i, r in enumerate(raws)]
        heat = np.zeros((H // 4, W // 4), np.float32)

        rays = [[0.0, d["id"], int(d["c"][0]), int(d["c"][1]),
                 round(d["dx"], 3), round(d["dy"], 3), 0.0,
                 1 if d["sig"] == "body" else 0, d["k"], d["cone"]]
                for d in dets if d["dx"] is not None and (d["dx"] or d["dy"])]
        persons = {}
        for i, d in enumerate(dets):
            p = {"first_t": 0, "frames": self.min_sightings, "dwell": defaultdict(float),
                 "episodes": defaultdict(int), "first_look": {}, "v": [], "v_look": [],
                 "sig_sec": defaultdict(float)}
            for z in zs:
                if (d["sig"] in ("head", "body")
                        and self.looks_at(d, z)):
                    p["dwell"][z["id"]] = 1.0
                    p["episodes"][z["id"]] = 1
                    p["sig_sec"][d["sig"]] += 1
                    d["zone"] = z["id"]
                    self._heat_add(heat, d, z, W, H)
            persons[i] = p

        annotated = self._draw(frame, dets, zs, heat, None, len(dets), tiled, blur=face_blur)
        out_path = Path(path).with_suffix(".annotated.jpg")
        cv2.imwrite(str(out_path), annotated)
        job["out_image"] = str(out_path)
        job["sim_frame"] = str(path)  # fotoda arka plan = orijinal görsel
        job["preview"] = self._jpeg_b64(annotated, 1280)
        job["progress"] = 100
        sim = {"w": W, "h": H, "dt": 1.0, "cone_deg": self.cone_deg, "min_dwell": 0.5,
               "k": self.persp_k, "auto_cone": self.auto_cone,
               "persons": {str(k): 0.0 for k in persons}, "rays": rays}
        report = self._report(persons, zs, {}, 0, len(dets), cost_map or {},
                              still=True, elapsed=0, sim=sim)
        report["scan_mode"] = "tiled multi-scan (crowd)" if tiled else "single-pass"
        report["calibration"] = self._cal.state()
        foot_samples = [(d["box"][0] + d["box"][2] / 2.0,
                         d["box"][1] + float(d["box"][3]), float(d["box"][3]))
                        for d in dets if d.get("fb_ok") and d["box"][3] >= 40]
        if len(foot_samples) < 4:   # tam-boy yoksa eski davranış (foto)
            foot_samples = [(d["box"][0] + d["box"][2] / 2.0,
                             d["box"][1] + float(d["box"][3]), float(d["box"][3]))
                            for d in dets]
        for i, d in enumerate(dets):  # fotoda kişi ayağı = tek örnek
            persons[i]["foot_sum"] = [foot_samples[i][0], foot_samples[i][1], 1]
        self._attach_scene3d(report, frame, foot_samples, persons, zs, job)
        if demographics and persons:
            id_persons = {d["id"]: persons[i] for i, d in enumerate(dets)}
            if self._gender_pass(frame, dets, id_persons):
                from server import demographics as demo
                report["audience"] = demo.aggregate(persons, zs, 0.5)
        # foto çıktısını 3D overlay ile yeniden bas (ızgara + halkalar + metre etiketi)
        try:
            from server.scene3d import SceneModel
            if report["scene3d"].get("enabled") and foot_samples:
                sc = SceneModel().build(frame, foot_samples)
                if sc.enabled and sc.reliable():
                    sc._grid = sc.grid_segments()
                    zq = {z["id"]: sc.zone_quad(z) for z in zs}
                    for q in zq.values():
                        if q:
                            q["_mesh"], q["_nseg"] = sc.zone_mesh(q)
                    annotated = self._draw(frame, dets, zs, heat, None, len(dets), tiled,
                                           scene=sc, zquads=zq, blur=face_blur)
                    cv2.imwrite(str(out_path), annotated)
                    job["preview"] = self._jpeg_b64(annotated, 1280)
        except Exception:
            pass
        return report

    # ---------- audience (opt-in, toplu) ----------
    def _gender_pass(self, frame, dets, persons):
        """Görünür-yüzlü kişilere gender oyu ekler. Hata olursa özelliği kapatır."""
        self._demo_err = None
        try:
            from server import demographics as demo
            batch, refs = [], []
            for d in dets:
                p = persons.get(d["id"])
                if p is None:
                    continue
                gv = p.setdefault("gender_votes", {})
                av = p.setdefault("age_votes", {})
                if sum(gv.values()) >= 4:      # kişi başına yeterli oy toplandı
                    continue
                fb = d.get("face_box")
                if not fb or fb[2] < demo.MIN_FACE_PX or fb[3] < demo.MIN_FACE_PX:
                    continue
                x, y, w, h = [int(v) for v in fb]
                crop = frame[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]
                if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                    continue
                batch.append(crop)
                refs.append((gv, av))
            if batch:
                for (lbl, score, age_b, age_s), (gv, av) in zip(demo.classify(batch), refs):
                    if score >= demo.MIN_SCORE:
                        gv[lbl] = gv.get(lbl, 0.0) + score
                    if age_b and age_s >= demo.AGE_MIN_SCORE:
                        av[age_b] = av.get(age_b, 0.0) + age_s
            return True
        except Exception as e:
            self._demo_err = f"{type(e).__name__}: {e}"
            return False

    # ---------- scene3d ----------
    def _attach_scene3d(self, report, frame, foot_samples, persons, zs, job, prebuilt=None):
        """3D sahne kalibrasyonu: derinlik + odak + zemin düzlemi + bölge boyutu +
        izleme mesafeleri. Başarısızlıkta rapor sadece enabled:false taşır."""
        if frame is None or not foot_samples:
            report["scene3d"] = {"enabled": False, "note": "no frame/person samples"}
            return
        try:
            from server.scene3d import SceneModel
            job["status"] = "scene-3d"
            if prebuilt is not None and prebuilt.enabled:
                sm = prebuilt.refit(foot_samples)  # tüm örneklerle güveni tazele (ucuz)
            else:
                sm = SceneModel().build(frame, foot_samples)
            report["scene3d"] = sm.state()
            if not sm.enabled:
                return
            # rekonstrüksiyon görüntüsü (tanı amaçlı — foto dahil her iş için)
            if not job.get("scene_view"):
                view = sm.render_view()
                if view is not None:
                    vp = Path(job.get("out_video") or job.get("out_image") or "x").parent / "input.scene.jpg"
                    cv2.imwrite(str(vp), view)
                    job["scene_view"] = str(vp)
            # GÜVEN KAPISI: kalibrasyon güvenilmezse 3D türevli rakamlar RAPORA GİRMEZ
            if not sm.reliable():
                return
            # bölge 3D'si + izleme mesafesi (bakanların ortalama ayak konumundan)
            for z, zr in zip(zs, report["zones"]):
                q = sm.zone_quad(z)
                if not q:
                    continue
                zr["size_m"] = [q["w_m"], q["h_m"]]
                zr["zone_depth_m"] = q["depth_m"]
                if q.get("tilt_deg") is not None:
                    zr["surface_tilt_deg"] = q["tilt_deg"]
                dists = []
                for p in persons.values():
                    if p["dwell"][z["id"]] < self.min_dwell:
                        continue
                    fs = p.get("foot_sum")
                    if not fs or not fs[2]:
                        continue
                    pos = sm.person_pos(fs[0] / fs[2], fs[1] / fs[2])
                    if pos is not None:
                        import numpy as _np
                        dists.append(float(_np.linalg.norm(q["center"] - pos)))
                if dists:
                    zr["avg_view_distance_m"] = round(sum(dists) / len(dists), 1)
        except Exception as e:
            report["scene3d"] = {"enabled": False, "note": f"{type(e).__name__}: {e}"}

    # ---------- internals ----------
    def _prep_zones(self, zones, W, H):
        out = []
        for i, z in enumerate(zones):
            x, y, w, h = z["x"] * W, z["y"] * H, z["w"] * W, z["h"] * H
            poly = z.get("poly")
            poly_px = None
            center = (x + w / 2, y + h / 2)
            if poly and len(poly) == 4:
                poly_px = [(float(px) * W, float(py) * H) for px, py in poly]
                center = (sum(p[0] for p in poly_px) / 4, sum(p[1] for p in poly_px) / 4)
            out.append({
                "id": z.get("id", i), "label": z.get("label", f"Zone {i+1}"),
                "type": z.get("type", "billboard"),
                "rect": (int(x), int(y), int(w), int(h)),
                "norm": (z["x"], z["y"], z["w"], z["h"]),
                "poly_px": poly_px,
                "poly_norm": poly if poly and len(poly) == 4 else None,
                "center": center,
                "color": ZONE_COLORS[i % len(ZONE_COLORS)],
            })
        return out

    def _heat_add(self, heat, d, z, W, H):
        zc = z["center"]
        t = max(0.0, (zc[0] - d["c"][0]) * d["dx"] + (zc[1] - d["c"][1]) * d["dy"])
        px = d["c"][0] + d["dx"] * t
        py = d["c"][1] + d["dy"] * t
        x, y, w, h = z["rect"]
        px = min(max(px, x), x + w) / 4
        py = min(max(py, y), y + h) / 4
        cv2.circle(heat, (int(px), int(py)), max(6, W // 200), 1.0, -1)

    def _live(self, persons, zs, t):
        live = {"t": round(t, 1), "traffic": len(persons), "zones": {}}
        for z in zs:
            att = sum(p["dwell"][z["id"]] for p in persons.values())
            lk = sum(1 for p in persons.values() if p["dwell"][z["id"]] >= self.min_dwell)
            live["zones"][str(z["id"])] = {"label": z["label"], "lookers": lk, "att": round(att, 1)}
        return live

    def _report(self, persons, zs, timeline, duration, peak, cost_map, still, elapsed, sim=None):
        traffic = len(persons)
        zones_out = []
        for z in zs:
            zid = z["id"]
            dwells = [p["dwell"][zid] for p in persons.values() if p["dwell"][zid] >= self.min_dwell]
            imp = len(dwells)
            att = sum(p["dwell"][zid] for p in persons.values())
            engaged = sum(1 for d in dwells if d >= 1.0)
            deep = sum(1 for d in dwells if d >= 3.0)
            ttfl = [p["first_look"][zid] - p["first_t"] for p in persons.values()
                    if zid in p["first_look"]]
            glances = [p["episodes"][zid] for p in persons.values() if p["episodes"][zid] > 0]
            v_all = [v for p in persons.values() for v in p["v"]]
            v_look = [v for p in persons.values() for v in p["v_look"]]
            slowdown = 0.0
            if v_all and v_look:
                va, vl = np.mean(v_all), np.mean(v_look)
                slowdown = max(0.0, (1 - vl / va) * 100) if va > 0 else 0.0

            rate = imp / traffic if traffic else 0
            avg_dwell = att / imp if imp else 0
            aqs = 100 * (0.30 * rate
                         + 0.30 * min(avg_dwell / 5, 1)
                         + 0.20 * (deep / imp if imp else 0)
                         + 0.20 * min(slowdown / 50, 1)) if not still else \
                  100 * (0.6 * rate + 0.4 * (imp > 0))

            cost = float(cost_map.get(str(zid), 0) or 0)
            hist = [0] * 5
            for d in dwells:
                hist[0 if d < 1 else 1 if d < 2 else 2 if d < 3 else 3 if d < 5 else 4] += 1

            ci_lo, ci_hi = _wilson(imp, traffic)

            # kanıt aralıkları: en uzun bakışlar (denetlenebilir ölçüm)
            evidence = []
            for pid_, p in persons.items():
                for s, e in p.get("intervals", {}).get(zid, []):
                    if e - s >= self.min_dwell:
                        evidence.append({"pid": pid_, "start": round(s, 1), "dur": round(e - s, 1)})
            evidence.sort(key=lambda x: -x["dur"])

            zones_out.append({
                "id": zid, "label": z["label"], "type": z["type"],
                "color": "#%02x%02x%02x" % (z["color"][2], z["color"][1], z["color"][0]),
                "norm": list(z["norm"]),
                "poly": z.get("poly_norm"),
                "traffic": traffic,
                "impressions": imp,
                "impressions_ci": [round(ci_lo * traffic), round(ci_hi * traffic)],
                "attention_rate": round(rate * 100, 1),
                "attention_rate_ci": [round(ci_lo * 100, 1), round(ci_hi * 100, 1)],
                "evidence": evidence[:8],
                "attentive_seconds": round(att, 1),
                "avg_dwell": round(avg_dwell, 2),
                "max_dwell": round(max(dwells), 2) if dwells else 0,
                "engaged": engaged, "deep": deep,
                "time_to_first_look": round(float(np.mean(ttfl)), 2) if ttfl else None,
                "glances_per_looker": round(float(np.mean(glances)), 2) if glances else 0,
                "stopping_power": round(float(slowdown), 1),
                "dwell_histogram": hist,
                "signal_share": self._sig_share(persons, zid),
                "aqs": round(float(aqs), 1),
                "cost": cost,
                "reach_cpm": round(cost / (imp / 1000), 2) if cost and imp else None,
                "attention_cpm": round(cost / (att / 1000), 2) if cost and att else None,
                "timeline": [{"t": b, "sec": round(v.get(zid, 0), 2)}
                             for b, v in sorted(timeline.items())],
            })

        report = {
            "method": "orientation-based attention (head-pose primary, on-device)",
            "model": "YOLO11-pose x + ByteTrack",
            "still": still,
            "duration": round(duration, 1),
            "traffic": traffic,
            "peak_concurrency": peak,
            "processing_seconds": round(elapsed, 1),
            "zones": zones_out,
        }
        if sim is not None:
            report["sim"] = sim
        return json.loads(json.dumps(report, default=lambda o: round(float(o), 3)))

    def _sig_share(self, persons, zid):
        tot = defaultdict(float)
        for p in persons.values():
            if p["dwell"][zid] > 0:
                for k, v in p["sig_sec"].items():
                    tot[k] += v
        s = sum(tot.values()) or 1
        return {k: round(v / s * 100, 1) for k, v in tot.items()}

    # ---------- rendering ----------
    def _draw(self, frame, dets, zs, heat, t, traffic, tiled=False, scene=None,
              zquads=None, blur=True):
        out = frame.copy()
        H, W = out.shape[:2]
        sc = max(W / 1280, 0.8)

        # --- GDPR: yüzleri pikselleştir (tüm yazılı çıktılar; kaynak kare dokunulmaz) ---
        if blur:
            for d in dets:
                fb = d.get("face_box")
                if not fb:
                    continue
                x0 = max(0, int(fb[0])); y0 = max(0, int(fb[1]))
                x1 = min(W, int(fb[0] + fb[2])); y1 = min(H, int(fb[1] + fb[3]))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                roi = out[y0:y1, x0:x1]
                small = cv2.resize(roi, (max(2, (x1 - x0) // 10), max(2, (y1 - y0) // 10)),
                                   interpolation=cv2.INTER_LINEAR)
                out[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0),
                                               interpolation=cv2.INTER_NEAREST)

        # --- AR katmanı: 3D zemin ızgarası gerçek yüzeye oturur (güven-renkli) ---
        if scene is not None and scene.enabled and getattr(scene, "_grid", None):
            gcol = ((154, 211, 47) if scene.confidence >= 70      # yeşil: güvenilir
                    else (39, 159, 239) if scene.confidence >= 40  # amber: orta
                    else (74, 75, 226))                            # kırmızı: düşük
            ov = out.copy()
            for (a, b, major) in scene._grid:
                cv2.line(ov, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         gcol, 2 if major else 1, cv2.LINE_AA)
            out = cv2.addWeighted(ov, 0.25, out, 0.75, 0)
            # kişilerin ayağında zemine yapışık AR çapa halkaları
            for d in dets:
                bx, by, bw, bh = d["box"]
                ring = scene.ground_ring(bx + bw / 2.0, by + float(bh))
                if ring:
                    cv2.polylines(out, [np.array(ring, np.int32)], True, gcol, 2, cv2.LINE_AA)

        if heat.max() > 0:
            hm = cv2.GaussianBlur(heat, (0, 0), 6)
            hm = (np.clip(hm / (hm.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
            hm_c = cv2.applyColorMap(hm, cv2.COLORMAP_TURBO)
            hm_c = cv2.resize(hm_c, (W, H))
            mask = cv2.resize(hm, (W, H)).astype(np.float32) / 255 * 0.55
            out = (out * (1 - mask[..., None]) + hm_c * mask[..., None]).astype(np.uint8)

        for z in zs:
            x, y, w, h = z["rect"]
            ppx = z.get("poly_px")
            pts = np.array(ppx, np.int32) if ppx else None
            ov = out.copy()
            if pts is not None:
                cv2.fillPoly(ov, [pts], z["color"])
            else:
                cv2.rectangle(ov, (x, y), (x + w, y + h), z["color"], -1)
            out = cv2.addWeighted(ov, 0.12, out, 0.88, 0)
            q = (zquads or {}).get(z["id"])
            # bölge yüzeye 3D oturmuş: tel-kafes + yüzey normali oku
            if q and q.get("_mesh"):
                ovm = out.copy()
                for (a, b) in q["_mesh"]:
                    cv2.line(ovm, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                             z["color"], 1, cv2.LINE_AA)
                out = cv2.addWeighted(ovm, 0.45, out, 0.55, 0)
                if q.get("_nseg"):
                    (p1, p2) = q["_nseg"]
                    cv2.arrowedLine(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                                    (12, 12, 12), max(3, int(2 * sc) + 2), cv2.LINE_AA, tipLength=0.3)
                    cv2.arrowedLine(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                                    z["color"], max(2, int(2 * sc)), cv2.LINE_AA, tipLength=0.3)
            if pts is not None:
                cv2.polylines(out, [pts], True, z["color"], max(2, int(2 * sc)), cv2.LINE_AA)
            else:
                cv2.rectangle(out, (x, y), (x + w, y + h), z["color"], max(2, int(2 * sc)))
            n_look = sum(1 for d in dets if d["zone"] == z["id"])
            label = f'{z["label"]}  |  {n_look} looking'
            if q:
                label += f'  |  {q["w_m"]}x{q["h_m"]}m @ {q["depth_m"]}m'
                if q.get("tilt_deg") is not None:
                    label += f'  |  tilt {q["tilt_deg"]:.0f}°'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, 2)
            cv2.rectangle(out, (x, y - th - 14), (x + tw + 14, y), z["color"], -1)
            cv2.putText(out, label, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 * sc, WHITE, 2, cv2.LINE_AA)

        for d in dets:
            x, y, w, h = d["box"]
            col = GREEN if d["zone"] is not None else GRAY
            L = max(10, int(min(w, h) * 0.22))
            th = max(2, int(2 * sc))
            for (cx_, cy_, sx, sy) in ((x, y, 1, 1), (x + w, y, -1, 1),
                                       (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
                cv2.line(out, (cx_, cy_), (cx_ + sx * L, cy_), col, th)
                cv2.line(out, (cx_, cy_), (cx_, cy_ + sy * L), col, th)
            drew3d = False
            if "head3" in d and scene is not None:
                # 3D bakış ışını: kafa konumundan dünya-uzayında 2.5m ileri
                p1 = scene.project(d["head3"])
                p2 = scene.project(d["head3"] + d["dir3"] * 2.5)
                if p1 and p2:
                    seg = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    if seg <= max(h * 2.5, 140 * sc):   # patlayan projeksiyon -> 2D'ye düş
                        a1 = (int(p1[0]), int(p1[1]))
                        a2 = (int(p2[0]), int(p2[1]))
                        cv2.arrowedLine(out, a1, a2, (12, 12, 12), th + 4, cv2.LINE_AA, tipLength=0.28)
                        cv2.arrowedLine(out, a1, a2, col, th + 1, cv2.LINE_AA, tipLength=0.28)
                        drew3d = True
            if not drew3d and d["dx"] is not None and (d["dx"] or d["dy"]):
                ln = max(w * 1.3, 56 * sc)
                sx0, sy0 = int(d["c"][0]), int(d["c"][1])
                ex, ey = int(d["c"][0] + d["dx"] * ln), int(d["c"][1] + d["dy"] * ln)
                # koyu kontur + renkli gövde: her zeminde okunur ok
                cv2.arrowedLine(out, (sx0, sy0), (ex, ey), (12, 12, 12),
                                th + 4, cv2.LINE_AA, tipLength=0.3)
                cv2.arrowedLine(out, (sx0, sy0), (ex, ey), col,
                                th + 1, cv2.LINE_AA, tipLength=0.3)
            tag = f'#{d["id"]} {d["sig"]} {int(d["conf"]*100)}%'
            cv2.putText(out, tag, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * sc, col, 1, cv2.LINE_AA)

        hud = f'OCULIQ  ·  on-device  ·  persons {traffic}'
        if tiled:
            hud += '  ·  tiled scan'
        if scene is not None and scene.enabled:
            hud += f'  ·  3D locked {scene.confidence:.0f}%'
        if t is not None:
            hud += f'  ·  t={t:.1f}s'
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, 2)
        cv2.rectangle(out, (12, 12), (tw + 32, th + 28), DARK, -1)
        cv2.putText(out, hud, (22, th + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * sc, (143, 196, 37), 2, cv2.LINE_AA)
        return out

    def _writer(self, path, W, H, fps):
        for fourcc in ("avc1", "mp4v"):
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))
            if w.isOpened():
                return w
        raise RuntimeError("no available video codec")

    def _jpeg_b64(self, frame, max_w):
        H, W = frame.shape[:2]
        if W > max_w:
            frame = cv2.resize(frame, (max_w, int(H * max_w / W)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        return base64.b64encode(buf).decode() if ok else None
