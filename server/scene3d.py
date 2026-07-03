"""Scene3D — ARKit-vari sahne kalibrasyonu, tamamen lokal.

Tek temiz kareden (kamera sabit):
  1. Depth Anything V2 (metrik, outdoor) -> derinlik haritasi (metre)
  2. Odak uzakligi tahmini: tespit edilen insanlarin piksel boyu + derinligi,
     gercek insan boyu (~1.70m) uzerinden f = h_px * Z / H  (medyan)
  3. KENDINI DOGRULAYAN kalibrasyon: f sabitlenince her kisinin boyu geri
     hesaplanir; boylar 1.70m etrafinda ne kadar siki kumeleniyorsa kalibrasyon
     o kadar guvenilir -> calib_confidence (0-100). Kor guven degil, olculmus guven.
  4. Zemin duzlemi: ayak noktalarinin 3D geri-projeksiyonuna duzlem oturt
     -> kamera yuksekligi + egimi.
  5. Bolge 3D'si: cizilen dikdortgenin medyan derinligiyle koseleri geri-projekte
     et -> gercek boyut (metre) + izleme mesafeleri.

Derinlik modeli is basina BIR KEZ calisir (~1-3s); sonuc sahne profilidir.
"""
import math
import os
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))

MEAN_PERSON_M = 1.70
_pipe = None


def _depth_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else \
                     (0 if torch.cuda.is_available() else -1)
        except Exception:
            device = -1
        _pipe = pipeline("depth-estimation",
                         model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
                         device=device)
    return _pipe


class SceneModel:
    """Bir kameranin 3D sahne profili. build() basarisiz olursa enabled=False."""

    def __init__(self):
        self.enabled = False
        self.W = self.H = 0
        self.f = None            # odak (px)
        self.cx = self.cy = 0.0
        self.depth = None        # (H, W) metre
        self.ground = None       # (n, d): n·P + d = 0, |n|=1, kamera orijinde
        self.cam_height = None
        self.tilt_deg = None
        self.confidence = 0.0
        self.height_mean = None
        self.height_std = None
        self.samples = 0
        self.note = ""

    # ---------- kurulum ----------
    def build(self, frame_bgr, person_samples):
        """frame_bgr: temiz kare (BGR). person_samples: [(foot_u, foot_v, h_px), ...]"""
        try:
            self._build(frame_bgr, person_samples)
        except Exception as e:
            self.enabled = False
            self.note = f"scene3d unavailable: {type(e).__name__}: {e}"
        return self

    def _build(self, frame_bgr, person_samples):
        from PIL import Image
        import cv2
        self.H, self.W = frame_bgr.shape[:2]
        self.cx, self.cy = self.W / 2.0, self.H / 2.0

        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        out = _depth_pipe()(img)
        d = np.asarray(out["predicted_depth"], dtype=np.float32)
        if d.shape != (self.H, self.W):
            d = cv2.resize(d, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        self.depth = d

        # --- odak tahmini + boy-tabanli oz-dogrulama ---
        obs = []  # (h_px, Z)
        for (fu, fv, h_px) in person_samples:
            if h_px < 24:
                continue
            u = int(np.clip(fu, 2, self.W - 3))
            # govde ortasindaki derinlik ayaktan daha kararli
            v = int(np.clip(fv - h_px * 0.5, 2, self.H - 3))
            Z = float(np.median(d[max(0, v - 4):v + 5, max(0, u - 4):u + 5]))
            if 0.5 < Z < 200:
                obs.append((float(h_px), Z))
        self.samples = len(obs)
        if self.samples >= 3:
            f_est = [h * Z / MEAN_PERSON_M for h, Z in obs]
            self.f = float(np.median(f_est))
            heights = [h * Z / self.f for h, Z in obs]
            self.height_mean = float(np.mean(heights))
            self.height_std = float(np.std(heights))
            # std 0m -> %100, 0.35m -> %0 (kaba ama durust bir olcek)
            self.confidence = round(max(0.0, min(1.0, 1.0 - self.height_std / 0.35)) * 100, 1)
        else:
            self.f = 0.9 * self.W  # tipik CCTV varsayilani
            self.note = "insufficient people for focal self-check"
            self.confidence = 0.0

        # --- zemin duzlemi: ayak noktalarinin 3D'sine oturt ---
        pts = []
        for (fu, fv, h_px) in person_samples:
            u = int(np.clip(fu, 2, self.W - 3))
            v = int(np.clip(fv, 2, self.H - 3))
            Z = float(np.median(d[max(0, v - 3):v + 4, max(0, u - 3):u + 4]))
            if 0.5 < Z < 200:
                pts.append(self.backproject(u, v, Z))
        if len(pts) >= 4:
            P = np.array(pts)
            centroid = P.mean(axis=0)
            _, _, vh = np.linalg.svd(P - centroid)
            n = vh[-1]
            if n[1] > 0:      # normal yukari baksin (görüntü y asagi -> dunya y asagi)
                n = -n
            dplane = -float(n @ centroid)
            self.ground = (n.astype(float), dplane)
            self.cam_height = abs(dplane)  # kamera orijinde: |n·0 + d|
            # egim: zemin normali ile dusey (0,-1,0) arasindaki aci
            self.tilt_deg = round(math.degrees(
                math.acos(max(-1, min(1, float(-n[1]))))), 1)
        else:
            self.note = (self.note + "; " if self.note else "") + "ground plane: too few foot points"

        self.enabled = True

    # ---------- geometri ----------
    def backproject(self, u, v, Z):
        return np.array([(u - self.cx) * Z / self.f, (v - self.cy) * Z / self.f, Z])

    def person_pos(self, foot_u, foot_v):
        """Ayak pikselini zemin duzlemine isinla kesistir (govde derinligine bagimli degil)."""
        if self.ground is None:
            return None
        ray = np.array([(foot_u - self.cx) / self.f, (foot_v - self.cy) / self.f, 1.0])
        n, dpl = self.ground
        denom = float(n @ ray)
        if abs(denom) < 1e-6:
            return None
        t = -dpl / denom
        if t <= 0 or t > 300:
            return None
        return ray * t

    def zone_quad(self, rect_px):
        """Cizilen 2D dikdortgen -> 3D dortgen (medyan derinlik) + metre boyutlari."""
        if self.depth is None:
            return None
        x, y, w, h = [int(v) for v in rect_px]
        x2, y2 = min(x + w, self.W - 1), min(y + h, self.H - 1)
        x, y = max(x, 0), max(y, 0)
        if x2 <= x or y2 <= y:
            return None
        Z = float(np.median(self.depth[y:y2, x:x2]))
        if not (0.5 < Z < 300):
            return None
        c = [self.backproject(u, v, Z) for (u, v) in
             ((x, y), (x2, y), (x2, y2), (x, y2))]
        wm = float(np.linalg.norm(c[1] - c[0]))
        hm = float(np.linalg.norm(c[3] - c[0]))
        center = (c[0] + c[2]) / 2
        return {"corners": c, "center": center, "w_m": round(wm, 2), "h_m": round(hm, 2),
                "depth_m": round(Z, 1)}

    # ---------- rapor ----------
    def state(self):
        if not self.enabled:
            return {"enabled": False, "note": self.note}
        s = {"enabled": True,
             "model": "Depth Anything V2 metric (outdoor, small)",
             "focal_px": round(self.f, 1) if self.f else None,
             "calib_confidence": self.confidence,
             "samples": self.samples}
        if self.height_mean is not None:
            s["person_height_m"] = {"mean": round(self.height_mean, 2),
                                    "std": round(self.height_std, 2)}
        if self.cam_height is not None:
            s["camera_height_m"] = round(self.cam_height, 2)
            s["camera_tilt_deg"] = self.tilt_deg
        if self.note:
            s["note"] = self.note
        return s
