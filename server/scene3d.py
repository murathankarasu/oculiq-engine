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
        """Cizilen 2D dikdortgen -> GERCEK yuzeye oturtulmus yonlu 3D dortgen.

        Bolge icindeki derinlik noktalarina duzlem oturtulur (Z = aX + bY + c,
        aykiri ayiklamali — onunden gecen insanlar/parazit elenir); koseler kendi
        piksel isinlariyla o duzleme yerlestirilir. Acili duvar/billboard artik
        acili temsil edilir: normal + egim (tilt) cikar. Fit tutmazsa medyan
        derinlikli kameraya-paralel dortgene dusulur."""
        if self.depth is None or not self.f:
            return None
        x, y, w, h = [int(v) for v in rect_px]
        x2, y2 = min(x + w, self.W - 1), min(y + h, self.H - 1)
        x, y = max(x, 0), max(y, 0)
        if x2 <= x + 2 or y2 <= y + 2:
            return None

        us = np.unique(np.linspace(x, x2, min(28, x2 - x)).astype(int))
        vs = np.unique(np.linspace(y, y2, min(28, y2 - y)).astype(int))
        pts = []
        for v in vs:
            for u in us:
                Z = float(self.depth[v, u])
                if 0.5 < Z < 300:
                    pts.append([(u - self.cx) * Z / self.f,
                                (v - self.cy) * Z / self.f, Z])
        if len(pts) < 12:
            return None
        P = np.array(pts)
        Zmed = float(np.median(P[:, 2]))
        P = P[np.abs(P[:, 2] - Zmed) < max(1.5, Zmed * 0.2)]  # kaba aykırı ayıklama
        if len(P) < 12:
            return None

        plane = None
        try:
            A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
            coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
            res = np.abs(A @ coef - P[:, 2])
            thr = max(0.25, 1.5 * float(np.median(res)))
            inl = res < thr
            if int(inl.sum()) >= 12:
                coef, *_ = np.linalg.lstsq(A[inl], P[inl, 2], rcond=None)
            plane = tuple(float(v) for v in coef)
        except Exception:
            plane = None

        def corner(u, v):
            if plane is not None:
                a, b, c = plane
                den = 1.0 - a * (u - self.cx) / self.f - b * (v - self.cy) / self.f
                if den > 1e-4:
                    Z = c / den
                    if 0.5 < Z < 300:
                        return self.backproject(u, v, Z)
            return self.backproject(u, v, Zmed)

        cns = [corner(x, y), corner(x2, y), corner(x2, y2), corner(x, y2)]
        center = (cns[0] + cns[2]) / 2
        wm = float(np.linalg.norm(cns[1] - cns[0]))
        hm = float(np.linalg.norm(cns[3] - cns[0]))

        nrm, tilt = None, None
        if plane is not None:
            nv = np.array([plane[0], plane[1], -1.0])
            nv /= np.linalg.norm(nv)
            if float(nv @ center) > 0:      # normal kameraya baksın
                nv = -nv
            view = center / (np.linalg.norm(center) or 1.0)
            tilt = round(math.degrees(math.acos(max(-1.0, min(1.0, float(-nv @ view))))), 1)
            nrm = nv
        return {"corners": cns, "center": center, "normal": nrm, "tilt_deg": tilt,
                "w_m": round(wm, 2), "h_m": round(hm, 2),
                "depth_m": round(float(center[2]), 1)}

    def zone_mesh(self, quad, nu=6, nv=4):
        """Yüzeye oturan 3D tel-kafes + normal oku (görsel kanıt katmanı)."""
        c0, c1, c2, c3 = quad["corners"]
        lines = []
        for i in range(nu + 1):
            s = i / nu
            a = self.project(c0 + (c1 - c0) * s)
            b = self.project(c3 + (c2 - c3) * s)
            if a and b:
                lines.append((a, b))
        for j in range(nv + 1):
            s = j / nv
            a = self.project(c0 + (c3 - c0) * s)
            b = self.project(c1 + (c2 - c1) * s)
            if a and b:
                lines.append((a, b))
        nseg = None
        if quad.get("normal") is not None:
            p1 = self.project(quad["center"])
            p2 = self.project(quad["center"] + quad["normal"] * max(0.6, quad["w_m"] * 0.3))
            if p1 and p2:
                nseg = (p1, p2)
        return lines, nseg

    def refit(self, person_samples):
        """Derinliği yeniden hesaplamadan odak+zemini TÜM örneklerle tazele (ucuz)."""
        if self.depth is None:
            return self
        d = self.depth
        obs = []
        for (fu, fv, h_px) in person_samples:
            if h_px < 24:
                continue
            u = int(np.clip(fu, 2, self.W - 3))
            v = int(np.clip(fv - h_px * 0.5, 2, self.H - 3))
            Z = float(np.median(d[max(0, v - 4):v + 5, max(0, u - 4):u + 5]))
            if 0.5 < Z < 200:
                obs.append((float(h_px), Z))
        if len(obs) >= 3:
            self.samples = len(obs)
            self.f = float(np.median([h * Z / MEAN_PERSON_M for h, Z in obs]))
            heights = [h * Z / self.f for h, Z in obs]
            self.height_mean = float(np.mean(heights))
            self.height_std = float(np.std(heights))
            self.confidence = round(max(0.0, min(1.0, 1.0 - self.height_std / 0.35)) * 100, 1)
        return self

    # ---------- görselleştirme (AR-tarzı) ----------
    def project(self, P):
        if P[2] <= 0.05:
            return None
        return (self.cx + self.f * P[0] / P[2], self.cy + self.f * P[1] / P[2])

    def _plane_axes(self):
        n, _ = self.ground
        n = np.asarray(n, dtype=float)
        fwd = np.array([0.0, 0.0, 1.0])
        e1 = fwd - float(fwd @ n) * n
        nrm = np.linalg.norm(e1)
        if nrm < 1e-6:
            return None, None, n
        e1 /= nrm
        e2 = np.cross(n, e1)
        return e1, e2, n

    def grid_segments(self, spacing=1.0, extent=14):
        """Zemin düzlemine metrik ızgara — sabit kamerada BİR KEZ hesaplanır.
        Gerçek yüzeye oturuyorsa kalibrasyon gözle doğrulanmış demektir."""
        if self.ground is None or not self.f:
            return []
        n, dpl = self.ground
        n = np.asarray(n, dtype=float)
        ray = np.array([0.0, (self.H * 0.85 - self.cy) / self.f, 1.0])
        denom = float(n @ ray)
        if abs(denom) < 1e-6:
            return []
        t = -dpl / denom
        if t <= 0 or t > 300:
            return []
        p0 = ray * t
        e1, e2, _ = self._plane_axes()
        if e1 is None:
            return []
        segs = []
        R = extent
        inb = lambda q: (q is not None and -self.W * 0.5 < q[0] < self.W * 1.5
                         and -self.H * 0.5 < q[1] < self.H * 1.5)
        for (a_ax, b_ax) in ((e1, e2), (e2, e1)):
            for i in range(-R, R + 1):
                prev = None
                for j in range(-R, R + 1):
                    q = self.project(p0 + b_ax * (i * spacing) + a_ax * (j * spacing))
                    ok = inb(q)
                    if prev is not None and ok:
                        segs.append((prev, q, i % 5 == 0))
                    prev = q if ok else None
        return segs

    def ground_ring(self, foot_u, foot_v, radius=0.35, npts=16):
        """Kişinin ayağında zemine yapışık AR çapa halkası (piksel poligonu)."""
        pos = self.person_pos(foot_u, foot_v)
        if pos is None or self.ground is None:
            return None
        e1, e2, _ = self._plane_axes()
        if e1 is None:
            return None
        pts = []
        for a in np.linspace(0, 2 * math.pi, npts, endpoint=False):
            q = self.project(pos + e1 * (math.cos(a) * radius) + e2 * (math.sin(a) * radius))
            if q is None:
                return None
            pts.append(q)
        return pts

    # ---------- 3D bakış (Faz 2) ----------
    def up(self):
        """Zemin normali, kameradan yukarı bakacak şekilde."""
        if self.ground is None:
            return None
        n, dpl = self.ground
        n = np.asarray(n, dtype=float)
        return n if dpl > 0 else -n

    def lateral_axes(self):
        """Dünya eksenleri: L (görüntü-sağı, zemine izdüşük), C (kameraya doğru, zeminde)."""
        e1, e2, n = self._plane_axes()
        if e1 is None:
            return None, None
        ex = np.array([1.0, 0.0, 0.0])
        L = ex - float(ex @ n) * n
        nrm = np.linalg.norm(L)
        if nrm < 1e-6:
            return None, None
        return L / nrm, -e1   # e1 kameradan uzağa; C = kameraya doğru

    def head_pos(self, foot_u, foot_v, eye_h=1.55):
        pos = self.person_pos(foot_u, foot_v)
        u = self.up()
        if pos is None or u is None:
            return None
        return pos + u * eye_h

    def gaze_dir3d(self, dx_img, dy_img, k, sig):
        """Görüntü-uzayı yön vektörü -> dünya 3D bakış yönü.
        body: zemin azimutu (yanal + derinlik), dikey bileşen yok (ölçülemiyor).
        head: yanal + kameraya-doğru + dikey (burun ofsetinden pitch)."""
        L, C = self.lateral_axes()
        u = self.up()
        if L is None or u is None:
            return None
        if sig == "body":
            ay = dy_img / max(k, 1e-6)      # + = kameraya doğru
            v = dx_img * L + ay * C
        else:
            lat = max(-1.0, min(1.0, dx_img))
            fwd = math.sqrt(max(0.0, 1.0 - lat * lat))
            v = lat * L + fwd * C + (-dy_img) * u
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else None

    @staticmethod
    def ang3d(a, b):
        na = np.linalg.norm(a) or 1e-9
        nb = np.linalg.norm(b) or 1e-9
        c = float(a @ b) / (na * nb)
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    def looks_at_3d(self, head3, dir3, quad, noise_deg, sig):
        """3D bakış testi — bölge GERÇEK yüzeyine oturmuş yönlü dörtgendir.

        1. Arka-taraf eleme: yüzeyin görünmez tarafındaki kişi bakamaz.
        2. head: ışın-düzlem KESİŞİMİ — bakış ışınının düzlemi deldiği nokta,
           dörtgenin içinde mi (gürültü marjı = mesafe * tan(noise))?
        3. body: gerçek zemin azimutu (dikey ölçülemez), yüzey genişliğine göre.
        Normal yoksa (fit başarısız) açısal teste düşülür."""
        to = quad["center"] - head3
        dist = float(np.linalg.norm(to))
        if dist < 0.3:
            return False
        n = quad.get("normal")
        if n is not None and float(n @ (head3 - quad["center"])) < 0:
            return False   # yüzeyin arkasından görünmez

        if sig == "body":
            u = self.up()
            a = dir3 - float(dir3 @ u) * u
            b = to - float(to @ u) * u
            half = math.degrees(math.atan((quad["w_m"] / 2) / dist))
            return self.ang3d(a, b) <= noise_deg + min(half, 25.0)

        # head: ışın-düzlem kesişimi (en doğru test)
        if n is not None:
            den = float(n @ dir3)
            if abs(den) > 1e-4:
                tt = float(n @ to) / den
                if tt <= 0:
                    return False           # bakış yüzeyden uzağa
                p = head3 + dir3 * tt
                e_u = quad["corners"][1] - quad["corners"][0]
                e_v = quad["corners"][3] - quad["corners"][0]
                wu = float(np.linalg.norm(e_u)) or 1e-6
                hv = float(np.linalg.norm(e_v)) or 1e-6
                rel = p - quad["corners"][0]
                du = float(rel @ (e_u / wu))
                dv = float(rel @ (e_v / hv))
                m = dist * math.tan(math.radians(noise_deg))
                return -m <= du <= wu + m and -m <= dv <= hv + m

        diag = math.hypot(quad["w_m"], quad["h_m"]) / 2
        half = math.degrees(math.atan(diag / dist))
        return self.ang3d(dir3, to) <= noise_deg + min(half, 25.0)

    def depth_grid(self, gw=96):
        """What-if için kaba derinlik ızgarası (tarayıcıya gider, ~birkaç KB)."""
        if self.depth is None:
            return None
        import cv2
        gh = max(8, int(round(gw * self.H / self.W)))
        g = cv2.resize(self.depth, (gw, gh), interpolation=cv2.INTER_AREA)
        return {"gw": gw, "gh": gh,
                "z": [[round(float(v), 1) for v in row] for row in g]}

    # ---------- güven kapısı ----------
    def reliable(self):
        """3D geometri kararlara girmeye layık mı? Değilse motor 2.5D'ye düşer.
        Bu, '3D sapıtırsa geri dön' mekanizmasının ta kendisi — otomatik."""
        return (self.enabled and self.confidence >= 40.0
                and self.ground is not None
                and self.cam_height is not None and 1.2 <= self.cam_height <= 40.0)

    # ---------- LiDAR-tarzı sahne görünümü ----------
    def render_view(self):
        """Derinlik haritasını AR/LiDAR taraması gibi renklendir + ızgarayı bindir."""
        import cv2
        if self.depth is None:
            return None
        d = np.clip(self.depth, 0.5, 80.0)
        dn = (np.log(d) - math.log(0.5)) / (math.log(80.0) - math.log(0.5))
        img = cv2.applyColorMap((np.clip(1 - dn, 0, 1) * 255).astype(np.uint8),
                                cv2.COLORMAP_TURBO)
        for (a, b, major) in (self.grid_segments() or []):
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     (255, 255, 255), 2 if major else 1, cv2.LINE_AA)
        tag = f"depth reconstruction · f={self.f:.0f}px · conf {self.confidence:.0f}%"
        cv2.rectangle(img, (10, 10), (26 + 8 * len(tag), 40), (12, 12, 12), -1)
        cv2.putText(img, tag, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # ---------- rapor ----------
    def state(self):
        if not self.enabled:
            return {"enabled": False, "note": self.note}
        s = {"enabled": True,
             "model": "Depth Anything V2 metric (outdoor, small)",
             "focal_px": round(self.f, 1) if self.f else None,
             "calib_confidence": self.confidence,
             "reliable": self.reliable(),
             "gate": "active" if self.reliable() else "fallback-2.5d (low confidence)",
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
