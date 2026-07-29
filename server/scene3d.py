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


# Derinlik modeli — VARSAYILAN İÇ MEKAN (retail sahneleri iç mekan; metrik derinlik
# domaine duyarlı). DOOH/açık hava için OCULIQ_DEPTH_MODEL env ile Outdoor'a alınabilir.
# Model değişikliği ölçümü etkiler → docs/SPEC.md ve tools/regress.py kapsamında.
_DEPTH_MODELS = {
    "indoor":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
}
DEPTH_VARIANT = os.environ.get("OCULIQ_DEPTH_MODEL", "indoor").lower()
if DEPTH_VARIANT not in _DEPTH_MODELS:
    DEPTH_VARIANT = "indoor"
DEPTH_MODEL = _DEPTH_MODELS[DEPTH_VARIANT]


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
        _pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device=device)
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
        # Çoklu-kare medyan derinlik: liste verilirse (≤3 kare) her karenin
        # derinliği hesaplanır ve PİKSEL BAZINDA MEDYAN alınır — önden geçen
        # kişi/geçici engel tek karenin haritasını bozar, medyanı bozamaz.
        frames = frame_bgr if isinstance(frame_bgr, list) else [frame_bgr]
        frames = [f for f in frames if f is not None][:3] or [frame_bgr]
        self.H, self.W = frames[0].shape[:2]
        self.cx, self.cy = self.W / 2.0, self.H / 2.0

        maps = []
        for fb in frames:
            img = Image.fromarray(cv2.cvtColor(fb, cv2.COLOR_BGR2RGB))
            out = _depth_pipe()(img)
            di = np.asarray(out["predicted_depth"], dtype=np.float32)
            if di.shape != (self.H, self.W):
                di = cv2.resize(di, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            maps.append(di)
        d = maps[0] if len(maps) == 1 else np.median(np.stack(maps), axis=0).astype(np.float32)
        self.depth_frames_used = len(maps)
        self.depth = d

        # --- KALIBRASYON: ayak-ZEMIN derinligi + kisi-basina medyan boy + iterasyon ---
        # Derinlik haritasi TEK andan; kisi baska anda baska yerde olabilir. Govde
        # derinligi okumak ARKA PLANI okur (buyuk hata kaynagiydi). Ayagin altindaki
        # zemin ise statiktir: her an gecerli. Ayrica ayni kisi cok kez olculup
        # MEDYANI alinir — kare gurultusu boy sacilimini sisiremez.
        by_pid = {}
        for s in person_samples:
            fu, fv, h_px = s[0], s[1], s[2]
            pid = s[3] if len(s) > 3 else len(by_pid)
            if h_px < 24:
                continue
            u = int(np.clip(fu, 2, self.W - 3))
            v = int(np.clip(fv, 2, self.H - 3))
            Z = float(np.median(d[max(0, v - 2):v + 3, max(0, u - 4):u + 5]))
            if 0.5 < Z < 200:
                by_pid.setdefault(pid, []).append((float(h_px), Z, u, v))
        self.samples = sum(len(v) for v in by_pid.values())
        self.persons_used = len(by_pid)
        self.inliers = 0

        if self.samples >= 3:
            flat = [(h, Z) for lst in by_pid.values() for (h, Z, _, _) in lst]
            f = float(np.median([h * Z / MEAN_PERSON_M for h, Z in flat]))
        else:
            f = 0.9 * self.W
            self.note = "insufficient people for focal self-check"
        self.f = f

        per_heights = []
        for _ in range(2):   # f <-> zemin duzlemi iterasyonu
            # zemin duzlemi (ayak noktalarindan, mevcut f ile)
            pts = [self.backproject(u, v, Z)
                   for lst in by_pid.values() for (_, Z, u, v) in lst]
            if len(pts) >= 4:
                P = np.array(pts)
                centroid = P.mean(axis=0)
                _, _, vh = np.linalg.svd(P - centroid)
                n = vh[-1]
                if n[1] > 0:
                    n = -n
                self.ground = (n.astype(float), -float(n @ centroid))
                self.cam_height = abs(self.ground[1])
                self.tilt_deg = round(math.degrees(
                    math.acos(max(-1, min(1, float(-n[1]))))), 1)
            # kisi basina medyan boy (zemin-duzlemi derinligiyle; yoksa harita Z)
            per = {}
            for pid, lst in by_pid.items():
                hs = []
                for (h, Zmap, u, v) in lst:
                    Zp = None
                    if self.ground is not None:
                        pos = self.person_pos(u, v)
                        if pos is not None:
                            Zp = float(pos[2])
                    Z = Zp if Zp and 0.5 < Zp < 200 else Zmap
                    hs.append(h * Z / self.f)
                if hs:
                    per[pid] = float(np.median(hs))
            inl = {p: h for p, h in per.items() if 1.2 <= h <= 2.1}
            use = inl if len(inl) >= 2 else per
            self.inliers = len(inl)
            per_heights = list(use.values())
            if per_heights:
                med = float(np.median(per_heights))
                self.f *= med / MEAN_PERSON_M   # boy medyani 1.70m'ye otursun

        if per_heights:
            self.height_mean = float(np.mean(per_heights))
            self.height_std = float(np.std(per_heights))
            # DOGAL boy sacilimi ~0.10m — onu cezalandirma; yalnizca FAZLASINI olc
            excess = max(0.0, self.height_std - 0.10)
            spread_conf = max(0.0, min(1.0, 1.0 - excess / 0.18))
            ratio = (len(inl) / max(len(per), 1)) if per_heights else 0.0
            ratio_pen = min(1.0, ratio / 0.5)
            few_pen = 1.0 if self.persons_used >= 3 else 0.5
            self.confidence = round(spread_conf * ratio_pen * few_pen * 100, 1)
        else:
            self.confidence = 0.0
            self.note = (self.note + "; " if self.note else "") + "no usable height samples"
        if self.ground is None:
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

    def zone_quad(self, z_or_rect):
        """Cizilen bolge (4-kose poligon ya da dikdortgen) -> GERCEK yuzeye
        oturtulmus yonlu 3D dortgen.

        Bolge icindeki derinlik noktalarina duzlem oturtulur (Z = aX + bY + c,
        aykiri ayiklamali — onunden gecen insanlar/parazit elenir); koseler kendi
        piksel isinlariyla o duzleme yerlestirilir. Acili duvar/billboard artik
        acili temsil edilir: normal + egim (tilt) cikar. Fit tutmazsa medyan
        derinlikli kameraya-paralel dortgene dusulur."""
        rect_px = z_or_rect["rect"] if isinstance(z_or_rect, dict) else z_or_rect
        if self.depth is None or not self.f:
            return None
        x, y, w, h = [int(v) for v in rect_px]
        x2, y2 = min(x + w, self.W - 1), min(y + h, self.H - 1)
        x, y = max(x, 0), max(y, 0)
        if x2 <= x + 2 or y2 <= y + 2:
            return None

        # Duzlem fit'i YALNIZCA cizilen poligonun ICINDEKI piksellerden beslenir.
        # (Onceden sinirlayici dikdortgen orneklenirdi: acili/egik bir yuzeyde
        # kutunun buyuk kismi bolgenin DISINDA kalir — komsu raf, zemin, arka
        # plan fit'i baska bir duzleme cekip gercek olcuyu bozardi.)
        ppx0 = z_or_rect.get("poly_px") if isinstance(z_or_rect, dict) else None
        mask = None
        if ppx0 and len(ppx0) >= 3:
            import cv2
            mask = np.zeros((self.H, self.W), np.uint8)
            cv2.fillPoly(mask, [np.array(ppx0, np.int32)], 1)

        us = np.unique(np.linspace(x, x2, min(28, x2 - x)).astype(int))
        vs = np.unique(np.linspace(y, y2, min(28, y2 - y)).astype(int))
        pts = []
        for v in vs:
            for u in us:
                if mask is not None and not mask[v, u]:
                    continue
                Z = float(self.depth[v, u])
                if 0.5 < Z < 300:
                    pts.append([(u - self.cx) * Z / self.f,
                                (v - self.cy) * Z / self.f, Z])
        if len(pts) < 12 and mask is not None:      # cok ince poligon: kutuya don
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

        # ZEMIN CAPASI: yuzeyin alt kenari zemine deger (raf/stand/vitrin).
        # Zemin duzlemi insan boylarindan DOGRULANMIS kalibrasyonun urunudur;
        # derinlik haritasi ise arkadan isikli/parlak yuzeylerde ciddi sasar
        # (gercek 4m'lik bir stand 12m okunabiliyor -> olcu 3 katina cikiyordu).
        # Ikisi carpici bicimde ayrisirsa zemine guvenilir, haritaya guvenilmez.
        ppx_all = z_or_rect.get("poly_px") if isinstance(z_or_rect, dict) else None
        base_pts = (sorted(ppx_all, key=lambda p: -p[1])[:2] if ppx_all
                    else [(x, y2), (x2, y2)])
        bu = sum(p[0] for p in base_pts) / len(base_pts)
        bv = sum(p[1] for p in base_pts) / len(base_pts)
        Zg = None
        if self.ground is not None and bv < self.H - 3:
            gp = self.person_pos(bu, bv)          # alt kenardan zemine isin
            if gp is not None and 0.5 < float(gp[2]) < 300:
                Zg = float(gp[2])
        self.zone_depth_source = "depth-map"
        if Zg is not None and (Zmed > 1.5 * Zg or Zmed < 0.66 * Zg):
            Zmed = Zg                              # zemin cozumu kazanir
            self.zone_depth_source = "ground-anchored"
            plane_reject = True
        else:
            plane_reject = False

        P = P[np.abs(P[:, 2] - Zmed) < max(1.5, Zmed * 0.2)]  # kaba aykırı ayıklama
        if len(P) < 12:
            P = np.array(pts)                      # capa sonrasi az kaldiysa geri al
        if len(P) < 12:
            return None

        plane = None
        try:
            if plane_reject:
                raise ValueError("depth map distrusted for this surface")
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

        ppx = z_or_rect.get("poly_px") if isinstance(z_or_rect, dict) else None
        cpx = (list(ppx) if ppx else
               [(x, y), (x2, y), (x2, y2), (x, y2)])
        cpx = [(int(np.clip(u, 0, self.W - 1)), int(np.clip(v, 0, self.H - 1)))
               for (u, v) in cpx]
        cns = [corner(u, v) for (u, v) in cpx]

        # SAGLAMA: metrik derinligin kucuk bir gradyan hatasi, egik duzlemde uzak
        # koseleri cok uzaga iter ve olcu katlanir. Fit'ten cikan olcu, kameraya
        # PARALEL varsayimin olcusunun 1.8 katini asiyorsa fit'e guvenilmez —
        # medyan derinlikli paralel dortgene donulur (olculmus guven ilkesi).
        par = [self.backproject(u, v, Zmed) for (u, v) in cpx]
        def _edges(c):
            return ((np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2.0,
                    (np.linalg.norm(c[2] - c[1]) + np.linalg.norm(c[3] - c[0])) / 2.0)
        fa, fb = _edges(cns)
        pa, pb = _edges(par)
        if pa > 1e-6 and pb > 1e-6 and (fa > 1.8 * pa or fb > 1.8 * pb):
            cns, plane = par, None       # duzlem reddedildi: paralel dortgen
        center = sum(cns) / len(cns)
        # Genislik/yukseklik kose SIRASINA gore degil, kenarin DUNYA-YUKARI ile
        # hizasina gore ayrilir: cizim sirasinda koseler aci-sirali geldiginden
        # cns[0] sol-ust olmak zorunda degil (eskiden w/h yer degistirebiliyordu).
        e_a = (np.linalg.norm(cns[1] - cns[0]) + np.linalg.norm(cns[2] - cns[3])) / 2.0
        e_b = (np.linalg.norm(cns[2] - cns[1]) + np.linalg.norm(cns[3] - cns[0])) / 2.0
        up = self.up()
        if up is not None:
            va = (cns[1] - cns[0]) + (cns[2] - cns[3])
            vb = (cns[2] - cns[1]) + (cns[3] - cns[0])
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na > 1e-6 and nb > 1e-6:
                da = abs(float(va @ up) / na)      # 1'e yakin = dikey kenar
                db = abs(float(vb @ up) / nb)
                wm, hm = (e_a, e_b) if da < db else (e_b, e_a)
            else:
                wm, hm = e_a, e_b
        else:
            wm, hm = float(e_a), float(e_b)
        wm, hm = float(wm), float(hm)

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
                "depth_m": round(float(center[2]), 1),
                # olcunun NEREDEN geldigi rapora tasinir: derinlik haritasindan mi,
                # yoksa (harita guvenilmezken) dogrulanmis zemin duzleminden mi
                "depth_source": getattr(self, "zone_depth_source", "depth-map")}

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
        """Derinligi yeniden hesaplamadan kalibrasyonu TUM orneklerle tazele."""
        if self.depth is None:
            return self
        return self._recalibrate(person_samples)

    def _recalibrate(self, person_samples):
        d = self.depth
        by_pid = {}
        for s in person_samples:
            fu, fv, h_px = s[0], s[1], s[2]
            pid = s[3] if len(s) > 3 else len(by_pid)
            if h_px < 24:
                continue
            u = int(np.clip(fu, 2, self.W - 3))
            v = int(np.clip(fv, 2, self.H - 3))
            Z = float(np.median(d[max(0, v - 2):v + 3, max(0, u - 4):u + 5]))
            if 0.5 < Z < 200:
                by_pid.setdefault(pid, []).append((float(h_px), Z, u, v))
        if not by_pid:
            return self
        self.samples = sum(len(v) for v in by_pid.values())
        self.persons_used = len(by_pid)
        per_heights = []
        for _ in range(2):
            per = {}
            for pid, lst in by_pid.items():
                hs = []
                for (h, Zmap, u, v) in lst:
                    Zp = None
                    if self.ground is not None:
                        pos = self.person_pos(u, v)
                        if pos is not None:
                            Zp = float(pos[2])
                    Z = Zp if Zp and 0.5 < Zp < 200 else Zmap
                    hs.append(h * Z / self.f)
                if hs:
                    per[pid] = float(np.median(hs))
            inl = {p: h for p, h in per.items() if 1.2 <= h <= 2.1}
            use = inl if len(inl) >= 2 else per
            self.inliers = len(inl)
            per_heights = list(use.values())
            if per_heights:
                self.f *= float(np.median(per_heights)) / MEAN_PERSON_M
        if per_heights:
            self.height_mean = float(np.mean(per_heights))
            self.height_std = float(np.std(per_heights))
            excess = max(0.0, self.height_std - 0.10)
            spread_conf = max(0.0, min(1.0, 1.0 - excess / 0.18))
            ratio = (self.inliers / max(len(per_heights), 1))
            ratio_pen = min(1.0, ratio / 0.5)
            few_pen = 1.0 if self.persons_used >= 3 else 0.5
            self.confidence = round(spread_conf * ratio_pen * few_pen * 100, 1)
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

    def _visible(self, P, q, tol=0.30):
        """P (3D, kamera uzayi) noktasi q pikselinde GERCEKTEN gorunur mu?
        Sahne derinligi o pikselde P'den belirgin kucukse, nokta bir yuzeyin
        ARKASINDA demektir (duvar/raf/mobilya) — cizilmemeli. Zemin izgarasi
        ve capa halkalari eskiden bu testi yapmadigi icin duvarlarin uzerine
        tasiyordu."""
        if q is None:
            return False
        if self.depth is None:
            return True
        u, v = int(round(q[0])), int(round(q[1]))
        if not (0 <= u < self.W and 0 <= v < self.H):
            return False
        Zs = float(self.depth[v, u])
        if not (0.3 < Zs < 300):
            return True
        return float(P[2]) <= Zs * (1.0 + tol) + tol

    def sight_blocked(self, A, B, samples=14, tol=0.35):
        """Goz (A) -> yuzey (B) dogrusu boyunca STATIK bir engel var mi?

        Derinlik haritasi boyunca isin yurutur: isin uzerindeki bir nokta, o
        pikseldeki sahne yuzeyinden belirgin DAHA UZAKTAYSA, arada bir sey
        duruyor demektir. Insan-tabanli occlusion testi yalnizca kisileri
        goruyordu; bu test rafi, direkgi, bench'i, tezgahi da yakalar
        ("hedefin onunde baska bir obje var" problemi).

        Sinir (durustce): derinlik haritasi kalibrasyon anindan alinmis tek
        goruntudur — statik mobilyayi dogru, sonradan yer degistiren insanlari
        yaklasik temsil eder. Bu yuzden iki test birbirini tamamlar.
        Hedefe cok yaklasan son parca atlanir; yoksa yuzeyin kendisi engel sayilir.
        """
        if self.depth is None or not self.f:
            return False
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        za, zb = float(A[2]), float(B[2])
        if abs(zb - za) < 0.4:        # yuzeyin hemen onunde: engel olacak yer yok
            return False
        hits = 0
        for i in range(2, samples - 2):          # bas ve son parcalari atla
            P = A + (B - A) * (i / float(samples))
            q = self.project(P)
            if q is None:
                continue
            u, v = int(round(q[0])), int(round(q[1]))
            if not (0 <= u < self.W and 0 <= v < self.H):
                continue
            Zs = float(self.depth[v, u])
            if not (0.3 < Zs < 300):
                continue
            # BAKANIN KENDI GOVDESI engel degildir: sahne yuzeyi kisinin kendi
            # derinligindeyse (ya da hedefin derinligindeyse) atlanir. Bu kontrol
            # olmadan raf onunde duran kisi kendi vucudunu engel sayip
            # "bakmiyor" olarak elenmisti.
            if abs(Zs - za) < 0.6 or abs(Zs - zb) < 0.6:
                continue
            if float(P[2]) > Zs * (1.0 + tol) + tol:
                hits += 1
                if hits >= 2:        # tek ornek gurultu olabilir; iki ornek = engel
                    return True
        return False

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
                    P = p0 + b_ax * (i * spacing) + a_ax * (j * spacing)
                    q = self.project(P)
                    # Izgara SALT GORSEL: burada siki tolerans kullanilir (0.10),
                    # cunku yanlis cizilen bir cizgi kalibrasyonu bozuk gosterir.
                    # Olcum yollarindaki tolerans (0.30) bilerek muhafazakar
                    # birakildi — orada gercek bir bakisi elemek daha pahali.
                    ok = inb(q) and self._visible(P, q, tol=0.10)
                    if prev is not None and ok:
                        segs.append((prev, q, i % 5 == 0))
                    prev = q if ok else None
        return segs

    def ground_ring(self, foot_u, foot_v, radius=0.35, npts=16):
        """Kişinin ayağında zemine yapışık AR çapa halkası (piksel poligonu).
        Ayak noktası sahne derinliğiyle çelişiyorsa (ayna yansıması, kısmi/hatalı
        kutu, mobilya üstü) halka ÇİZİLMEZ — yanlış yerde halka, kalibrasyonun
        bozuk olduğu izlenimi verir."""
        pos = self.person_pos(foot_u, foot_v)
        if pos is None or self.ground is None:
            return None
        # ayak noktasının zemin çözümü, o pikseldeki gerçek derinlikle tutmalı
        if self.depth is not None:
            u = int(np.clip(round(foot_u), 0, self.W - 1))
            v = int(np.clip(round(foot_v), 0, self.H - 1))
            Zs = float(self.depth[v, u])
            if 0.3 < Zs < 300 and not (0.6 * Zs <= float(pos[2]) <= 1.6 * Zs + 0.5):
                return None
        e1, e2, _ = self._plane_axes()
        if e1 is None:
            return None
        pts = []
        for a in np.linspace(0, 2 * math.pi, npts, endpoint=False):
            P = pos + e1 * (math.cos(a) * radius) + e2 * (math.sin(a) * radius)
            q = self.project(P)
            if q is None or not self._visible(P, q, tol=0.45):
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
            # Dikey bileşen burun ofsetinden gelir ve gürültülüdür: kamera açısı
            # ya da eğik duruş, bakışı tavana/zemine savurabiliyordu. Perakendede
            # bakış esasen yataydır — pitch ±25° ile sınırlanır (yön uydurulmaz,
            # yalnızca fiziksel olarak makul aralığa kırpılır).
            v = lat * L + fwd * C + (-dy_img) * u
            n0 = np.linalg.norm(v)
            if n0 > 1e-6:
                v = v / n0
                vert = float(v @ u)
                lim = math.sin(math.radians(25.0))
                if abs(vert) > lim:
                    horiz = v - vert * u
                    hn = np.linalg.norm(horiz)
                    if hn > 1e-6:
                        v = horiz / hn * math.cos(math.radians(25.0)) \
                            + u * (lim if vert > 0 else -lim)
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
             "model": f"Depth Anything V2 metric ({DEPTH_VARIANT}, small)"
                      + (f", {self.depth_frames_used}-frame median"
                         if getattr(self, "depth_frames_used", 1) > 1 else ""),
             "focal_px": round(self.f, 1) if self.f else None,
             "calib_confidence": self.confidence,
             "inliers": getattr(self, "inliers", 0),
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
