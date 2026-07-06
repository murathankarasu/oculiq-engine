"""Audience insights (beta) — toplu, opt-in, lokal gender tahmini.

Korkuluklar (KVKK/GDPR ve dürüstlük):
  - Varsayilan KAPALI; kullanici acikca secer.
  - Yalnizca yeterince buyuk/gorunur yuzler siniflandirilir (kalite kapisi);
    kapsama orani raporda acikca yazilir.
  - Cikti YALNIZCA toplu yuzdedir (bolge basina kadin/erkek/bilinmeyen);
    kisi bazli etiket hicbir export'a girmez, goruntu saklanmaz.
  - Model lokal calisir (FairFace-egitimli ViT); goruntu makineden cikmaz.
Bu bir TAHMINDIR — raporda "estimate" olarak etiketlenir.
"""
import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))

MODEL = "dima806/fairface_gender_image_detection"
AGE_MODEL = "dima806/fairface_age_image_detection"
MIN_FACE_PX = 24          # bundan kucuk yuz siniflandirilmaz (kalite kapisi)
MIN_SCORE = 0.65          # dusuk guvenli tahmin oy sayilmaz
AGE_MIN_SCORE = 0.35      # yas kovalari cok sinifli -> esik daha dusuk
# FairFace yas etiketleri -> kaba kovalar (rapor icin yeterli granularite)
AGE_MAP = {"0-2": "0-12", "3-9": "0-12", "10-19": "13-19", "20-29": "20-29",
           "30-39": "30-39", "40-49": "40-49", "50-59": "50-59",
           "60-69": "60+", "more than 70": "60+"}
AGE_ORDER = ["0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60+"]
_pipe = None
_pipe_age = None


def _device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1


def _gp():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("image-classification", model=MODEL, device=_device())
    return _pipe


def _ga():
    global _pipe_age
    if _pipe_age is None:
        from transformers import pipeline
        _pipe_age = pipeline("image-classification", model=AGE_MODEL, device=_device())
    return _pipe_age


def preload():
    """Her iki modeli iş başında yükle (ilk sefer indirir — sessiz gecikme olmasın)."""
    _gp()
    _ga()


def classify(crops_bgr):
    """BGR yüz kırpıntıları -> [(gender, g_score, age_bucket, a_score)]."""
    import cv2
    from PIL import Image
    imgs = []
    for c in crops_bgr:
        if min(c.shape[:2]) < 96:   # kucuk yuzleri buyut (ViT 224'e gider)
            s = 128.0 / min(c.shape[:2])
            c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)),
                           interpolation=cv2.INTER_CUBIC)
        imgs.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
    g_outs = _gp()(imgs, top_k=1)
    a_outs = _ga()(imgs, top_k=1)
    res = []
    for g, a in zip(g_outs, a_outs):
        gb = g[0] if isinstance(g, list) else g
        ab = a[0] if isinstance(a, list) else a
        label = "female" if "female" in gb["label"].lower() else "male"
        bucket = AGE_MAP.get(ab["label"], None)
        res.append((label, float(gb["score"]), bucket, float(ab["score"])))
    return res


def aggregate(persons, zs, min_dwell):
    """Kişi başına biriken oylardan YALNIZCA toplu bölünmeler üret."""
    def split(pids):
        s = {"female": 0, "male": 0, "unknown": 0}
        for pid in pids:
            gv = persons[pid].get("gender_votes") or {}
            tot = sum(gv.values())
            if tot < 0.9:
                s["unknown"] += 1
                continue
            top = max(gv, key=gv.get)
            s[top if gv[top] / tot >= 0.6 else "unknown"] += 1
        return s

    def age_split(pids):
        s = {b: 0 for b in AGE_ORDER}
        s["unknown"] = 0
        for pid in pids:
            av = persons[pid].get("age_votes") or {}
            tot = sum(av.values())
            if tot < 0.6:
                s["unknown"] += 1
                continue
            top = max(av, key=av.get)
            if av[top] / tot >= 0.5:
                s[top] += 1
            else:
                s["unknown"] += 1
        return s

    ids = list(persons)
    traffic = split(ids)
    n = len(ids) or 1
    known = traffic["female"] + traffic["male"]
    out = {
        "enabled": True,
        "method": "on-device face-crop gender+age estimate (FairFace ViT) — aggregate only",
        "coverage_pct": round(known / n * 100, 1),
        "traffic_split": traffic,
        "age_split": age_split(ids),
        "age_order": AGE_ORDER,
        "zones": {},
        "disclosure": ("Estimates from visible faces only; aggregate percentages, "
                       "no per-person data stored or exported, no images retained."),
    }
    if known == 0:
        out["note"] = ("No classifiable faces in this footage — faces were too small, "
                       "occluded or not frontal enough (min face 24px). "
                       "Closer/higher-res footage improves coverage.")
    for z in zs:
        lookers = [pid for pid in ids if persons[pid]["dwell"][z["id"]] >= min_dwell]
        if lookers:
            out["zones"][str(z["id"])] = {"lookers_split": split(lookers),
                                          "n": len(lookers)}
    return out
