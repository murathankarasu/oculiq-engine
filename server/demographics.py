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
MIN_FACE_PX = 24          # bundan kucuk yuz siniflandirilmaz (kalite kapisi)
MIN_SCORE = 0.65          # dusuk guvenli tahmin oy sayilmaz
_pipe = None


def _gp():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else \
                     (0 if torch.cuda.is_available() else -1)
        except Exception:
            device = -1
        _pipe = pipeline("image-classification", model=MODEL, device=device)
    return _pipe


def classify(crops_bgr):
    """BGR yüz kırpıntıları -> [(label, score)]; label: female|male."""
    import cv2
    from PIL import Image
    imgs = []
    for c in crops_bgr:
        if min(c.shape[:2]) < 96:   # kucuk yuzleri buyut (ViT 224'e gider)
            s = 128.0 / min(c.shape[:2])
            c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)),
                           interpolation=cv2.INTER_CUBIC)
        imgs.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
    outs = _gp()(imgs, top_k=1)
    res = []
    for o in outs:
        best = o[0] if isinstance(o, list) else o
        label = "female" if "female" in best["label"].lower() else "male"
        res.append((label, float(best["score"])))
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

    ids = list(persons)
    traffic = split(ids)
    n = len(ids) or 1
    known = traffic["female"] + traffic["male"]
    out = {
        "enabled": True,
        "method": "on-device face-crop gender estimate (FairFace ViT) — aggregate only",
        "coverage_pct": round(known / n * 100, 1),
        "traffic_split": traffic,
        "zones": {},
        "disclosure": ("Estimates from visible faces only; aggregate percentages, "
                       "no per-person data stored or exported, no images retained."),
    }
    for z in zs:
        lookers = [pid for pid in ids if persons[pid]["dwell"][z["id"]] >= min_dwell]
        if lookers:
            out["zones"][str(z["id"])] = {"lookers_split": split(lookers),
                                          "n": len(lookers)}
    return out
