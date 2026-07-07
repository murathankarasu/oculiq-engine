"""RTMO — kalabalik-optimize tek-gecis coklu-kisi pose (OpenMMLab, Apache-2.0).

Kalabalik modunda 7-gecisli YOLO parcalamasinin yerini alir: tek inferans,
CrowdPose-sinifi isabet, CPU'da ~230ms. Cikti bizim raw formatina cevrilir;
kimlikleri yine SimpleTracker verir. Yuklenemezse motor sessizce eski
parcali YOLO yoluna doner (davranis kaybi yok).
"""
import os
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
os.environ.setdefault("XDG_CACHE_HOME", str(MODELS_DIR))

RTMO_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
            "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip")


class RtmoDetector:
    def __init__(self, kpt_thr=0.3, person_thr=0.5):
        from rtmlib import RTMO
        self.kpt_thr = kpt_thr
        self.person_thr = person_thr
        self.m = RTMO(onnx_model=RTMO_URL, model_input_size=(640, 640),
                      backend="onnxruntime", device="cpu")

    def detect(self, frame_bgr):
        """-> engine raw formati: [{box, conf, kp, kc}] (COCO-17)."""
        kpts, scores = self.m(frame_bgr)
        raws = []
        if kpts is None or len(kpts) == 0:
            return raws
        for kp, kc in zip(kpts, scores):
            kc = np.asarray(kc, dtype=float)
            kp = np.asarray(kp, dtype=float)
            conf = float(np.mean(np.sort(kc)[-8:]))   # en iyi 8 eklemin ortalamasi
            if conf < self.person_thr:
                continue
            vis = kp[kc >= self.kpt_thr]
            if len(vis) < 4:
                continue
            if int((kc >= self.kpt_thr).sum()) < 6:
                continue   # çok az güvenilir eklem -> gürültü pozu
            x1, y1 = vis.min(axis=0)
            x2, y2 = vis.max(axis=0)
            w, h = float(x2 - x1), float(y2 - y1)
            if w < 6 or h < 12:
                continue
            if w > 1.6 * h:
                continue   # insandan geniş kutu = dağılmış/karışmış keypoint seti
            pad = 0.12 * h   # kafa ustu / ayak alti payi (kutu tam govdeye yaklassin)
            raws.append({"box": (float(x1 - 0.08 * w), float(y1 - pad),
                                 w * 1.16, h + 2 * pad),
                         "conf": conf, "kp": kp, "kc": kc})
        return raws
