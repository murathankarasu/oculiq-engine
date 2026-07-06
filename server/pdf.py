"""Oculiq branded PDF report — pure black, Inter thin, "Oculiq." wordmark.

Mirrors the web report: KPIs, annotated frame (heatmap), per-zone funnel,
metric grid with CI, timeline + dwell histogram, AQS ring, dual CPM,
zone comparison. Generated fully on-device with reportlab.
"""
import io
import time
from pathlib import Path

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

ASSETS = Path(__file__).parent / "assets"
PAGE_W, PAGE_H = A4          # 595 x 842 pt
M = 46                       # margin
CW = PAGE_W - 2 * M          # content width

F = {"light": "Helvetica", "reg": "Helvetica", "med": "Helvetica-Bold", "bold": "Helvetica-Bold"}
for key, name in (("light", "Inter-Light"), ("reg", "Inter-Regular"),
                  ("med", "Inter-Medium"), ("bold", "Inter-Bold")):
    f = ASSETS / "fonts" / f"{name}.ttf"
    if f.exists():
        try:
            pdfmetrics.registerFont(TTFont(name, str(f)))
            F[key] = name
        except Exception:
            pass

WHITE, MUTED, FAINT = 1.0, 0.62, 0.38


def _hex(h):
    h = h.lstrip("#")
    return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255


def _grab_frame(job):
    """Isı haritalı temsili kare: video ortası ya da işlenmiş foto."""
    if job.get("out_video"):
        cap = cv2.VideoCapture(job["out_video"])
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.55))
        ok, fr = cap.read()
        cap.release()
        if ok:
            ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok2:
                return ImageReader(io.BytesIO(buf.tobytes())), fr.shape[1], fr.shape[0]
    p = job.get("out_image") or job.get("sim_frame")
    if p and Path(p).exists():
        img = cv2.imread(p)
        if img is not None:
            ok2, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok2:
                return ImageReader(io.BytesIO(buf.tobytes())), img.shape[1], img.shape[0]
    return None, 0, 0


class _Doc:
    def __init__(self, out_path):
        self.c = Canvas(out_path, pagesize=A4)
        self.page = 0
        self.y = 0.0
        self._new_page()

    def _new_page(self):
        if self.page:
            self._footer()
            self.c.showPage()
        self.page += 1
        c = self.c
        c.setFillGray(0)
        c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        self.y = PAGE_H - M

    def _footer(self):
        c = self.c
        c.setStrokeGray(WHITE)
        c.setStrokeAlpha(0.12)
        c.setLineWidth(0.6)
        c.line(M, M - 14, PAGE_W - M, M - 14)
        c.setStrokeAlpha(1)
        c.setFillGray(FAINT)
        c.setFont(F["reg"], 7)
        c.drawString(M, M - 26, "Oculiq. — attention intelligence · 100% on-device · oculiq.studio")
        c.drawRightString(PAGE_W - M, M - 26, f"{self.page}")

    def need(self, h):
        if self.y - h < M + 10:
            self._new_page()

    def hairline_box(self, x, y, w, h, alpha=0.14):
        c = self.c
        c.setStrokeGray(WHITE)
        c.setStrokeAlpha(alpha)
        c.setLineWidth(0.7)
        c.rect(x, y, w, h, fill=0, stroke=1)
        c.setStrokeAlpha(1)

    def label(self, x, y, s, size=6.5, gray=MUTED):
        self.c.setFillGray(gray)
        self.c.setFont(F["reg"], size)
        self.c.drawString(x, y, s.upper())

    def finish(self):
        self._footer()
        self.c.save()


def build_pdf(job, out_path):
    rep = job["report"]
    d = _Doc(out_path)
    c = d.c

    # ---------- header ----------
    c.setFillGray(WHITE)
    c.setFont(F["bold"], 19)
    c.drawString(M, d.y - 14, "Oculiq.")
    badge = "MEASURED · ON-DEVICE"
    c.setFont(F["reg"], 6.5)
    bw = c.stringWidth(badge, F["reg"], 6.5) + 16
    d.hairline_box(PAGE_W - M - bw, d.y - 16, bw, 14, 0.3)
    c.setFillGray(WHITE)
    c.drawString(PAGE_W - M - bw + 8, d.y - 11.5, badge)
    d.y -= 46

    c.setFillGray(WHITE)
    c.setFont(F["light"], 29)
    c.drawString(M, d.y, "Attention report")
    d.y -= 15
    c.setFillGray(MUTED)
    c.setFont(F["reg"], 8)
    parts = [rep["method"], rep["model"], rep.get("scan_mode", ""),
             "snapshot" if rep["still"] else f"{rep['duration']}s footage",
             f"generated {time.strftime('%d %b %Y %H:%M')}"]
    line = ""
    for p in [p for p in parts if p]:
        cand = (line + "  ·  " + p) if line else p
        if c.stringWidth(cand, F["reg"], 8) > CW:   # taşarsa alt satıra geç
            c.drawString(M, d.y, line)
            d.y -= 11
            line = p
        else:
            line = cand
    c.drawString(M, d.y, line)
    d.y -= 26

    # ---------- KPI row ----------
    kpis = [("Passersby (traffic)", f"{rep['traffic']}"),
            ("Peak concurrency", f"{rep['peak_concurrency']}")]
    if rep.get("avg_concurrency") is not None:
        kpis.append(("Avg crowd density", f"{rep['avg_concurrency']}"))
    kpis.append(("Zones analyzed", f"{len(rep['zones'])}"))
    kw = (CW - (len(kpis) - 1) * 8) / len(kpis)
    for i, (k, v) in enumerate(kpis):
        x = M + i * (kw + 8)
        d.hairline_box(x, d.y - 52, kw, 52)
        d.label(x + 10, d.y - 16, k)
        c.setFillGray(WHITE)
        c.setFont(F["light"], 20)
        c.drawString(x + 10, d.y - 42, v)
    d.y -= 66

    # ---------- annotated frame ----------
    img, iw, ih = _grab_frame(job)
    if img and iw:
        fh = CW * ih / iw
        fh = min(fh, 300)
        fw = fh * iw / ih
        d.need(fh + 20)
        c.drawImage(img, M + (CW - fw) / 2, d.y - fh, fw, fh)
        d.hairline_box(M + (CW - fw) / 2, d.y - fh, fw, fh, 0.2)
        d.y -= fh + 10
        c.setFillGray(FAINT)
        c.setFont(F["reg"], 7)
        c.drawString(M, d.y, "Annotated frame — zones, orientation arrows, attention heatmap (turbo)")
        d.y -= 20

    # ---------- zones ----------
    for z in rep["zones"]:
        _zone_block(d, z, rep["still"])

    # ---------- comparison ----------
    if len(rep["zones"]) > 1:
        key = "impressions" if rep["still"] else "attentive_seconds"
        mx = max(1, max(z[key] for z in rep["zones"]))
        win = max(rep["zones"], key=lambda z: z["aqs"])
        d.need(60 + 20 * len(rep["zones"]))
        c.setFillGray(WHITE)
        c.setFont(F["med"], 12)
        c.drawString(M, d.y, "Zone comparison")
        c.setFillGray(MUTED)
        c.setFont(F["reg"], 7.5)
        c.drawString(M + 110, d.y + 0.5,
                     f"({'impressions' if rep['still'] else 'attentive seconds'} — winner by AQS: {win['label']})")
        d.y -= 20
        for z in rep["zones"]:
            r, g, b = _hex(z["color"])
            c.setFillGray(MUTED)
            c.setFont(F["reg"], 8)
            c.drawString(M, d.y - 3, z["label"][:22] + (" ★" if z["id"] == win["id"] else ""))
            bx, bw_ = M + 130, CW - 200
            c.setFillGray(WHITE)
            c.setFillAlpha(0.06)
            c.rect(bx, d.y - 7, bw_, 11, fill=1, stroke=0)
            c.setFillAlpha(1)
            c.setFillColorRGB(r, g, b)
            c.rect(bx, d.y - 7, bw_ * z[key] / mx, 11, fill=1, stroke=0)
            c.setFillGray(WHITE)
            c.setFont(F["med"], 8.5)
            c.drawRightString(M + CW, d.y - 3, f"{z[key]}{'' if rep['still'] else 's'}")
            d.y -= 19
        d.y -= 8

    # ---------- audience ----------
    aud = rep.get("audience")
    if aud and aud.get("enabled"):
        d.need(70)
        c.setFillGray(WHITE)
        c.setFont(F["med"], 12)
        c.drawString(M, d.y, "Audience insights")
        c.setFillGray(FAINT)
        c.setFont(F["reg"], 7)
        c.drawString(M + 108, d.y + 0.5,
                     f"estimates, aggregate only — coverage {aud.get('coverage_pct', 0)}%")
        d.y -= 16
        ts = aud.get("traffic_split", {})
        c.setFillGray(MUTED)
        c.setFont(F["reg"], 8.5)
        c.drawString(M, d.y, f"Gender (traffic): {ts.get('female', 0)} female · "
                             f"{ts.get('male', 0)} male · {ts.get('unknown', 0)} unknown")
        d.y -= 12
        ag = aud.get("age_split")
        if ag:
            parts = [f"{b}: {ag.get(b, 0)}" for b in aud.get("age_order", []) if ag.get(b, 0)]
            if ag.get("unknown"):
                parts.append(f"?: {ag['unknown']}")
            c.drawString(M, d.y, "Age (est.): " + (" · ".join(parts) if parts else "insufficient data"))
            d.y -= 12
        d.y -= 8

    # ---------- AI insights ----------
    ins = job.get("insights")
    if ins and ins.get("text"):
        d.need(80)
        c.setFillGray(WHITE)
        c.setFont(F["med"], 12)
        c.drawString(M, d.y, "AI insights")
        c.setFillGray(FAINT)
        c.setFont(F["reg"], 7)
        c.drawString(M + 70, d.y + 0.5, f"generated by {ins.get('provider', '')} — numbers only, footage never leaves the device")
        d.y -= 18
        _insights_text(d, ins["text"])
        d.y -= 8

    # ---------- honesty note ----------
    d.need(40)
    c.setFillGray(FAINT)
    c.setFont(F["reg"], 7)
    c.drawString(M, d.y, "Orientation-based attention: head-pose primary, body-orientation fallback; every measurement carries")
    d.y -= 10
    c.drawString(M, d.y, "a confidence score and rates ship with Wilson 95% CIs. Attention CPM = cost / (attentive seconds / 1000).")

    d.finish()


def _insights_text(d, text):
    """Mini markdown -> PDF: **başlık** satırları medium, '- ' maddeler girintili, sarma."""
    c = d.c
    for raw in text.split("\n"):
        ln = raw.strip()
        if not ln:
            d.y -= 4
            continue
        header = ln.startswith("**") and ln.endswith("**")
        bullet = ln.startswith("- ")
        ln = ln.replace("**", "")
        if bullet:
            ln = "•  " + ln[2:]
        font = F["med"] if header else F["reg"]
        size = 8.5 if header else 8
        indent = 0 if header else (10 if bullet else 0)
        c.setFillGray(WHITE if header else MUTED)
        c.setFont(font, size)
        words = ln.split(" ")
        line = ""
        for w_ in words:
            cand = (line + " " + w_) if line else w_
            if c.stringWidth(cand, font, size) > CW - indent:
                d.need(12)
                c.setFont(font, size)
                c.drawString(M + indent, d.y, line)
                d.y -= 11
                line = ("   " + w_) if bullet else w_
            else:
                line = cand
        if line:
            d.need(12)
            c.setFont(font, size)
            c.drawString(M + indent, d.y, line)
            d.y -= 12 if header else 11
        if header:
            d.y -= 2


def _zone_block(d, z, still):
    c = d.c
    block_h = 150 if still else 232
    d.need(block_h)
    r, g, b = _hex(z["color"])

    # title + AQS ring
    c.setFillColorRGB(r, g, b)
    c.rect(M, d.y - 9, 9, 9, fill=1, stroke=0)
    c.setFillGray(WHITE)
    c.setFont(F["med"], 13)
    c.drawString(M + 16, d.y - 9, z["label"])
    c.setFillGray(FAINT)
    c.setFont(F["reg"], 7)
    c.drawString(M + 20 + c.stringWidth(z["label"], F["med"], 13), d.y - 9, z["type"].upper())

    ring_x, ring_y, ring_r = M + CW - 18, d.y - 12, 13
    c.setStrokeGray(WHITE)
    c.setStrokeAlpha(0.15)
    c.setLineWidth(3)
    c.circle(ring_x, ring_y, ring_r, fill=0, stroke=1)
    c.setStrokeAlpha(1)
    c.setStrokeColorRGB(r, g, b)
    c.arc(ring_x - ring_r, ring_y - ring_r, ring_x + ring_r, ring_y + ring_r,
          90, -3.6 * z["aqs"])
    c.setFillGray(WHITE)
    c.setFont(F["med"], 8)
    c.drawCentredString(ring_x, ring_y - 2.5, f"{round(z['aqs'])}")
    d.label(ring_x - ring_r - 52, ring_y - 2, "AQS score", 6)
    d.y -= 26

    # funnel
    stages = [("Traffic", z["traffic"]), ("Impressions", z["impressions"])]
    if not still:
        stages += [("Engaged ≥1s", z["engaged"]), ("Deep ≥3s", z["deep"])]
    base = max(1, z["traffic"])
    for i, (lbl, v) in enumerate(stages):
        c.setFillGray(MUTED)
        c.setFont(F["reg"], 7.5)
        c.drawString(M, d.y - 3, lbl)
        bx, bw_ = M + 82, CW - 150
        c.setFillGray(WHITE)
        c.setFillAlpha(0.05)
        c.rect(bx, d.y - 6.5, bw_, 10, fill=1, stroke=0)
        c.setFillAlpha(max(0.25, 1 - i * 0.22))
        c.setFillGray(WHITE)
        c.rect(bx, d.y - 6.5, bw_ * v / base, 10, fill=1, stroke=0)
        c.setFillAlpha(1)
        c.setFillGray(WHITE)
        c.setFont(F["med"], 8.5)
        c.drawRightString(M + CW, d.y - 3, f"{v}")
        d.y -= 16
    d.y -= 6

    # metric grid
    ci = z.get("attention_rate_ci")
    cells = [("Attention rate" + (f"  (CI {ci[0]}–{ci[1]}%)" if ci else ""), f"{z['attention_rate']}%", True)]
    if not still:
        cells += [
            ("Attentive seconds", f"{z['attentive_seconds']}s", False),
            ("Avg / max dwell", f"{z['avg_dwell']}s / {z['max_dwell']}s", False),
            ("Time to first look", "—" if z.get("time_to_first_look") is None else f"{z['time_to_first_look']}s", False),
            ("Glances per looker", f"{z['glances_per_looker']}", False),
            ("Stopping power", f"{z['stopping_power']}% slowdown", False),
        ]
    cols = 3
    cw_ = (CW - (cols - 1) * 8) / cols
    rows = -(-len(cells) // cols)
    for i, (k, v, star) in enumerate(cells):
        x = M + (i % cols) * (cw_ + 8)
        yy = d.y - (i // cols) * 40
        if star:
            c.setFillGray(WHITE)
            c.rect(x, yy - 34, cw_, 34, fill=1, stroke=0)
            c.setFillGray(0.25)
            c.setFont(F["reg"], 6)
            c.drawString(x + 8, yy - 12, k.upper())
            c.setFillGray(0)
            c.setFont(F["med"], 13)
            c.drawString(x + 8, yy - 28, v)
        else:
            d.hairline_box(x, yy - 34, cw_, 34)
            d.label(x + 8, yy - 12, k, 6)
            c.setFillGray(WHITE)
            c.setFont(F["light"], 13)
            c.drawString(x + 8, yy - 28, v)
    d.y -= rows * 40 + 8

    # charts: timeline + histogram
    if not still:
        ch_w = (CW - 10) / 2
        ch_h = 64
        tl = z.get("timeline") or []
        x0 = M
        d.hairline_box(x0, d.y - ch_h, ch_w, ch_h)
        d.label(x0 + 8, d.y - 12, "Attention over time")
        if tl:
            mx_t = max(1, max(p["t"] for p in tl))
            mx_v = max(0.1, max(p["sec"] for p in tl))
            pts = [(x0 + 10 + (p["t"] / mx_t) * (ch_w - 20),
                    d.y - ch_h + 8 + (p["sec"] / mx_v) * (ch_h - 30)) for p in tl]
            c.setStrokeColorRGB(r, g, b)
            c.setLineWidth(1.4)
            for i in range(1, len(pts)):
                c.line(*pts[i - 1], *pts[i])
            if len(pts) == 1:
                c.circle(pts[0][0], pts[0][1], 2, fill=1, stroke=0)
        x1 = M + ch_w + 10
        d.hairline_box(x1, d.y - ch_h, ch_w, ch_h)
        d.label(x1 + 8, d.y - 12, "Dwell distribution")
        hist = z.get("dwell_histogram") or [0] * 5
        labels = ["<1s", "1-2", "2-3", "3-5", "5s+"]
        mx_h = max(1, max(hist))
        bw2 = (ch_w - 40) / 5
        for i, v in enumerate(hist):
            bx = x1 + 14 + i * bw2
            bh = (v / mx_h) * (ch_h - 34)
            c.setFillColorRGB(r, g, b)
            c.setFillAlpha(0.45 + 0.55 * (v / mx_h))
            c.rect(bx, d.y - ch_h + 14, bw2 - 8, max(bh, 0.5), fill=1, stroke=0)
            c.setFillAlpha(1)
            c.setFillGray(FAINT)
            c.setFont(F["reg"], 5.5)
            c.drawCentredString(bx + (bw2 - 8) / 2, d.y - ch_h + 5, labels[i])
        d.y -= ch_h + 10

    # dual CPM
    cw2 = (CW - 8) / 2
    d.hairline_box(M, d.y - 34, cw2, 34)
    d.label(M + 8, d.y - 12, "Reach CPM (per 1k lookers)", 6)
    c.setFillGray(WHITE)
    c.setFont(F["light"], 12)
    c.drawString(M + 8, d.y - 28, f"${z['reach_cpm']}" if z.get("reach_cpm") is not None else "—")
    x = M + cw2 + 8
    c.setFillGray(WHITE)
    c.rect(x, d.y - 34, cw2, 34, fill=1, stroke=0)
    c.setFillGray(0.25)
    c.setFont(F["reg"], 6)
    c.drawString(x + 8, d.y - 12, "ATTENTION CPM (PER 1K ATTENTIVE SEC)")
    c.setFillGray(0)
    c.setFont(F["med"], 12)
    c.drawString(x + 8, d.y - 28, f"${z['attention_cpm']}" if z.get("attention_cpm") is not None else "—")
    d.y -= 52
