"""Oculiq attention-event dataset — the compounding data asset (çift-amaçlı).

Every analysis (batch + live) emits anonymous per-(person, surface) *attention
episodes*: the behavioral signal (dwell, glances, gaze signal grade, approach
speed, reach) plus the surface context (type, size, view distance). No identity,
no per-person timestamps — aggregate-safe, k-anon friendly (Spec v1.0 §9).

Dual purpose:
  1. Retail benchmark — the normative "how does a shelf/window/display behave"
     dataset (our moat). Feeds portfolio rank + category comparisons.
  2. Model seed — each episode is a labeled attention data point (context →
     behavior). Substrate-independent features that a future world-attention
     model (physical + AR/3D) trains on. See memory: oculiq-ar-dual-purpose.

Stored in the same SQLite file as the hourly aggregates (data/metrics.db).
"""
import sqlite3
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DB = DATA / "metrics.db"

SCHEMA_VERSION = 1
# model-hazır özellik seti — versiyonla; şema değişirse SCHEMA_VERSION artar
FIELDS = ("zone_type", "dwell", "glances", "signal", "approach_speed",
          "reached", "view_distance_m", "zone_w_m", "zone_h_m")


def _db():
    DATA.mkdir(exist_ok=True)
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS attention_events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source     TEXT,            -- job id (batch) ya da camera id (live)
        kind       TEXT,            -- 'batch' | 'live'
        spec       TEXT,            -- measurement spec sürümü
        schema_v   INTEGER,         -- bu veri şemasının sürümü
        zone_type  TEXT,
        dwell      REAL,            -- attentive saniye (bu kişi, bu yüzey)
        glances    INTEGER,         -- ayrı bakış epizodu sayısı
        signal     TEXT,            -- baskın sinyal: 'head' | 'body'
        approach_speed REAL,        -- ortalama hız (boy/sn); yoksa NULL
        reached    INTEGER,         -- rafa uzandı mı (0/1)
        view_distance_m REAL,       -- 3D varsa; yoksa NULL
        zone_w_m   REAL,            -- yüzey genişliği (m); yoksa NULL
        zone_h_m   REAL,
        created_at INTEGER)""")
    return con


def record(source, kind, spec, episodes):
    """Bir analizin dikkat epizotlarını kalıcı hale getir. episodes: list[dict]."""
    if not episodes:
        return 0
    now = int(time.time())
    rows = [(source, kind, str(spec), SCHEMA_VERSION,
             e.get("zone_type"), e.get("dwell"), e.get("glances"), e.get("signal"),
             e.get("approach_speed"),
             1 if e.get("reached") else 0,
             e.get("view_distance_m"), e.get("zone_w_m"), e.get("zone_h_m"), now)
            for e in episodes]
    con = _db()
    with con:
        con.executemany(
            "INSERT INTO attention_events "
            "(source,kind,spec,schema_v,zone_type,dwell,glances,signal,"
            " approach_speed,reached,view_distance_m,zone_w_m,zone_h_m,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.close()
    return len(rows)


def stats():
    """Normatif benchmark: yüzey tipi başına epizot sayısı, ortalama dwell,
    reach oranı, kafa-sinyali payı. Retail kıyas + model veri sağlığı göstergesi."""
    con = _db()
    total = con.execute("SELECT COUNT(*) FROM attention_events").fetchone()[0]
    by_type = con.execute("""
        SELECT zone_type, COUNT(*), AVG(dwell), AVG(reached),
               AVG(CASE WHEN signal='head' THEN 1.0 ELSE 0.0 END),
               AVG(approach_speed)
        FROM attention_events GROUP BY zone_type ORDER BY COUNT(*) DESC""").fetchall()
    sources = con.execute("SELECT COUNT(DISTINCT source) FROM attention_events").fetchone()[0]
    con.close()
    return {
        "total_episodes": total,
        "sources": sources,
        "schema_version": SCHEMA_VERSION,
        "by_zone_type": [
            {"zone_type": r[0], "episodes": r[1],
             "avg_dwell": round(r[2], 2) if r[2] is not None else None,
             "reach_rate": round(r[3] * 100, 1) if r[3] is not None else None,
             "head_signal_share": round(r[4] * 100, 1) if r[4] is not None else None,
             "avg_approach_speed": round(r[5], 3) if r[5] is not None else None}
            for r in by_type],
    }


def percentile_rank(zone_type, dwell):
    """Bir yüzeyin ortalama dwell'i, aynı tip yüzeyler arasında yüzde kaçından iyi.
    Portfolio-rank / kategori kıyasının veri-tabanlı hâli (kimseye ait değil, agregat)."""
    con = _db()
    row = con.execute(
        "SELECT COUNT(*), SUM(CASE WHEN dwell < ? THEN 1 ELSE 0 END) "
        "FROM attention_events WHERE zone_type=?", (dwell, zone_type)).fetchone()
    con.close()
    n, below = row[0] or 0, row[1] or 0
    if n < 20:                       # az veri: dürüstlük — kıyas verme
        return None
    return round(below / n * 100)
