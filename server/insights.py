"""AI insights — raporun rakamlarini yorumlayan LLM katmani.

Saglayici secimi: OCULIQ_LLM env (openai|gemini) ya da mevcut anahtara gore otomatik
(OPENAI_API_KEY / GEMINI_API_KEY). Hicbiri yoksa yerel kural-tabanli ozet uretilir —
ozellik her zaman calisir, LLM varsa zenginlesir.

Gizlilik: LLM'e SADECE rapor rakamlari gider (sim/ray verisi ve goruntu asla gitmez).
"""
import json
import os
import urllib.request

SYSTEM = (
    "You are the senior analytics writer for Oculiq, an on-device attention measurement product "
    "for out-of-home advertising. You receive a full JSON attention report (funnel, dwell stats, "
    "timelines, dwell histograms, density timeline, CIs, signal mix, calibration, CPMs). "
    "Write a DETAILED, concrete analysis in English markdown with exactly these sections:\n"
    "**Executive summary** — 3-4 sentences: the headline result and what it means commercially.\n"
    "**Funnel & engagement** — traffic -> impressions -> engaged -> deep per zone; conversion "
    "percentages between stages; what drop-offs suggest.\n"
    "**Zone-by-zone breakdown** — one bullet block per zone: rate (with 95% CI), attentive seconds, "
    "avg/max dwell, time-to-first-look, glances per looker, stopping power, AQS; interpret each.\n"
    "**Temporal patterns** — use the timeline buckets and density_timeline: when attention peaked, "
    "quiet periods, whether crowd density and attention move together.\n"
    "**Audience behavior** — dwell histogram shape (glancers vs readers), re-look behavior, "
    "stopping power meaning.\n"
    "**Media value & pricing** — reach CPM vs attention CPM if costs present; which zone deserves "
    "premium pricing and why; if costs missing, say what adding them unlocks.\n"
    "**Recommendations** — 3-5 specific, actionable items (placement, creative, measurement).\n"
    "**Data quality & caveats** — CI width, signal mix (head vs body share and its confidence), "
    "auto-calibration status, sample duration; be honest about what is directional vs solid.\n"
    "450-650 words. Never invent numbers not present in the JSON. The method is orientation-based "
    "attention (not eye-tracking) — keep claims honest and cite the numbers you use."
)


def _strip(report):
    return {k: v for k, v in report.items() if k != "sim"}


def generate(report):
    payload = json.dumps(_strip(report))
    last_err = None
    for prov in _providers():
        try:
            if prov == "openai":
                return {"provider": "openai", "text": _openai(payload)}
            if prov == "gemini":
                return {"provider": "gemini", "text": _gemini(payload)}
        except Exception as e:
            last_err = f"{prov}: {e}"
    out = {"provider": "local", "text": _local(report)}
    if last_err:
        out["note"] = f"LLM unavailable ({last_err}) — local summary used."
    return out


def _providers():
    forced = os.environ.get("OCULIQ_LLM")
    if forced:
        return [forced]
    out = []
    if os.environ.get("OPENAI_API_KEY"):
        out.append("openai")
    if os.environ.get("GEMINI_API_KEY"):
        out.append("gemini")
    return out


def _openai(payload):
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": os.environ.get("OCULIQ_LLM_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": payload}],
            "temperature": 0.3, "max_tokens": 1600,
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def _gemini(payload):
    model = os.environ.get("OCULIQ_LLM_MODEL", "gemini-2.0-flash")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
           f":generateContent?key={os.environ['GEMINI_API_KEY']}")
    req = urllib.request.Request(
        url,
        data=json.dumps({
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"parts": [{"text": payload}]}],
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["candidates"][0]["content"]["parts"][0]["text"]


def _local(report):
    """Anahtar yokken: kural-tabanli, rakamlari dogru kullanan detayli analiz."""
    zs = report["zones"]
    if not zs:
        return "No zones analyzed."
    still = report["still"]
    best = max(zs, key=lambda z: z["aqs"])
    total_att = sum(z["attentive_seconds"] for z in zs)
    L = []

    L.append("**Executive summary**")
    L.append(
        f"{report['traffic']} unique people were tracked"
        + ("" if still else f" over {report['duration']}s of footage")
        + f" (peak {report['peak_concurrency']} concurrent"
        + (f", avg crowd {report['avg_concurrency']}" if report.get("avg_concurrency") is not None else "")
        + f"). The strongest placement was **{best['label']}** with AQS {best['aqs']} and a "
        f"{best['attention_rate']}% attention rate. Zones captured "
        f"{round(total_att, 1)} attentive seconds in total.")
    L.append("")

    if not still:
        L.append("**Funnel & engagement**")
        for z in zs:
            imp, eng, deep = z["impressions"], z["engaged"], z["deep"]
            e_pct = round(eng / imp * 100) if imp else 0
            d_pct = round(deep / imp * 100) if imp else 0
            L.append(
                f"- {z['label']}: {z['traffic']} → {imp} impressions "
                f"({z['attention_rate']}%) → {eng} engaged ≥1s ({e_pct}% of lookers) → "
                f"{deep} deep ≥3s ({d_pct}%). "
                + ("Strong retention once noticed." if imp and e_pct >= 60
                   else "Most looks stay brief — creative may not be holding attention." if imp
                   else "No qualified impressions at the current thresholds."))
        L.append("")

    L.append("**Zone-by-zone breakdown**")
    for z in zs:
        ci = z.get("attention_rate_ci")
        parts = [f"rate {z['attention_rate']}%" + (f" (95% CI {ci[0]}–{ci[1]}%)" if ci else "")]
        if not still:
            parts += [f"{z['attentive_seconds']}s attention",
                      f"dwell avg {z['avg_dwell']}s / max {z['max_dwell']}s"]
            if z.get("time_to_first_look") is not None:
                parts.append(f"first look after {z['time_to_first_look']}s")
            parts.append(f"{z['glances_per_looker']} glances/looker")
            if z["stopping_power"] > 0:
                parts.append(f"{z['stopping_power']}% slowdown while looking")
        parts.append(f"AQS {z['aqs']}")
        L.append(f"- **{z['label']}** ({z['type']}): " + ", ".join(parts) + ".")
    L.append("")

    if not still:
        L.append("**Temporal patterns**")
        for z in zs:
            tl = z.get("timeline") or []
            if tl:
                pk = max(tl, key=lambda p: p["sec"])
                if pk["sec"] > 0:
                    L.append(f"- {z['label']}: attention peaked at t={pk['t']}–{pk['t']+2}s "
                             f"({pk['sec']}s of attention in that window).")
        dt_ = report.get("density_timeline") or []
        if dt_:
            pkd = max(dt_, key=lambda p: p["avg"])
            L.append(f"- Crowd density peaked around t={pkd['t']}s ({pkd['avg']} people on average).")
        L.append("")

        L.append("**Audience behavior**")
        for z in zs:
            h = z.get("dwell_histogram") or [0] * 5
            tot = sum(h)
            if tot:
                labels = ["<1s glancers", "1–2s scanners", "2–3s readers", "3–5s engagers", "5s+ dwellers"]
                dom = max(range(5), key=lambda i: h[i])
                L.append(f"- {z['label']}: dominant group is {labels[dom]} ({h[dom]}/{tot} lookers). "
                         + ("Re-look behavior present." if z["glances_per_looker"] > 1.2 else ""))
        L.append("")

    L.append("**Media value & pricing**")
    priced = [z for z in zs if z.get("attention_cpm") is not None]
    if priced:
        cheap = min(priced, key=lambda z: z["attention_cpm"])
        L.append(f"- Best attention value: {cheap['label']} at ${cheap['attention_cpm']} per 1k "
                 f"attentive seconds (reach CPM ${cheap['reach_cpm']}).")
        for z in priced:
            if z is not cheap:
                L.append(f"- {z['label']}: attention CPM ${z['attention_cpm']}, reach CPM ${z['reach_cpm']}.")
    else:
        L.append("- Add slot costs to unlock reach-CPM vs attention-CPM comparison — the divergence "
                 "between the two is where under/over-priced inventory shows up.")
    L.append(f"- {best['label']} justifies premium positioning on current AQS ranking.")
    L.append("")

    L.append("**Recommendations**")
    if len(zs) > 1:
        worst = min(zs, key=lambda z: z["aqs"])
        L.append(f"- Prioritize {best['label']} for premium campaigns; test repositioning "
                 f"{worst['label']} (AQS {worst['aqs']}) in the what-if simulator before physical changes.")
    L.append("- Use the what-if simulator to compare alternative placements against the recorded gaze rays.")
    if not still and any((z.get("time_to_first_look") or 99) > 2 for z in zs):
        L.append("- Slow time-to-first-look suggests weak stopping power at approach — consider higher-contrast creative.")
    L.append("- Re-run at different dayparts to build a temporal baseline before pricing decisions.")
    L.append("")

    L.append("**Data quality & caveats**")
    sig = best.get("signal_share", {})
    body_share = sig.get("body", 0)
    cal = report.get("calibration") or {}
    L.append(
        f"- Orientation-based measurement (not eye-tracking). {body_share}% of the attention signal "
        "comes from body orientation (confidence 0.5) — treat rates as directional; the 95% CIs are "
        "the honest range.")
    L.append(
        f"- Perspective auto-calibration: {'active (horizon fit from ' + str(cal.get('samples')) + ' samples)' if cal.get('auto') else 'fallback defaults in use (scene too sparse to fit)'}"
        + (f"; scan mode: {report.get('scan_mode')}" if report.get("scan_mode") else "") + ".")
    if not still and report["duration"] < 30:
        L.append(f"- Short sample ({report['duration']}s): results are indicative — capture longer "
                 "footage for pricing-grade numbers.")
    return "\n".join(L)
