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
    "You write the plain-language read-out for Oculiq, which measures where people actually look "
    "in a physical shop, on the shop's own camera. You are writing for the shop owner or the "
    "brand's marketing manager — a smart person with no analytics background. They should never "
    "have to look a word up.\n\n"
    "HOW TO WRITE\n"
    "- Short sentences. Everyday words. Write like you are standing in the shop explaining what "
    "you saw, not like a dashboard.\n"
    "- The FIRST time any measure appears, explain it in the same sentence, in passing. "
    "Not '\"AQS 62\"' but '\"a placement score of 62 out of 100 — that score blends how many "
    "people looked, how long they stayed, and whether they came back for a second look\"'. Same for "
    "attentive seconds ('total seconds of human attention'), capture rate ('of everyone who walked "
    "past, the share who actually came in'), hesitation ('looked properly, more than once, and "
    "still walked away empty-handed'), reach ('reached a hand toward the shelf'), CPM ('what you "
    "paid per thousand'), and the 95% range ('with this much footage the true figure sits "
    "somewhere in this band').\n"
    "- Use round numbers in the prose. Say 'about a quarter' and give the exact figure once.\n"
    "- No bullet lists of raw metrics. Write paragraphs that make a point, and put the number "
    "inside the sentence that needs it.\n\n"
    "SECTIONS (use these exact headings)\n"
    "**What happened** — 3-4 sentences a person could repeat out loud from memory.\n"
    "**Where people looked** — walk through each surface as a story: how many walked past, how "
    "many noticed it, how long they stayed, whether they came back. Say what that pattern means "
    "about the surface, not just what the numbers are.\n"
    "**What the shopping behaviour says** — use whatever is in the JSON: people who came in vs "
    "walked past, how long visits lasted, people who left quickly, people who looked at the window "
    "and then came in, people who reached for the product, and people who studied something "
    "repeatedly and still left with nothing. That last group is the most commercially interesting "
    "one — name them plainly as interested-but-not-convinced.\n"
    "**What to change** — 2-4 specific things to do, each tied to the number that justifies it. "
    "A shopkeeper should be able to act on one of them tomorrow morning without buying anything.\n"
    "**How much to trust this** — plain talk. How long the footage was, how wide the 95% range is, "
    "whether the camera view was good enough, what would make the numbers firmer. If something is "
    "too thin to act on, say so in one clear sentence.\n\n"
    "RULES\n"
    "- Never invent a number that is not in the JSON. If a measure is missing, say what it would "
    "have told them and move on.\n"
    "- This measures head and body direction, NOT eye movement. Never say eye-tracking, never say "
    "you know what someone read or felt. 'Faced the shelf' is true; 'was interested in the price "
    "tag' is not.\n"
    "- Small counts deserve hedging, not confidence. Four people is an anecdote.\n"
    "- 400-600 words."
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


def _plural(n, one, many=None):
    return one if n == 1 else (many or one + "s")


def _local(report):
    """Anahtar yokken calisan ozet — LLM'le AYNI dilde: jargonsuz, tam cumleli.

    Bu metni okuyan kisi analitik bilmiyor. Her olcu ilk gectigi cumlede
    aciklanir; hicbir kisaltma tek basina birakilmaz."""
    zs = report["zones"]
    if not zs:
        return "No zones were marked on this footage, so there is nothing to report yet. Draw a surface over a window, shelf or screen and run it again."
    still = report["still"]
    best = max(zs, key=lambda z: z["aqs"])
    total_att = sum(z["attentive_seconds"] for z in zs)
    n = report["traffic"]
    L = []

    # ---------- ne oldu ----------
    L.append("**What happened**")
    dur = "" if still else f" in {report['duration']:.0f} seconds of footage"
    s = (f"{n} {_plural(n, 'person', 'people')} passed through the camera's view{dur}"
         f", at most {report['peak_concurrency']} of them in shot at once. ")
    if report.get("capture_rate") is not None:
        s += (f"About {report['capture_rate']:.0f} in every 100 of them walked in through the "
              f"door you marked — that is the share of passers-by you actually captured. ")
    s += (f"Across everything you marked, people spent {total_att:.0f} seconds "
          f"facing your surfaces. ")
    if n < 15:
        s += ("That is a small number of people, so read everything below as a first impression "
              "rather than a finding.")
    else:
        s += f"The surface that did best was **{best['label']}**."
    L.append(s)
    L.append("")

    # ---------- nereye baktilar ----------
    L.append("**Where people looked**")
    first = True
    for z in zs:
        imp = z["impressions"]
        ci = z.get("attention_rate_ci")
        s = f"**{z['label']}** was noticed by {imp} of the {z['traffic']} {_plural(z['traffic'], 'person', 'people')} who came within range"
        s += f" — {z['attention_rate']:.0f}%."
        if ci and first:
            s += (f" With this much footage the true figure sits somewhere between {ci[0]:.0f}% "
                  f"and {ci[1]:.0f}%, so treat it as a range, not a single number.")
        if not still and imp:
            s += (f" The people who did look stayed {z['avg_dwell']:.1f} seconds on average, "
                  f"and the longest single look ran {z['max_dwell']:.1f} seconds.")
            if z.get("glances_per_looker", 0) > 1.2:
                s += (f" They came back for another look {z['glances_per_looker']:.1f} times each on "
                      "average, which usually means the surface is doing its job of pulling people "
                      "back rather than being read once and forgotten.")
            ttfl = z.get("time_to_first_look")
            if ttfl is not None:
                if ttfl < 1:
                    s += " People turned toward it almost the moment they came into view."
                else:
                    s += (f" On average it took {ttfl:.0f} seconds from entering the camera's view "
                          "before anyone turned toward it")
                    s += ("." if ttfl <= 3 else
                          " — slow enough that many people are probably well past it before it registers.")
        if z.get("benchmark_percentile") is not None:
            s += (f" Compared with every other {z['type']} we have measured, this one holds attention "
                  f"longer than {z['benchmark_percentile']}% of them.")
        L.append(s)
        L.append("")
        first = False

    # ---------- alisveris davranisi ----------
    beh = []
    for z in zs:
        if z.get("reaches"):
            beh.append(f"{z['reachers']} {_plural(z['reachers'], 'person', 'people')} actually "
                       f"reached a hand toward **{z['label']}** — the closest thing to a purchase "
                       f"signal a camera can see without watching the till.")
        if z.get("hesitations"):
            beh.append(f"{z['hesitations']} {_plural(z['hesitations'], 'person', 'people')} studied "
                       f"**{z['label']}** properly, came back to it more than once, and still walked "
                       f"away without touching anything. These are your interested-but-not-convinced "
                       f"shoppers, and they are the cheapest sale in the shop to win back.")
    for l in (report.get("visits") or {}).get("lines") or []:
        if l.get("visits"):
            beh.append(f"{l['visits']} {_plural(l['visits'], 'visit')} started and finished inside "
                       f"the footage. Half of them lasted under {l['median_duration_s']:.0f} seconds"
                       + (f", and {l['bounce_rate']:.0f}% of visitors were back out of the door "
                          f"within {l['bounce_threshold_s']:.0f} seconds without looking at anything "
                          f"you marked." if l.get("bounce_rate") is not None else "."))
        if l.get("last_surface_before_exit"):
            a = l["last_surface_before_exit"][0]
            beh.append(f"The last thing most leaving shoppers faced was **{a['label']}** "
                       f"({a['visits']} of them). Whatever is there is the final impression people "
                       f"take out of the door.")
        for w in l.get("window_conversion") or []:
            if not w["lookers"]:
                continue
            line = (f"{w['lookers']} {_plural(w['lookers'], 'person', 'people')} stopped at "
                    f"**{w['label']}** from outside, and {w['entered_after_looking']} of them came "
                    f"in afterwards — {w['conversion_rate']:.0f}%.")
            # Yorum sonuca gore degisir: %0'a "vitrin isini yapiyor" demek yanlis olurdu.
            if w["entered_after_looking"] and w["conversion_rate"] >= 20:
                line += " That is the window earning its place, measured rather than guessed."
            elif w["lookers"] < 5:
                line += (" Too few people to read anything into yet — but this is the number that "
                         "tells you whether the window sells or just decorates.")
            else:
                line += (" People are stopping but not coming in, so the window is winning the "
                         "glance and losing the step through the door.")
            beh.append(line)
    if beh:
        L.append("**What the shopping behaviour says**")
        L.extend(x + "\n" for x in beh)

    # ---------- ne degistirmeli ----------
    L.append("**What to change**")
    acts = []
    hes = max(zs, key=lambda z: z.get("hesitations") or 0)
    if hes.get("hesitations"):
        acts.append(f"Put a price, a size guide or a staff member within arm's reach of "
                    f"**{hes['label']}**. {hes['hesitations']} {_plural(hes['hesitations'], 'person', 'people')} "
                    f"looked hard at it and left anyway — they had the interest and lost it "
                    f"somewhere between looking and asking.")
    if len(zs) > 1:
        worst = min(zs, key=lambda z: z["aqs"])
        # Yalnizca gercek bir fark varsa soyle: 25% ile 24% arasindaki bir farki
        # "cogu kisi kaciriyor" diye sunmak uydurma bir sorun yaratirdi.
        if worst is not best and best["aqs"] - worst["aqs"] >= 10:
            act = (f"**{worst['label']}** is getting noticeably less attention than "
                   f"**{best['label']}** ({worst['attention_rate']:.0f}% of people versus "
                   f"{best['attention_rate']:.0f}%).")
            # What-if simulatoru YALNIZCA video analizinde var; canlida kayit
            # tutulmadigi icin isin yok. Olmayan bir ekrana yonlendirmeyiz.
            act += (" Before moving anything physically, drag it around in the what-if simulator "
                    "on this page — it replays the real looks that were recorded and shows what a "
                    "different position would have caught."
                    if report.get("sim") else
                    " Record a short clip of this camera and open it as a video analysis: that "
                    "gives you the what-if simulator, where you can move the surface and see what "
                    "a different position would have caught before you shift anything.")
            acts.append(act)
    slow = [z for z in zs if (z.get("time_to_first_look") or 0) > 3]
    if slow and not still:
        acts.append(f"People take {slow[0]['time_to_first_look']:.0f} seconds to notice "
                    f"**{slow[0]['label']}**. Anything that reads from further away — bigger type, "
                    f"more contrast, a light — buys back those seconds.")
    if report.get("capture_rate") is not None and report["capture_rate"] < 15:
        acts.append(f"Only {report['capture_rate']:.0f}% of passers-by come in. The window is doing "
                    f"the recruiting here, so it is worth more of your attention than anything "
                    f"inside the shop.")
    if not acts:
        acts.append("Nothing in this footage is clearly broken. The useful next step is more "
                    "footage at different times of day, so you can see whether these patterns hold.")
    acts.append("Check the hour-by-hour panel below once a few days have built up — it names the "
                "hour of the day where you lose the most people, which is the one thing here you "
                "can schedule staff around."
                if report.get("mode") == "live" else
                "Run this again at a different time of day. One clip is a snapshot; the pattern "
                "across a week is what you can actually price and plan against.")
    L.extend(x + "\n" for x in acts)

    # ---------- ne kadar guvenmeli ----------
    L.append("**How much to trust this**")
    mh = report.get("measurement_health") or {}
    sig = best.get("signal_share", {})
    t = ("This is measured from which way people's heads and bodies were turned — not from their "
         "eyes. It tells you someone faced a surface, never what they read or how they felt. ")
    if sig.get("body", 0) > 40:
        t += (f"For {sig['body']:.0f}% of the measurements only the body direction was visible, "
              "which is a weaker signal than a clear view of the head — those looks are counted "
              "with lower confidence rather than dropped. ")
    if mh.get("direction_share") is not None and mh["direction_share"] < 60:
        t += (f"We could only read a direction for {mh['direction_share']:.0f}% of the people we "
              "saw. A camera angle closer to eye level would raise that. ")
    if not still and report["duration"] < 60:
        t += (f"The clip is {report['duration']:.0f} seconds long, which is enough to see whether "
              "the system works but not enough to price anything. ")
    if n < 15:
        t += (f"Above all, {n} people is a handful. Percentages calculated from a handful move "
              "wildly with one extra person, so use the direction of these numbers, not their "
              "exact value.")
    L.append(t.strip())
    return "\n".join(L)
