// Oculiq web app — upload → zones → analyze (SSE live) → advanced report.
// All inference happens in the local Python backend; this file is pure UI.

const $ = (id) => document.getElementById(id);
const video = $("video"), photo = $("photo"), draw = $("draw");
const dctx = draw.getContext("2d");

const state = {
  file: null, isVideo: true, zones: [], nextId: 0,
  jobId: null, es: null,
};
const COLORS = ["#1d9e75", "#378add", "#d85a30", "#7f77dd", "#ba7517"];

/* ---------- step navigation ---------- */
const STEPS = ["upload", "zones", "analyze", "report", "history"];
function goto(step) {
  STEPS.forEach((s) => {
    $("step-" + s).classList.toggle("on", s === step);
    const btn = document.querySelector(`.nav-item[data-step="${s}"]`);
    btn.classList.toggle("on", s === step);
    if (step !== "history" && s !== "history"
        && STEPS.indexOf(s) < STEPS.indexOf(step)) btn.classList.add("done");
  });
  document.querySelector(`.nav-item[data-step="${step}"]`).disabled = false;
}
document.querySelectorAll(".nav-item").forEach((b) =>
  b.addEventListener("click", () => {
    if (b.disabled) return;
    goto(b.dataset.step);
    if (b.dataset.step === "history") showHistory();
  }));

/* ---------- settings / gear ---------- */
$("gearBtn").onclick = async () => {
  const p = $("settings");
  p.classList.toggle("hidden");
  if (!p.classList.contains("hidden")) refreshSettings();
};
$("setClose").onclick = () => $("settings").classList.add("hidden");

async function refreshSettings() {
  try {
    const s = await (await fetch("/api/storage")).json();
    $("storageLine").textContent = `${s.analyses} stored ${s.analyses === 1 ? "analysis" : "analyses"} · ${fmtBytes(s.bytes)}`;
  } catch { $("storageLine").textContent = "unavailable"; }
  try {
    const c = await (await fetch("/api/config")).json();
    $("llmLine").textContent = c.llm === "local"
      ? "No API key set — local rule-based summary in use."
      : `Provider: ${c.llm} · model: ${c.model}`;
  } catch { $("llmLine").textContent = "unavailable"; }
}

$("wipeBtn").onclick = async () => {
  if (!confirm("Delete ALL stored analyses? This cannot be undone.")) return;
  const r = await (await fetch("/api/jobs", { method: "DELETE" })).json();
  $("storageLine").textContent = `${r.deleted} analyses deleted.`;
  refreshSettings();
  if ($("step-history").classList.contains("on")) showHistory();
};

const fmtBytes = (b) => b > 1e9 ? (b / 1e9).toFixed(2) + " GB" : b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.round(b / 1e3) + " KB";

/* ---------- history / workspace ---------- */
async function showHistory() {
  const el = $("historyList");
  el.innerHTML = '<p class="hist-empty">Loading…</p>';
  let list = [];
  try { list = await (await fetch("/api/history")).json(); } catch { }
  if (!list.length) {
    el.innerHTML = '<p class="hist-empty">No analyses yet — run one from Upload.</p>';
    return;
  }
  el.innerHTML = list.map((m) => `
    <div class="hist-row">
      <div class="hi-main">
        <b>${m.zones.map((z) => esc(z.label)).join(", ") || "—"}</b>
        <small>${new Date(m.created * 1000).toLocaleString("en-GB")} · ${m.still ? "photo" : fmt(m.duration, 0) + "s video"} · ${esc(m.scan_mode || "")}</small>
      </div>
      <div class="hi-stats">
        <span><b>${fmt(m.traffic)}</b> traffic</span>
        <span><b>${m.zones.length ? fmt(Math.max(...m.zones.map((z) => z.aqs)), 0) : "—"}</b> best AQS</span>
      </div>
      <button class="sm" data-open="${m.job_id}">Open report</button>
      <button class="rm" data-del="${m.job_id}" title="delete this analysis">×</button>
    </div>`).join("");
  el.querySelectorAll("[data-open]").forEach((b) =>
    b.onclick = () => loadReport(b.dataset.open));
  el.querySelectorAll("[data-del]").forEach((b) =>
    b.onclick = async () => {
      if (!confirm("Delete this analysis?")) return;
      await fetch(`/api/jobs/${b.dataset.del}`, { method: "DELETE" });
      showHistory();
    });
}

/* ---------- upload ---------- */
const dz = $("dropzone");
$("pickBtn").onclick = (e) => { e.stopPropagation(); $("fileInput").click(); };
dz.onclick = () => $("fileInput").click();
$("fileInput").onchange = (e) => { if (e.target.files[0]) loadMedia(e.target.files[0]); };
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("over"));
dz.addEventListener("drop", (e) => {
  e.preventDefault(); dz.classList.remove("over");
  if (e.dataTransfer.files[0]) loadMedia(e.dataTransfer.files[0]);
});

function loadMedia(file) {
  state.file = file;
  state.isVideo = file.type.startsWith("video");
  state.zones = []; renderZoneList();
  const url = URL.createObjectURL(file);
  $("canvasWrap").classList.toggle("photo", !state.isVideo);
  if (state.isVideo) {
    video.src = url;
    video.onloadedmetadata = () => { initStage(video.videoWidth, video.videoHeight); video.currentTime = 0; };
  } else {
    photo.src = url;
    photo.onload = () => initStage(photo.naturalWidth, photo.naturalHeight);
  }
}
function initStage(W, H) {
  draw.width = W; draw.height = H;
  renderZones();
  goto("zones");
}

/* ---------- zone drawing ---------- */
let drawing = false, dragStart = null, dragCur = null, pending = null;

$("zoneBtn").onclick = () => {
  draw.style.pointerEvents = "auto"; draw.style.cursor = "crosshair";
};
const normPt = (e) => {
  const r = draw.getBoundingClientRect();
  return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
};
draw.addEventListener("pointerdown", (e) => { drawing = true; dragStart = normPt(e); dragCur = dragStart; });
draw.addEventListener("pointermove", (e) => { if (drawing) { dragCur = normPt(e); renderZones(); } });
draw.addEventListener("pointerup", () => {
  if (!drawing) return;
  drawing = false;
  const a = dragStart, b = dragCur;
  const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
  const w = Math.abs(a.x - b.x), h = Math.abs(a.y - b.y);
  draw.style.pointerEvents = "none"; draw.style.cursor = "default";
  if (w < 0.01 || h < 0.01) { renderZones(); return; }
  pending = { x, y, w, h };
  openPop(x, y, w, h);
});

function openPop(x, y, w) {
  const pop = $("zonePop");
  const wrap = $("canvasWrap").getBoundingClientRect();
  pop.classList.remove("hidden");
  pop.style.left = Math.min(x * wrap.width + 8, wrap.width - 246) + "px";
  pop.style.top = Math.min((y) * wrap.height + 8, wrap.height - 150) + "px";
  $("zpLabel").value = `Zone ${state.zones.length + 1}`;
  $("zpCost").value = "";
  $("zpLabel").focus(); $("zpLabel").select();
}
$("zpSave").onclick = () => {
  if (!pending) return;
  state.zones.push({
    id: state.nextId++, ...pending,
    label: $("zpLabel").value.trim() || `Zone ${state.zones.length + 1}`,
    type: $("zpType").value,
    cost: parseFloat($("zpCost").value) || 0,
    color: COLORS[state.zones.length % COLORS.length],
  });
  pending = null;
  $("zonePop").classList.add("hidden");
  renderZones(); renderZoneList();
  $("toAnalyze").disabled = false;
};
$("zpCancel").onclick = () => { pending = null; $("zonePop").classList.add("hidden"); renderZones(); };
$("zpLabel").addEventListener("keydown", (e) => { if (e.key === "Enter") $("zpSave").click(); });

function renderZones() {
  const W = draw.width, H = draw.height;
  dctx.clearRect(0, 0, W, H);
  for (const z of state.zones) {
    dctx.strokeStyle = z.color; dctx.lineWidth = Math.max(2, W / 480);
    dctx.strokeRect(z.x * W, z.y * H, z.w * W, z.h * H);
    dctx.fillStyle = z.color;
    dctx.font = `600 ${Math.max(13, W / 70)}px sans-serif`;
    dctx.fillText(z.label, z.x * W + 6, z.y * H - 7);
  }
  if (drawing && dragStart && dragCur) {
    dctx.setLineDash([7, 5]); dctx.strokeStyle = "#fff"; dctx.lineWidth = 2;
    dctx.strokeRect(Math.min(dragStart.x, dragCur.x) * W, Math.min(dragStart.y, dragCur.y) * H,
      Math.abs(dragStart.x - dragCur.x) * W, Math.abs(dragStart.y - dragCur.y) * H);
    dctx.setLineDash([]);
  }
  if (pending) {
    dctx.setLineDash([7, 5]); dctx.strokeStyle = "#fff"; dctx.lineWidth = 2;
    dctx.strokeRect(pending.x * W, pending.y * H, pending.w * W, pending.h * H);
    dctx.setLineDash([]);
  }
}

function renderZoneList() {
  const ul = $("zoneList");
  if (!state.zones.length) { ul.innerHTML = '<li class="empty">No zones yet</li>'; return; }
  ul.innerHTML = state.zones.map((z) =>
    `<li><span class="zdot" style="background:${z.color}"></span>` +
    `<span class="zl-main"><b>${esc(z.label)}</b><small>${z.type}${z.cost ? " · $" + z.cost + "/day" : ""}</small></span>` +
    `<button class="rm" data-rm="${z.id}" title="remove">×</button></li>`).join("");
  ul.querySelectorAll("[data-rm]").forEach((b) => b.onclick = () => {
    state.zones = state.zones.filter((z) => z.id != b.dataset.rm);
    renderZones(); renderZoneList();
    $("toAnalyze").disabled = !state.zones.length;
  });
}
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---------- analyze ---------- */
$("toAnalyze").onclick = startAnalysis;
$("cancelBtn").onclick = async () => {
  if (state.jobId) await fetch(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
};

async function startAnalysis() {
  goto("analyze");
  $("anSub").textContent = "Loading model on this device… first run downloads nothing, everything is local.";
  $("livePreview").removeAttribute("src");

  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("zones", JSON.stringify(state.zones.map(({ id, label, type, x, y, w, h }) => ({ id, label, type, x, y, w, h }))));
  fd.append("costs", JSON.stringify(Object.fromEntries(state.zones.map((z) => [z.id, z.cost]))));
  fd.append("crowd_mode", $("crowdMode").checked ? "on" : "auto");

  let res;
  try {
    res = await fetch("/api/analyze", { method: "POST", body: fd });
  } catch { $("anSub").textContent = "Local server unreachable — is run.sh running?"; return; }
  if (!res.ok) { $("anSub").textContent = "Upload failed: " + (await res.text()); return; }
  const { job_id } = await res.json();
  state.jobId = job_id;

  const es = new EventSource(`/api/jobs/${job_id}/events`);
  state.es = es;
  es.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.status === "error") { es.close(); $("anSub").textContent = "Engine error: " + (m.error || "unknown"); return; }
    if (m.status === "processing") $("anSub").textContent = "YOLO11-pose x · ByteTrack IDs · orientation cone test — live preview below.";
    setProgress(m.progress || 0);
    if (m.frame) $("livePreview").src = "data:image/jpeg;base64," + m.frame;
    if (m.live) renderLiveCards(m.live);
    if (m.status === "done") { es.close(); loadReport(job_id); }
  };
  es.onerror = () => { /* transient; EventSource retries */ };
}

function setProgress(p) {
  $("pct").textContent = p;
  const C = 326.7;
  $("ringFg").style.strokeDashoffset = C - (C * p) / 100;
}

function renderLiveCards(live) {
  let html = `<div class="live-card"><span class="k">passersby</span><span class="v">${live.traffic}</span></div>`;
  for (const zid in live.zones) {
    const z = live.zones[zid];
    html += `<div class="live-card"><span class="k">${esc(z.label)}</span><span class="v">${z.lookers} <small>${z.att}s</small></span></div>`;
  }
  $("liveCards").innerHTML = html;
}

/* ---------- report ---------- */
async function loadReport(jobId) {
  const rep = await (await fetch(`/api/jobs/${jobId}/report`)).json();
  renderReport(rep, jobId);
  goto("report");
  requestAnimationFrame(() => requestAnimationFrame(animateReport));
  addPortfolioBadges(rep);
}

// Portföy benchmark'ı: bu bölgenin AQS'i, kayıtlı tüm analizlerdeki bölgelere göre yüzde kaçta?
async function addPortfolioBadges(rep) {
  let hist = [];
  try { hist = await (await fetch("/api/history")).json(); } catch { return; }
  const pool = hist.flatMap((m) => m.zones.map((z) => z.aqs));
  if (pool.length < 4) return;
  document.querySelectorAll(".zone-report .zr-top").forEach((el, i) => {
    const z = rep.zones[i];
    if (!z) return;
    const rank = pool.filter((a) => a > z.aqs).length + 1;
    const span = document.createElement("span");
    span.className = "pf-badge";
    span.textContent = `portfolio rank #${rank} of ${pool.length} zones`;
    el.querySelector(".aqs-chip").before(span);
  });
}

window.oculiqLoadReport = loadReport; // dış erişim: rapor deep-link / debug

const fmt = (n, d = 0) => Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });

function renderReport(rep, jobId) {
  const dlHref = rep.still ? `/api/jobs/${jobId}/image` : `/api/jobs/${jobId}/video`;
  const dlLabel = rep.still ? "Download annotated image" : "Download annotated video";

  let html = `
  <div class="rep-head">
    <div>
      <span class="badge"><span class="dot-live"></span>measured · on-device</span>
      <h2>Attention report</h2>
      <div class="sub">${esc(rep.method)} · ${esc(rep.model)} · ${esc(rep.scan_mode || "single-pass")}${rep.calibration && rep.calibration.auto ? " · auto-calibrated" : ""} · ${rep.still ? "snapshot" : fmt(rep.duration, 1) + "s footage"} · processed in ${fmt(rep.processing_seconds, 1)}s</div>
    </div>
    <div class="rep-actions">
      <button id="dlPdf" class="primary">Export PDF</button>
      <button id="dlMedia">${dlLabel}</button>
      <button id="dlJson">JSON</button>
      <button id="dlCsv">CSV</button>
    </div>
  </div>
  <div class="kpi-grid">
    <div class="kpi"><div class="k">Passersby (traffic)</div><div class="v" data-count="${rep.traffic}">0</div></div>
    <div class="kpi"><div class="k">Peak concurrency</div><div class="v" data-count="${rep.peak_concurrency}">0</div></div>
    ${rep.avg_concurrency != null ? `<div class="kpi"><div class="k">Avg crowd density</div><div class="v">${fmt(rep.avg_concurrency, 1)}<small> ppl</small></div></div>` : ""}
    ${rep.zones.length ? `<div class="kpi"><div class="k">Best zone (AQS)</div><div class="v">${esc(best(rep).label)} <small>${best(rep).aqs}</small></div></div>` : ""}
    <div class="kpi"><div class="k">Zones analyzed</div><div class="v" data-count="${rep.zones.length}">0</div></div>
  </div>`;

  html += densitySvg(rep);
  html += glossaryHtml();

  for (const z of rep.zones) html += zoneReport(z, rep.still);

  if (rep.zones.length > 1) {
    const key = rep.still ? "impressions" : "attentive_seconds";
    const max = Math.max(1, ...rep.zones.map((z) => z[key]));
    const win = rep.zones.reduce((a, b) => (a.aqs >= b.aqs ? a : b));
    html += `<div class="cmp-section"><h3>Zone comparison</h3>
      <div class="sub">${rep.still ? "impressions" : "attentive seconds"} — winner by AQS: <b>${esc(win.label)}</b></div>` +
      rep.zones.map((z) =>
        `<div class="cmp-bar"><span>${esc(z.label)}${z.id === win.id ? '<span class="winner">★ winner</span>' : ""}</span>` +
        `<div class="track"><div data-w="${Math.round((z[key] / max) * 100)}" style="background:${z.color}"></div></div>` +
        `<b>${fmt(z[key], rep.still ? 0 : 1)}${rep.still ? "" : "s"}</b></div>`).join("") +
      donutSvg(rep) +
      `</div>`;
  }

  if (rep.sim && rep.sim.rays && rep.sim.rays.length) {
    html += `
    <div class="sim-section">
      <h3>What-if simulator <span class="new-tag">UNIQUE</span></h3>
      <div class="sub">Drag the virtual placement — or draw a new one anywhere. Metrics recompute instantly from ${fmt(rep.sim.rays.length)} recorded gaze rays. No re-processing.</div>
      <div class="sim-grid">
        <div class="sim-wrap"><canvas id="simCanvas"></canvas></div>
        <div>
          <div class="sim-metrics" id="simMetrics"></div>
          <div class="cone-cal">
            <label>Cone calibration <b id="coneVal">${rep.sim.cone_deg}°</b></label>
            <input type="range" id="coneSlider" min="15" max="60" step="1" value="${rep.sim.cone_deg}" />
            <small>Attention-cone half angle — tune per camera, watch metrics react. Analysis used ${rep.sim.cone_deg}°.</small>
          </div>
        </div>
      </div>
    </div>`;
  }

  html += `
  <div class="ins-section">
    <h3>AI insights <span class="new-tag">LLM</span></h3>
    <div class="sub">Numbers-only interpretation — footage never leaves this machine. Uses GPT/Gemini when an API key is set, local summary otherwise.</div>
    <div id="insBody"><button id="insBtn">Generate insights</button></div>
  </div>`;

  if (!rep.still) {
    html += `<div class="player-section hidden" id="playerSection">
      <h3>Evidence player</h3>
      <video id="evPlayer" src="/api/jobs/${jobId}/video" controls></video>
    </div>`;
  }

  html += `<p class="dl-note">Orientation-based attention: head-pose is the primary signal with per-measurement confidence — honest metrics, no eye-tracking overclaim. Attention CPM = cost ÷ (attentive seconds ÷ 1000).</p>`;
  $("reportRoot").innerHTML = html;

  $("dlPdf").onclick = () => location.assign(`/api/jobs/${jobId}/pdf`);
  $("dlMedia").onclick = () => location.assign(dlHref);
  $("dlJson").onclick = () => download("oculiq-report.json", JSON.stringify(rep, null, 2), "application/json");
  $("dlCsv").onclick = () => download("oculiq-report.csv", toCsv(rep), "text/csv");

  document.querySelectorAll(".ev-chip").forEach((b) => b.onclick = () => {
    const ps = $("playerSection");
    ps.classList.remove("hidden");
    const v = $("evPlayer");
    v.currentTime = parseFloat(b.dataset.t);
    v.play();
    ps.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  if (rep.sim && rep.sim.rays && rep.sim.rays.length) initSim(rep, jobId);
  initInsights(jobId);
}

/* ---------- AI insights ---------- */
async function initInsights(jobId) {
  const body = $("insBody");
  const show = (res) => {
    body.innerHTML = `<div class="ins-text">${mdLite(res.text)}</div>` +
      `<div class="ins-meta">generated by ${esc(res.provider)}${res.note ? " · " + esc(res.note) : ""}` +
      ` · <button id="insRe" class="link-btn">regenerate</button></div>`;
    $("insRe").onclick = run;
  };
  const run = async () => {
    body.innerHTML = '<div class="ins-loading">Analyzing the numbers…</div>';
    try {
      const res = await (await fetch(`/api/jobs/${jobId}/insights`, { method: "POST" })).json();
      show(res);
    } catch {
      body.innerHTML = '<div class="ins-loading">Insight generation failed — is the server running?</div>';
    }
  };
  try {
    const r = await fetch(`/api/jobs/${jobId}/insights`);
    if (r.ok) { show(await r.json()); return; }
  } catch { }
  $("insBtn").onclick = run;
}

// minik markdown: **bold**, "- " bullets, satırlar
function mdLite(t) {
  const esc2 = esc(t);
  const lines = esc2.split("\n");
  let html = "", inList = false;
  for (const ln of lines) {
    const b = ln.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
    if (ln.trim().startsWith("- ")) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${b.trim().slice(2)}</li>`;
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (ln.trim()) html += `<p>${b}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

/* ---------- what-if simulator ---------- */
function initSim(rep, jobId) {
  const sim = rep.sim;
  const canvas = $("simCanvas"), ctx = canvas.getContext("2d");
  canvas.width = sim.w; canvas.height = sim.h;
  const bg = new Image();
  bg.src = `/api/jobs/${jobId}/frame`;

  const z0 = rep.zones[0];
  let coneDeg = sim.cone_deg;
  const compute = (rect) => simCompute(rect, sim, coneDeg);
  const base = z0 ? simCompute(normToPx(z0.norm, sim), sim, sim.cone_deg) : null;
  let vz = z0
    ? normToPx(z0.norm, sim)
    : { x: sim.w * 0.35, y: sim.h * 0.3, w: sim.w * 0.3, h: sim.h * 0.25 };
  let mode = null, grab = null, anchor = null;

  const pt = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) / r.width * sim.w, y: (e.clientY - r.top) / r.height * sim.h };
  };
  const inside = (p) => p.x >= vz.x && p.x <= vz.x + vz.w && p.y >= vz.y && p.y <= vz.y + vz.h;

  canvas.style.touchAction = "none";
  canvas.onpointerdown = (e) => {
    canvas.setPointerCapture(e.pointerId);
    const p = pt(e);
    if (inside(p)) { mode = "move"; grab = { dx: p.x - vz.x, dy: p.y - vz.y }; }
    else { mode = "draw"; anchor = p; vz = { x: p.x, y: p.y, w: 1, h: 1 }; }
  };
  canvas.onpointermove = (e) => {
    if (!mode) return;
    const p = pt(e);
    if (mode === "move") {
      vz.x = Math.min(Math.max(p.x - grab.dx, 0), sim.w - vz.w);
      vz.y = Math.min(Math.max(p.y - grab.dy, 0), sim.h - vz.h);
    } else {
      vz = { x: Math.min(anchor.x, p.x), y: Math.min(anchor.y, p.y),
             w: Math.abs(p.x - anchor.x), h: Math.abs(p.y - anchor.y) };
    }
    schedule();
  };
  canvas.onpointerup = () => { mode = null; schedule(); };

  let raf = null;
  function schedule() {
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = null; render(); showMetrics(); });
  }

  function render() {
    ctx.clearRect(0, 0, sim.w, sim.h);
    if (bg.complete) ctx.drawImage(bg, 0, 0, sim.w, sim.h);
    const lw = Math.max(2, sim.w / 480);
    for (const z of rep.zones) {
      const r = normToPx(z.norm, sim);
      ctx.setLineDash([8, 6]); ctx.strokeStyle = "rgba(255,255,255,.45)"; ctx.lineWidth = lw;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,.6)";
      ctx.font = `${Math.max(13, sim.w / 80)}px sans-serif`;
      ctx.fillText(z.label, r.x + 6, r.y - 8);
    }
    ctx.fillStyle = "rgba(255,255,255,.14)";
    ctx.fillRect(vz.x, vz.y, vz.w, vz.h);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = lw * 1.4;
    ctx.strokeRect(vz.x, vz.y, vz.w, vz.h);
    ctx.fillStyle = "#fff";
    ctx.font = `600 ${Math.max(13, sim.w / 80)}px sans-serif`;
    ctx.fillText("VIRTUAL", vz.x + 6, vz.y - 8);
  }
  bg.onload = () => schedule();

  const slider = $("coneSlider");
  if (slider) {
    slider.oninput = () => {
      coneDeg = parseFloat(slider.value);
      $("coneVal").textContent = coneDeg + "°";
      schedule();
    };
  }

  function showMetrics() {
    const m = compute(vz);
    const d = (cur, b, suffix = "") => {
      if (!base || b == null || cur == null) return "";
      const diff = cur - b;
      if (Math.abs(diff) < 0.005) return `<small>±0</small>`;
      return `<small>${diff > 0 ? "+" : ""}${fmt(diff, 1)}${suffix} vs ${esc(z0.label)}</small>`;
    };
    $("simMetrics").innerHTML =
      row("Impressions", fmt(m.imp), d(m.imp, base?.imp)) +
      row("Attention rate", fmt(m.rate, 1) + "%", d(m.rate, base?.rate, "pp")) +
      (rep.still ? "" : row("Attentive seconds", fmt(m.att, 1) + "s", d(m.att, base?.att, "s"))) +
      (rep.still ? "" : row("Avg dwell", fmt(m.avg, 2) + "s", d(m.avg, base?.avg, "s"))) +
      (rep.still || m.ttfl == null ? "" : row("Time to first look", fmt(m.ttfl, 1) + "s", ""));
  }
  const row = (k, v, delta) =>
    `<div class="sim-row"><span class="k">${k}</span><span class="v">${v} ${delta}</span></div>`;

  schedule();
}

const normToPx = (n, sim) => ({ x: n[0] * sim.w, y: n[1] * sim.h, w: n[2] * sim.w, h: n[3] * sim.h });

function simCompute(rect, sim, coneDeg) {
  const cos = Math.cos(((coneDeg ?? sim.cone_deg) * Math.PI) / 180);
  const k = sim.k || 1;
  const zc = { x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 };
  const dwell = {}, first = {}, look = {};
  let att = 0;
  for (const [t, pid, x, y, dx0, dy0, , sig, rk] of sim.rays) {
    let dx = dx0, dy = dy0, c2 = cos;
    let vx = zc.x - x, vy = zc.y - y;
    if (sig === 1) {  // body: zemin-azimut uzayı (ışının kendi k'sı) + dar koni
      const kk = rk || k;
      dy = dy0 / kk; vy = vy / kk;
      c2 = Math.cos(((coneDeg ?? sim.cone_deg) * 0.7 * Math.PI) / 180);
    }
    const na = Math.hypot(dx, dy) || 1;
    const n = Math.hypot(vx, vy) || 1;
    const hit = (dx * vx + dy * vy) / (na * n) > c2;
    if (hit) {
      dwell[pid] = (dwell[pid] || 0) + sim.dt;
      att += sim.dt;
      if (!look[pid] && first[pid] == null) first[pid] = t;
    }
    look[pid] = hit;
  }
  const lookers = Object.keys(dwell).filter((p) => dwell[p] >= sim.min_dwell);
  const imp = lookers.length;
  const traffic = Object.keys(sim.persons).length || 1;
  const ttfls = lookers.map((p) => (first[p] ?? 0) - (sim.persons[p] ?? 0)).filter((v) => v >= 0);
  return {
    imp,
    rate: (imp / traffic) * 100,
    att,
    avg: imp ? lookers.reduce((s, p) => s + dwell[p], 0) / imp : 0,
    ttfl: ttfls.length ? ttfls.reduce((a, b) => a + b, 0) / ttfls.length : null,
  };
}

const best = (rep) => rep.zones.reduce((a, b) => (a.aqs >= b.aqs ? a : b));

// AI'sız metrik açıklamaları — tooltip + sözlük tek kaynaktan
const GLOSS = {
  "Traffic": "Unique people tracked in the scene — including those facing away from the camera.",
  "Impressions": "People whose orientation stayed on the zone for at least 0.4s. Shown with a ±95% confidence range.",
  "Attention rate": "Impressions ÷ traffic. The share of passersby who actually looked.",
  "Attentive seconds": "Total seconds of measured attention on the zone, summed across all lookers.",
  "Avg dwell": "Average continuous attention per looker. Max dwell is the single longest look.",
  "Time to first look": "How quickly the zone captures people after they enter the scene. Lower = stronger pull.",
  "Glances / looker": "Average number of separate looks per looker. Above 1 means people look back again.",
  "Stopping power": "How much passersby slow down while looking (walking-speed drop, %). Physical proof of engagement.",
  "AQS": "Attention Quality Score (0–100): a composite of attention rate, dwell, deep engagement and stopping power.",
  "Reach CPM": "Cost per 1,000 people who looked (classic reach pricing).",
  "Attention CPM": "Cost per 1,000 attentive seconds — pricing by actual attention, Oculiq's core currency.",
  "Signal mix": "Where the measurement came from: head pose (confidence 0.85) vs body orientation (confidence 0.5).",
  "95% CI": "Wilson confidence interval — the honest statistical range of the rate given the sample size.",
};

function glossaryHtml() {
  return `<details class="gloss"><summary>What do these metrics mean? (no AI — plain definitions)</summary><dl>` +
    Object.entries(GLOSS).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") +
    `</dl></details>`;
}

function densitySvg(rep) {
  const pts = rep.density_timeline || [];
  if (pts.length < 2) return "";
  const maxV = Math.max(1, ...pts.map((p) => p.avg));
  const maxT = Math.max(1, ...pts.map((p) => p.t));
  const X = (t) => 8 + (t / maxT) * 664, Y = (v) => 74 - (v / maxV) * 56;
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p.t).toFixed(1)},${Y(p.avg).toFixed(1)}`).join(" ");
  const area = line + ` L${X(pts.at(-1).t).toFixed(1)},74 L${X(pts[0].t).toFixed(1)},74 Z`;
  return `<div class="wide-chart"><h4>Scene activity — avg people over time</h4>
    <svg viewBox="0 0 680 88">
      <path d="${area}" fill="#ffffff" opacity=".08"/>
      <path d="${line}" fill="none" stroke="#ffffff" stroke-width="1.6" opacity=".8"/>
      <text x="8" y="86" fill="rgba(255,255,255,.35)" font-size="9">0s</text>
      <text x="648" y="86" fill="rgba(255,255,255,.35)" font-size="9">${maxT}s</text>
      <text x="8" y="14" fill="rgba(255,255,255,.35)" font-size="9">peak ${maxV}</text>
    </svg></div>`;
}

function donutSvg(rep) {
  const zones = rep.zones.filter((z) => z.attentive_seconds > 0);
  const total = zones.reduce((s, z) => s + z.attentive_seconds, 0);
  if (!total || zones.length < 2) return "";
  const C = 2 * Math.PI * 34;
  let off = 0, segs = "";
  for (const z of zones) {
    const frac = z.attentive_seconds / total;
    segs += `<circle cx="45" cy="45" r="34" fill="none" stroke="${z.color}" stroke-width="13"
      stroke-dasharray="${(frac * C).toFixed(1)} ${C.toFixed(1)}" stroke-dashoffset="${(-off * C).toFixed(1)}"
      transform="rotate(-90 45 45)"/>`;
    off += frac;
  }
  const legend = zones.map((z) =>
    `<span><span class="zdot" style="background:${z.color}"></span>${esc(z.label)} — ${Math.round(z.attentive_seconds / total * 100)}%</span>`).join("");
  return `<div class="donut-wrap"><svg width="90" height="90" viewBox="0 0 90 90">${segs}</svg>
    <div class="donut-legend"><span style="color:var(--text)">Attention share</span>${legend}</div></div>`;
}

function sigMixHtml(z) {
  const s = z.signal_share || {};
  const head = s.head || 0, body = s.body || 0, rest = Math.max(0, 100 - head - body);
  if (!head && !body) return "";
  return `<div class="sigmix">
    <div class="lbl"><span title="${GLOSS["Signal mix"]}">Signal mix</span><span>head ${head}% · body ${body}%</span></div>
    <div class="bar-line">
      <div style="width:${head}%;background:#fff"></div>
      <div style="width:${body}%;background:rgba(255,255,255,.38)"></div>
      <div style="width:${rest}%;background:rgba(255,255,255,.12)"></div>
    </div></div>`;
}

function zoneReport(z, still) {
  const aqsC = 2 * Math.PI * 19;
  return `
  <div class="zone-report">
    <div class="zr-top">
      <span class="zdot" style="background:${z.color}"></span>
      <h3>${esc(z.label)}</h3><span class="ztype">${z.type}</span>
      <div class="aqs-chip">
        <svg viewBox="0 0 46 46"><circle cx="23" cy="23" r="19" fill="none" stroke="var(--surface3)" stroke-width="5"/>
        <circle cx="23" cy="23" r="19" fill="none" stroke="${z.color}" stroke-width="5" stroke-linecap="round"
          stroke-dasharray="${aqsC}" stroke-dashoffset="${aqsC}" data-aqs="${aqsC * (1 - z.aqs / 100)}" style="transition: stroke-dashoffset 1.2s ease"/></svg>
        <div class="lbl"><b>${z.aqs}</b><span>AQS score</span></div>
      </div>
    </div>

    <div class="funnel">
      ${fstage("Traffic", z.traffic, z.traffic, "")}
      ${fstage("Impressions", z.impressions, z.traffic, z.impressions_ci ? `±${Math.max(z.impressions - z.impressions_ci[0], z.impressions_ci[1] - z.impressions)}` : "")}
      ${still ? "" : fstage("Engaged ≥1s", z.engaged, z.traffic, "")}
      ${still ? "" : fstage("Deep ≥3s", z.deep, z.traffic, "")}
    </div>

    <div class="metric-grid">
      <div class="mcard star" title="${GLOSS["Attention rate"]} ${GLOSS["95% CI"]}"><div class="k">Attention rate · 95% CI ${z.attention_rate_ci ? z.attention_rate_ci[0] + "–" + z.attention_rate_ci[1] + "%" : ""}</div><div class="v">${z.attention_rate}<small>%</small></div></div>
      ${still ? "" : `<div class="mcard" title="${GLOSS["Attentive seconds"]}"><div class="k">Attentive seconds</div><div class="v">${fmt(z.attentive_seconds, 1)}<small>s</small></div></div>`}
      ${still ? "" : `<div class="mcard" title="${GLOSS["Avg dwell"]}"><div class="k">Avg dwell</div><div class="v">${fmt(z.avg_dwell, 2)}<small>s</small></div></div>`}
      ${still ? "" : `<div class="mcard" title="${GLOSS["Avg dwell"]}"><div class="k">Max dwell</div><div class="v">${fmt(z.max_dwell, 1)}<small>s</small></div></div>`}
      ${still || z.time_to_first_look == null ? "" : `<div class="mcard" title="${GLOSS["Time to first look"]}"><div class="k">Time to first look <span class="new-tag">NEW</span></div><div class="v">${fmt(z.time_to_first_look, 1)}<small>s</small></div></div>`}
      ${still ? "" : `<div class="mcard" title="${GLOSS["Glances / looker"]}"><div class="k">Glances / looker <span class="new-tag">NEW</span></div><div class="v">${fmt(z.glances_per_looker, 1)}</div></div>`}
      ${still ? "" : `<div class="mcard" title="${GLOSS["Stopping power"]}"><div class="k">Stopping power <span class="new-tag">NEW</span></div><div class="v">${fmt(z.stopping_power, 0)}<small>% slowdown</small></div></div>`}
    </div>
    ${sigMixHtml(z)}

    ${still ? "" : `<div class="charts">
      <div class="chart-box"><h4>Attention over time</h4>${timelineSvg(z)}</div>
      <div class="chart-box"><h4>Dwell distribution</h4>${histSvg(z)}</div>
    </div>`}

    <div class="cpm-row">
      <div class="cpm-box"><div class="k">Reach CPM (per 1k lookers)</div><div class="v">${z.reach_cpm != null ? "$" + fmt(z.reach_cpm, 2) : "— add cost"}</div></div>
      <div class="cpm-box hero"><div class="k">Attention CPM (per 1k attentive sec)</div><div class="v">${z.attention_cpm != null ? "$" + fmt(z.attention_cpm, 2) : "— add cost"}</div></div>
    </div>

    ${!still && z.evidence && z.evidence.length ? `
    <div class="evidence">
      <h4>Evidence <span class="new-tag">AUDITABLE</span></h4>
      <div class="ev-chips">${z.evidence.map((e) =>
        `<button class="ev-chip" data-t="${e.start}">#${e.pid} · ${e.dur}s @ ${tstamp(e.start)}</button>`).join("")}</div>
    </div>` : ""}
  </div>`;
}

const tstamp = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const fstage = (label, v, base, extra) =>
  `<div class="fstage"><span class="fl">${label}</span><div class="fb"><div data-w="${base ? Math.round((v / base) * 100) : 0}"></div></div><span class="fv">${fmt(v)} <small>${extra}</small></span></div>`;

function timelineSvg(z) {
  const pts = z.timeline || [];
  if (!pts.length) return '<svg viewBox="0 0 300 80"><text x="10" y="45" fill="var(--muted)" font-size="11">no data</text></svg>';
  const maxV = Math.max(0.1, ...pts.map((p) => p.sec));
  const maxT = Math.max(1, ...pts.map((p) => p.t));
  const X = (t) => 6 + (t / maxT) * 288, Y = (v) => 70 - (v / maxV) * 58;
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p.t).toFixed(1)},${Y(p.sec).toFixed(1)}`).join(" ");
  const area = line + ` L${X(pts.at(-1).t).toFixed(1)},70 L${X(pts[0].t).toFixed(1)},70 Z`;
  return `<svg viewBox="0 0 300 84">
    <path d="${area}" fill="${z.color}" opacity=".16"/>
    <path d="${line}" fill="none" stroke="${z.color}" stroke-width="2" stroke-linejoin="round"/>
    <text x="6" y="82" fill="var(--muted)" font-size="9">0s</text>
    <text x="272" y="82" fill="var(--muted)" font-size="9">${maxT}s</text></svg>`;
}

function histSvg(z) {
  const h = z.dwell_histogram || [0, 0, 0, 0, 0];
  const labels = ["<1s", "1–2", "2–3", "3–5", "5s+"];
  const maxV = Math.max(1, ...h);
  return `<svg viewBox="0 0 300 84">` + h.map((v, i) => {
    const bh = (v / maxV) * 56;
    return `<rect x="${14 + i * 58}" y="${66 - bh}" width="40" height="${bh}" rx="4" fill="${z.color}" opacity="${0.45 + 0.55 * (v / maxV)}"/>
      <text x="${34 + i * 58}" y="80" fill="var(--muted)" font-size="9" text-anchor="middle">${labels[i]}</text>
      <text x="${34 + i * 58}" y="${60 - bh}" fill="var(--text)" font-size="9" text-anchor="middle">${v || ""}</text>`;
  }).join("") + `</svg>`;
}

function animateReport() {
  document.querySelectorAll("[data-count]").forEach((el) => {
    const target = parseFloat(el.dataset.count);
    const t0 = performance.now();
    const tick = (t) => {
      const k = Math.min((t - t0) / 900, 1);
      el.firstChild.textContent = fmt(Math.round(target * (1 - Math.pow(1 - k, 3))));
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  document.querySelectorAll(".fb > div, .cmp-bar .track > div").forEach((el) => { el.style.width = el.dataset.w + "%"; });
  document.querySelectorAll("[data-aqs]").forEach((el) => { el.style.strokeDashoffset = el.dataset.aqs; });
}

function download(name, text, mime) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = name;
  a.click();
}

function toCsv(rep) {
  const cols = ["label", "type", "traffic", "impressions", "attention_rate", "attentive_seconds",
    "avg_dwell", "max_dwell", "engaged", "deep", "time_to_first_look", "glances_per_looker",
    "stopping_power", "aqs", "cost", "reach_cpm", "attention_cpm"];
  return [cols.join(","), ...rep.zones.map((z) => cols.map((c) => z[c] ?? "").join(","))].join("\n");
}
