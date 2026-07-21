// Oculiq web app — upload → zones → analyze (SSE live) → advanced report.
// All inference happens in the local Python backend; this file is pure UI.

const $ = (id) => document.getElementById(id);
const video = $("video"), photo = $("photo"), draw = $("draw");
const dctx = draw.getContext("2d");

const state = {
  file: null, isVideo: true, zones: [], nextId: 0,
  jobId: null, es: null,
};
const COLORS = ["#ff6b5e", "#4f8a69", "#517a95", "#8c6ca0", "#c08a45"];
let renderedLogCount = 0;

/* ---------- step navigation ---------- */
const STEPS = ["upload", "zones", "analyze", "report", "history", "live"];
const FLOW = ["upload", "zones", "analyze", "report"];   // sadece akış adımları "done" işaretlenir
function goto(step) {
  STEPS.forEach((s) => {
    $("step-" + s).classList.toggle("on", s === step);
    const btn = document.querySelector(`.nav-item[data-step="${s}"]`);
    btn.classList.toggle("on", s === step);
    if (FLOW.includes(step) && FLOW.includes(s)
        && FLOW.indexOf(s) < FLOW.indexOf(step)) btn.classList.add("done");
  });
  document.querySelector(`.nav-item[data-step="${step}"]`).disabled = false;
}
document.querySelectorAll(".nav-item").forEach((b) =>
  b.addEventListener("click", () => {
    if (b.disabled) return;
    goto(b.dataset.step);
    if (b.dataset.step === "history") showHistory();
    if (b.dataset.step === "live") showLive();
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

/* ---------- zone drawing: reklamın 4 KÖŞESİNE tıkla (serbest dörtgen) ----------
   Açılı çekilen billboard ekranda yamuktur; eksen-hizalı dikdörtgen fazla/eksik
   alan kapsar. 4 köşe = gerçek yüzey; 3D yerleşim de gerçek köşeleri kullanır. */
let placing = false, placingLine = false, corners = [], hoverPt = null, pending = null;

$("zoneBtn").onclick = () => {
  placing = true; placingLine = false; corners = []; hoverPt = null;
  $("drawHint").textContent = "click the 4 corners of the ad surface";
  draw.style.pointerEvents = "auto"; draw.style.cursor = "crosshair";
  renderZones();
};
$("lineBtn").onclick = () => {
  placing = true; placingLine = true; corners = []; hoverPt = null;
  $("drawHint").textContent = "click 2 points across the entrance — the arrow shows IN (draw in reverse to flip)";
  draw.style.pointerEvents = "auto"; draw.style.cursor = "crosshair";
  renderZones();
};
const normPt = (e) => {
  const r = draw.getBoundingClientRect();
  return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
};
draw.addEventListener("pointermove", (e) => {
  if (placing && corners.length) { hoverPt = normPt(e); renderZones(); }
});
draw.addEventListener("pointerdown", (e) => {
  if (!placing) return;
  corners.push(normPt(e));
  renderZones();
  if (placingLine && corners.length === 2) {
    placing = false; placingLine = false; hoverPt = null;
    draw.style.pointerEvents = "none"; draw.style.cursor = "default";
    const [p1, p2] = corners;
    if (Math.hypot(p2.x - p1.x, p2.y - p1.y) < 0.02) { corners = []; renderZones(); return; }
    // kenar uyarısı: çizgi kadraj kenarına yapışıksa geçiş onaylanamaz (Spec §5 histerezis)
    const E = 0.08, hint = $("drawHint");
    const hug = (f) => f(p1) && f(p2);
    if (hug((p) => p.y > 1 - E) || hug((p) => p.y < E) || hug((p) => p.x < E) || hug((p) => p.x > 1 - E)) {
      hint.textContent = "⚠ line hugs the frame edge — crossings can't be confirmed there; draw it further inside, across the walking path";
      hint.style.color = "#e0a43c";
    } else {
      hint.textContent = "click the 4 corners of the ad surface";
      hint.style.color = "";
    }
    const x = Math.min(p1.x, p2.x), y = Math.min(p1.y, p2.y);
    pending = { line: [[p1.x, p1.y], [p2.x, p2.y]], x, y,
                w: Math.abs(p2.x - p1.x) || 0.01, h: Math.abs(p2.y - p1.y) || 0.01 };
    corners = [];
    openPop(x, y, pending.w, pending.h);
    $("zpLabel").value = `Entrance ${state.zones.filter((z) => z.type === "line").length + 1}`;
    $("zpType").value = "billboard"; $("zpType").disabled = true;   // tip: line (kayıtta zorlanır)
    $("zpCost").style.display = "none";
    return;
  }
  if (!placingLine && corners.length === 4) {
    placing = false; hoverPt = null;
    draw.style.pointerEvents = "none"; draw.style.cursor = "default";
    // açıya göre sırala (basit/kesişmesiz çokgen garantisi)
    const cx = corners.reduce((s, p) => s + p.x, 0) / 4;
    const cy = corners.reduce((s, p) => s + p.y, 0) / 4;
    const poly = [...corners].sort((a, b) =>
      Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
    const xs = poly.map((p) => p.x), ys = poly.map((p) => p.y);
    const x = Math.min(...xs), y = Math.min(...ys);
    const w = Math.max(...xs) - x, h = Math.max(...ys) - y;
    if (w < 0.01 || h < 0.01) { corners = []; renderZones(); return; }
    pending = { poly: poly.map((p) => [p.x, p.y]), x, y, w, h };
    corners = [];
    openPop(x, y, w, h);
  }
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
  const isLine = !!pending.line;
  state.zones.push({
    id: state.nextId++, ...pending,
    label: $("zpLabel").value.trim() || `Zone ${state.zones.length + 1}`,
    type: isLine ? "line" : $("zpType").value,
    cost: isLine ? 0 : parseFloat($("zpCost").value) || 0,
    color: COLORS[state.zones.length % COLORS.length],
  });
  pending = null;
  $("zonePop").classList.add("hidden");
  $("zpType").disabled = false; $("zpCost").style.display = "";
  renderZones(); renderZoneList();
  $("toAnalyze").disabled = false;
};
$("zpCancel").onclick = () => {
  pending = null; $("zonePop").classList.add("hidden");
  $("zpType").disabled = false; $("zpCost").style.display = "";
  renderZones();
};
$("zpLabel").addEventListener("keydown", (e) => { if (e.key === "Enter") $("zpSave").click(); });

function polyPath(ctx2, poly, W, H) {
  ctx2.beginPath();
  poly.forEach(([px, py], i) => i ? ctx2.lineTo(px * W, py * H) : ctx2.moveTo(px * W, py * H));
  ctx2.closePath();
}

function renderZones() {
  const W = draw.width, H = draw.height;
  dctx.clearRect(0, 0, W, H);
  for (const z of state.zones) {
    dctx.strokeStyle = z.color; dctx.lineWidth = Math.max(2, W / 480);
    dctx.font = `600 ${Math.max(13, W / 70)}px sans-serif`;
    if (z.line) {
      // giriş çizgisi + IN oku (p1→p2'nin sol normali)
      const [[x1, y1], [x2, y2]] = z.line;
      dctx.lineWidth = Math.max(3, W / 320);
      dctx.beginPath(); dctx.moveTo(x1 * W, y1 * H); dctx.lineTo(x2 * W, y2 * H); dctx.stroke();
      const mx = (x1 + x2) / 2 * W, my = (y1 + y2) / 2 * H;
      const dx = x2 * W - x1 * W, dy = y2 * H - y1 * H;
      const ll = Math.hypot(dx, dy) || 1;
      const nx = -dy / ll, ny = dx / ll, al = Math.max(26, ll * 0.18);
      dctx.beginPath(); dctx.moveTo(mx, my); dctx.lineTo(mx + nx * al, my + ny * al); dctx.stroke();
      const a = Math.atan2(ny, nx);
      dctx.beginPath();
      dctx.moveTo(mx + nx * al, my + ny * al);
      dctx.lineTo(mx + nx * al - 9 * Math.cos(a - 0.45), my + ny * al - 9 * Math.sin(a - 0.45));
      dctx.lineTo(mx + nx * al - 9 * Math.cos(a + 0.45), my + ny * al - 9 * Math.sin(a + 0.45));
      dctx.closePath(); dctx.fillStyle = z.color; dctx.fill();
      dctx.fillText(`${z.label} (in →)`, mx + 8, my - 8);
      continue;
    }
    if (z.type === "staff") dctx.setLineDash([8, 6]);
    if (z.poly) { polyPath(dctx, z.poly, W, H); dctx.stroke(); }
    else dctx.strokeRect(z.x * W, z.y * H, z.w * W, z.h * H);
    dctx.setLineDash([]);
    dctx.fillStyle = z.color;
    dctx.fillText(z.type === "staff" ? `${z.label} (excluded)` : z.label, z.x * W + 6, z.y * H - 7);
  }
  if (placing && corners.length) {
    dctx.setLineDash([7, 5]); dctx.strokeStyle = "#fff"; dctx.lineWidth = 2;
    dctx.beginPath();
    corners.forEach((p, i) => i ? dctx.lineTo(p.x * W, p.y * H) : dctx.moveTo(p.x * W, p.y * H));
    if (hoverPt) dctx.lineTo(hoverPt.x * W, hoverPt.y * H);
    dctx.stroke();
    dctx.setLineDash([]);
    dctx.fillStyle = "#fff";
    for (const p of corners) { dctx.beginPath(); dctx.arc(p.x * W, p.y * H, Math.max(4, W / 300), 0, 6.283); dctx.fill(); }
  }
  if (pending && pending.poly) {
    dctx.setLineDash([7, 5]); dctx.strokeStyle = "#fff"; dctx.lineWidth = 2;
    polyPath(dctx, pending.poly, W, H); dctx.stroke();
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
  resetPipelineUI();

  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("zones", JSON.stringify(state.zones.map(({ id, label, type, x, y, w, h, poly, line }) => ({ id, label, type, x, y, w, h, poly, line }))));
  fd.append("costs", JSON.stringify(Object.fromEntries(state.zones.map((z) => [z.id, z.cost]))));
  fd.append("crowd_mode", $("crowdMode").checked ? "on" : "auto");
  fd.append("demographics", $("demoMode").checked ? "on" : "off");
  fd.append("face_blur", $("blurMode").checked ? "on" : "off");

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
    if (m.log) renderProcLog(m.log);
    if (m.status === "done") { es.close(); loadReport(job_id); }
  };
  es.onerror = () => { /* transient; EventSource retries */ };
}

function setProgress(p) {
  $("pct").textContent = p;
  $("progressFill").style.width = `${Math.max(0, Math.min(100, p))}%`;
  if (p >= 92) setPipelineStage(3, "Building audited report");
  else if (p > 0) setPipelineStage(2, "Measuring attention signals");
}

function renderProcLog(log) {
  const el = $("procLog");
  if (!el || !log.length) return;
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
  if (log.length < renderedLogCount) {
    renderedLogCount = 0;
    el.innerHTML = "";
  }
  if (!renderedLogCount) el.innerHTML = "";
  el.querySelectorAll(".is-latest").forEach((line) => line.classList.remove("is-latest"));

  const fresh = log.slice(renderedLogCount);
  const frag = document.createDocumentFragment();
  fresh.forEach((entry, i) => {
    const line = document.createElement("div");
    const level = ["ok", "warn", "err"].includes(entry.lv) ? entry.lv : "info";
    line.className = `pl-line pl-${level}`;
    line.style.animationDelay = `${Math.min(i * 45, 180)}ms`;
    line.innerHTML = `<span class="pl-t">${esc(String(entry.t || "--:--:--"))}</span><span class="pl-msg">${esc(entry.m)}</span>`;
    frag.appendChild(line);
  });
  el.appendChild(frag);
  const latest = el.lastElementChild;
  if (latest) latest.classList.add("is-latest");
  renderedLogCount = log.length;
  $("logCount").textContent = renderedLogCount;

  const current = log[log.length - 1];
  const stage = inferPipelineStage(current.m);
  setPipelineStage(stage.index, stage.label);
  updateLogNarrative(current, stage);
  if (atBottom) el.scrollTop = el.scrollHeight;
}

function resetPipelineUI() {
  renderedLogCount = 0;
  $("pct").textContent = "0";
  $("progressFill").style.width = "0%";
  $("logCount").textContent = "0";
  $("procLog").innerHTML = '<div class="log-placeholder"><span></span><p>Waiting for the first local event…</p></div>';
  setPipelineStage(0, "Preparing local engine");
  const narrative = $("logNarrative");
  narrative.querySelector(".narrative-icon").textContent = "01";
  narrative.querySelector("h3").textContent = "Preparing the pipeline";
  narrative.querySelector("p").textContent = "The engine will explain each step here as your footage is processed.";
}

function inferPipelineStage(message = "") {
  const m = message.toLowerCase();
  if (/report|rendered|done in|saved/.test(m)) return { index: 3, label: "Building audited report" };
  if (/\d+%|tracked|attention|reconstruct|3d scene|staff exclusion|reach/.test(m)) return { index: 2, label: "Measuring attention signals" };
  if (/model|yolo|bytetrack|scan mode|video |classif|crowd engine|entrance counting|face blur/.test(m)) return { index: 1, label: "Detecting people and pose" };
  return { index: 0, label: "Securing local input" };
}

function setPipelineStage(activeIndex, label) {
  ["stageIngest", "stageDetect", "stageMeasure", "stageReport"].forEach((id, index) => {
    const stage = $(id);
    stage.classList.toggle("active", index === activeIndex);
    stage.classList.toggle("done", index < activeIndex);
  });
  if (label) $("activeStageLabel").textContent = label;
}

function updateLogNarrative(entry, stage) {
  const m = (entry.m || "").toLowerCase();
  let title = ["Input secured", "People and pose detection", "Attention measurement", "Report assembly"][stage.index];
  let detail = [
    "The media is registered inside this local job. Nothing is uploaded or shared.",
    "The engine locates people, assigns stable IDs and reads pose landmarks frame by frame.",
    "Orientation signals are tested against your marked surfaces and converted into auditable attention events.",
    "Measured events are being consolidated into metrics, evidence clips and the final report."
  ][stage.index];

  if (m.includes("loading detection model")) {
    title = "Loading the vision model";
    detail = "YOLO11-pose and ByteTrack are starting locally to detect people and keep identities stable across frames.";
  } else if (m.includes("detection model ready")) {
    title = "Vision model ready";
    detail = "Detection is active. The next frames can now be tracked without sending footage off this device.";
  } else if (m.startsWith("video ")) {
    title = "Reading the source accurately";
    detail = "Resolution, duration and real frame timestamps are mapped so sampling remains reliable—even with variable frame rates.";
  } else if (m.includes("scan mode")) {
    title = "Choosing the scan strategy";
    detail = m.includes("tiled")
      ? "The frame is split into overlapping tiles so smaller or distant people remain detectable in crowded footage."
      : "A single-pass scan is sufficient for this scene, keeping local processing efficient.";
  } else if (m.includes("face blurring")) {
    title = "Privacy layer is active";
    detail = "Faces are blurred in generated evidence while anonymous pose and orientation signals remain measurable.";
  } else if (/\d+%/.test(m)) {
    title = "Following attention over time";
    detail = "Stable track IDs are accumulating dwell, direction and zone intersections. The live counters update as evidence grows.";
  } else if (m.includes("reconstructing 3d")) {
    title = "Reconstructing scene depth";
    detail = "Perspective cues are being used to understand real spatial placement and improve surface-direction measurements.";
  } else if (m.includes("annotated video rendered")) {
    title = "Evidence replay is ready";
    detail = "The annotated output has been rendered. Oculiq is now turning the event stream into report metrics.";
  } else if (m.includes("done in")) {
    title = "Analysis complete";
    detail = "The local report and its replayable evidence have been saved to this workspace.";
  } else if (entry.lv === "warn") {
    title = "Fallback applied safely";
    detail = "A preferred signal was unavailable, so the engine continued with a supported local fallback and recorded the limitation.";
  } else if (entry.lv === "err") {
    title = "Pipeline needs attention";
    detail = "Processing stopped at this event. The technical message on the left identifies the local cause.";
  }

  const narrative = $("logNarrative");
  narrative.querySelector(".narrative-icon").textContent = String(stage.index + 1).padStart(2, "0");
  narrative.querySelector("h3").textContent = title;
  narrative.querySelector("p").textContent = detail;
  narrative.classList.remove("updated");
  requestAnimationFrame(() => narrative.classList.add("updated"));
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
      <div class="sub">${esc(rep.method)} · ${esc(rep.model)} · ${esc(rep.scan_mode || "single-pass")}${rep.calibration && rep.calibration.auto ? " · auto-calibrated" : ""}${rep.scene3d && rep.scene3d.enabled ? " · 3D scene (" + fmt(rep.scene3d.calib_confidence, 0) + "% conf)" : ""}${rep.scene3d && rep.scene3d.gaze3d_pct != null ? " · 3D gaze " + fmt(rep.scene3d.gaze3d_pct, 0) + "%" : ""}${rep.scene3d && rep.scene3d.enabled && !rep.scene3d.reliable ? " · 3D→2.5D fallback" : ""} · ${rep.still ? "snapshot" : fmt(rep.duration, 1) + "s footage"} · processed in ${fmt(rep.processing_seconds, 1)}s</div>
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
    ${rep.scene3d && rep.scene3d.enabled ? `<div class="kpi" title="${GLOSS["3D calibration"]}"><div class="k">3D calibration <span class="new-tag">SCENE</span></div><div class="v">${fmt(rep.scene3d.calib_confidence, 0)}<small>% conf${rep.scene3d.camera_height_m ? " · cam " + fmt(rep.scene3d.camera_height_m, 1) + "m" : ""}</small></div></div>` : ""}
    ${rep.capture_rate != null ? `<div class="kpi"><div class="k">Capture rate <span class="new-tag">NEW</span></div><div class="v">${fmt(rep.capture_rate, 1)}<small>%</small></div></div>` : ""}
    ${rep.staff_excluded != null ? `<div class="kpi" title="Persons spending ≥30% of visible time (or ≥60s) inside a staff area are excluded from all metrics (Spec v1.0 §8)."><div class="k">Staff excluded</div><div class="v" data-count="${rep.staff_excluded}">0</div></div>` : ""}
    ${rep.zones.length ? `<div class="kpi"><div class="k">Best zone (AQS)</div><div class="v">${esc(best(rep).label)} <small>${best(rep).aqs}</small></div></div>` : ""}
    <div class="kpi"><div class="k">Zones analyzed</div><div class="v" data-count="${rep.zones.length}">0</div></div>
  </div>`;

  if (rep.lines && rep.lines.length) {
    html += `<div class="wide-chart"><h4>Entrance counting <span class="new-tag">NEW</span> — line crossings by tracked foot position (Spec v1.0 §5)</h4>
      <div class="cpm-row">` + rep.lines.map((l) => `
        <div class="cpm-box"><div class="k">${esc(l.label)}</div>
          <div class="v">${l.enters} in · ${l.exits} out</div>
          <div class="k">capture rate ${fmt(l.capture_rate, 1)}%${l.capture_rate_ci ? " · 95% CI " + l.capture_rate_ci[0] + "–" + l.capture_rate_ci[1] + "%" : ""}</div>
        </div>`).join("") + `</div></div>`;
  }
  if (rep.lines_note) {
    html += `<div class="wide-chart"><h4>Entrance counting</h4><p class="aud-note">${esc(rep.lines_note)}</p></div>`;
  }

  if (rep.measurement_health) {
    const h = rep.measurement_health, sm = h.signal_mix || {};
    const chip = (k, v, warn) =>
      `<span class="lv-kpi" ${warn ? 'style="color:var(--danger)"' : ""}><b>${v}</b> <small>${k}</small></span>`;
    html += `<div class="wide-chart"><h4>Measurement health <span class="new-tag">TRANSPARENCY</span> — a weak scene is disclosed, never hidden (Spec §10)</h4>
      <div class="cam-live" style="margin-top:4px">
        ${chip("direction signal", fmt(h.direction_share, 0) + "%", h.direction_share < 60)}
        ${chip("head / body / away", `${fmt(sm.head, 0)} / ${fmt(sm.body, 0)} / ${fmt(sm.away, 0)}%`, (sm.away || 0) > 30)}
        ${chip("detector confidence", fmt(h.avg_det_conf, 2), h.avg_det_conf < 0.4)}
        ${chip("tracks", `${h.tracks_seen} seen · ${h.tracks_stitched} stitched · ${h.ghosts_dropped} ghosts`, false)}
        ${chip("3D gaze path", fmt(h.gaze3d_pct, 0) + "%", false)}
      </div></div>`;
  }
  if (rep.reach_note) {
    html += `<div class="wide-chart"><h4>Shelf interaction</h4><p class="aud-note">${esc(rep.reach_note)}</p></div>`;
  }

  html += densitySvg(rep);
  if (rep.scene3d && rep.scene3d.enabled) {
    html += `<div class="wide-chart"><h4>3D scene reconstruction <span class="new-tag">DEPTH</span> — ${rep.scene3d.reliable ? "active" : "low confidence · engine fell back to 2.5D"}</h4>
      <img class="scene-view" src="/api/jobs/${jobId}/scene" onerror="this.parentElement.style.display='none'" alt="depth reconstruction" /></div>`;
  }
  html += audienceHtml(rep);
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
      <div class="sub">Drag the white box — or draw a new one anywhere. Impressions recompute live from ${fmt(rep.sim.rays.length)} recorded gaze rays${rep.sim.s3 ? " in true 3D (ray ↔ surface geometry)" : ""}; watch gaze flow into the placement. Cone is automatic (per-person + zone size). No re-processing.</div>
      <div class="sim-grid">
        <div class="sim-wrap">
          <canvas id="simCanvas"></canvas>
          <div class="sim-flow" id="simFlow">— rays flowing in</div>
        </div>
        <div>
          <div class="sim-metrics" id="simMetrics"></div>
          <div class="cone-cal">
            <label>Cone sensitivity <b id="coneVal">auto</b></label>
            <input type="range" id="coneSlider" min="-10" max="10" step="1" value="0" />
            <small>Cone is computed automatically. Nudge ± only to override the auto-calibration for this camera.</small>
          </div>
          <button id="simReset" class="sm" style="margin-top:10px">Reset to ${rep.zones[0] ? esc(rep.zones[0].label) : "original"}</button>
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

/* ---------- what-if simulator (animated, auto-cone) ---------- */
const angBetween = (ax, ay, bx, by) => {
  const na = Math.hypot(ax, ay) || 1e-9, nb = Math.hypot(bx, by) || 1e-9;
  return Math.acos(Math.max(-1, Math.min(1, (ax * bx + ay * by) / (na * nb))));
};

// 3x3 Gauss çözücü (düzlem LSQ normal denklemleri için)
function solve3(M, R) {
  const A = M.map((r, i) => [...r, R[i]]);
  for (let i = 0; i < 3; i++) {
    let p = i;
    for (let r = i + 1; r < 3; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
    if (Math.abs(A[p][i]) < 1e-9) return null;
    [A[i], A[p]] = [A[p], A[i]];
    for (let r = 0; r < 3; r++) {
      if (r === i) continue;
      const f = A[r][i] / A[i][i];
      for (let c = i; c < 4; c++) A[r][c] -= f * A[i][c];
    }
  }
  return [A[0][3] / A[0][0], A[1][3] / A[1][1], A[2][3] / A[2][2]];
}
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len3 = (a) => Math.hypot(a[0], a[1], a[2]);

// Dikdörtgenin 3D karşılığı — motorla aynı: derinlik ızgarasına DÜZLEM oturt
// (Z = aX + bY + c), köşeleri o düzleme ışınla yerleştir -> yönlü dörtgen + normal.
function rectQuad3d(rect, sim) {
  const s3 = sim.s3;
  if (!s3) return null;
  const { gw, gh, z } = s3.grid;
  const xs = Math.max(0, Math.floor((rect.x / sim.w) * gw));
  const xe = Math.min(gw - 1, Math.ceil(((rect.x + rect.w) / sim.w) * gw));
  const ys = Math.max(0, Math.floor((rect.y / sim.h) * gh));
  const ye = Math.min(gh - 1, Math.ceil(((rect.y + rect.h) / sim.h) * gh));
  const pts = [];
  for (let gy = ys; gy <= ye; gy++)
    for (let gx = xs; gx <= xe; gx++) {
      const Z = z[gy] && z[gy][gx];
      if (!(Z > 0.5 && Z < 300)) continue;
      const u = ((gx + 0.5) / gw) * sim.w, v = ((gy + 0.5) / gh) * sim.h;
      pts.push([((u - s3.cx) * Z) / s3.f, ((v - s3.cy) * Z) / s3.f, Z]);
    }
  if (!pts.length) return null;
  const zsort = pts.map((p) => p[2]).sort((a, b) => a - b);
  const Zmed = zsort[zsort.length >> 1];
  const inl = pts.filter((p) => Math.abs(p[2] - Zmed) < Math.max(1.5, Zmed * 0.2));

  let plane = null;
  if (inl.length >= 6) {
    let Sxx = 0, Sxy = 0, Sx = 0, Syy = 0, Sy = 0, S1 = 0, Sxz = 0, Syz = 0, Sz = 0;
    for (const [X, Y, Z2] of inl) {
      Sxx += X * X; Sxy += X * Y; Sx += X; Syy += Y * Y; Sy += Y; S1++;
      Sxz += X * Z2; Syz += Y * Z2; Sz += Z2;
    }
    plane = solve3([[Sxx, Sxy, Sx], [Sxy, Syy, Sy], [Sx, Sy, S1]], [Sxz, Syz, Sz]);
  }
  const corner = (u, v) => {
    if (plane) {
      const [a, b, c] = plane;
      const den = 1 - (a * (u - s3.cx)) / s3.f - (b * (v - s3.cy)) / s3.f;
      if (den > 1e-4) {
        const Z = c / den;
        if (Z > 0.5 && Z < 300) return [((u - s3.cx) * Z) / s3.f, ((v - s3.cy) * Z) / s3.f, Z];
      }
    }
    return [((u - s3.cx) * Zmed) / s3.f, ((v - s3.cy) * Zmed) / s3.f, Zmed];
  };
  const c0 = corner(rect.x, rect.y), c1 = corner(rect.x + rect.w, rect.y);
  const c2 = corner(rect.x + rect.w, rect.y + rect.h), c3c = corner(rect.x, rect.y + rect.h);
  const cen = [(c0[0] + c2[0]) / 2, (c0[1] + c2[1]) / 2, (c0[2] + c2[2]) / 2];
  const eu = sub3(c1, c0), ev = sub3(c3c, c0);
  let n = null;
  if (plane) {
    n = [plane[0], plane[1], -1];
    const L = len3(n);
    n = n.map((v) => v / L);
    if (dot3(n, cen) > 0) n = n.map((v) => -v);   // normal kameraya baksın
  }
  return { c3: cen, wm: len3(eu), hm: len3(ev), corners: [c0, c1, c2, c3c], eu, ev, n };
}

// Bir ışın verilen dikdörtgene bakıyor mu? Python ile birebir: 3D varsa gerçek
// ışın-yüzey açısı (motor looks_at_3d), yoksa 2.5D azimut koni testi.
function rayHit(r, rect, sim, bias, quad3) {
  if (quad3 && r.length >= 16) {
    const sig = r[7], rcone = r[9] || 14;
    let ax = r[13], ay = r[14], az = r[15];
    let bx = quad3.c3[0] - r[10], by = quad3.c3[1] - r[11], bz = quad3.c3[2] - r[12];
    const dist = Math.hypot(bx, by, bz) || 1e-6;
    let half;
    if (sig === 1) {           // body: gerçek azimut — dikey bileşen düşülür
      const u = sim.s3.up;
      const da = ax * u[0] + ay * u[1] + az * u[2];
      ax -= da * u[0]; ay -= da * u[1]; az -= da * u[2];
      const db = bx * u[0] + by * u[1] + bz * u[2];
      bx -= db * u[0]; by -= db * u[1]; bz -= db * u[2];
      half = (Math.atan(quad3.wm / 2 / dist) * 180) / Math.PI;
    } else {                    // head: tam 3D açı (dikey dahil)
      half = (Math.atan(Math.hypot(quad3.wm, quad3.hm) / 2 / dist) * 180) / Math.PI;
    }
    const na = Math.hypot(ax, ay, az) || 1e-9;
    const nb = Math.hypot(bx, by, bz) || 1e-9;
    const ang = (Math.acos(Math.max(-1, Math.min(1,
      (ax * bx + ay * by + az * bz) / (na * nb)))) * 180) / Math.PI;
    return ang <= rcone + Math.min(half, 25) + bias;
  }
  return rayHit2d(r, rect, sim, bias);
}

function rayHit2d(r, rect, sim, bias) {
  const [t, pid, x, y, dx0, dy0, v, sig, rk, rcone] = r;
  const k = sig === 1 ? (rk || sim.k || 1) : 1;
  const zc = { x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 };
  const ddx = dx0, ddy = dy0 / k;
  const vx = zc.x - x, vy = (zc.y - y) / k;
  let cone;
  if (sim.auto_cone) {
    const half = angBetween(rect.x - x, vy, rect.x + rect.w - x, vy) * 180 / Math.PI / 2;
    cone = (rcone || 14) + Math.min(half, 20) + bias;
  } else {
    cone = sim.cone_deg * (sig === 1 ? 0.7 : 1) + bias;
  }
  const ang = angBetween(ddx, ddy, vx, vy) * 180 / Math.PI;
  return ang <= Math.max(2, cone);
}

function simCompute(rect, sim, bias) {
  const dwell = {}, first = {}, look = {};
  const quad3 = rectQuad3d(rect, sim);   // 3D bölge — kare başına BİR kez
  let att = 0;
  for (const r of sim.rays) {
    const pid = r[1], t = r[0];
    const hit = rayHit(r, rect, sim, bias, quad3);
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
    imp, rate: (imp / traffic) * 100, att,
    avg: imp ? lookers.reduce((s, p) => s + dwell[p], 0) / imp : 0,
    ttfl: ttfls.length ? ttfls.reduce((a, b) => a + b, 0) / ttfls.length : null,
  };
}

const normToPx = (n, sim) => ({ x: n[0] * sim.w, y: n[1] * sim.h, w: n[2] * sim.w, h: n[3] * sim.h });

function initSim(rep, jobId) {
  const sim = rep.sim;
  const canvas = $("simCanvas"), ctx = canvas.getContext("2d");
  canvas.width = sim.w; canvas.height = sim.h;
  const bg = new Image(); bg.src = `/api/jobs/${jobId}/frame`;
  const lw = Math.max(2, sim.w / 480);
  const fs = Math.max(13, sim.w / 78);

  const z0 = rep.zones[0];
  let bias = 0;
  const base = z0 ? simCompute(normToPx(z0.norm, sim), sim, 0) : null;
  const origVz = z0 ? normToPx(z0.norm, sim)
    : { x: sim.w * 0.35, y: sim.h * 0.3, w: sim.w * 0.3, h: sim.h * 0.25 };
  let vz = { ...origVz };
  let mode = null, grab = null, anchor = null;

  // çizim için ışın örneklemi (perf); her ışının anlık isabeti + akış parçacığı
  const step = sim.rays.length > 480 ? Math.ceil(sim.rays.length / 480) : 1;
  const drawRays = sim.rays.filter((_, i) => i % step === 0);
  let hits = new Array(drawRays.length).fill(false);
  let particles = [];
  const shown = { imp: 0, rate: 0, att: 0, avg: 0, ttfl: 0 };  // sayaç animasyonu için

  const pt = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) / r.width * sim.w, y: (e.clientY - r.top) / r.height * sim.h };
  };
  const inside = (p) => p.x >= vz.x && p.x <= vz.x + vz.w && p.y >= vz.y && p.y <= vz.y + vz.h;

  canvas.style.touchAction = "none";
  canvas.style.cursor = "grab";
  canvas.onpointerdown = (e) => {
    canvas.setPointerCapture(e.pointerId);
    const p = pt(e);
    if (inside(p)) { mode = "move"; grab = { dx: p.x - vz.x, dy: p.y - vz.y }; canvas.style.cursor = "grabbing"; }
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
    recompute();
  };
  canvas.onpointerup = () => { mode = null; canvas.style.cursor = "grab"; recompute(); };

  function recompute() {
    const zc = { x: vz.x + vz.w / 2, y: vz.y + vz.h / 2 };
    const q3 = rectQuad3d(vz, sim);
    particles = [];
    for (let i = 0; i < drawRays.length; i++) {
      const h = rayHit(drawRays[i], vz, sim, bias, q3);
      hits[i] = h;
      if (h && particles.length < 180) {
        const r = drawRays[i];
        particles.push({ x0: r[2], y0: r[3], x1: zc.x, y1: zc.y, t: Math.random() });
      }
    }
    const m = simCompute(vz, sim, bias);
    $("simFlow").textContent = `${particles.length ? [...hits].filter(Boolean).length : 0} / ${drawRays.length} rays flowing in`;
    showMetrics(m);
  }

  function showMetrics(m) {
    const d = (cur, b, suffix = "") => {
      if (!base || b == null) return "";
      const diff = cur - b;
      const cls = diff > 0.05 ? "up" : diff < -0.05 ? "down" : "flat";
      const s = Math.abs(diff) < 0.05 ? "±0" : `${diff > 0 ? "+" : ""}${fmt(diff, 1)}${suffix}`;
      return `<small class="delta ${cls}">${s} vs ${esc(z0.label)}</small>`;
    };
    $("simMetrics").innerHTML =
      row("Impressions", "imp", fmt(shown.imp), d(m.imp, base?.imp)) +
      row("Attention rate", "rate", fmt(shown.rate, 1) + "%", d(m.rate, base?.rate, "pp")) +
      (rep.still ? "" : row("Attentive seconds", "att", fmt(shown.att, 1) + "s", d(m.att, base?.att, "s"))) +
      (rep.still ? "" : row("Avg dwell", "avg", fmt(shown.avg, 2) + "s", d(m.avg, base?.avg, "s"))) +
      (rep.still || m.ttfl == null ? "" : row("Time to first look", "ttfl", fmt(shown.ttfl, 1) + "s", ""));
    target.imp = m.imp; target.rate = m.rate; target.att = m.att;
    target.avg = m.avg; target.ttfl = m.ttfl ?? 0;
  }
  const target = { ...shown };
  const row = (k, id, v, delta) =>
    `<div class="sim-row"><span class="k">${k}</span><span class="v"><span data-sv="${id}">${v}</span> ${delta}</span></div>`;

  const slider = $("coneSlider");
  if (slider) slider.oninput = () => {
    bias = parseFloat(slider.value);
    $("coneVal").textContent = bias === 0 ? "auto" : (bias > 0 ? "+" : "") + bias + "°";
    recompute();
  };
  const rst = $("simReset");
  if (rst) rst.onclick = () => { vz = { ...origVz }; if (slider) { slider.value = 0; bias = 0; $("coneVal").textContent = "auto"; } recompute(); };

  // sürekli animasyon döngüsü: akan bakış parçacıkları + nabız + sayaç yumuşatma
  let running = true;
  function frame(now) {
    if (!running || !document.getElementById("simCanvas")) return;
    ctx.clearRect(0, 0, sim.w, sim.h);
    if (bg.complete && bg.naturalWidth) ctx.drawImage(bg, 0, 0, sim.w, sim.h);
    else { ctx.fillStyle = "#111"; ctx.fillRect(0, 0, sim.w, sim.h); }
    ctx.save(); ctx.globalAlpha = 0.5; ctx.fillStyle = "#000"; ctx.fillRect(0, 0, sim.w, sim.h); ctx.restore();

    // ışınlar: kısa yön çizgileri (isabet = beyaz parlak, ıskalama = soluk)
    ctx.lineWidth = Math.max(1, lw * 0.5);
    for (let i = 0; i < drawRays.length; i++) {
      const r = drawRays[i], L = Math.max(sim.w * 0.02, 22);
      ctx.strokeStyle = hits[i] ? "rgba(255,255,255,.5)" : "rgba(255,255,255,.08)";
      ctx.beginPath(); ctx.moveTo(r[2], r[3]);
      ctx.lineTo(r[2] + r[4] * L, r[3] + r[5] * L); ctx.stroke();
    }

    // diğer bölgeler (referans, kesikli)
    for (const z of rep.zones) {
      const r = normToPx(z.norm, sim);
      ctx.setLineDash([8, 6]); ctx.strokeStyle = "rgba(255,255,255,.4)"; ctx.lineWidth = lw;
      if (z.poly) {
        ctx.beginPath();
        z.poly.forEach(([px, py], i) => i ? ctx.lineTo(px * sim.w, py * sim.h) : ctx.moveTo(px * sim.w, py * sim.h));
        ctx.closePath(); ctx.stroke();
      } else ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.font = `${fs}px sans-serif`;
      ctx.fillText(z.label, r.x + 6, r.y - 8);
    }

    // akan bakış parçacıkları (reklamın içine akan dikkat)
    const zc = { x: vz.x + vz.w / 2, y: vz.y + vz.h / 2 };
    ctx.fillStyle = "#2fd39a";
    for (const p of particles) {
      p.t += 0.012 + p.t * 0.02;
      if (p.t >= 1) p.t = 0;
      const e = 1 - Math.pow(1 - p.t, 2);
      const px = p.x0 + (zc.x - p.x0) * e, py = p.y0 + (zc.y - p.y0) * e;
      ctx.globalAlpha = 0.25 + 0.7 * (1 - p.t);
      ctx.beginPath(); ctx.arc(px, py, Math.max(2, sim.w / 520), 0, 6.283); ctx.fill();
    }
    ctx.globalAlpha = 1;

    // sanal bölge — nabız atan parlak çerçeve
    const pulse = 0.5 + 0.5 * Math.sin(now / 380);
    ctx.fillStyle = `rgba(47,211,154,${0.08 + 0.06 * pulse})`;
    ctx.fillRect(vz.x, vz.y, vz.w, vz.h);
    ctx.strokeStyle = `rgba(47,211,154,${0.6 + 0.4 * pulse})`;
    ctx.lineWidth = lw * 1.6; ctx.strokeRect(vz.x, vz.y, vz.w, vz.h);
    ctx.fillStyle = "#2fd39a"; ctx.font = `600 ${fs}px sans-serif`;
    ctx.fillText("VIRTUAL", vz.x + 6, vz.y - 8);

    // sayaç yumuşatma (ekrandaki değer hedefe doğru akar)
    let changed = false;
    for (const key of ["imp", "rate", "att", "avg", "ttfl"]) {
      const diff = target[key] - shown[key];
      if (Math.abs(diff) > 0.001) { shown[key] += diff * 0.2; changed = true; }
      else shown[key] = target[key];
    }
    if (changed) {
      const set = (id, v) => { const el = document.querySelector(`[data-sv="${id}"]`); if (el) el.textContent = v; };
      set("imp", fmt(shown.imp)); set("rate", fmt(shown.rate, 1) + "%");
      set("att", fmt(shown.att, 1) + "s"); set("avg", fmt(shown.avg, 2) + "s");
      set("ttfl", fmt(shown.ttfl, 1) + "s");
    }
    requestAnimationFrame(frame);
  }

  bg.onload = () => recompute();
  recompute();
  requestAnimationFrame(frame);
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
  "3D calibration": "The scene is reconstructed in 3D (metric depth + ground plane). Confidence is self-verified: reconstructed people should cluster around 1.70m — the tighter the cluster, the more trustworthy the geometry.",
  "Zone size": "Physical size of the ad surface in meters, reconstructed from the 3D scene.",
  "View distance": "Average real-world distance (meters) between lookers and the ad surface.",
  "Audience insights": "Opt-in, on-device gender estimate from visible faces only. Aggregate percentages — never per-person, no images stored. Compare traffic vs lookers to see who the creative attracts.",
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

function audienceHtml(rep) {
  const a = rep.audience;
  if (a && a.enabled === false) {
    return `<div class="wide-chart aud"><h4>Audience insights <span class="new-tag">BETA</span></h4>
      <p class="aud-note">Could not run: ${esc(a.note || "unknown")}. Faces may be too small/occluded, or the classifier failed to load.</p></div>`;
  }
  if (!a) {
    return `<div class="wide-chart aud"><h4>Audience insights</h4>
      <p class="aud-note">Not run for this analysis — enable the "Audience insights" checkbox on the Ad zones step. First run downloads the classifier models (needs internet, a few minutes).</p></div>`;
  }
  if (!a.enabled) return "";
  const bar = (s) => {
    if (s && s.suppressed) return `<div class="bar-line" style="height:14px"><div style="width:100%;background:rgba(255,255,255,.06)"></div></div>`;
    const n = (s.female || 0) + (s.male || 0) + (s.unknown || 0) || 1;
    const f = (s.female / n) * 100, m = (s.male / n) * 100, u = (s.unknown / n) * 100;
    return `<div class="bar-line" style="height:14px">
      <div style="width:${f}%;background:#fff" title="female"></div>
      <div style="width:${m}%;background:rgba(255,255,255,.45)" title="male"></div>
      <div style="width:${u}%;background:rgba(255,255,255,.12)" title="unknown"></div></div>`;
  };
  const legend = (s) => s && s.suppressed
    ? `<small>hidden — group of ${s.n} too small to report safely (k-anonymity, min 5)</small>`
    : `<small>${s.female} female · ${s.male} male · ${s.unknown} unknown</small>`;
  let rows = `<div class="aud-row"><span class="k" title="${GLOSS["Audience insights"]}">Traffic</span>${bar(a.traffic_split)}${legend(a.traffic_split)}</div>`;
  if (a.age_split && a.age_split.suppressed) {
    rows += `<div class="aud-row"><span class="k">Age (est.)</span><div class="bar-line" style="height:14px"><div style="width:100%;background:rgba(255,255,255,.06)"></div></div><small>hidden — group too small (k-anonymity)</small></div>`;
  } else if (a.age_split && a.age_order) {
    const shades = ["#ffffff", "rgba(255,255,255,.8)", "rgba(255,255,255,.62)", "rgba(255,255,255,.46)", "rgba(255,255,255,.32)", "rgba(255,255,255,.2)", "rgba(255,255,255,.55)"];
    const tot = a.age_order.reduce((s, b) => s + (a.age_split[b] || 0), 0) + (a.age_split.unknown || 0);
    if (tot) {
      let segs = "", leg = [];
      a.age_order.forEach((b, i) => {
        const v = a.age_split[b] || 0;
        if (!v) return;
        segs += `<div style="width:${(v / tot) * 100}%;background:${shades[i % shades.length]}" title="${b}: ${v}"></div>`;
        leg.push(`${b}:${v}`);
      });
      const u = a.age_split.unknown || 0;
      if (u) segs += `<div style="width:${(u / tot) * 100}%;background:rgba(255,255,255,.08)" title="unknown: ${u}"></div>`;
      rows += `<div class="aud-row"><span class="k">Age (est.)</span><div class="bar-line" style="height:14px">${segs}</div><small>${leg.join(" · ")}${u ? " · ?:" + u : ""}</small></div>`;
    }
  }
  for (const z of rep.zones) {
    const zi = a.zones[String(z.id)];
    if (zi) rows += `<div class="aud-row"><span class="k">${esc(z.label)} lookers</span>${bar(zi.lookers_split)}${legend(zi.lookers_split)}</div>`;
  }
  return `<div class="wide-chart aud">
    <h4>Audience insights <span class="new-tag">BETA</span> — coverage ${fmt(a.coverage_pct, 0)}% of people had a classifiable face</h4>
    ${rows}
    <p class="aud-note">${a.note ? esc(a.note) + " " : ""}${esc(a.disclosure)}</p></div>`;
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
      ${!still && z.reaches != null ? fstage("Reached", z.reachers, z.traffic, "") : ""}
    </div>

    <div class="metric-grid">
      <div class="mcard star" title="${GLOSS["Attention rate"]} ${GLOSS["95% CI"]}"><div class="k">Attention rate · 95% CI ${z.attention_rate_ci ? z.attention_rate_ci[0] + "–" + z.attention_rate_ci[1] + "%" : ""}</div><div class="v">${z.attention_rate}<small>%</small></div></div>
      ${still ? "" : `<div class="mcard" title="${GLOSS["Attentive seconds"]}"><div class="k">Attentive seconds</div><div class="v">${fmt(z.attentive_seconds, 1)}<small>s</small></div></div>`}
      ${still ? "" : `<div class="mcard" title="${GLOSS["Avg dwell"]}"><div class="k">Avg dwell</div><div class="v">${fmt(z.avg_dwell, 2)}<small>s</small></div></div>`}
      ${still ? "" : `<div class="mcard" title="${GLOSS["Avg dwell"]}"><div class="k">Max dwell</div><div class="v">${fmt(z.max_dwell, 1)}<small>s</small></div></div>`}
      ${still || z.time_to_first_look == null ? "" : `<div class="mcard" title="${GLOSS["Time to first look"]}"><div class="k">Time to first look <span class="new-tag">NEW</span></div><div class="v">${fmt(z.time_to_first_look, 1)}<small>s</small></div></div>`}
      ${still ? "" : `<div class="mcard" title="${GLOSS["Glances / looker"]}"><div class="k">Glances / looker <span class="new-tag">NEW</span></div><div class="v">${fmt(z.glances_per_looker, 1)}</div></div>`}
      ${still ? "" : `<div class="mcard" title="${GLOSS["Stopping power"]}"><div class="k">Stopping power <span class="new-tag">NEW</span></div><div class="v">${fmt(z.stopping_power, 0)}<small>% slowdown</small></div></div>`}
      ${!still && z.reaches != null ? `<div class="mcard" title="A wrist keypoint inside the shelf zone in ≥3 consecutive samples (Spec v1.0)."><div class="k">Reaches <span class="new-tag">NEW</span></div><div class="v">${z.reaches}</div></div>` : ""}
      ${!still && z.reach_rate != null ? `<div class="mcard" title="Share of lookers who reached toward the shelf."><div class="k">Reach rate</div><div class="v">${fmt(z.reach_rate, 1)}<small>% of lookers</small></div></div>` : ""}
      ${z.size_m ? `<div class="mcard" title="${GLOSS["Zone size"]}"><div class="k">Zone size <span class="new-tag">3D</span></div><div class="v">${fmt(z.size_m[0], 1)}×${fmt(z.size_m[1], 1)}<small>m${z.surface_tilt_deg != null ? " · " + fmt(z.surface_tilt_deg, 0) + "° tilt" : ""}</small></div></div>` : ""}
      ${z.avg_view_distance_m != null ? `<div class="mcard" title="${GLOSS["View distance"]}"><div class="k">Avg view distance <span class="new-tag">3D</span></div><div class="v">${fmt(z.avg_view_distance_m, 1)}<small>m</small></div></div>` : ""}
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
    ${!still && z.reach_evidence && z.reach_evidence.length ? `
    <div class="evidence">
      <h4>Reach evidence <span class="new-tag">AUDITABLE</span></h4>
      <div class="ev-chips">${z.reach_evidence.map((e) =>
        `<button class="ev-chip" data-t="${e.t}">reach #${e.pid} @ ${tstamp(e.t)}</button>`).join("")}</div>
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

/* ================= LIVE (Faz 2): continuous measurement ================= */
let lvPoll = null, lvCams = [], lvEdit = null;   // lvEdit: {cam, zones, mode, pts}

async function lvBench() {
  const el = $("benchPanel");
  if (!el) return;
  try {
    const s = await (await fetch("/api/dataset/stats")).json();
    if (!s.total_episodes) {
      el.innerHTML = '<div class="wide-chart"><h4>Attention dataset <span class="new-tag">ASSET</span></h4>' +
        '<p class="aud-note">No episodes yet. Every analysis (batch + live) adds anonymous attention episodes — the normative benchmark and the seed for the world attention model.</p></div>';
      return;
    }
    const rows = s.by_zone_type.map((z) =>
      `<tr><td>${esc(z.zone_type)}</td><td>${z.episodes}</td><td>${fmt(z.avg_dwell, 1)}s</td>` +
      `<td>${z.reach_rate != null ? fmt(z.reach_rate, 0) + "%" : "—"}</td>` +
      `<td>${z.head_signal_share != null ? fmt(z.head_signal_share, 0) + "%" : "—"}</td></tr>`).join("");
    el.innerHTML = `<div class="wide-chart"><h4>Attention dataset <span class="new-tag">ASSET</span> — ${s.total_episodes} episodes from ${s.sources} source${s.sources === 1 ? "" : "s"} · schema v${s.schema_version}</h4>
      <table class="bench-tbl"><thead><tr><th>Surface</th><th>Episodes</th><th>Avg dwell</th><th>Reach</th><th>Head signal</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="aud-note">Anonymous, aggregate — the normative benchmark (retail moat) and the seed corpus for the cross-environment attention model.</p></div>`;
  } catch { el.innerHTML = ""; }
}

async function showLive() {
  await lvRefresh();
  lvBench();
  clearInterval(lvPoll);
  lvPoll = setInterval(() => {
    if (!$("step-live").classList.contains("on")) { clearInterval(lvPoll); return; }
    lvTick();
  }, 3000);
}

async function lvRefresh() {
  lvCams = await (await fetch("/api/cameras")).json();
  const el = $("liveList");
  if (!lvCams.length) {
    el.innerHTML = '<p class="aud-note">No cameras yet. Add an RTSP URL, a webcam index (0), or a file path with "loop" for a fake-live test.</p>';
    return;
  }
  el.innerHTML = lvCams.map((c) => `
    <div class="cam-card" data-cam="${c.id}">
      <div class="cam-head">
        <span class="lv-dot ${c.status === "live" ? "on" : ""}"></span>
        <b>${esc(c.name || c.id)}</b>
        <small>${esc(String(c.url))}${c.loop ? " · loop" : ""} · ${(c.zones || []).length} zones</small>
        <span class="spacer"></span>
        <button class="primary sm" data-act="watch">▶ Watch live</button>
        ${c.status === "live"
          ? `<button class="sm" data-act="stop">Stop</button>`
          : `<button class="sm" data-act="start" ${!(c.zones || []).length ? "disabled title='draw zones first'" : ""}>Start</button>`}
        <button class="sm" data-act="zones">Zones</button>
        <button class="sm" data-act="del" title="remove">×</button>
      </div>
      <div class="cam-live" id="lv-${c.id}"><small>${c.status === "live" ? "…" : "stopped"}</small></div>
      <div class="cam-chart" id="lvc-${c.id}"></div>
    </div>`).join("");
  el.querySelectorAll("[data-act]").forEach((b) => b.onclick = () => lvAction(b));
  lvTick();
}

async function lvAction(btn) {
  const id = btn.closest(".cam-card").dataset.cam;
  const act = btn.dataset.act;
  if (act === "del") {
    if (!confirm("Remove this camera? Stored hourly aggregates are kept.")) return;
    await fetch(`/api/cameras/${id}`, { method: "DELETE" });
    return lvRefresh();
  }
  if (act === "start" || act === "stop") {
    await fetch(`/api/cameras/${id}/${act}`, { method: "POST" });
    setTimeout(lvRefresh, 800);
    return;
  }
  if (act === "zones") lvOpenModal(lvCams.find((c) => c.id === id));
  if (act === "watch") lvWatch(lvCams.find((c) => c.id === id));
}

/* -- live watch modal: annotated + face-blurred frame, ~1s refresh -- */
let lvWatchTimer = null;
async function lvWatch(cam) {
  $("lvwName").textContent = cam.name || cam.id;
  $("lvwImg").removeAttribute("src");
  $("lvwHint").textContent = "starting camera…";
  $("lvWatchModal").classList.remove("hidden");
  if (cam.status !== "live") {                 // izlemek için kamerayı otomatik başlat
    await fetch(`/api/cameras/${cam.id}/start`, { method: "POST" });
    setTimeout(lvRefresh, 1500);
  }
  clearInterval(lvWatchTimer);
  const tick = () => {
    const img = new Image();
    const url = `/api/cameras/${cam.id}/live_frame?ts=${Date.now()}`;
    img.onload = () => { $("lvwImg").src = url; $("lvwHint").textContent = "live · face-blurred · on-device"; };
    img.onerror = () => { $("lvwHint").textContent = "waiting for first frame…"; };
    img.src = url;
  };
  tick();
  lvWatchTimer = setInterval(tick, 1000);
}
function lvWatchClose() {
  clearInterval(lvWatchTimer); lvWatchTimer = null;
  $("lvWatchModal").classList.add("hidden");
}
$("lvwClose").onclick = lvWatchClose;

async function lvTick() {
  for (const c of lvCams) {
    const box = $("lv-" + c.id);
    if (!box) continue;
    try {
      const lv = await (await fetch(`/api/live/${c.id}`)).json();
      if (lv.status !== "live") {
        box.innerHTML = `<small>${lv.status || "stopped"}${lv.error ? " · " + esc(lv.error) : ""}</small>`;
      } else {
        const zbits = Object.values(lv.zones || {}).map((z) =>
          `<span class="lv-kpi"><b>${z.lookers}</b> looking · ${fmt(z.att, 0)}s <small>${esc(z.label)}</small></span>`);
        const lbits = Object.entries(lv.lines || {}).map(([zid, l]) =>
          `<span class="lv-kpi"><b>${l.in}</b> in · ${l.out} out</span>`);
        box.innerHTML = `<span class="lv-kpi"><b>${lv.traffic}</b> tracked</span>` + zbits.join("") + lbits.join("");
      }
    } catch { box.innerHTML = "<small>unreachable</small>"; }
    lvChart(c);   // grafik durumdan bağımsız: geçmiş agregatlar her zaman görünür
  }
}

async function lvChart(c) {
  const el = $("lvc-" + c.id);
  if (!el) return;
  const since = Math.floor(Date.now() / 1000) - 24 * 3600;
  const rows = await (await fetch(`/api/timeseries?camera=${c.id}&since=${since}`)).json();
  const cam = rows.filter((r) => r.zone_id === "_cam");
  if (!cam.length) { el.innerHTML = ""; return; }
  const att = {};
  rows.forEach((r) => { if (r.zone_id !== "_cam") att[r.hour_ts] = (att[r.hour_ts] || 0) + r.attentive_sec; });
  const maxT = Math.max(1, ...cam.map((r) => r.traffic));
  const bw = 100 / Math.max(cam.length, 12);
  const bars = cam.map((r, i) => {
    const h = r.traffic / maxT * 34;
    const hh = new Date(r.hour_ts * 1000).getHours();
    return `<rect x="${i * bw + 0.5}" y="${38 - h}" width="${bw - 1}" height="${h}" rx="0.8" fill="var(--acc, #8fda3c)" opacity=".85"><title>${hh}:00 — ${r.traffic} tracked · ${fmt(att[r.hour_ts] || 0, 0)}s attention · ${r.enters} in</title></rect>`;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 100 40" preserveAspectRatio="none" style="width:100%;height:52px">${bars}</svg>
    <small class="lv-axis">last 24h · hourly tracked people (hover for detail)</small>`;
}

/* -- add camera -- */
$("lvAdd").onclick = async () => {
  const name = $("lvName").value.trim(), url = $("lvUrl").value.trim();
  if (!url) return $("lvUrl").focus();
  await fetch("/api/cameras", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: name || url, url, loop: $("lvLoop").checked, sample_fps: 5 }) });
  $("lvName").value = ""; $("lvUrl").value = "";
  lvRefresh();
};

/* -- zone drawing modal (surfaces + entrance lines on a captured frame) -- */
function lvSetMode(mode) {
  lvEdit.mode = mode; lvEdit.pts = [];
  $("lvmSurface").classList.toggle("is-on", mode === "poly");
  $("lvmLine").classList.toggle("is-on", mode === "line");
  lvHint(mode === "line"
    ? "click 2 points across the entrance — arrow shows IN (draw in reverse to flip)"
    : "click the 4 corners of the surface");
}
function lvHint(text, warn) {
  const h = $("lvmHint"); h.textContent = text; h.classList.toggle("warn", !!warn);
}
function lvmRenderList() {
  const ul = $("lvmZoneList");
  if (!lvEdit.zones.length) { ul.innerHTML = '<li class="empty">None yet — draw one</li>'; return; }
  ul.innerHTML = lvEdit.zones.map((z, i) =>
    `<li><span class="zdot" style="background:${COLORS[i % COLORS.length]}"></span>` +
    `<span class="zl-main"><b>${esc(z.label)}</b><small>${esc(z.type)}</small></span>` +
    `<button class="rm" data-del="${z.id}" title="remove">×</button></li>`).join("");
  ul.querySelectorAll("[data-del]").forEach((b) => b.onclick = () => {
    lvEdit.zones = lvEdit.zones.filter((z) => String(z.id) !== b.dataset.del);
    lvmRenderList(); lvDraw();
  });
}
function lvOpenModal(cam) {
  lvEdit = { cam, zones: JSON.parse(JSON.stringify(cam.zones || [])), mode: null, pts: [] };
  $("lvmName").textContent = cam.name || cam.id;
  $("lvmSurface").classList.remove("is-on"); $("lvmLine").classList.remove("is-on");
  lvHint("loading frame…");
  $("lvModal").classList.remove("hidden");
  lvmRenderList();
  const img = $("lvmImg");
  img.onload = () => {
    const cv = $("lvmCanvas");
    cv.width = img.naturalWidth; cv.height = img.naturalHeight;
    lvHint("pick a tool, then click on the frame");
    lvDraw();
  };
  img.onerror = () => { lvHint("source not reachable — check the URL", true); };
  img.src = `/api/cameras/${cam.id}/frame?ts=${Date.now()}`;
}
$("lvmClose").onclick = () => { $("lvModal").classList.add("hidden"); lvEdit = null; };
$("lvmSurface").onclick = () => lvSetMode("poly");
$("lvmLine").onclick = () => lvSetMode("line");

$("lvmCanvas").addEventListener("pointerdown", (e) => {
  if (!lvEdit || !lvEdit.mode) return;
  const r = e.target.getBoundingClientRect();
  lvEdit.pts.push({ x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height });
  const pts = lvEdit.pts;
  if (lvEdit.mode === "line" && pts.length === 2) {
    const [p1, p2] = pts;
    lvEdit.zones.push({ id: Date.now() % 1e9, label: `Entrance ${lvEdit.zones.filter((z) => z.type === "line").length + 1}`,
      type: "line", line: [[p1.x, p1.y], [p2.x, p2.y]],
      x: Math.min(p1.x, p2.x), y: Math.min(p1.y, p2.y),
      w: Math.abs(p2.x - p1.x) || 0.01, h: Math.abs(p2.y - p1.y) || 0.01 });
    lvEdit.mode = null; lvEdit.pts = [];
    $("lvmLine").classList.remove("is-on");
    lvHint("saved locally — draw more or Save & start");
    lvmRenderList();
  }
  if (lvEdit.mode === "poly" && pts.length === 4) {
    const cx = pts.reduce((s, p) => s + p.x, 0) / 4, cy = pts.reduce((s, p) => s + p.y, 0) / 4;
    const poly = [...pts].sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
    const xs = poly.map((p) => p.x), ys = poly.map((p) => p.y);
    const type = $("lvmType").value;
    lvEdit.zones.push({ id: Date.now() % 1e9, label: `${type} ${lvEdit.zones.length + 1}`, type,
      poly: poly.map((p) => [p.x, p.y]),
      x: Math.min(...xs), y: Math.min(...ys),
      w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) });
    lvEdit.mode = null; lvEdit.pts = [];
    $("lvmSurface").classList.remove("is-on");
    lvHint("saved locally — draw more or Save & start");
    lvmRenderList();
  }
  lvDraw();
});

function lvDraw() {
  const cv = $("lvmCanvas"), ctx2 = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx2.clearRect(0, 0, W, H);
  ctx2.font = `600 ${Math.max(13, W / 70)}px sans-serif`;
  for (const [i, z] of lvEdit.zones.entries()) {
    const col = COLORS[i % COLORS.length];
    ctx2.strokeStyle = col; ctx2.fillStyle = col; ctx2.lineWidth = Math.max(2, W / 480);
    if (z.line) {
      const [[x1, y1], [x2, y2]] = z.line;
      ctx2.lineWidth = Math.max(3, W / 320);
      ctx2.beginPath(); ctx2.moveTo(x1 * W, y1 * H); ctx2.lineTo(x2 * W, y2 * H); ctx2.stroke();
      const mx = (x1 + x2) / 2 * W, my = (y1 + y2) / 2 * H;
      const dx = (x2 - x1) * W, dy = (y2 - y1) * H, ll = Math.hypot(dx, dy) || 1;
      ctx2.beginPath(); ctx2.moveTo(mx, my);
      ctx2.lineTo(mx - dy / ll * 30, my + dx / ll * 30); ctx2.stroke();
      ctx2.fillText(`${z.label} (in →)`, mx + 8, my - 8);
    } else if (z.poly) {
      ctx2.beginPath();
      z.poly.forEach(([px, py], j) => j ? ctx2.lineTo(px * W, py * H) : ctx2.moveTo(px * W, py * H));
      ctx2.closePath();
      if (z.type === "staff") ctx2.setLineDash([8, 6]);
      ctx2.stroke(); ctx2.setLineDash([]);
      ctx2.fillText(z.label, z.x * W + 6, z.y * H - 7);
    }
  }
  if (lvEdit.pts.length) {
    ctx2.fillStyle = "#fff";
    for (const p of lvEdit.pts) { ctx2.beginPath(); ctx2.arc(p.x * W, p.y * H, Math.max(4, W / 300), 0, 6.283); ctx2.fill(); }
  }
}

$("lvmSave").onclick = async () => {
  if (!lvEdit) return;
  const cam = { ...lvEdit.cam, zones: lvEdit.zones };
  delete cam.status;
  await fetch("/api/cameras", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cam) });
  await fetch(`/api/cameras/${cam.id}/stop`, { method: "POST" });
  await fetch(`/api/cameras/${cam.id}/start`, { method: "POST" });
  $("lvModal").classList.add("hidden"); lvEdit = null;
  setTimeout(lvRefresh, 800);
};
