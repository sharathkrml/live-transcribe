const $ = (id) => document.getElementById(id);

const el = {
  video: $("video"),
  overlay: $("overlay"),
  pulse: $("pulse"),
  fileName: $("file-name"),
  session: $("session"),
  profile: $("profile"),
  chooser: $("chooser"),
  transcript: $("transcript"),
  panelEmpty: $("panel-empty"),
  follow: $("follow"),
  status: $("status"),
  meter: $("meter"),
  meterTrack: document.querySelector(".meter-track"),
  meterCells: $("meter-cells"),
  meterHead: $("meter-head"),
  clock: $("clock"),
  note: $("note"),
  empty: $("empty"),
  open: $("open"),
  emptyOpen: $("empty-open"),
  fallback: $("fallback"),
  fallbackPath: $("fallback-path"),
  export: $("export"),
  veil: $("veil"),
  veilFile: $("veil-file"),
  veilTitle: $("veil-title"),
  veilHint: $("veil-hint"),
  veilBar: $("veil-bar"),
  veilFill: $("veil-fill"),
  steps: $("steps"),
  banner: $("banner"),
  bannerText: $("banner-text"),
  save: $("save"),
};

const S = {
  ws: null,
  cues: [],
  keys: new Set(),
  rows: new Map(),
  lookahead: 10,
  duration: 0,
  activeKey: null,
  hint: 0,
  lastSent: 0,
  dragging: false,
  hasMedia: false,
  finished: false,
  busy: false,
  buckets: 0,
  chunksPerBucket: 1,
  pendingVideo: null,
  mediaInfo: null,
  phase: "idle",
  warming: false,
  videoReady: false,
  playback: null,
};

// ------------------------------------------------------------- transport

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  S.ws = ws;

  ws.onopen = () => {
    if (S.hasMedia) sendPlayhead(true);
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "hello") {
      S.lookahead = msg.lookahead;
      if (msg.profiles) fillProfiles(msg.profiles, msg.profile);
      if (!msg.native_picker) showFallback();
      if (msg.media) restore(msg.media);
    } else if (msg.type === "cues") {
      addCues(msg.items);
    } else if (msg.type === "state") {
      renderState(msg);
    }
  };

  ws.onclose = () => {
    setStatus("Disconnected — reconnecting…");
    setTimeout(connect, 1000);
  };
}

function sendPlayhead(force) {
  if (!S.ws || S.ws.readyState !== 1) return;
  const now = performance.now();
  if (!force && now - S.lastSent < 250) return;
  S.lastSent = now;
  S.ws.send(JSON.stringify({ type: "playhead", time: el.video.currentTime }));
}

// ------------------------------------------------------------------ cues

function addCues(items) {
  if (!items || !items.length) return;
  const fresh = [];
  for (const cue of items) {
    const key = cue.start.toFixed(2);
    if (S.keys.has(key)) continue;
    S.keys.add(key);
    S.cues.push({ ...cue, key });
    fresh.push(cue);
  }
  if (!fresh.length) return;
  S.cues.sort((a, b) => a.start - b.start);
  fresh.sort((a, b) => a.start - b.start);

  const nearBottom =
    el.transcript.scrollHeight - el.transcript.scrollTop - el.transcript.clientHeight < 90;
  for (const cue of fresh) el.transcript.appendChild(row(cue));
  el.panelEmpty.hidden = true;
  el.export.hidden = false;
  if (nearBottom && el.follow.checked) el.transcript.scrollTop = el.transcript.scrollHeight;
  if (!S.warming) hideBanner();
}

function row(cue) {
  const li = document.createElement("li");
  li.dataset.key = cue.key;

  const time = document.createElement("span");
  time.className = "t";
  time.textContent = clock(cue.start);

  const body = document.createElement("div");
  body.className = "x";
  const src = document.createElement("div");
  src.className = "src";
  src.textContent = cue.source;
  body.appendChild(src);
  if (cue.target && cue.target !== cue.source) {
    const tgt = document.createElement("div");
    tgt.className = "tgt";
    tgt.textContent = cue.target;
    body.appendChild(tgt);
  }

  li.append(time, body);
  li.onclick = () => seekTo(cue.start + 0.01);
  S.rows.set(cue.key, li);
  return li;
}

function backfill() {
  fetch("/api/cues")
    .then((r) => r.json())
    .then((d) => addCues(d.items))
    .catch(() => {});
}

function refreshState() {
  fetch("/api/state")
    .then((r) => r.json())
    .then((s) => {
      if (s && s.chunks_total !== undefined) renderState(s);
    })
    .catch(() => {});
}

function clearCues() {
  S.cues = [];
  S.keys = new Set();
  S.rows = new Map();
  S.activeKey = null;
  S.hint = 0;
  S.buckets = 0;
  S.finished = false;
  el.transcript.replaceChildren();
  el.meterCells.replaceChildren();
  el.panelEmpty.hidden = false;
  el.export.hidden = true;
  el.save.hidden = true;
  el.overlay.textContent = "";
  el.overlay.classList.remove("show");
}

function findCue(t) {
  const cues = S.cues;
  for (let i = Math.max(0, S.hint - 2); i < cues.length; i++) {
    if (cues[i].start > t) return -1;
    if (t <= cues[i].end + 0.6) return i;
  }
  return -1;
}

// -------------------------------------------------------------- playback

function seekTo(time) {
  const t = Math.max(0, Math.min(S.duration || el.video.duration || 0, time));
  if (S.dragging && el.video.fastSeek) el.video.fastSeek(t);
  else el.video.currentTime = t;
  paintHead(t);
  if (!S.dragging) sendPlayhead(true);
}

function paintHead(t) {
  const d = S.duration || 0;
  const pct = d ? Math.min(100, Math.max(0, (t / d) * 100)) : 0;
  el.meterHead.style.left = `${pct}%`;
  el.meterHead.style.opacity = d ? "1" : "0";
  el.meter.setAttribute("aria-valuenow", Math.round(pct));
  el.meter.setAttribute("aria-valuetext", clock(t));
  el.clock.textContent = `${clock(t)} / ${clock(d)}`;
}

// ----------------------------------------------------------------- meter

function timelineSeek(event) {
  const rect = el.meterTrack.getBoundingClientRect();
  const pct = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
  seekTo(pct * (S.duration || 0));
}

el.meter.addEventListener("pointerdown", (event) => {
  if (!S.duration) return;
  S.dragging = true;
  el.meter.dataset.active = "true";
  el.meter.setPointerCapture(event.pointerId);
  timelineSeek(event);
});

el.meter.addEventListener("pointermove", (event) => {
  if (S.dragging) timelineSeek(event);
});

const endDrag = (event) => {
  if (!S.dragging) return;
  S.dragging = false;
  el.meter.dataset.active = "false";
  if (el.meter.hasPointerCapture(event.pointerId)) el.meter.releasePointerCapture(event.pointerId);
  sendPlayhead(true);
};
el.meter.addEventListener("pointerup", endDrag);
el.meter.addEventListener("pointercancel", endDrag);

el.meter.addEventListener("keydown", (event) => {
  if (!S.duration) return;
  const step = event.shiftKey ? 30 : 5;
  const map = { ArrowLeft: -step, ArrowRight: step };
  if (event.key === "Home") seekTo(0);
  else if (event.key === "End") seekTo(S.duration);
  else if (map[event.key]) seekTo(el.video.currentTime + map[event.key]);
  else return;
  event.preventDefault();
});

// ------------------------------------------------------------- rendering

function tick() {
  const t = el.video.currentTime;
  const idx = findCue(t);
  if (idx >= 0) {
    S.hint = idx;
    const cue = S.cues[idx];
    el.overlay.textContent = cue.target || cue.source;
    el.overlay.classList.add("show");
    setActive(cue.key);
  } else {
    el.overlay.classList.remove("show");
    setActive(null);
  }
  if (!S.dragging) paintHead(t);
  if (!el.video.paused) sendPlayhead(false);
  requestAnimationFrame(tick);
}

function setActive(key) {
  if (key === S.activeKey) return;
  if (S.activeKey && S.rows.has(S.activeKey)) S.rows.get(S.activeKey).classList.remove("active");
  S.activeKey = key;
  if (key && S.rows.has(key)) {
    const li = S.rows.get(key);
    li.classList.add("active");
    if (el.follow.checked) li.scrollIntoView({ block: "nearest" });
  }
}

function renderState(state) {
  if (!S.hasMedia) {
    el.pulse.classList.remove("live", "done");
    if (S.phase === "idle") {
      el.note.textContent = "No video open";
      el.note.classList.remove("ahead");
    }
    return;
  }
  S.finished = !!state.finished;
  S.warming = !!state.warming;
  if (state.playback) S.playback = state.playback;

  paintCells(state);

  if (S.pendingVideo && S.playback && S.playback.ready) {
    if (S.playback.error) {
      setStatus(`Couldn't convert for playback: ${S.playback.error}`, true);
    }
    attachVideo(S.pendingVideo, S.mediaInfo);
  }

  el.pulse.classList.toggle("live", !S.finished);
  el.pulse.classList.toggle("done", !!S.finished);

  paintPhase();
  el.save.hidden = !S.finished || !el.veil.hidden;
  if (!el.veil.hidden) return;

  const ahead = state.ahead ?? 0;
  const done = state.chunks_done ?? 0;
  const total = state.chunks_total ?? 0;
  el.note.classList.toggle("ahead", (S.finished || ahead >= S.lookahead) && !S.warming);
  if (state.error) {
    el.note.textContent = "Transcription error";
    setStatus(state.error, true);
  } else if (S.finished) {
    el.note.textContent = "Fully transcribed";
  } else if (S.warming) {
    el.note.textContent = "Loading model…";
  } else {
    el.note.textContent = total ? `Transcribing ${done}/${total}` : "Transcribing…";
  }
}

// One cell per chunk. Above ~120 chunks we bucket them so the DOM and the
// gaps stay sane; a bucket lights as soon as any chunk in it lands.
function buildCells(total) {
  S.buckets = Math.min(total, 120);
  S.chunksPerBucket = total / S.buckets;
  el.meterCells.replaceChildren();
  for (let i = 0; i < S.buckets; i++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    el.meterCells.appendChild(cell);
  }
}

function paintCells(state) {
  if (!S.buckets) return;
  const done = new Set();
  for (const idx of state.cached || []) {
    done.add(Math.min(S.buckets - 1, Math.floor(idx / S.chunksPerBucket)));
  }
  const cells = el.meterCells.children;
  for (let i = 0; i < cells.length; i++) cells[i].classList.toggle("done", done.has(i));
}

function setStatus(text, isError) {
  el.status.textContent = text || "";
  el.status.classList.toggle("error", !!isError);
}

function clock(t) {
  const total = Math.max(0, Math.floor(t || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = h ? String(m).padStart(2, "0") : m;
  return `${h ? `${h}:` : ""}${mm}:${String(s).padStart(2, "0")}`;
}

function currentLabel() {
  const opt = el.profile.selectedOptions[0];
  return (opt && opt.textContent) || "model";
}

function selectedProfile() {
  const checked = el.chooser.querySelector("input:checked");
  return (checked && checked.value) || el.profile.value;
}

function syncChooser(name) {
  if (!name) return;
  el.profile.value = name;
  const radio = el.chooser.querySelector(`input[value="${CSS.escape(name)}"]`);
  if (radio) radio.checked = true;
}

function fillProfiles(profiles, current) {
  if (!el.profile.options.length) {
    for (const p of profiles) {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.label;
      el.profile.appendChild(opt);
    }
  }
  if (!el.chooser.children.length) {
    for (const p of profiles) {
      const lab = document.createElement("label");
      lab.className = "choice";
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "profile-pick";
      radio.value = p.name;
      const name = document.createElement("span");
      name.className = "choice-label";
      name.textContent = p.label;
      const detail = document.createElement("span");
      detail.className = "choice-detail";
      detail.textContent = p.detail || "";
      lab.append(radio, name, detail);
      el.chooser.appendChild(lab);
    }
    el.chooser.addEventListener("change", () => syncChooser(selectedProfile()));
  }
  syncChooser(current);
}

function showVeil({ title, file, hint, steps, bar } = {}) {
  el.empty.hidden = true;
  el.veil.hidden = false;
  el.veilTitle.textContent = title || "Preparing";
  el.veilFile.textContent = file || "";
  el.veilFile.hidden = !file;
  el.veilHint.textContent = hint || "";
  el.veilHint.hidden = !hint;
  el.steps.hidden = !steps;
  if (steps) {
    for (const li of el.steps.children) {
      li.className = steps[li.dataset.step] || "";
    }
  }
  if (bar == null) {
    el.veilBar.hidden = true;
  } else {
    el.veilBar.hidden = false;
    el.veilFill.style.width = `${Math.round(Math.min(1, Math.max(0, bar)) * 100)}%`;
  }
}

function hideVeil() {
  el.veil.hidden = true;
  if (!S.hasMedia) el.empty.hidden = false;
}

function showBanner(text) {
  el.bannerText.textContent = text;
  el.banner.hidden = false;
}

function hideBanner() {
  el.banner.hidden = true;
}

function paintPhase() {
  const converting =
    S.pendingVideo && S.playback && !S.playback.ready && !S.playback.error;
  const file = el.fileName.textContent;

  if (S.phase === "idle") {
    hideVeil();
    hideBanner();
    el.save.hidden = true;
    el.note.textContent = "No video open";
    el.note.classList.remove("ahead");
    el.panelEmpty.hidden = false;
    el.panelEmpty.textContent = "Open a video to start.";
    return;
  }

  if (S.phase === "picking") {
    showVeil({
      title: "Choose a video",
      hint: "The file panel should be in front of the browser",
    });
    el.note.textContent = "Waiting for a file…";
    el.panelEmpty.hidden = false;
    el.panelEmpty.textContent = "Waiting for a video.";
    return;
  }

  if (S.phase === "opening") {
    showVeil({
      title: "Opening",
      file,
      hint: "Extracting audio and planning chunks…",
      steps: { audio: "active", play: "", model: "" },
    });
    el.note.textContent = "Opening…";
    el.panelEmpty.hidden = false;
    el.panelEmpty.textContent = "Extracting audio…";
    return;
  }

  if (S.phase === "switching") {
    showVeil({
      title: `Switching to ${currentLabel()}`,
      file,
      hint: "Reloading models and restarting transcription…",
      steps: { audio: "done", play: "done", model: "active" },
    });
    el.note.textContent = "Switching model…";
    el.panelEmpty.hidden = false;
    el.panelEmpty.textContent = "Reloading model…";
    return;
  }

  if (converting) {
    const pct = Math.round((S.playback.progress || 0) * 100);
    const why = S.playback.reason || "video";
    showVeil({
      title: "Preparing playback",
      file,
      hint: `Converting ${why} so the browser can play it… ${pct}%`,
      steps: { audio: "done", play: "active", model: S.warming ? "active" : "" },
      bar: S.playback.progress || 0,
    });
    el.note.textContent = `Converting… ${pct}%`;
    el.panelEmpty.hidden = false;
    el.panelEmpty.textContent = "Preparing playback…";
    return;
  }

  if (S.hasMedia && !S.videoReady) {
    showVeil({
      title: "Loading video",
      file,
      hint: S.warming ? `Loading ${currentLabel()} in the background…` : "",
      steps: { audio: "done", play: "active", model: S.warming ? "active" : "" },
    });
    el.note.textContent = "Loading video…";
    return;
  }

  hideVeil();

  if (S.hasMedia && S.warming) {
    showBanner(`Loading ${currentLabel()}… first run may download the model`);
    if (!S.cues.length) {
      el.panelEmpty.hidden = false;
      el.panelEmpty.textContent = "Model is loading…";
    }
  } else {
    hideBanner();
    if (!S.cues.length) {
      el.panelEmpty.hidden = false;
      el.panelEmpty.textContent = S.hasMedia
        ? "Lines land here as they’re transcribed."
        : "Open a video to start.";
    }
  }
}

// ------------------------------------------------------------------ open

function showFallback() {
  el.open.hidden = true;
  el.emptyOpen.hidden = true;
  el.fallback.hidden = false;
  $("empty-hint").textContent = "Paste a path in the field above · local files only";
  el.fallbackPath.focus();
}

async function pick() {
  if (S.busy) return;
  if (el.fallback.hidden === false) {
    return open(el.fallbackPath.value.trim());
  }
  S.busy = true;
  S.phase = "picking";
  paintPhase();
  try {
    const res = await fetch("/api/pick", { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const { path } = await res.json();
    if (path) await open(path);
    else {
      S.phase = S.hasMedia ? "playing" : "idle";
      setStatus("");
      paintPhase();
    }
  } catch (err) {
    S.phase = S.hasMedia ? "playing" : "idle";
    setStatus(String(err.message || err), true);
    paintPhase();
  } finally {
    S.busy = false;
  }
}

async function open(path) {
  if (!path) return;
  S.phase = "opening";
  el.fileName.textContent = path.split("/").pop();
  el.fileName.title = path;
  paintPhase();
  el.open.disabled = true;
  el.emptyOpen.disabled = true;
  try {
    const res = await fetch("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, profile: selectedProfile() }),
    });
    if (!res.ok) throw new Error(await res.text());
    const info = await res.json();
    if (info.profile) syncChooser(info.profile);
    applyMedia(path, info, `/media?t=${Date.now()}`);
  } catch (err) {
    S.phase = S.hasMedia ? "playing" : "idle";
    setStatus(String(err.message || err), true);
    paintPhase();
  } finally {
    el.open.disabled = false;
    el.emptyOpen.disabled = false;
  }
}

function restore(media) {
  applyMedia(media.path, media, "/media");
}

function applyMedia(path, info, url) {
  clearCues();
  S.hasMedia = true;
  S.duration = info.duration;
  S.mediaInfo = info;
  S.playback = info.playback || { ready: true };
  S.videoReady = false;
  S.warming = true;
  S.phase = "playing";
  buildCells(info.chunks || 1);
  el.fileName.textContent = path.split("/").pop();
  el.fileName.title = path;
  el.session.hidden = false;
  if (el.fallback.hidden) el.open.hidden = false;
  el.empty.hidden = true;
  el.video.removeAttribute("src");
  backfill();
  refreshState();

  const playback = S.playback;
  if (playback.ready) {
    if (playback.error) {
      setStatus(`Couldn't convert for playback: ${playback.error}`, true);
    }
    attachVideo(url, info);
  } else {
    S.pendingVideo = url;
    paintPhase();
  }
}

function attachVideo(url, info) {
  S.pendingVideo = null;
  S.videoReady = false;
  el.video.src = url;
  el.video.load();
  setStatus(
    `${info.chunks} chunks · ${clock(info.duration)}` +
      (info.playback && info.playback.converted ? " · converted for playback" : "")
  );
  paintPhase();
}

// ----------------------------------------------------------------- wiring

el.open.onclick = pick;
el.emptyOpen.onclick = pick;
el.fallback.onsubmit = (event) => {
  event.preventDefault();
  open(el.fallbackPath.value.trim());
};

el.profile.onchange = async () => {
  syncChooser(el.profile.value);
  if (!S.hasMedia) return;
  const label = currentLabel();
  S.phase = "switching";
  S.videoReady = false;
  S.warming = true;
  paintPhase();
  try {
    const res = await fetch("/api/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: el.profile.value }),
    });
    if (!res.ok) throw new Error(await res.text());
    const info = await res.json();
    if (info.duration != null) {
      applyMedia(info.path, info, `/media?t=${Date.now()}`);
    } else {
      S.phase = "playing";
      paintPhase();
    }
  } catch (err) {
    S.phase = S.hasMedia ? "playing" : "idle";
    setStatus(String(err.message || err), true);
    paintPhase();
  }
};

for (const btn of document.querySelectorAll(".seg button")) {
  btn.onclick = () => (location.href = `/api/export?fmt=${btn.dataset.fmt}`);
}

el.video.addEventListener("loadedmetadata", () => {
  if (el.video.duration && isFinite(el.video.duration)) S.duration = el.video.duration;
  paintHead(el.video.currentTime);
  sendPlayhead(true);
});
el.video.addEventListener("canplay", () => {
  S.videoReady = true;
  paintPhase();
});
el.video.addEventListener("seeked", () => {
  S.hint = 0;
  sendPlayhead(true);
  backfill();
});

const MEDIA_ERRORS = {
  1: "loading was aborted",
  2: "a network error occurred",
  3: "the video could not be decoded",
  4: "this format isn't supported by the browser",
};
el.video.addEventListener("error", () => {
  if (!el.video.error) return;
  S.videoReady = true;
  setStatus(
    `Playback failed: ${MEDIA_ERRORS[el.video.error.code] || el.video.error.message}`,
    true
  );
  paintPhase();
});

document.addEventListener("keydown", (event) => {
  const tag = (document.activeElement || {}).tagName || "";
  const typing = /INPUT|TEXTAREA|SELECT/.test(tag);

  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "o") {
    event.preventDefault();
    pick();
    return;
  }
  if (event.code === "Space" && S.hasMedia && el.video.src && !typing && tag !== "BUTTON") {
    event.preventDefault();
    el.video.paused ? el.video.play() : el.video.pause();
  }
});

connect();
requestAnimationFrame(tick);
