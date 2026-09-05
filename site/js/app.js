// Application wiring: mesh editor + demo + settings panel.
import { ASSETS, GROUP_COLORS, clamp } from "./config.js";
import { MeshEditor } from "./mesh3d.js";
import { Demo } from "./demo.js";

const $ = (id) => document.getElementById(id);

class Drag {
  constructor(el) {
    this.el = el;
    this.type = el.dataset.type === "int" ? "int" : "float";
    this.decimals = el.dataset.decimals ? Number(el.dataset.decimals) : this.type === "int" ? 0 : 2;
    this.min = el.dataset.min !== undefined ? Number(el.dataset.min) : -Infinity;
    this.max = el.dataset.max !== undefined ? Number(el.dataset.max) : Infinity;
    this.step = el.dataset.step ? Number(el.dataset.step) : this.type === "int" ? 1 : 0.01;
    this.suffix = el.dataset.suffix || "";
    this.value = this.clamp(this.type === "int" ? Math.round(Number(el.dataset.value || 0)) : Number(el.dataset.value || 0));
    this.active = false;
    this.onInput = null;
    el.setAttribute("role", "slider");
    el.setAttribute("tabindex", "0");
    if (el.dataset.label) el.setAttribute("aria-label", el.dataset.label);
    this._render();
    el.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      try { el.setPointerCapture(e.pointerId); } catch (_) {}
      this.active = true;
      this._startX = e.clientX;
      this._startVal = this.value;
      el.classList.add("active");
    });
    el.addEventListener("pointermove", (e) => {
      if (!this.active) return;
      const dx = e.clientX - this._startX;
      let v;
      if (this.type === "int") v = this._startVal + (dx / 8) * this.step;
      else if (isFinite(this.min) && isFinite(this.max)) v = this._startVal + (dx / 500) * (this.max - this.min);
      else v = this._startVal + (dx / 8) * this.step;
      this._set(v, true);
    });
    const end = (e) => {
      if (!this.active) return;
      this.active = false;
      el.classList.remove("active");
      try { el.releasePointerCapture(e.pointerId); } catch (_) {}
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);
    el.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" || e.key === "ArrowUp" ? this.step : e.key === "ArrowLeft" || e.key === "ArrowDown" ? -this.step : 0;
      if (!d) return;
      e.preventDefault();
      this._set(this.value + d, true);
    });
  }
  clamp(v) { return Math.max(this.min, Math.min(this.max, v)); }
  _set(v, emit) {
    v = this.clamp(v);
    if (this.type === "int") v = Math.round(v);
    if (v === this.value) return;
    this.value = v;
    this._render();
    if (emit && this.onInput) this.onInput(v);
  }
  getValue() { return this.value; }
  setValue(v) { this._set(Number(v), false); }
  setRange(min, max) { this.min = min; this.max = max; this._set(this.clamp(this.value), false); }
  _render() {
    this.el.textContent = (this.type === "int" ? String(Math.round(this.value)) : this.value.toFixed(this.decimals)) + this.suffix;
    this.el.setAttribute("aria-valuenow", String(this.value));
    this.el.setAttribute("aria-valuemin", isFinite(this.min) ? String(this.min) : "");
    this.el.setAttribute("aria-valuemax", isFinite(this.max) ? String(this.max) : "");
  }
}

const drags = {};
function initDrags() {
  document.querySelectorAll(".drag").forEach((el) => { drags[el.id] = new Drag(el); });
}

const editor = new MeshEditor($("mesh3d"), ASSETS.faceMesh);
const demo = new Demo({
  video: $("video"),
  canvas: $("video-canvas"),
  overlay: $("video-overlay"),
  cropCanvas: $("crop-canvas"),
  onStatus: handleStatus,
});

async function boot() {
  try {
    await editor.init();
  } catch (e) {
    console.error("mesh load failed", e);
    $("mesh3d-status").textContent = "Failed to load mesh.";
  }
  editor.onchange = () => {
    $("query-count").textContent = editor.meta.n + " pts";
    if (sendEnabled) sendQueries();
  };
  await loadPresetAndSend("wflw_98");
  await demo.setModel(ASSETS.model); // bundled model
  $("model-status").textContent = ASSETS.model.split("/").pop();
  bindUI();
  setMode("add");
  updateSendButton();
  requestAnimationFrame(animateSlidersSync);
}

let sendEnabled = true;

async function loadPresetAndSend(name) {
  await editor.loadPreset(name);
  if (sendEnabled) sendQueries();
}

function sendQueries() {
  demo.setQueries(editor.getQueries());
}

function setSendEnabled(on) {
  sendEnabled = on;
  if (on) sendQueries();
  updateSendButton();
}

function updateSendButton() {
  const b = $("btn-send-queries");
  b.classList.toggle("active", sendEnabled);
  b.setAttribute("aria-pressed", String(sendEnabled));
}

// ---- UI bindings ------------------------------------------------------------
function bindUI() {
  // Presets
  $("preset-select").addEventListener("change", (e) => loadPresetAndSend(e.target.value));
  $("btn-send-queries").addEventListener("click", () => setSendEnabled(!sendEnabled));

  // Editor mode buttons (clicking the active tool disables it for pure orbit/pan)
  const modes = { "btn-add": "add", "btn-move": "move", "btn-del": "delete" };
  for (const [id, m] of Object.entries(modes)) $(id).addEventListener("click", () => setMode(editor.mode === m ? "orbit" : m));
  $("btn-clear").addEventListener("click", () => editor.clear());
  $("btn-resetview").addEventListener("click", () => editor.resetView());

  // Source
  $("btn-webcam").addEventListener("click", async () => {
    setCamError("");
    try {
      await demo.startWebcam();
      setWebcamBtn(true);
    } catch (e) {
      setCamError(camErrorMsg(e));
    }
  });
  $("btn-stop").addEventListener("click", () => { demo.stopWebcam(); setWebcamBtn(false); setCamError(""); });
  $("btn-file").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => {
    if (e.target.files[0]) {
      setCamError("");
      demo.loadFile(e.target.files[0]);
      setWebcamBtn(false);
    }
  });

  // Play / pause
  function syncPlayBtn() {
    $("btn-playpause").textContent = demo.video.paused ? "Play" : "Pause";
  }
  demo.video.addEventListener("play", syncPlayBtn);
  demo.video.addEventListener("pause", syncPlayBtn);
  $("btn-playpause").addEventListener("click", () => {
    const v = demo.video;
    if (v.paused) v.play().catch(() => {});
    else v.pause();
  });

  // Model
  $("btn-model").addEventListener("click", () => $("model-file").click());
  $("model-file").addEventListener("change", async (e) => {
    if (!e.target.files[0]) return;
    const f = e.target.files[0];
    const url = URL.createObjectURL(f);
    $("model-status").textContent = "Loading " + f.name + "…";
    await demo.setModel(url);
    $("model-status").textContent = f.name;
  });

  // Settings → demo
  const numericIds = [
    "in-iter", "in-radius", "in-cutoff",
    "t-minsize", "t-maxvel", "t-maxsizevel",
    "t-smooth-trans", "t-smooth-size",
    "t-vel-ema", "t-lost-var", "t-var-ema",
    "t-lost-grace", "t-lost-smooth", "t-margin",
    "d-cross", "d-pointscale", "d-bbox",
  ];
  for (const id of numericIds) drags[id].onInput = () => pushSettings();
  ["in-fhidden", "in-fpreds", "d-ellipses", "d-cropbox"].forEach((id) =>
    $(id).addEventListener("change", pushSettings)
  );
  $("auto-follow").addEventListener("change", (e) => {
    if (e.target.checked) {
      const c = demo.settings.crop;
      demo.tracker.manualSet(c.x, c.y, c.size);
    } else if (demo.ready && demo.video.videoWidth) {
      demo.settings.crop = {
        x: demo.tracker.crop_x,
        y: demo.tracker.crop_y,
        size: demo.tracker.crop_size,
      };
    }
    pushSettings();
  });

  // Crop drags
  const applyCrop = (key, v) => {
    const size = drags["crop-size"].getValue();
    const x = drags["crop-x"].getValue();
    const y = drags["crop-y"].getValue();
    if (key === "size") demo.manualCrop(x, y, v);
    else if (key === "x") demo.manualCrop(v, y, size);
    else demo.manualCrop(x, v, size);
    pushSettings();
  };
  drags["crop-size"].onInput = (v) => applyCrop("size", v);
  drags["crop-x"].onInput = (v) => applyCrop("x", v);
  drags["crop-y"].onInput = (v) => applyCrop("y", v);

  $("btn-recenter").addEventListener("click", () => {
    demo.recenter();
    if (!$("auto-follow").checked) $("auto-follow").checked = true;
    pushSettings();
  });

  function pushSettings() {
    demo.setSettings({
      iterations: dv("in-iter"), gatingRadius: dv("in-radius"), gatingCutoff: dv("in-cutoff"),
      forwardHidden: $("in-fhidden").checked, forwardPredictions: $("in-fpreds").checked,
      autoFollow: $("auto-follow").checked,
      minSize: dv("t-minsize"), maxVelocity: dv("t-maxvel"), maxSizeVelocity: dv("t-maxsizevel"),
      smoothingTranslation: dv("t-smooth-trans"), smoothingSize: dv("t-smooth-size"),
      velocityEmaAlpha: dv("t-vel-ema"), trackingLostVar: dv("t-lost-var"), varianceEmaAlpha: dv("t-var-ema"),
      lostGracePeriod: dv("t-lost-grace"), lostSmoothingTranslation: dv("t-lost-smooth"), marginRatio: dv("t-margin"),
      crossSize: dv("d-cross"), pointScale: dv("d-pointscale"), bboxWidth: dv("d-bbox"),
      showEllipses: $("d-ellipses").checked, showCropBox: $("d-cropbox").checked,
    });
  }

  document.querySelectorAll("[data-scroll]").forEach((a) =>
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const t = $(a.dataset.scroll.slice(1));
      if (t) t.scrollIntoView({ behavior: "smooth" });
    })
  );

  pushSettings();
}

function dv(id) { return drags[id] ? drags[id].getValue() : 0; }
const MODE_BUTTONS = { add: "btn-add", move: "btn-move", delete: "btn-del" };

function setMode(m) {
  editor.setMode(m);
  for (const [mode, id] of Object.entries(MODE_BUTTONS)) {
    const on = editor.mode === mode;
    const el = $(id);
    el.classList.toggle("active", on);
    el.setAttribute("aria-pressed", String(on));
  }
}

function setWebcamBtn(on) {
  $("btn-webcam").classList.toggle("hidden", on);
  $("btn-stop").classList.toggle("hidden", !on);
}

function setCamError(msg) {
  const el = $("webcam-error");
  el.textContent = msg || "";
  el.classList.toggle("show", !!msg);
}
function camErrorMsg(e) {
  switch (e && e.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Camera permission was denied. Allow access in your browser and try again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No camera was found on this device.";
    case "NotReadableError":
    case "TrackStartError":
      return "The camera is in use by another application. Close it and try again.";
    case "SecurityError":
      return "Camera access is blocked here — use HTTPS or localhost.";
    default:
      return "Could not start the camera: " + ((e && e.message) || "unknown error");
  }
}

// Live-sync crop drags to the active crop (tracker when auto-follow is on,
// manual otherwise). Skipped while the user is dragging one.
function animateSlidersSync() {
  requestAnimationFrame(animateSlidersSync);
  if (drags["crop-size"].active || drags["crop-x"].active || drags["crop-y"].active) return;
  const vw = demo.video.videoWidth || demo.tracker._width || 0;
  const vh = demo.video.videoHeight || demo.tracker._height || 0;
  const c = demo._lastStatusCrop || demo.settings.crop;
  if (!vw || !vh || !c) return;
  const size = Math.round(clamp(c.size, 8, Math.min(vw, vh)));
  drags["crop-size"].setRange(8, Math.min(vw, vh));
  drags["crop-size"].setValue(size);
  drags["crop-x"].setRange(0, Math.max(0, vw - size));
  drags["crop-x"].setValue(c.x);
  drags["crop-y"].setRange(0, Math.max(0, vh - size));
  drags["crop-y"].setValue(c.y);
}

function handleStatus(s) {
  demo._lastStatusCrop = s.crop;
  $("btn-playpause").classList.toggle("hidden", s.source === "none");
  $("st-fps").textContent = s.ready && s.modelReady ? s.fps.toFixed(1) : "—";
  $("st-ms").textContent = s.ready && s.modelReady && s.ms ? s.ms.toFixed(0) + " ms" : "—";
  $("st-provider").textContent = s.provider;
  const locked = s.locked;
  const st = $("st-state");
  if (s.auto) {
    st.textContent = locked ? "TRACKING" : "LOST";
    st.className = "stat-value state " + (locked ? "locked" : "lost");
  } else {
    st.textContent = "MANUAL";
    st.className = "stat-value state manual";
  }
  $("st-var").textContent = s.meanVar != null ? s.meanVar.toFixed(1) : "—";
  $("st-source").textContent = s.source === "none" ? "no source" : s.source;
}

// Build the mesh legend chips from group colors.
function buildLegend() {
  const el = $("mesh-legend");
  el.innerHTML = "";
  for (const [g, c] of Object.entries(GROUP_COLORS)) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<i style="background:${c}"></i>${g}`;
    el.appendChild(chip);
  }
}

buildLegend();
initDrags();
drags["mesh-radius"].onInput = (v) => editor.setRadiusScale(v);
boot();
