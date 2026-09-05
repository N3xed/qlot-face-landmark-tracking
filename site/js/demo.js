// Live demo: webcam/file source, crop extraction, ONNX inference, tracker, overlay.
import { clamp } from "./config.js";
import { Inference } from "./inference.js";
import { Tracker } from "./tracker.js";

export class Demo {
  constructor({ video, canvas, overlay, cropCanvas, snapshotCanvas, onStatus }) {
    this.video = video;
    this.canvas = canvas;
    this.overlay = overlay;
    this.cropCanvas = cropCanvas;
    // A dedicated buffer (never the crop canvas) that holds the frozen base frame.
    this.snapshot = snapshotCanvas || document.createElement("canvas");
    this.ctx = canvas.getContext("2d");
    this.octx = overlay.getContext("2d");
    this._cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });
    this._snapCtx = this.snapshot.getContext("2d", { willReadFrequently: true });
    this.onStatus = onStatus;
    this._cropMoved = false;
    this._cropCentered = false;

    this.inference = new Inference();
    this.tracker = new Tracker({ width: 640, height: 480 });

    this.settings = {
      modelUrl: "",
      iterations: 1,
      gatingRadius: 0.5,
      gatingCutoff: 0.05,
      forwardHidden: true,
      forwardPredictions: true,
      autoFollow: false,
      crop: { x: 0, y: 0, size: 128 },
      minSize: 30,
      maxVelocity: 40,
      maxSizeVelocity: 20,
      smoothingTranslation: 0.12,
      smoothingSize: 0.12,
      velocityEmaAlpha: 0.4,
      trackingLostVar: 80,
      varianceEmaAlpha: 0.5,
      lostGracePeriod: 60,
      lostSmoothingTranslation: 0.05,
      marginRatio: 0.12,
      crossSize: 3,
      pointScale: 1,
      bboxWidth: 1,
      showEllipses: false,
      showCropBox: true,
    };

    this.queries = null; // { pts: Float32Array, n, meta }
    this.stream = null;
    this.fileUrl = null;
    this.source = "none";
    this.ready = false; // video has frames
    this.busy = false;
    this.fps = 0;
    this.msAvg = 0;
    this._fpsAcc = 0;
    this._fpsCount = 0;
    this._fpsT = 0;
    this._fpsT0 = 0;
    this._lastStatus = 0;
    this._pausedDirty = false;
    this._lastCrop = null;
    this._lastSample = null;

    this._bindVideo();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  _bindVideo() {
    this.video.addEventListener("loadedmetadata", () => {
      this.syncSize();
      this.ready = true;
      this._pausedDirty = true;
    });
    this.video.addEventListener("resize", () => {
      this.syncSize();
      this._pausedDirty = true;
    });
    this.video.addEventListener("loadeddata", () => {
      // Default the manual crop to a centered, face-sized window so the model
      // sees the face instead of the top-left corner. Skipped once the user has
      // moved the crop sliders themselves.
      if (!this._cropMoved) this._centerCrop();
      this._pausedDirty = true;
    });
    this.video.addEventListener("play", () => {
      this.ready = true;
      this._pausedDirty = true;
    });
    this.video.addEventListener("pause", () => (this._pausedDirty = true));
  }

  syncSize() {
    const w = this.video.videoWidth;
    const h = this.video.videoHeight;
    if (!w || !h) return false;
    const changed =
      w !== this.canvas.width || h !== this.canvas.height ||
      w !== this.tracker._width || h !== this.tracker._height;
    if (!changed) return true;
    this.canvas.width = this.overlay.width = w;
    this.canvas.height = this.overlay.height = h;
    if (this.snapshot !== this.cropCanvas) {
      this.snapshot.width = w;
      this.snapshot.height = h;
    }
    this.tracker.setVideoSize(w, h);
    this._cropMoved = false;
    this._cropCentered = false;
    this._lastCrop = null;
    this._lastSample = null;
    this._pausedDirty = true;
    return true;
  }

  _centerCrop() {
    if (this._cropCentered) return;
    const w = this.video.videoWidth, h = this.video.videoHeight;
    if (!w || !h) return;
    const size = Math.round(Math.min(w, h) * 0.6);
    const x = Math.round((w - size) / 2);
    const y = Math.round((h - size) / 2);
    this.settings.crop = { x, y, size };
    if (this.settings.autoFollow) this.tracker.manualSet(x, y, size);
    this._cropCentered = true;
  }

  // ---- source control -------------------------------------------------------
  async startWebcam() {
    this._clearSource();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    this.stream = stream;
    this.video.srcObject = stream;
    this.source = "webcam";
    this.video.load();
    await this.video.play().catch(() => {});
  }

  stopWebcam() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this._clearSource();
  }

  loadFile(file) {
    this._clearSource();
    this.fileUrl = URL.createObjectURL(file);
    this.video.src = this.fileUrl;
    this.video.muted = true;
    this.video.loop = true;
    this.source = "file";
    this.video.load();
    this.video.play().catch(() => {});
  }

  _clearSource() {
    this.video.pause();
    this.video.srcObject = null;
    this.video.removeAttribute("src");
    if (this.fileUrl) {
      URL.revokeObjectURL(this.fileUrl);
      this.fileUrl = null;
    }
    this.source = "none";
    this.ready = false;
    this._pausedDirty = true;
    this._cropMoved = false;
    this._cropCentered = false;
    this._lastCrop = null;
    this._lastSample = null;
  }

  // ---- external config ------------------------------------------------------
  setSettings(patch) {
    Object.assign(this.settings, patch);
    const t = this.tracker;
    t.min_size = this.settings.minSize;
    t.max_velocity = this.settings.maxVelocity;
    t.max_size_velocity = this.settings.maxSizeVelocity;
    t.smoothing_translation = this.settings.smoothingTranslation;
    t.smoothing_size = this.settings.smoothingSize;
    t.velocity_ema_alpha = this.settings.velocityEmaAlpha;
    t.tracking_lost_var = this.settings.trackingLostVar;
    t.variance_ema_alpha = this.settings.varianceEmaAlpha;
    t.lost_grace_period = this.settings.lostGracePeriod;
    t.lost_smoothing_translation = this.settings.lostSmoothingTranslation;
    t.margin_ratio = this.settings.marginRatio;
    this._pausedDirty = true;
  }

  async setModel(url) {
    if (url === this.settings.modelUrl && this.inference.ready) return;
    this.settings.modelUrl = url;
    try {
      await this.inference.loadModel(url);
      this.inference.resetState();
    } catch (e) {
      console.error("model load failed", e);
    }
    this._pausedDirty = true;
  }

  setQueries(q) {
    this.queries = q ? { pts: new Float32Array(q.pts), n: q.n, meta: q.meta } : null;
    this.inference.setNumQueries(q ? q.n : 0);
    this._pausedDirty = true;
  }

  manualCrop(x, y, size) {
    this._cropMoved = true;
    this.settings.crop = { x, y, size };
    if (this.settings.autoFollow) this.tracker.manualSet(x, y, size);
    this._pausedDirty = true;
  }

  recenter() {
    this.tracker.recenter();
  }

  dispose() {
    this.stopWebcam();
  }

  // ---- frame loop -----------------------------------------------------------
  _loop() {
    requestAnimationFrame(this._loop);
    if (this.busy) return;
    const v = this.video;
    if (!this.ready || !v.videoWidth || v.readyState < 2) {
      this._drawIdle();
      this._status();
      return;
    }
    if (v.paused && !this._pausedDirty) return;
    this.syncSize();
    // Fallback in case the "loadeddata" event did not fire (e.g. headless).
    if (!this._cropCentered && !this._cropMoved) this._centerCrop();
    this.busy = true;
    this._frame()
      .catch((e) => console.error("frame error", e))
      .finally(() => {
        this.busy = false;
        if (v.paused) this._pausedDirty = false;
      });
  }

  async _frame() {
    const v = this.video;
    const vw = v.videoWidth, vh = v.videoHeight;
    this.syncSize();
    if (!this._cropCentered && !this._cropMoved) this._centerCrop();
    const s = this.settings;

    let crop;
    if (s.autoFollow) crop = { x: this.tracker.crop_x, y: this.tracker.crop_y, size: this.tracker.crop_size };
    else crop = { ...s.crop };
    crop = this._clampCrop(crop, vw, vh);

    // Freeze the current frame into the snapshot so the base image and the
    // landmark overlay come from the same instant (the video otherwise advances
    // while inference is awaited, making the overlay lag the live picture).
    if (vw && vh) {
      if (this.snapshot.width !== vw || this.snapshot.height !== vh) {
        this.snapshot.width = vw;
        this.snapshot.height = vh;
      }
      this._snapCtx.drawImage(v, 0, 0, vw, vh);
      this.ctx.drawImage(this.snapshot, 0, 0, vw, vh);
    }

    // Inference (only when a model is loaded and we have queries).
    let sample = null;
    let ms = 0;
    if (this.inference.ready && this.queries && this.queries.n > 0) {
      this._extractCrop(crop);
      const image = Inference.imageFromCanvas(this.cropCanvas);
      const res = await this.inference.run({
        image,
        queries: this.queries.pts,
        n: this.queries.n,
        gatingRadius: s.gatingRadius,
        gatingCutoff: s.gatingCutoff,
        forwardHidden: s.forwardHidden,
        forwardPredictions: s.forwardPredictions,
      });
      ms = res.durationMs;
      sample = this._makeSample(res.pred, crop, vw, vh);
    }

    this._lastCrop = crop;
    this._lastSample = sample || this._lastSample;
    if (s.autoFollow && sample && !v.paused) this.tracker.track(sample);
    this._drawOverlay(crop, this._lastSample, vw, vh);
    if (!v.paused) this._fpsTick(ms);
    this._status(v.paused ? 0 : ms);
  }

  _clampCrop(c, vw, vh) {
    const size = clamp(c.size, 8, Math.min(vw, vh));
    const x = clamp(c.x, 0, Math.max(0, vw - size));
    const y = clamp(c.y, 0, Math.max(0, vh - size));
    return { x, y, size };
  }

  _extractCrop(crop) {
    const c = this._cropCtx;
    // Draw from the snapshot (the same oriented frame shown to the user) rather
    // than the raw <video> element, so rotated/portrait sources stay aligned.
    const vw = this.snapshot.width, vh = this.snapshot.height;
    c.fillStyle = "#000";
    c.fillRect(0, 0, 224, 224);
    const sx = clamp(crop.x, 0, vw);
    const sy = clamp(crop.y, 0, vh);
    const sw = clamp(crop.size, 1, vw - sx);
    const sh = clamp(crop.size, 1, vh - sy);
    c.drawImage(this.snapshot, sx, sy, sw, sh, 0, 0, 224, 224);
  }

  _makeSample(pred, crop, vw, vh) {
    const n = this.queries.n;
    const frame = new Float32Array(n * 2);
    const genvar = new Float32Array(n);
    const k = crop.size / 224;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const x = pred[i * 7], y = pred[i * 7 + 1];
      frame[i * 2] = crop.x + x * k;
      frame[i * 2 + 1] = crop.y + y * k;
      const lsx = pred[i * 7 + 2], lsy = pred[i * 7 + 3], rho = Math.tanh(pred[i * 7 + 4]);
      const gv = Math.exp(lsx + lsy) * Math.sqrt(Math.max(1 - rho * rho, 1e-12));
      genvar[i] = gv;
      sum += gv;
    }
    return { n, frame, genvar, meanGenvar: sum / Math.max(n, 1), pred, crop };
  }

  _fpsTick(ms) {
    const now = performance.now();
    if (this._fpsT0 === 0) {
      this._fpsT0 = now;
      this._fpsT = now;
    }
    this._fpsAcc += ms;
    this._fpsCount++;
    if (now - this._fpsT >= 500) {
      this.fps = this._fpsCount ? (this._fpsCount * 1000) / (now - this._fpsT0) : 0;
      this.msAvg = this._fpsAcc / Math.max(this._fpsCount, 1);
      this._fpsCount = 0;
      this._fpsAcc = 0;
      this._fpsT0 = now;
      this._fpsT = now;
    }
  }

  _drawIdle() {
    const w = this.canvas.width || 640, h = this.canvas.height || 480;
    this.ctx.fillStyle = "#0e1320";
    this.ctx.fillRect(0, 0, w, h);
    this.ctx.fillStyle = "#66788f";
    this.ctx.font = `${Math.max(14, w / 30)}px system-ui, sans-serif`;
    this.ctx.textAlign = "center";
    this.ctx.fillText("No source — start the webcam or load a video", w / 2, h / 2);
    this.octx.clearRect(0, 0, w, h);
  }

  _drawOverlay(crop, sample, vw, vh) {
    const c = this.octx;
    const s = this.settings;
    c.clearRect(0, 0, vw, vh);

    if (s.showCropBox) {
      c.strokeStyle = s.autoFollow && sample ? (this.tracker.is_lost ? "#ff6b6b" : "#54d68a") : "#4c8dff";
      c.lineWidth = Math.max(1, 2 * (s.bboxWidth || 1));
      c.strokeRect(crop.x, crop.y, crop.size, crop.size);
    }

    if (sample) {
      const ps = s.pointScale || 1;
      const cs = (s.crossSize || 1) * ps;
      c.strokeStyle = "#4c8dff";
      c.lineWidth = Math.max(1, (s.crossSize || 1) * 0.5 * ps);
      for (let i = 0; i < sample.n; i++) {
        const x = sample.frame[i * 2], y = sample.frame[i * 2 + 1];
        c.beginPath();
        c.moveTo(x - cs, y);
        c.lineTo(x + cs, y);
        c.moveTo(x, y - cs);
        c.lineTo(x, y + cs);
        c.stroke();
        if (s.showEllipses) this._ellipse(c, sample, i, crop, ps);
      }
    }
  }

  _ellipse(c, sample, i, crop, ps = 1) {
    const p = sample.pred;
    const sx = Math.exp(p[i * 7 + 2]);
    const sy = Math.exp(p[i * 7 + 3]);
    const r = Math.tanh(p[i * 7 + 4]);
    const a = sx * sx, b = sy * sy, cc = r * sx * sy;
    const k = (crop.size / 224) * 2; // 2-sigma
    const trace = a + b;
    const det = Math.max(a * b - cc * cc, 1e-9);
    const e1 = (trace + Math.sqrt(Math.max(trace * trace - 4 * det, 0))) / 2;
    const e2 = (trace - Math.sqrt(Math.max(trace * trace - 4 * det, 0))) / 2;
    const ang = 0.5 * Math.atan2(2 * cc, a - b);
    const x = sample.frame[i * 2], y = sample.frame[i * 2 + 1];
    c.save();
    c.translate(x, y);
    c.rotate(ang);
    c.beginPath();
    c.ellipse(0, 0, Math.max(k * Math.sqrt(Math.max(e1, 0)), 1), Math.max(k * Math.sqrt(Math.max(e2, 0)), 1), 0, 0, Math.PI * 2);
    c.strokeStyle = "rgba(255,80,80,0.85)";
    c.lineWidth = Math.max(1.5, ps * 1.6);
    c.stroke();
    c.restore();
  }

  _status(ms) {
    const now = performance.now();
    if (now - this._lastStatus < 100) return;
    this._lastStatus = now;
    const t = this.tracker;
    const crop = this.settings.autoFollow
      ? { x: t.crop_x, y: t.crop_y, size: t.crop_size }
      : this.settings.crop;
    this.onStatus &&
      this.onStatus({
        ready: this.ready,
        source: this.source,
        provider: this.inference.provider,
        auto: this.settings.autoFollow,
        modelReady: this.inference.ready,
        fps: this.fps,
        ms: this.msAvg,
        locked: !t.is_lost,
        meanVar: t.mean_var,
        crop,
        videoW: this.video.videoWidth,
        videoH: this.video.videoHeight,
        queryCount: this.queries ? this.queries.n : 0,
      });
  }
}
