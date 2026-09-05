// Auto-following crop tracker: exact port of src/demo.py Tracker, minus the face detector.
// Locked state is driven purely by landmark-prediction bbox; lost state pans out;
// relock is automatic when prediction variance recovers.
import { clamp } from "./config.js";

const DEFAULTS = {
  margin_ratio: 0.12,
  tracking_lost_var: 40.0,
  max_velocity: 40.0,
  max_size_velocity: 20.0,
  smoothing_translation: 0.3,
  smoothing_size: 0.12,
  lost_smoothing_translation: 0.05,
  velocity_ema_alpha: 0.4,
  min_size: 30,
  variance_ema_alpha: 0.5,
  lost_grace_period: 60,
  lost_velocity_decay: 0.5,
};

export class Tracker {
  constructor({ width = 640, height = 480 }) {
    this._width = width;
    this._height = height;
    this.max_size = Math.min(width, height);
    this.crop_size = this.max_size;
    this.crop_x = 0;
    this.crop_y = 0;
    Object.assign(this, DEFAULTS);
    this.velocity_x = 0;
    this.velocity_y = 0;
    this.size_velocity = 0;
    this.is_lost = true;
    this.mean_var = null;
    this._frames_lost_counter = 0;
    this._reinit_mean_var_next_frame = true;
  }

  setVideoSize(w, h) {
    this._width = w;
    this._height = h;
    this.max_size = Math.min(w, h);
    this.reset();
  }

  reset() {
    this.crop_size = this.max_size;
    this.crop_x = 0;
    this.crop_y = 0;
    this.is_lost = true;
    this.mean_var = null;
    this._frames_lost_counter = 0;
    this.velocity_x = this.velocity_y = this.size_velocity = 0;
    this._reinit_mean_var_next_frame = true;
  }

  resetVelocity() {
    this.velocity_x = this.velocity_y = this.size_velocity = 0;
  }

  _smoothSaturate(v, max) {
    if (max <= 0) return 0;
    return v / (1 + Math.abs(v) / max);
  }

  _smoothStep(error) {
    const maxError = Math.min(this._width, this._height) / 2;
    return error / (1 + Math.abs(error) / maxError);
  }

  // Manual crop override (dragging sliders / recenters): adopt it and re-baseline variance.
  manualSet(x, y, size) {
    this.crop_x = clamp(x, 0, this._width);
    this.crop_y = clamp(y, 0, this._height);
    this.crop_size = clamp(size, this.min_size, this.max_size);
    this.resetVelocity();
    this._reinit_mean_var_next_frame = true;
  }

  recenter() {
    this.crop_x = Math.max(0, (this._width - this.max_size) / 2);
    this.crop_y = Math.max(0, (this._height - this.max_size) / 2);
    this.crop_size = this.max_size;
    this.resetVelocity();
    this._reinit_mean_var_next_frame = true;
  }

  // sample: { n, frame: Float32Array(n*2), meanGenvar: number }
  track(sample) {
    const nextVar = sample.meanGenvar;
    if (this._reinit_mean_var_next_frame || this.mean_var == null) {
      this.mean_var = nextVar;
      this._reinit_mean_var_next_frame = false;
    } else {
      this.mean_var = (1 - this.variance_ema_alpha) * this.mean_var + this.variance_ema_alpha * nextVar;
    }

    if (!this.is_lost) {
      if (this.mean_var > this.tracking_lost_var) {
        this._frames_lost_counter++;
        this.velocity_x *= this.lost_velocity_decay;
        this.velocity_y *= this.lost_velocity_decay;
        this.size_velocity *= this.lost_velocity_decay;
      } else {
        this._frames_lost_counter = 0;
      }
      if (this._frames_lost_counter > this.lost_grace_period) {
        this.resetVelocity();
        this.is_lost = true;
        this._frames_lost_counter = 0;
      }
    }
    if (this.mean_var <= this.tracking_lost_var) this.is_lost = false;

    if (this.is_lost) {
      // Pan out toward a centered full-size window (no face detector).
      this.crop_size += this._smoothStep(this.max_size - this.crop_size) * this.smoothing_size;
      this.crop_x += this._smoothStep(this._width / 5 - this.crop_x) * this.lost_smoothing_translation;
      this.crop_y += this._smoothStep(0.0 - this.crop_y) * this.lost_smoothing_translation;
      this.crop_size = clamp(this.crop_size, this.min_size, this.max_size);
    } else {
      // Follow the landmark-prediction bounding box.
      const { n, frame } = sample;
      if (n > 0) {
        let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
        for (let i = 0; i < n; i++) {
          const x = frame[i * 2], y = frame[i * 2 + 1];
          if (x < minx) minx = x;
          if (x > maxx) maxx = x;
          if (y < miny) miny = y;
          if (y > maxy) maxy = y;
        }
        const w = maxx - minx, h = maxy - miny;
        const cx = minx + w / 2, cy = miny + h / 2;
        const pw = w * (1 + this.margin_ratio * 2);
        const ph = h * (1 + this.margin_ratio * 2);
        let target = Math.max(pw, ph, this.min_size);
        target = Math.min(target, this.max_size);
        const targetX = cx - target / 2;
        const targetY = cy - target / 2;

        const baseSize = (target - this.crop_size) * this.smoothing_size;
        this.size_velocity = (1 - this.velocity_ema_alpha) * this.size_velocity + this.velocity_ema_alpha * baseSize;
        this.crop_size = clamp(
          this.crop_size + this._smoothSaturate(this.size_velocity, this.max_size_velocity),
          this.min_size,
          this.max_size
        );

        const bsx = (targetX - this.crop_x) * this.smoothing_translation;
        const bsy = (targetY - this.crop_y) * this.smoothing_translation;
        this.velocity_x = (1 - this.velocity_ema_alpha) * this.velocity_x + this.velocity_ema_alpha * bsx;
        this.velocity_y = (1 - this.velocity_ema_alpha) * this.velocity_y + this.velocity_ema_alpha * bsy;
        this.crop_x += this._smoothSaturate(this.velocity_x, this.max_velocity);
        this.crop_y += this._smoothSaturate(this.velocity_y, this.max_velocity);
      }
    }
  }
}
