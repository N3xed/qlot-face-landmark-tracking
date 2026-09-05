// Central configuration and shared helpers for the site.
export const ASSETS = {
  modelSvg: "assets/model.svg",
  updateSvg: "assets/update_prediction.svg",
  videoPlaceholder: "assets/video_placeholder.png",
  paperPdf: "assets/qlot-paper.pdf",
  supplementPdf: "assets/qlot-supplement.pdf",
  faceMesh: "data/face_mesh.obj",
  model: "models/qlot-final.onnx",
  queryPresets: {
    wflw_98: "data/queries_wflw_98.json",
    synth_70: "data/queries_synth_70.json",
    ibug_68: "data/queries_ibug_68.json",
  },
};

// Colors used to distinguish landmark groups in the 3D editor and legend.
export const GROUP_COLORS = {
  jaw: "#e05c5c",
  brows: "#e0a03c",
  nose: "#5cb85c",
  eyes: "#4ca6e0",
  mouth: "#c76ce0",
  pupils: "#f2f2f2",
};

export const IM_SIZE = 224; // model input / output pixel space
export const HIDDEN_DIM = 128;

// Default per-query prefill when there is no previous prediction (pixel space).
export function defaultPrefillLandmark() {
  const ln = Math.log(IM_SIZE / 2); // ln(112)
  return new Float32Array([IM_SIZE / 2 - 0.5, IM_SIZE / 2 - 0.5, ln, ln, 0, 0, 0]);
}

// Build the (N,7) prefill_starting_landmarks tensor data for the first frame.
export function defaultPrefillLandmarks(n) {
  const one = defaultPrefillLandmark();
  const out = new Float32Array(n * 7);
  for (let i = 0; i < n; i++) out.set(one, i * 7);
  return out;
}

export function groupColor(name) {
  return GROUP_COLORS[name] || "#cccccc";
}

// Clamp helper.
export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}
