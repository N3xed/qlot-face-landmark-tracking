// ONNX inference engine (onnxruntime-web), WASM-JSEP runtime (WebGPU disabled, see loadModel).
// Works without Cross-Origin-Opener/Policy headers, so it runs on plain GitHub Pages.
import { IM_SIZE, HIDDEN_DIM, defaultPrefillLandmarks } from "./config.js";

const ort = globalThis.ort;

export class Inference {
  constructor() {
    this.session = null;
    this.provider = "—";
    this.n = 0;
    this.state = { hidden: null, preds: null, valid: false };
    if (ort) {
      // ORT resolves the .mjs import and the internal .wasm fetch against different
      // bases, so a relative wasmPaths breaks one of them. Derive an ABSOLUTE dir
      // from the ort.all script's own URL; this works on any deploy root.
      let wasmBase = "./";
      try {
        const s = document.querySelector('script[src*="ort.all"]');
        if (s) wasmBase = new URL(".", s.src).href;
      } catch (e) {}
      ort.env.wasm.wasmPaths = wasmBase;
      ort.env.wasm.jsepWasm = true; // no SharedArrayBuffer required
      ort.env.wasm.numThreads = 1;
    }
  }

  get ready() {
    return !!this.session;
  }

  async loadModel(url) {
    const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
    this.session = null;
    const attempts = [];
    // WebGPU disabled: on tested hardware the EP keeps falling back to CPU at WASM-like speed.
    // if (typeof navigator !== "undefined" && navigator.gpu) attempts.push({ providers: ["WebGPU"], name: "WebGPU" });
    attempts.push({ providers: ["WASM"], name: "WASM" });
    let lastErr;
    for (const a of attempts) {
      try {
        const session = await ort.InferenceSession.create(bytes, { providers: a.providers });
        this.session = session;
        this.provider = a.name;
        this.state = { hidden: null, preds: null, valid: false };
        return a.name;
      } catch (e) {
        lastErr = e;
      }
    }
    this.session = null;
    this.provider = "failed";
    throw lastErr || new Error("could not create session");
  }

  resetState() {
    this.state = { hidden: null, preds: null, valid: false };
  }

  setNumQueries(n) {
    if (n !== this.n) {
      this.n = n;
      this.resetState();
    }
  }

  // Build a CHW float [0,1] tensor from a 224x224 RGBA canvas.
  static imageFromCanvas(canvas) {
    const ctx = canvas.getContext("2d");
    const d = ctx.getImageData(0, 0, IM_SIZE, IM_SIZE).data;
    const plane = IM_SIZE * IM_SIZE;
    const chw = new Float32Array(3 * plane);
    for (let p = 0, s = 0; p < plane; p++, s += 4) {
      chw[p] = d[s] / 255;
      chw[plane + p] = d[s + 1] / 255;
      chw[2 * plane + p] = d[s + 2] / 255;
    }
    return chw;
  }

  // Run one step. `image` is CHW Float32 (3*224*224), `queries` is flat (n*3).
  async run({ image, queries, n, gatingRadius, gatingCutoff, forwardHidden, forwardPredictions }) {
    if (!this.session) throw new Error("model not loaded");
    this.setNumQueries(n);
    const st = this.state;
    const hidden = forwardHidden && st.valid ? st.hidden : new Float32Array(n * HIDDEN_DIM);
    const landmarks = forwardPredictions && st.valid ? st.preds : defaultPrefillLandmarks(n);
    const feeds = {
      image: new ort.Tensor(image, [1, 3, IM_SIZE, IM_SIZE]),
      query_points: new ort.Tensor(queries, [1, n, 3]),
      gating_cutoff: new ort.Tensor(new Float32Array([gatingCutoff]), [1]),
      gating_radius: new ort.Tensor(new Float32Array([gatingRadius]), [1]),
      prefill_hidden_state: new ort.Tensor(hidden, [1, n, HIDDEN_DIM]),
      prefill_starting_landmarks: new ort.Tensor(landmarks, [1, n, 7]),
    };
    const t0 = performance.now();
    const out = await this.session.run(feeds);
    const durationMs = performance.now() - t0;
    const pred = out.predictions.data;
    const hiddenOut = out.hidden_state.data;
    if (forwardHidden) st.hidden = new Float32Array(hiddenOut);
    if (forwardPredictions) st.preds = new Float32Array(pred);
    st.valid = true;
    return { pred: new Float32Array(pred), hidden: new Float32Array(hiddenOut), durationMs, provider: this.provider };
  }
}
