// 3D face-mesh viewer + query-point editor (three.js, full-resolution OBJ).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { ASSETS, GROUP_COLORS } from "./config.js";

const CUSTOM_COLOR = "#ffd24c";

export class MeshEditor {
  constructor(container, objUrl) {
    this.container = container;
    this.objUrl = objUrl;
    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this.controls = null;
    this.mesh = null;
    this.pointsLayer = null;
    this.queryPoints = null;
    this.raycaster = new THREE.Raycaster();
    this.maxDim = 1;
    this.radiusScale = 1;
    this.mode = "add";
    this.selected = -1;
    this.dragging = false;
    this.down = null; // {x,y,t}
    // current queries as a flat Float32Array (N*3) + group name per index
    this.pts = new Float32Array(0);
    this.groups = [];
    this.meta = { name: "custom", n: 0, indices: {} };
    this.ready = false;
  }

  async init() {
    const w = this.container.clientWidth || 420;
    const h = this.container.clientHeight || 360;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0f16);
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.05, 100);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;

    // Lights are unnecessary for MeshBasicMaterial, but keep a subtle grid for depth.
    this.grid = new THREE.GridHelper(4, 20, 0x243044, 0x182231);
    this.grid.position.y = -1.7;
    this.scene.add(this.grid);

    await this.loadMesh();
    this.bindPointer();
    this.ready = true;
    this._resizeObserver = new ResizeObserver(() => this.onResize());
    this._resizeObserver.observe(this.container);
    this.animate();
  }

  loadMesh() {
    return new Promise((resolve, reject) => {
      new OBJLoader().load(
        this.objUrl,
        (obj) => {
          // OBJLoader.parse returns a Group; the actual mesh is a child. Resolve it.
          let geometry = obj && obj.geometry ? obj.geometry : null;
          if (!geometry && obj) obj.traverse((o) => { if (!geometry && o.isMesh && o.geometry) geometry = o.geometry; });
          if (!geometry) { reject(new Error("mesh: no geometry found in OBJ")); return; }

          geometry.computeBoundingBox();
          geometry.computeBoundingSphere();
          const box = geometry.boundingBox;
          const center = new THREE.Vector3();
          box.getCenter(center);
          const size = new THREE.Vector3();
          box.getSize(size);
          const maxDim = Math.max(size.x, size.y, size.z) || 1;

          // Solid, faint head for depth.
          this.mesh = new THREE.Mesh(
            geometry,
            new THREE.MeshBasicMaterial({ color: 0x46587a, side: THREE.DoubleSide, transparent: true, opacity: 0.28 })
          );
          this.scene.add(this.mesh);

          // Dense vertex point cloud = the visible full-resolution surface.
          const pos = geometry.getAttribute("position");
          const ptsGeo = new THREE.BufferGeometry();
          ptsGeo.setAttribute("position", pos);
          if (geometry.index) ptsGeo.setIndex(geometry.index);
          this.pointsLayer = new THREE.Points(
            ptsGeo,
            new THREE.PointsMaterial({ color: 0x8fa6c8, size: 0.006, sizeAttenuation: true, transparent: true, opacity: 0.55 })
          );
          this.scene.add(this.pointsLayer);

          // Query points rendered as selectable spheres (InstancedMesh).
          this.maxDim = maxDim;
          this.sphereRadius = maxDim * 0.005;
          const sGeo = new THREE.SphereGeometry(1, 18, 14);
          const sMat = new THREE.MeshBasicMaterial();
          this.queryPoints = new THREE.InstancedMesh(sGeo, sMat, 2048);
          this.queryPoints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
          this.queryPoints.count = 0;
          this.queryPoints.frustumCulled = false;
          this.scene.add(this.queryPoints);

          // Camera framing: look at the head, slightly above and in front.
          this.center = center;
          this.fitRadius = (maxDim * 1.6) / 2;
          const dist = this.fitRadius / Math.tan((this.camera.fov * Math.PI) / 360);
          this.camera.position.set(center.x, center.y + 0.12 * maxDim, center.z + dist);
          this.controls.target.copy(center);
          this.controls.update();
          resolve();
        },
        undefined,
        (err) => reject(err)
      );
    });
  }

  ndcFromEvent(e) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
  }

  surfacePoint(e) {
    if (!this.mesh) return null;
    this.raycaster.far = Infinity;
    this.raycaster.setFromCamera(this.ndcFromEvent(e), this.camera);
    const hits = this.raycaster.intersectObject(this.mesh, false);
    if (hits.length > 0 && hits[0].point) return hits[0].point.clone();
    return null;
  }

  _queryVisible(i) {
    const p = new THREE.Vector3(this.pts[i * 3], this.pts[i * 3 + 1], this.pts[i * 3 + 2]);
    const dist = this.camera.position.distanceTo(p);
    if (dist < 1e-5) return true;
    const dir = p.clone().sub(this.camera.position).normalize();
    this.raycaster.ray.origin.copy(this.camera.position);
    this.raycaster.ray.direction.copy(dir);
    this.raycaster.far = dist - this.maxDim * 0.01;
    const hits = this.raycaster.intersectObject(this.mesh, false);
    this.raycaster.far = Infinity;
    return !hits.length;
  }

  pickQuery(e, threshold = 26) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    let best = -1;
    let bestD = threshold;
    let bestCam = Infinity;
    const v = new THREE.Vector3();
    for (let i = 0; i < this.meta.n; i++) {
      v.set(this.pts[i * 3], this.pts[i * 3 + 1], this.pts[i * 3 + 2]);
      if (!this._queryVisible(i)) continue;
      const cam = this.camera.position.distanceTo(v);
      v.project(this.camera);
      if (v.z > 1) continue;
      const sx = (v.x * 0.5 + 0.5) * rect.width;
      const sy = (-v.y * 0.5 + 0.5) * rect.height;
      const d = Math.hypot(sx - x, sy - y);
      if (d < bestD || (d === bestD && cam < bestCam)) {
        bestD = d;
        bestCam = cam;
        best = i;
      }
    }
    return best;
  }

  bindPointer() {
    const el = this.renderer.domElement;
    el.addEventListener("pointerdown", (e) => {
      this.down = { x: e.clientX, y: e.clientY, t: performance.now() };
      if (this.mode === "move") {
        const i = this.pickQuery(e);
        if (i >= 0) {
          this.selected = i;
          this.dragging = true;
          this.controls.enabled = false;
          try { el.setPointerCapture(e.pointerId); } catch (_) {}
          this.rebuildQueries();
          this.emitChange();
        }
      }
    });
    el.addEventListener("pointermove", (e) => {
      if (this.dragging && this.selected >= 0) {
        const p = this.surfacePoint(e);
        if (p) {
          const i = this.selected * 3;
          const old = new THREE.Vector3(this.pts[i], this.pts[i + 1], this.pts[i + 2]);
          if (p.distanceTo(old) < this.maxDim * 0.25) {
            this.pts[i] = p.x;
            this.pts[i + 1] = p.y;
            this.pts[i + 2] = p.z;
            this.rebuildQueries();
            this.emitChange();
          }
        }
      }
    });
    el.addEventListener("pointerup", (e) => {
      try { el.releasePointerCapture(e.pointerId); } catch (_) {}
      const moved = this.down ? Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) : 0;
      this.dragging = false;
      this.controls.enabled = true;
      this.down = null;
      if (moved > 6) return; // was an orbit / drag gesture
      if (this.mode === "add") {
        const p = this.surfacePoint(e);
        if (p) this.addQuery(p);
      } else if (this.mode === "delete") {
        const i = this.pickQuery(e);
        if (i >= 0) this.deleteQuery(i);
      } else if (this.mode === "move") {
        const i = this.pickQuery(e);
        this.selected = i >= 0 ? i : -1;
        this.rebuildQueries();
        this.emitChange();
      }
    });
    el.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  addQuery(p) {
    this.pts = grow(this.pts, 3);
    this.pts[this.pts.length - 3] = p.x;
    this.pts[this.pts.length - 2] = p.y;
    this.pts[this.pts.length - 1] = p.z;
    this.groups.push("custom");
    this.rebuildQueries();
    this.emitChange();
  }

  deleteQuery(i) {
    const out = new Float32Array(this.pts.length - 3);
    out.set(this.pts.subarray(0, i * 3), 0);
    out.set(this.pts.subarray(i * 3 + 3), i * 3);
    this.pts = out;
    this.groups.splice(i, 1);
    this.rebuildQueries();
    this.emitChange();
  }

  clear() {
    this.pts = new Float32Array(0);
    this.groups = [];
    this.meta = { name: "empty", n: 0, indices: {} };
    this.selected = -1;
    this.rebuildQueries();
    this.emitChange();
  }

  async loadPreset(name) {
    const data = await (await fetch(ASSETS.queryPresets[name])).json();
    this.pts = new Float32Array(data.points.flat());
    this.groups = new Array(this.pts.length / 3).fill("custom");
    const indices = {};
    for (const [g, list] of Object.entries(data.indices || {})) indices[g] = list.slice();
    for (const [g, list] of Object.entries(indices)) {
      for (const idx of list) this.groups[idx] = g;
    }
    this.meta = { name, n: this.pts.length / 3, indices };
    this.selected = -1;
    this.rebuildQueries();
    this.emitChange();
  }

  setQueriesFromExternal(pts, meta) {
    this.pts = new Float32Array(pts);
    this.groups = new Array(pts.length / 3).fill("custom");
    this.meta = meta;
    this.rebuildQueries();
    this.emitChange();
  }

  rebuildQueries() {
    if (!this.queryPoints) return;
    const n = this.pts.length / 3;
    this.meta.n = n;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const pos = new THREE.Vector3();
    const scl = new THREE.Vector3();
    const tmp = new THREE.Color();
    for (let i = 0; i < n; i++) {
      pos.set(this.pts[i * 3], this.pts[i * 3 + 1], this.pts[i * 3 + 2]);
      const r = i === this.selected ? this.sphereRadius * this.radiusScale * 1.7 : this.sphereRadius * this.radiusScale;
      scl.setScalar(r);
      m.compose(pos, q, scl);
      this.queryPoints.setMatrixAt(i, m);
      const c = this.groups[i] === "custom" ? CUSTOM_COLOR : GROUP_COLORS[this.groups[i]] || CUSTOM_COLOR;
      tmp.set(i === this.selected ? "#ffffff" : c);
      this.queryPoints.setColorAt(i, tmp);
    }
    this.queryPoints.count = n;
    this.queryPoints.instanceMatrix.needsUpdate = true;
    if (this.queryPoints.instanceColor) this.queryPoints.instanceColor.needsUpdate = true;
  }

  emitChange() {
    this.onchange && this.onchange({ pts: this.pts, meta: this.meta, n: this.meta.n });
  }

  getQueries() {
    return { pts: this.pts, n: this.meta.n, meta: this.meta };
  }

  setMode(m) {
    this.mode = m;
    if (m !== "move") {
      this.selected = -1;
      this.dragging = false;
      if (this.controls) this.controls.enabled = true;
      this.rebuildQueries();
    }
  }

  setRadiusScale(v) {
    this.radiusScale = Math.max(0.5, Math.min(10, v));
    if (!this.queryPoints) return;
    this.rebuildQueries();
  }

  resetView() {
    const maxDim = this.maxDim;
    const dist = this.fitRadius / Math.tan((this.camera.fov * Math.PI) / 360);
    this.camera.position.set(this.center.x, this.center.y + 0.12 * maxDim, this.center.z + dist);
    this.controls.target.copy(this.center);
    this.controls.update();
  }

  onResize() {
    if (!this.renderer) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  animate = () => {
    requestAnimationFrame(this.animate);
    if (this.controls) this.controls.update();
    if (this.renderer && this.scene && this.camera) this.renderer.render(this.scene, this.camera);
  };
}

function grow(arr, extra) {
  const out = new Float32Array(arr.length + extra);
  out.set(arr);
  return out;
}
