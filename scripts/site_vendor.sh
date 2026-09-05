#!/usr/bin/env bash
# One-time vendoring of third-party runtime deps into site/vendor/.
# No node/npm required -- downloads the npm registry tarballs and extracts them.
#
#   scripts/site_vendor.sh
#
# Re-running is safe; it overwrites the vendored files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="$ROOT/site"
VENDOR="$SITE/vendor"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

THREE_VERSION="0.161.0"
ORT_VERSION="1.29.0"

echo "==> three@${THREE_VERSION}"
curl -fsSL "https://registry.npmjs.org/three/-/three-${THREE_VERSION}.tgz" -o "$TMP/three.tgz"
tar -xzf "$TMP/three.tgz" -C "$TMP"
mkdir -p "$VENDOR/three/build" "$VENDOR/three/addons/controls" "$VENDOR/three/addons/loaders"
cp "$TMP/package/build/three.module.js" "$VENDOR/three/build/"
cp "$TMP/package/examples/jsm/controls/OrbitControls.js" "$VENDOR/three/addons/controls/"
cp "$TMP/package/examples/jsm/loaders/OBJLoader.js" "$VENDOR/three/addons/loaders/"

echo "==> onnxruntime-web@${ORT_VERSION}"
curl -fsSL "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-${ORT_VERSION}.tgz" -o "$TMP/ort.tgz"
tar -xzf "$TMP/ort.tgz" -C "$TMP"
mkdir -p "$VENDOR/ort"
# Only the jsep (SharedArrayBuffer-free) build is used: inference.js sets
# jsepWasm=true so WebGPU + the WASM fallback both run on ort.all.min.js ->
# jsep.mjs -> jsep.wasm. The non-jesp SIMD-threaded files are never referenced.
cp "$TMP/package/dist/ort.all.min.js" \
   "$TMP/package/dist/ort-wasm-simd-threaded.jsep.mjs" \
   "$TMP/package/dist/ort-wasm-simd-threaded.jsep.wasm" \
   "$VENDOR/ort/"

echo "==> vendored files:"
find "$VENDOR" -type f | sort | sed "s#$ROOT/##"
