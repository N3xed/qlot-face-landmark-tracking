#!/usr/bin/env python3
"""Generate the static assets for the paper website under site/.

Run with the project venv:
    .venv/bin/python scripts/site_build.py

It (re)generates:
  - SVG figures compiled from the paper Typst source (model + write-mix-read block)
  - PNG thumbnails for paper.pdf / supplement.pdf, the video placeholder, and the annotated-samples figure
  - the query-point JSON presets (canonical landmarks)
  - the full-resolution face mesh (.obj)
  - the downloadable PDFs and the ONNX model

The one-time third-party vendoring (three.js + onnxruntime-web) lives in
scripts/site_vendor.sh.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
FIGDIR = ROOT / "docs" / "paper" / "figures"
PAPER = ROOT / "docs" / "paper"
DATA = ROOT / "data"

ASSETS = SITE / "assets"
CSS = SITE / "css"
JS = SITE / "js"
SDATA = SITE / "data"
SML = SITE / "models"


def run(cmd, **kw):
    print("  $", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kw)


def ensure_dirs():
    for d in (ASSETS, CSS, JS, SDATA, SML):
        d.mkdir(parents=True, exist_ok=True)


def build_figures():
    print("[figures] compiling SVG figures with typst")
    model_typ = FIGDIR / "model.typ"
    run(["typst", "compile", str(model_typ), str(ASSETS / "model.svg"), "--pages", "1"])
    run(["typst", "compile", str(model_typ), str(ASSETS / "update_prediction.svg"), "--pages", "2"])


def pdf_to_png(src, dst, dpi=110, single=True):
    out_prefix = dst.with_suffix("")
    cmd = ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1"]
    if single:
        cmd.append("-singlefile")
    cmd += [str(src), str(out_prefix)]
    run(cmd)


def build_thumbnails():
    print("[thumbnails] rasterizing PDF first pages")
    pdf_to_png(PAPER / "paper.pdf", ASSETS / "thumb_paper.png", dpi=120)
    pdf_to_png(PAPER / "supplement.pdf", ASSETS / "thumb_supplement.png", dpi=120)
    pdf_to_png(FIGDIR / "fig_annotated_samples.pdf", ASSETS / "video_placeholder.png", dpi=130)
    pdf_to_png(FIGDIR / "fig_annotated_samples.pdf", ASSETS / "fig_annotated_samples.png", dpi=130)


def build_queries():
    print("[queries] copying canonical landmark presets")
    mapping = {
        "canonical_landmarks_98_wflw.json": "queries_wflw_98.json",
        "canonical_landmarks_70_synth.json": "queries_synth_70.json",
        "canonical_landmarks_68_ibug.json": "queries_ibug_68.json",
    }
    for src, dst in mapping.items():
        data = json.load(open(DATA / src))
        with open(SDATA / dst, "w") as f:
            json.dump(data, f)
        print(f"   {dst}: {len(data['points'])} points, groups={list(data.get('indices', {}).keys())}")


def build_mesh():
    print("[mesh] writing full-resolution face mesh (normal refs stripped for three.js OBJLoader)")
    src = DATA / "face_mesh.obj"
    dst = SDATA / "face_mesh.obj"
    with open(src, "r") as f_in, open(dst, "w") as f_out:
        for line in f_in:
            if line.startswith("f "):
                toks = line.split()
                toks = [t.split("//")[0] for t in toks[1:]]
                f_out.write("f " + " ".join(toks) + "\n")
            else:
                f_out.write(line)
    size = dst.stat().st_size / 1e6
    print(f"   face_mesh.obj: {size:.2f} MB")


def build_downloads():
    print("[downloads] copying paper + supplement PDFs")
    shutil.copy2(PAPER / "paper.pdf", ASSETS / "qlot-paper.pdf")
    shutil.copy2(PAPER / "supplement.pdf", ASSETS / "qlot-supplement.pdf")
    # Third-party attribution must travel with the served face_mesh.obj.
    shutil.copy2(ROOT / "NOTICE", SITE / "NOTICE")


def build_model():
    print("[model] copying ONNX model")
    src = ROOT / "models" / "qlot-final.onnx"
    shutil.copy2(src, SML / "qlot-final.onnx")
    print(f"   qlot-final.onnx: {src.stat().st_size / 1e6:.2f} MB")


def main():
    ensure_dirs()
    build_figures()
    build_thumbnails()
    build_queries()
    build_mesh()
    build_downloads()
    build_model()
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
