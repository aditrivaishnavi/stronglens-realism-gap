#!/usr/bin/env python3
"""
Generate an HTML gallery of all scored real Tier-A lenses and injection
examples (from the D06 grid pool) so the user can manually select which
ones to include in the paper figure.

Each image gets a unique ID for easy reference:
  - Real lenses:  REAL_001, REAL_002, ...
  - Injections:   INJ_075_001 (theta_E=0.75, item 1), INJ_150_003, etc.

Outputs:
  <out-dir>/gallery.html      — the gallery page
  <out-dir>/thumbs/            — thumbnail PNGs
  <out-dir>/gallery_data.json  — all metadata for programmatic use

Usage (on lambda3):
    cd /lambda/nfs/darkhaloscope-training-dc/stronglens_calibration
    PYTHONPATH=. python scripts/generate_gallery_html.py \
        --out-dir results/D06_corrected_priors/gallery

Date: 2026-02-17
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

THETA_E_VALS = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
N_PSF = 7
N_DEPTH = 5
STRIDE = N_PSF * N_DEPTH
TARGET_THETA_E = [0.75, 1.50, 2.50]

BASE = Path("/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration")
D06 = BASE / "results" / "D06_corrected_priors"
GRID_CUTOUTS = D06 / "grid_no_poisson" / "cutouts"
MANIFEST = BASE / "manifests" / "training_parity_70_30_v1.parquet"


def load_cutout_rgb(path, key="cutout"):
    data = np.load(path)
    img = None
    for k in [key, "injected", "cutout_grz", "image"]:
        if k in data:
            img = data[k]
            break
    if img is None:
        img = data[list(data.keys())[0]]
    if img.ndim == 3 and img.shape[2] == 3:
        img = img.transpose(2, 0, 1)
    return img


def cutout_to_display(img, clip_lo=0.5, clip_hi=99.5):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    rgb = img[:, :, ::-1].copy()
    lo = np.percentile(rgb[rgb > 0], clip_lo) if np.any(rgb > 0) else 0
    hi = np.percentile(rgb, clip_hi)
    if hi <= lo:
        hi = lo + 1
    rgb = (rgb - lo) / (hi - lo)
    rgb = np.clip(rgb, 0, 1)
    return rgb


def total_r_mag(img_3hw):
    r_flux = np.sum(img_3hw[1])
    if r_flux <= 0:
        return 99.0
    return 22.5 - 2.5 * np.log10(r_flux)


def save_thumb(img_3hw, path, size_px=200):
    """Save a single cutout as a PNG thumbnail."""
    rgb = cutout_to_display(img_3hw)
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.imshow(rgb, origin="lower", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str,
                        default=str(D06 / "gallery"))
    parser.add_argument("--max-inj-per-theta", type=int, default=50,
                        help="Max injections to score per theta_E")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    sys.path.insert(0, str(BASE))
    from dhs.scoring_utils import load_model_and_spec
    from dhs.preprocess import preprocess_stack

    ckpt_path = str(BASE / "checkpoints" /
                    "paperIV_efficientnet_v2_s_v4_finetune" / "best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pp_kwargs = load_model_and_spec(ckpt_path, device)
    model.eval()

    def score_img(img_3hw):
        pp = preprocess_stack(img_3hw, **pp_kwargs)
        inp = torch.from_numpy(pp).float()
        with torch.no_grad():
            return torch.sigmoid(model(inp.unsqueeze(0).to(device))).item()

    # ── Score all Tier-A lenses ──
    manifest = pd.read_parquet(MANIFEST)
    tier_a = manifest[
        (manifest["split"] == "val") & (manifest["tier"] == "A")
    ].copy()
    print(f"Scoring {len(tier_a)} Tier-A lenses...")

    real_data = []
    for i, (_, row) in enumerate(tier_a.iterrows()):
        cp = row["cutout_path"]
        if not os.path.exists(cp):
            continue
        img = load_cutout_rgb(cp)
        rmag = total_r_mag(img)
        score = score_img(img)
        uid = f"REAL_{i+1:03d}"
        thumb_name = f"{uid}.png"
        save_thumb(img, thumbs_dir / thumb_name)
        real_data.append({
            "uid": uid,
            "thumb": f"thumbs/{thumb_name}",
            "cutout_path": cp,
            "r_mag": round(rmag, 2),
            "score": round(score, 4),
        })
        if (i + 1) % 20 == 0:
            print(f"  Real: {i+1}/{len(tier_a)}")
    print(f"Scored {len(real_data)} real lenses")

    # ── Score injection pools ──
    inj_data = {}
    for target_te in TARGET_THETA_E:
        te_idx = THETA_E_VALS.index(target_te)
        cell_start = te_idx * STRIDE
        cell_end = (te_idx + 1) * STRIDE
        te_key = str(target_te)

        pool = []
        count = 0
        for cell_idx in range(cell_start, cell_end):
            for inj_idx in range(500):
                fname = f"cell{cell_idx:05d}_inj{inj_idx:05d}.npz"
                fpath = GRID_CUTOUTS / fname
                if not fpath.exists():
                    continue
                img = load_cutout_rgb(str(fpath))
                rmag = total_r_mag(img)
                if rmag < 16 or rmag > 23:
                    continue
                score = score_img(img)
                count += 1
                te_label = str(target_te).replace(".", "")
                uid = f"INJ_{te_label}_{count:03d}"
                thumb_name = f"{uid}.png"
                save_thumb(img, thumbs_dir / thumb_name)
                pool.append({
                    "uid": uid,
                    "thumb": f"thumbs/{thumb_name}",
                    "cutout_path": str(fpath),
                    "r_mag": round(rmag, 2),
                    "score": round(score, 4),
                    "theta_e": target_te,
                    "cell_idx": cell_idx,
                    "inj_idx": inj_idx,
                })
                if inj_idx >= 9:
                    break
            if count >= args.max_inj_per_theta:
                break

        inj_data[te_key] = pool
        print(f"theta_E={target_te}: scored {len(pool)} injections")

    # ── Save JSON ──
    gallery_json = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "real_lenses": real_data,
        "injections": inj_data,
    }
    json_path = out_dir / "gallery_data.json"
    with open(json_path, "w") as f:
        json.dump(gallery_json, f, indent=2, default=str)

    # ── Generate HTML ──
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Figure 5 Gallery — Real Lenses &amp; Injections</title>
<style>
  body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
  h1 { color: #e94560; }
  h2 { color: #0f3460; background: #e94560; padding: 8px 12px; border-radius: 4px; }
  .gallery { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 30px; }
  .card {
    background: #16213e; border: 2px solid #333; border-radius: 8px;
    padding: 6px; text-align: center; width: 180px;
    transition: border-color 0.2s;
  }
  .card:hover { border-color: #e94560; }
  .card img { width: 160px; height: 160px; border-radius: 4px; }
  .card .uid { font-weight: bold; font-size: 12px; color: #e94560; margin-top: 4px; }
  .card .meta { font-size: 11px; color: #aaa; margin-top: 2px; }
  .card .score { font-size: 13px; font-weight: bold; margin-top: 2px; }
  .score-high { color: #00ff88; }
  .score-low { color: #ff4444; }
  .score-mid { color: #ffaa00; }
  .section-header { margin-top: 40px; }
  .theta-label { color: #00d2ff; font-size: 14px; font-weight: bold;
    margin: 15px 0 5px 0; padding: 4px 8px; background: #0f3460;
    display: inline-block; border-radius: 4px; }
</style>
</head><body>
<h1>Figure 5 Gallery — Select Images by UID</h1>
<p>Pick UIDs for the paper figure. Each image shows: UID, CNN score (p), r-band magnitude.</p>
""")

    # Panel (a): Real lenses
    html_parts.append('<div class="section-header"><h2>(a) Real Tier-A Strong Lenses</h2></div>')
    html_parts.append(f'<p>{len(real_data)} lenses scored. Sorted by r-mag (bright first).</p>')
    html_parts.append('<div class="gallery">')
    for entry in sorted(real_data, key=lambda x: x["r_mag"]):
        sc = entry["score"]
        sc_class = "score-high" if sc > 0.9 else ("score-mid" if sc > 0.3 else "score-low")
        html_parts.append(f'''<div class="card">
  <img src="{entry['thumb']}" alt="{entry['uid']}">
  <div class="uid">{entry['uid']}</div>
  <div class="score {sc_class}">p = {sc:.4f}</div>
  <div class="meta">r = {entry['r_mag']:.2f}</div>
</div>''')
    html_parts.append('</div>')

    # Panel (b): Injections by theta_E
    for te_key in [str(t) for t in TARGET_THETA_E]:
        entries = inj_data[te_key]
        html_parts.append(f'<div class="section-header"><h2>(b) Injections — '
                          f'θ_E = {te_key} arcsec</h2></div>')
        html_parts.append(f'<p>{len(entries)} injections scored. Sorted by r-mag.</p>')
        html_parts.append('<div class="gallery">')
        for entry in sorted(entries, key=lambda x: x["r_mag"]):
            sc = entry["score"]
            sc_class = "score-high" if sc > 0.9 else ("score-mid" if sc > 0.3 else "score-low")
            html_parts.append(f'''<div class="card">
  <img src="{entry['thumb']}" alt="{entry['uid']}">
  <div class="uid">{entry['uid']}</div>
  <div class="score {sc_class}">p = {sc:.4f}</div>
  <div class="meta">r = {entry['r_mag']:.2f}  |  θ_E = {te_key}"</div>
</div>''')
        html_parts.append('</div>')

    html_parts.append('</body></html>')

    html_path = out_dir / "gallery.html"
    with open(html_path, "w") as f:
        f.write("\n".join(html_parts))
    print(f"\nHTML gallery saved to: {html_path}")
    print(f"JSON data saved to: {json_path}")
    print(f"Thumbnails in: {thumbs_dir}")


if __name__ == "__main__":
    main()
