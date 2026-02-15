"""
Generate LaTeX table for Appendix B: the 12 Tier-A lenses missed by
the CNN (score < 0.3 at threshold p > 0.3).

Cross-matches with desi_candidates.csv via RA/Dec to obtain z_lens.
Outputs: LaTeX snippet to stdout and JSON to results directory.
"""

import csv
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
GALLERY = BASE / "results/D06_corrected_priors/gallery/gallery_data.json"
DESI_CSV = BASE / "data/positives/desi_candidates.csv"
OUT_JSON = BASE / "results/D06_corrected_priors/gallery/missed_tier_a.json"
OUT_TEX = BASE / "results/D06_corrected_priors/gallery/missed_tier_a_table.tex"

THRESHOLD = 0.3
MATCH_RADIUS_DEG = 0.20  # ~12 arcmin, covers half-brick diagonal (~0.18 deg)


def parse_brick_radec(cutout_path):
    """Extract approximate RA, Dec from the Legacy Survey brick name."""
    fname = cutout_path.split("/")[-1]  # e.g. 0008m627_3575.npz
    brick = fname.split("_")[0]  # e.g. 0008m627
    if len(brick) < 8:
        return None, None
    ra = int(brick[:4]) / 10.0
    sign = brick[4]
    dec = int(brick[5:8]) / 10.0
    if sign == "m":
        dec = -dec
    return ra, dec


def angular_sep(ra1, dec1, ra2, dec2):
    """Simple angular separation in degrees (small-angle approximation)."""
    ddec = dec1 - dec2
    dra = (ra1 - ra2) * math.cos(math.radians((dec1 + dec2) / 2))
    return math.sqrt(dra**2 + ddec**2)


def load_desi_catalog():
    """Load DESI candidates with RA, Dec, z_lens."""
    rows = []
    with open(DESI_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "name": row["name"],
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
                "zlens": float(row["zlens"]) if row["zlens"] else None,
                "grading": row["grading"],
            })
    return rows


def match_desi(ra, dec, catalog):
    """Find the closest DESI candidate within MATCH_RADIUS_DEG."""
    best = None
    best_sep = MATCH_RADIUS_DEG
    for c in catalog:
        sep = angular_sep(ra, dec, c["ra"], c["dec"])
        if sep < best_sep:
            best_sep = sep
            best = c
    return best, best_sep


def format_score(score):
    if score >= 0.1:
        return f"{score:.3f}"
    if score >= 0.001:
        return f"{score:.4f}"
    return f"{score:.4f}"


def main():
    with open(GALLERY) as f:
        gallery = json.load(f)

    missed = [l for l in gallery["real_lenses"] if l["score"] < THRESHOLD]
    missed.sort(key=lambda x: x["score"])

    desi = load_desi_catalog()

    records = []
    for lens in missed:
        ra, dec = parse_brick_radec(lens["cutout_path"])
        desi_match, sep = match_desi(ra, dec, desi) if ra is not None else (None, None)

        rec = {
            "uid": lens["uid"],
            "r_mag": lens["r_mag"],
            "score": lens["score"],
            "brick_ra": ra,
            "brick_dec": dec,
        }
        if desi_match:
            rec["desi_name"] = desi_match["name"]
            rec["zlens"] = desi_match["zlens"]
            rec["grading"] = desi_match["grading"]
            rec["match_sep_deg"] = round(sep, 4)
        else:
            rec["desi_name"] = None
            rec["zlens"] = None
            rec["grading"] = None
            rec["match_sep_deg"] = None

        records.append(rec)

    with open(OUT_JSON, "w") as f:
        json.dump(records, f, indent=2)
    print(f"JSON written to {OUT_JSON}")

    n_matched = sum(1 for r in records if r["desi_name"])
    n_total = len(records)
    print(f"Matched {n_matched}/{n_total} to DESI candidates")

    tex_lines = []
    tex_lines.append(r"\begin{table}")
    tex_lines.append(r"\centering")
    tex_lines.append(
        r"\caption{The 12 Tier-A lenses missed by the CNN at the $p > 0.3$ "
        r"detection threshold.  Sorted by ascending CNN score.  $z_{\rm lens}$ "
        r"from the DESI Strong Lensing catalogue cross-matched by position "
        r"($< 12\;\mathrm{arcmin}$; matching brick centre to catalogue position).  All missed lenses have $r$-band magnitudes "
        r"$\leq 20$, consistent with being bright deflector galaxies; the CNN "
        r"scores are nevertheless extremely low ($\leq 0.19$), suggesting "
        r"morphological reasons (e.g.\ edge-on hosts, compact image "
        r"configurations) rather than faintness as the cause of failure.}"
    )
    tex_lines.append(r"\label{tab:missed}")
    tex_lines.append(r"\small")
    tex_lines.append(r"\begin{tabular}{llccl}")
    tex_lines.append(r"\hline")
    tex_lines.append(r"ID & DESI name & $r$ (mag) & CNN score & $z_{\rm lens}$ \\")
    tex_lines.append(r"\hline")

    for r in records:
        desi_name = r["desi_name"] if r["desi_name"] else "---"
        if len(desi_name) > 25:
            desi_name = desi_name.replace("DESI-", "")
        zlens_str = f"{r['zlens']:.3f}" if r["zlens"] else "---"
        score_str = format_score(r["score"])
        tex_lines.append(
            f"{r['uid']} & {desi_name} & ${r['r_mag']:.2f}$ & "
            f"${score_str}$ & ${zlens_str}$ \\\\"
        )

    tex_lines.append(r"\hline")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{table}")

    tex_content = "\n".join(tex_lines)
    with open(OUT_TEX, "w") as f:
        f.write(tex_content + "\n")
    print(f"LaTeX written to {OUT_TEX}")
    print()
    print(tex_content)


if __name__ == "__main__":
    main()
