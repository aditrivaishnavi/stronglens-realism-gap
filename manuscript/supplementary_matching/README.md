# Canonical matched-target visual and tabular supplement

This directory accompanies the final \(G_{\rm match}\) and \(C_{\rm bright}\)
descriptions.

- `canonical_gmatch_cbright_confirmed_target_atlas.pdf` displays all 43
  observed-reference and canonical matched-bare-target cutouts using the
  complete \(101\times101\)-pixel field and one fixed Legacy Survey RGB
  mapping.
- `canonical_matches.csv` gives the final score-blind assignments and five
  matching quantities.
- `canonical_balance_diagnostics.csv` gives the final balance summaries.
- `canonical_cbright_target_mapping.csv` binds the final matched targets to
  the \(C_{\rm bright}\) parent scenes.
- `canonical_atlas_receipt.json` records the authority and display hashes.
- `generate_gmatch_cbright_pair_atlas.py` regenerates the atlas from the
  retained authorities.
- `canonical_pair_pages/` contains the 43 atlas pair pages as PNG files using
  the same full-field display mapping.
- `component_checks/` records the catalogue-centre and injection-centre checks
  for the three visually queried systems.

These materials document approximate matching on recorded central-object and
observing quantities. They do not establish physical equivalence between
observed deflectors and injection targets.

The generator requires the retained repository authorities and pixel payloads;
the display PNGs and PDF are included so that those private paths are not
required for visual inspection. From the repository root, regenerate with:

```bash
uv run --with-requirements manuscript/supplementary_matching/requirements.txt \
  python manuscript/supplementary_matching/generate_gmatch_cbright_pair_atlas.py
```
