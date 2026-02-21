# Experiment Registry

Single source of truth for all training runs, diagnostic experiments, and
injection model variants. Updated as new runs are added.

**NFS base**: `/lambda/nfs/darkhaloscope-training-dc/stronglens-realism-gap/`

---

## Training Generations (EfficientNetV2-S)

| Gen ID | Config File | LR | Epochs | Init From | Annulus | Loss | Status | Best AUC | Notes |
|--------|-------------|-----|--------|-----------|---------|------|--------|----------|-------|
| gen1 | `paperIV_efficientnet_v2_s.yaml` | 3.88e-4 | 160 | ImageNet | (20,32) | Unweighted BCE | Trained | 0.9893 | Baseline step LR |
| gen2 | `paperIV_efficientnet_v2_s_v2.yaml` | 3.88e-4 | 160 | ImageNet | (20,32) | Unweighted BCE | Trained | 0.9915 | Freeze backbone 5ep + warmup |
| gen3 | `paperIV_efficientnet_v2_s_v3_cosine.yaml` | 3.88e-4 | 160 | ImageNet | (20,32) | Unweighted BCE | Trained | ~0.991 | Cosine LR schedule |
| gen4 | `paperIV_efficientnet_v2_s_v4_finetune.yaml` | 5e-5 | 60 | gen2/best.pt | (20,32) | Unweighted BCE | Trained | 0.9921 | **Current best.** Finetune from gen2 E19 |
| gen5a | `gen5a_efficientnet_annulus_fix.yaml` | 3.88e-4 | 160 | ImageNet | **(32.5,45)** | Unweighted BCE | Not trained | -- | From-scratch with corrected annulus |
| gen5b | `gen5b_efficientnet_annulus_ft.yaml` | 5e-5 | 60 | gen4/best.pt | **(32.5,45)** | Unweighted BCE | Not trained | -- | Finetune gen4 with corrected annulus |
| gen5c | `gen5c_efficientnet_weighted_loss.yaml` | 3.88e-4 | 160 | ImageNet | **(32.5,45)** | **Weighted** (A=1.0, B=0.5) | Not trained | -- | Corrected annulus + weighted loss |

## Training Generations (Bottlenecked ResNet)

| Gen ID | Config File | LR | Epochs | Annulus | Status | Notes |
|--------|-------------|-----|--------|---------|--------|-------|
| resnet-gen1 | `paperIV_bottlenecked_resnet.yaml` | 5e-4 | 160 | (20,32) | Trained | ~195K params baseline |
| resnet-gen2 | `paperIV_bottlenecked_resnet_v2_annulus_fix.yaml` | 5e-4 | 160 | **(32.5,45)** | Not trained | Corrected annulus |

## Other Training Configs (not part of main Paper IV line)

| Config | Purpose | Status |
|--------|---------|--------|
| `paperIV_resnet18.yaml` | ResNet-18 baseline comparison | Trained |
| `resnet18_baseline_v1.yaml` | Early ResNet-18 baseline | Trained |
| `paired_baseline.yaml` | R0 paired experiment | Completed |
| `paired_core_dropout.yaml` | R1 core dropout experiment | Completed |
| `unpaired_matched_raw.yaml` | R3 unpaired raw experiment | Completed |
| `unpaired_matched_raw_hardneg.yaml` | R4 unpaired + hard negatives | Completed |
| `unpaired_matched_residual.yaml` | R5 unpaired residual | Completed |
| `unpaired_matched_residual_hardneg.yaml` | R6 unpaired residual + hard neg | Completed |

---

## Injection Model Variants

| ID | Directory | Host Selection | Lens Params | Status | Key Finding |
|----|-----------|---------------|-------------|--------|-------------|
| IM1 | `injection_model_1/` | Random negatives from val split | Independent SIE | Active | Base selection function grid |
| IM2 | `injection_model_2/` | LRG-matched hosts (conditioned q/PA) | q_lens conditioned on q_host | Active | 0.77pp WORSE than IM1 -- host matching does not help |

---

## Diagnostic Runs

| Run ID | Date | Scripts | Checkpoint | Results Path | Status | Key Finding |
|--------|------|---------|------------|-------------|--------|-------------|
| D01 | 2026-02-14 | split_balance, masked_pixel, annulus_comparison, mismatched_scoring, beta_frac, embedding_umap | gen4 | `results/D01_20260214_pre_retrain_diagnostics/` | **Complete** | Annulus recall drop 3.6-3.8pp (GO). Linear probe AUC=0.991 (massive realism gap). Beta_frac cap ceiling ~35.5%. |
| D02 | 2026-02-14 | clip_range sweep (20/50/100), poisson_noise, poisson+clip50, tier_a_scoring, all_tier_scoring, unrestricted_bf, healpix_investigation, UMAP | gen4 | `results/D02_20260214_quick_tests/` | **Complete** | Clip_range=20 raises mag 18-19 from 17%→30.5%. **D02 Poisson numbers used BUGGY clamp(min=1.0) — apparent improvements were artifacts.** Tier-A recall 89.3% [82.6%,94.0%] at p>0.3. Zero Tier-A spatial leakage. healpix_128 NaN is manifest bug (ra/dec valid). |
| D03 | 2026-02-14 | selection_function_grid (with --add-poisson-noise), bright_arc Poisson+clip20 combined | gen4 | `results/D03_20260214_poisson_grid/` | **Complete (INVALIDATED)** | Poisson clamp bug: clamp(min=1.0) added noise to zero-flux pixels, inflating annulus MAD ~2.5x. Summary reporting bug: averaged across mag sub-bins. True marginal C=0.74% (818/110k), not 2.6%. Grid mismatch: old grid depth 24.0-25.5 vs D03 22.5-24.5. Superseded by D04. |
| D04 | 2026-02-14 | matched baseline + fixed Poisson grids + Poisson+clip20 | gen4 | `results/D04_20260214_matched_comparison/` | **Complete** | **Definitive result:** Baseline marginal completeness 3.41% (3,755/110,000). Fixed Poisson HURT: 2.35% (2,584/110,000), down 1.06pp. Poisson+clip20 bright-arc test worse than clip20 alone at every mag bin except 18-19. Matched parameters: depth 22.5-24.5, 500 inj/cell, 220/385 non-empty cells. |
| D05 | 2026-02-14 | Full re-eval: 6 bright-arc variants (incl. gain sweep), 2 grids, linear probe, Tier-A scoring | gen4 | `results/D05_20260214_full_reeval/` | **Complete** | **Independent verification of all D01-D04 results.** Baseline 3.41% matches D04 exactly. Poisson 2.37% (~D04 2.35%). Gain=1e12 recovers baseline exactly (proves Poisson code correct). NEW: Poisson-only bright-arc test shows degradation at every mag bin. Poisson+clip20 worse than clip20 alone (-16.5pp at mag 21-22). Linear probe AUC=0.996. Tier-A recall 89.3%. |

---

## Evaluation Runs

| Run ID | Date | Checkpoint | Injection Model | Results Path | Key Metrics |
|--------|------|------------|-----------------|-------------|-------------|
| E04-eval | 2026-02-11 | gen4 | -- | `results/eval_efficientnet_v2_s.json` | AUC=0.9921 |
| E04-IM1 | 2026-02-12 | gen4 | IM1 | `results/selection_function_v4_finetune/` | Completeness 3.5%, ceiling 30% |
| E04-IM2 | 2026-02-12 | gen4 | IM2 | `results/selection_function_model2/` | 0.77pp worse than IM1 |

---

## Shared Resources

| Resource | Path |
|----------|------|
| Training manifest (70/30) | `manifests/training_parity_70_30_v1.parquet` |
| Training manifest (70/15/15) | `manifests/training_v1.parquet` |
| Positive cutouts | `cutouts/positives/` |
| Negative cutouts | `cutouts/negatives/20260210_025117/` |

---

## Naming Convention

- **Training**: `gen<N><letter>` (gen1-gen4 = existing, gen5a/b/c = next)
- **Diagnostics**: `D<NN>` with date stamp in results directory
- **Evaluation**: `E<NN>-<suffix>` (e.g., E04-IM1)
- **Injection models**: `IM<N>` (IM1 = random hosts, IM2 = LRG-matched)
- **Results on NFS**: `results/<RunID>_<YYYYMMDD>_<description>/`
