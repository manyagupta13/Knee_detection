# RSNA Knee Abnormality Detection — Competition Plan & Phased Roadmap

Macro-averaged AUC over 12 findings · code competition · 9-hour offline compute limit ·
final submission 22 Oct 2026. Numbers below are measured in the Day-0 diagnostics run,
not assumed.

## The one fact that shapes everything

Only **58 of 4,407** training studies carry gold labels (1.3%). Every finding has between
9 and 35 positives in total. All 4,407 studies have a radiology report; 4,349 have a
report and no labels. **Reports do not exist in the test set.**

Therefore **text is a supervision channel, not a model input**. The whole competition is:
turn reports into labels for ~4,300 studies, then train an image-only model on those. A
gold-only image model is not viable — 2 positives per fold is noise.

## 1 · Competition at a glance

| Item | Detail |
|---|---|
| Task | Per-study probability for each of 12 knee-MRI findings |
| Metric | Macro-averaged AUC ROC over the 12 columns (all findings weighted equally) |
| Targets | ACL, MCL, Medial/Lateral Meniscus, Medial/Lateral/PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture |
| Train | 4,407 studies · 24,371 series · reports in ~10 languages · 58 gold-labeled |
| Test | ~1,300 studies · DICOM only · no reports |
| Compute | Notebook, internet OFF, ≤ 9 h GPU. External pretrained weights & data allowed |
| Output | `submission.csv` with 12 probability columns |
| Deadlines | Entry & merger 15 Oct · Final sub 22 Oct · Winners' pkg 5 Nov |
| Prizes | Main $9k–$5k (top 10) · Efficiency track $7k/$6k/$5k |

The Efficiency track ($18k, 3 places) rewards a fast model that merely beats the 0.5
baseline and draws far fewer serious entrants — a deliberate small variant is close to
free money.

## 2 · What the Day-0 diagnostics changed

**2.1 Label scarcity is extreme.** 58 gold studies, 9–35 positives per column. Text
pseudo-labeling is the critical path, not an enhancement.

**2.2 English is a minority (39%).** en 39.4 · es 15.6 · tr 10.4 · el 7.4 · hr/slavic 7.3 ·
de 5.9 · ru/bg 5.0 · nl 3.4 · fr 1.9 · pl 1.6. Per-language regex across ~10 languages is
more work than fine-tuning one multilingual model.

**2.3 Naive negation is unreliable — even in English.** When Medial Meniscus is absent the
term still appears in 62% of reports and the negation regex catches only 30% of those.
Effusion is negated in 77% of absent cases but also 53% of present cases ("no significant
effusion" vs "trace effusion"). The regex failure mode is not "misses rare phrasing" — it
is "systematically mislabels common phrasing."

**2.4 Decode is a non-issue.** 0.28 s/series, uncompressed, zero failures; full test at 4
series/study projects to ~0.4 h single-core. *Caveat:* the 60-series sample hit only
uncompressed Explicit VR LE. JPEG Lossless and JPEG 2000 also exist in the corpus and
neither pylibjpeg nor gdcm is in the base image. Re-run the probe on a transfer-syntax
stratified sample and pin a decoder before banking the headroom.

**2.5 A fixed canonical series set covers ~all studies.** Sagittal fluid-sensitive 94.2% ·
Coronal fluid 96.4% · Axial fluid 100% · Sagittal any 100%. Top protocol combination covers
57% of studies; the top four cover 92%.

## 3 · Strategy in one page

1. **Text → labels.** Highest leverage, cheapest compute, full internet (train-only).
2. **Data engineering.** Decode once, cache small. Canonical series + slice ordering +
   intensity normalization, one code path for train and test.
3. **Image → 12 labels.** Slice encoder → attention pool over slices → attention pool over
   series → 12 heads.

### Design invariants — protect these from day one

1. **One preprocessing function for train and test.** Share the exact code.
2. **Masked BCE from the start.** Loss backprops only on labeled columns — this is how
   gold, high-confidence text and low-confidence text labels later coexist in one run.
3. **Validate on gold only, never on pseudo-labels.** Pseudo-label AUC measures agreement
   with your text model, not truth.
4. **Commit a working offline submission in week 1.** Timeouts kill more entries than bad
   models.

## 4 · Phased plan

**Phase 0 · First submission (2 hours).** See [PHASE0.md](PHASE0.md). Precision-first
pseudo-labels for 2 columns; one small image model on the canonical series; offline timed
inference notebook. Not a leaderboard position.

**Phase 1 · Text pseudo-labeling (week 1–2, critical path).** Language-bucket every report.
Fine-tune XLM-R / a multilingual clinical encoder with 12 sigmoid heads on the 58 gold
studies, or run an LLM per report with a strict extraction schema — prefer this over regex
given §2.3. Emit soft labels + per-study confidence. Anchor medial/lateral/PF terms to
their modifiers. Absence-of-mention as negative is valid only where the probe showed ~0%
mention when absent.

*Exit criterion:* high agreement with gold on common findings and a documented, honest
lower number on rare ones. Everything downstream is bounded by label quality.

**Phase 2 · Data engineering (parallel, week 1–3).** Decode-once cache of per-series uint8
volumes ~320×320. Percentile-clip per series (1st–99th), not per slice — per-slice
normalization destroys fluid cues. Sort by `ImagePositionPatient` projected on the slice
normal, not `InstanceNumber`. Canonicalize laterality/orientation via
`ImageOrientationPatient` *before* any horizontal flip. Pin decoders for compressed
syntaxes. `preprocess_study()` returns the stacked multi-series form from day one.

**Phase 3 · Image model (week 3–8).**

| Component | Choice | Why |
|---|---|---|
| Slice encoder | ConvNeXt-T / EffNetV2-S @ 320–384px, ImageNet | 2.5D (3 adjacent slices as channels) buys most of 3D cheaply |
| Slice pool | Gated attention (ABMIL) | Findings are focal; most slices irrelevant |
| Series pool | Attention + plane/contrast embeddings | Tolerates heterogeneous protocols & missing series |
| Training | 2-stage: pseudo-labels → fine-tune on gold | Uses all data; gold refines |
| Loss | Masked BCE, soft targets | AUC is rank-based; skip focal loss |
| Augment | Heavy intensity (gamma, bias field, noise) | Scanner variation is the stated domain shift |

**Phase 4 · Validation.** 5-fold, grouped by patient where inferable,
multilabel-stratified. Report per-column gold AUC every run — macro-average hides
rare-column collapse. Trust CV loosely; rankings, not calibrated probabilities, transfer.
Ensemble by rank-average.

**Phase 5 · Inference & submission.** Multiprocess decode → queue; GPU consumes fp16
batched across slices. Build and time the real notebook in week 3 with a dummy model.
Exact filename, 12 columns in order, every test StudyInstanceUID present.

**Phase 6 · Efficiency track.** Score = score-gap × evaluation-seconds, minimized. One
sagittal series, 224px, EfficientNet-B0. The Phase-0 model is already close — keep that
checkpoint.

## 5 · Timeline

| Window | Focus | Exit signal |
|---|---|---|
| Next 2 h | Phase 0 valid offline submission | submission.csv commits & scores |
| Week 1–2 | Text labeler + preprocessing cache, in parallel | Labeler agrees with gold |
| Week 3 | Baseline single-plane model; notebook timed | End-to-end run inside budget |
| Week 4–6 | Multi-series architecture, series selection, augments | Multi-series > single on gold CV |
| Week 7–8 | Scale resolution/backbone; pseudo-label round 2 | Per-column AUC gains |
| Week 9 | Ensemble; efficiency variant | Rank-averaged LB gain |
| Week 10 | Buffer — reserve it | Final selections locked |

## 6 · Risk register

| Risk | Mitigation |
|---|---|
| Notebook times out on real test | Time it in week 3 with a dummy model; keep decode cheap |
| Missing JPEG2000 decoder → silent failures | Pin pylibjpeg-openjpeg / gdcm; re-benchmark stratified |
| Pseudo-labels systematically wrong (negation) | Multilingual model over regex; validate per-language on gold |
| Train/test preprocessing drift | One shared `preprocess_study()`; no test-only path |
| Overfitting to 58 gold labels | Train on pseudo-labels; gold = fine-tune + validation only |
| Rare-column collapse hidden by macro-avg | Log all 12 per-column AUCs every run |
| Laterality flip swaps medial/lateral | Canonicalize orientation before any horizontal flip |
| Prevalence shift between splits | Optimize ranking; ensemble by rank-average |

**Highest-leverage action right now:** get the multilingual text labeler to strong gold
agreement. Every image-model gain downstream is capped by the quality of those ~4,300
pseudo-labels.
