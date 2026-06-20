# E4B NLA Findings Log

Research findings from the E4B NLA (Natural Language Autoencoder / Activation Verbalizer) project.
Append-only. If a finding is overturned, append a new entry; do not edit the original.

---

## §F89 (carried from E2B) — Injection channel cleared; objective is the wall

**Date:** 2026-06-12 (E2B research, carried into E4B planning)

Stage-0 channel floor-probe CLEARED (10/10 trained hits, jitter-robust). Earlier "FAIL" on E2B
was a GC-poisoned in-process eval artifact. The injection channel (the ㊗ token hook mechanism,
INJ_ID=249568) is NOT the bottleneck.

Primary wall: **objective (VAE posterior collapse / Bowman 1511.06349 framing)**. SFT-only
cross-entropy can be minimized by finding a format-satisfying template independent of the
injected activation. Reconstruction-in-loop reward is the decisive missing component.

---

## §F90 (carried from E2B) — Balanced 15-domain corpus + anisotropy finding

**Date:** 2026-06-12 (E2B research)

New balanced 15-domain corpus: 1,356 train / 144 eval rows. FineWeb capped 10%; 5 new academic
domains; Gemini two-pass labeling with "N: KEEP" auditor-contamination found and fixed (exact 0
contaminated rows after fix).

Anisotropy finding: common-mean direction holds ~22% of activation energy; mean-centering removes
it → cheap mean-centered injection lever. Applied in all E4B ARM A and GRPO runs.

---

## §F91 — E4B ARM A: total mode collapse at scale; Gate R → GRPO on E2B

**Date:** 2026-06-19 to 2026-06-20
**Kernel:** calebdeleeuw/e4b-nla-train-0619
**Artifact:** results/e4b/arm_a_summary.json, results/e4b/gate_r_decision.json

### Setup

E4B ARM A = domain-aware InfoNCE (16 same-domain + 16 cross-domain negatives),
mean-centered injection, LoRA r=8, 1500 steps, seed 17, on the §F90 balanced corpus
(1,356 train / 144 eval). Model: google/gemma-4-E4B (D_MODEL=2560, INJECT_LAYER=24).

### Kill gate result: PASS (but misleading)

Linear regression on raw loss: slope = −0.00254/step, R² = 0.135 → satisfies both
thresholds (slope < −0.002, R² ≥ 0.10). Loss descends. Kill gate passes.

### Eval result: total mode collapse

| Metric | Value | Pre-registered bar |
|---|---|---|
| TF-IDF doc_top1 | 0.0 (p=1.0) | p < 0.05 sustained ≥ 2 ckpts |
| Semantic doc_top1 | 0.0 (p=1.0) | p < 0.05 sustained ≥ 2 ckpts |
| n_unique (n=68) | 1 / 68 | ≥ 8 / 10 |
| mode_collapsed | True | False |
| domain_locked | True | False |

All 68 eval outputs were identical:
> "The Zombie Chronicles review: watching the awful anthology film twice despite knowing the ending"

Gen-compare at step 250 (n=12) also showed n_unique=1. Collapse was present from the start.

### Interpretation

Loss descends because the model found a single B-movie review template that satisfies the
`<explanation>...</explanation>` format constraint. The contrastive loss is minimized by
formatting-correct constant output — the activation vector is ignored entirely.

This confirms §F89 at E4B scale: the SFT-only objective (even domain-aware InfoNCE) cannot
break posterior collapse. The injection channel is clear; the objective is the wall.

### Gate R decision

Option A: GRPO on E2B first (DAPO variant, TF-IDF retrieval reward).

Rationale: E2B is 4× cheaper for prototyping. A positive GRPO result enables a clean E4B
scale test. All SFT arms (B=doc-id contrastive, C=plain CE, D=mean-centered) are expected
to produce the same collapse for the same reason — burning GPU on them adds no information.

Stage 4 (reconstruction-reward / GRPO) is the pre-registered next step under §F89; the
E4B ARM A result is confirmatory data, not a surprise.

Pre-registered GRPO success bar: mean_hard_retrieval > 0.1 AND n_unique ≥ 5/20.

**Kernel running:** calebdeleeuw/e2b-nla-grpo-0620
**Script:** train_grpo_e2b.py (DAPO, G=4, 600 steps, combined reward 0.8*hard + 0.2*soft)
**HF output:** Solshine/e2b-nla-grpo-0620

---

## §F91-GRPO (pending)

E2B GRPO results pending from kernel calebdeleeuw/e2b-nla-grpo-0620. Will append once
the kernel completes. Watching for: mean_hard_retrieval, n_unique, verdict.
