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

## §F91-GRPO — E2B GRPO DAPO: reward-density bottleneck identified

**Date:** 2026-06-20
**Kernel:** calebdeleeuw/e2b-nla-grpo-0620 (v4, 600 steps)
**HF artifacts:** Solshine/e2b-nla-grpo-0620

### Setup

DAPO (no KL/reference model), G=4 completions/anchor, TF-IDF retrieval reward
(0.8 × hard_retrieval@1 + 0.2 × cosine_soft), E2B L23 activations, 163 docs,
mean-centered injection, LoRA r=16, lr=3e-5, seed 17.

### Training reward trajectory

| Steps | Mean reward |
|---|---|
| 1–100 | 0.0166 |
| 201–400 | 0.0206 |
| 401–600 | 0.0219 |
| Max (single step) | 0.4313 |

n_unique_5 (rolling 5-step diversity window, temperature=0.9): **sustained at 5 from step 5
onward**. The model maintained stochastic diversity throughout training — unlike ARM A
which collapsed at step 250.

### Greedy eval result (n=20)

| Metric | Value | Pre-registered bar |
|---|---|---|
| mean_hard_retrieval | 0.05 (1/20) | > 0.1 |
| mean_soft_cosine | 0.093 | — |
| n_unique (greedy) | 1 / 20 | >= 5 |
| verdict | **COLLAPSE** | PASS |

Collapse text: "Now define a precise definition of the term closer with regard to the
image and" — a TF-IDF-gaming template (academic register, occasional retrieval hit on
doc_00000773 which covers image proximity/closeness concepts).

### Interpretation

GRPO DAPO **fixes stochastic diversity** (n_unique_5=5 throughout) but fails to drive
consistent hard retrieval under greedy decoding. The reward signal is too sparse: most
training steps see 0 hard hits across G=4 rollouts (mean combined_reward ~0.02, which
at REWARD_SOFT=0.2 implies mean_hard ~0.0-0.025 per rollout).

The model learned a template that exploits the soft cosine component (academic vocabulary
scores ~0.09 across all docs, better than random), and occasionally lands a hard hit
(1/20 greedy, ~1/4 training steps got ≥1/4 rollouts with hard=1). But 600 steps of
sparse signal is insufficient to condition greedy output on the injected activation.

**Root cause: reward density, not diversity.** GRPO solved the diversity problem that SFT
had. The remaining bottleneck is that TF-IDF hard retrieval@1 across 163 documents is too
sparse a signal to bootstrap activation conditioning from a cold start.

### Next options

1. **Denser reward**: BM25 soft matching or n-gram overlap (BLEU/ROUGE) instead of
   binary retrieval@1 — would give gradient signal on every step
2. **Curriculum**: start with 10-20 highly-distinct docs to raise baseline hit rate, then
   expand corpus
3. **Warm start**: SFT on a handful of high-reward GRPO rollouts first, then GRPO
4. **Longer run**: 2000+ steps — the sparse signal may propagate given enough steps
   (the max single-step reward 0.43 shows the model CAN find correct content)
