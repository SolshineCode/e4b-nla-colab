"""Layer ceiling sweep for Gemma-4-E4B.

Which layer's last-token activation is most document-discriminative? One forward pass
per text captures all swept layers simultaneously via hooks — candidate count is nearly free.

Sweeps 6 fractional depths of N_LAYERS_E4B by default:
  {0.30, 0.40, 0.49, 0.57, 0.66, 0.75} x N_LAYERS

Same 200 texts from data/stage1/rl.parquet as the E2B sweep (cross-model comparable).
Also runs a linear probe (logistic regression, CV) as a secondary metric (secondary because
the E2B lesson: both L17 and L23 had ~0.99 probe AUC while conditioning differed — ceiling
is necessary-not-sufficient, but helps rule out layers with very low decodability).

SELECTION RULE (implemented here as guidance, not hard code):
  Pick the highest doc_top1 mid-depth layer (fraction nearest 0.49).
  Among layers within 0.10 of the top doc_top1: tie-break on probe CV, then mid-depth proximity.
  ALL < 0.4: print warning — stop GPU block and widen sweep.

Output:
  results/e4b/layer_ceiling_sweep_e4b.json

GPU required. Launch inside the gpu-grant window.

Usage:
    python -u eval_layer_ceiling_sweep_e4b.py \\
        --base-model google/gemma-4-E4B \\
        --n-layers <N_LAYERS_E4B> \\
        [--fractions 0.30,0.40,0.49,0.57,0.66,0.75] \\
        [--n-texts 200] [--n-perm 2000]
"""
from __future__ import annotations
import argparse, json, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

HERE = Path(__file__).parent
RL = HERE / "data/stage1/rl.parquet"
OUT = HERE / "results/e4b/layer_ceiling_sweep_e4b.json"
DEFAULT_FRACTIONS = [0.30, 0.40, 0.49, 0.57, 0.66, 0.75]
N_PERM_DEFAULT = 2000
RNG = np.random.RandomState(0)


def cosine_sim(acts: np.ndarray) -> np.ndarray:
    A = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-9)
    sim = A @ A.T
    np.fill_diagonal(sim, -np.inf)
    return sim


def doc_top1(sim: np.ndarray, doc: list) -> tuple[float, int]:
    n = sim.shape[0]
    docs = sorted(set(doc))
    cols = {d: [j for j in range(n) if doc[j] == d] for d in docs}
    didx = {d: k for k, d in enumerate(docs)}
    hit = 0
    for i in range(n):
        scores = np.array([sim[i, cols[d]].max() if cols[d] else -np.inf for d in docs])
        if int(np.argmax(scores)) == didx[doc[i]] and len(cols[doc[i]]) > 1:
            hit += 1
    return hit / n, len(docs)


def perm_test(sim: np.ndarray, doc: list, n_perm: int) -> dict:
    obs, nd = doc_top1(sim, doc)
    d = np.array(doc, dtype=object)
    null = np.array([doc_top1(sim, list(RNG.permutation(d)))[0] for _ in range(n_perm)])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"doc_top1": float(obs), "n_docs": nd, "chance": 1.0 / nd,
            "p_value": p, "z": float((obs - null.mean()) / (null.std() + 1e-12))}


def linear_probe_cv(acts: np.ndarray, doc: list, cv: int = 5) -> float:
    """5-fold cross-validated logistic regression accuracy on doc-id labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y = le.fit_transform(doc)
    counts = np.bincount(y)
    valid_classes = np.where(counts >= cv)[0]
    mask = np.isin(y, valid_classes)
    if mask.sum() < cv * 2:
        return float("nan")
    X = StandardScaler().fit_transform(acts[mask])
    y_masked = y[mask]
    clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs", multi_class="multinomial")
    scores = cross_val_score(clf, X, y_masked, cv=cv, scoring="accuracy", n_jobs=1)
    return float(scores.mean())


def select_layer(per_layer: dict) -> tuple[int, str]:
    """Return (best_layer, reason) per the plan's selection rule."""
    top_val = max(v["doc_top1"] for v in per_layer.values())
    candidates = {li: v for li, v in per_layer.items() if v["doc_top1"] >= top_val - 0.10}
    # Among candidates: prefer highest probe CV, then mid-depth proximity (target fraction 0.49)
    n_layers = per_layer.get("_n_layers")
    def score(li):
        cv = per_layer[li].get("probe_cv", 0.0) or 0.0
        depth = li / n_layers if n_layers else 0.5
        mid_dist = abs(depth - 0.49)
        return (cv, -mid_dist)
    best = max(candidates, key=score)
    reason = (f"doc_top1={per_layer[best]['doc_top1']:.3f} (within 0.10 of top {top_val:.3f}); "
              f"probe_cv={per_layer[best].get('probe_cv', 'N/A')}; "
              f"depth={best}/{n_layers}={best/n_layers:.2f}")
    return best, reason


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="google/gemma-4-E4B")
    ap.add_argument("--n-layers", type=int, required=True, help="Total transformer layers in E4B")
    ap.add_argument("--fractions", default=",".join(str(f) for f in DEFAULT_FRACTIONS),
                    help="Comma-separated fractional depths to sweep")
    ap.add_argument("--d-model", type=int, default=None,
                    help="Residual stream width (fill from nla_model_params or pass explicitly; "
                         "avoids model.config.text_config crash on E4B)")
    ap.add_argument("--n-texts", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--source", default=str(RL), help="Parquet with detokenized_text_truncated + doc_id")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    fractions = [float(f) for f in args.fractions.split(",")]
    layers = sorted(set(int(round(f * args.n_layers)) for f in fractions))
    layers = [li for li in layers if 0 <= li < args.n_layers]
    print(f"[sweep] base={args.base_model} n_layers={args.n_layers}")
    print(f"[sweep] sweeping layers: {layers} (fractions ~{[f'{li/args.n_layers:.2f}' for li in layers]})")

    df = pd.read_parquet(args.source).iloc[:args.n_texts]
    texts = list(df["detokenized_text_truncated"])
    doc = list(df["doc_id"])
    print(f"[sweep] {len(texts)} texts, {len(set(doc))} docs")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb,
        device_map={"": torch.cuda.current_device()})
    model.eval()

    # Verify the target_modules_regex matches E4B (warn on suspicious counts)
    try:
        import re
        from nla_model_params import PARAMS
        regex = PARAMS.get("google/gemma-4-e4b", {}).get("target_modules_regex")
        if regex:
            matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
            print(f"[sweep] LoRA target regex matches {len(matched)} modules (expect ~{args.n_layers * 7})")
    except Exception:
        pass

    layer_modules = model.model.language_model.layers
    captured: dict[int, np.ndarray] = {}
    handles = []
    for li in layers:
        def mk(li):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[li] = h[0, -1, :].detach().cpu().float().numpy()
            return hook
        handles.append(layer_modules[li].register_forward_hook(mk(li)))

    # Resolve d_model from args (passed explicitly) — do NOT use model.config.text_config
    # which is a Gemma-3/multimodal artifact absent on E4B (would crash or give wrong shape).
    # The --d-model arg is required and was verified against config.json in Phase 1.
    d_model_for_fill = args.d_model if hasattr(args, "d_model") else None
    if d_model_for_fill is None:
        # Last resort: try the text_config path then hidden_size directly
        d_model_for_fill = (getattr(getattr(model.config, "text_config", None), "hidden_size", None)
                            or getattr(model.config, "hidden_size", 2560))
    print(f"[sweep] d_model for zero-fill fallback: {d_model_for_fill}")

    acts = {li: [] for li in layers}
    with torch.no_grad():
        for t in texts:
            ids = tok.encode(t, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            captured.clear()
            model(input_ids=ids)
            for li in layers:
                if li in captured:
                    acts[li].append(captured[li].copy())
                else:
                    print(f"  WARNING: layer {li} not captured for text '{t[:40]}' — zero-filling")
                    acts[li].append(np.zeros(d_model_for_fill, dtype=np.float32))
    for h in handles:
        h.remove()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    per_layer: dict[int, dict] = {"_n_layers": args.n_layers}
    for li in layers:
        A = np.stack(acts[li])
        r = perm_test(cosine_sim(A), doc, args.n_perm)
        try:
            r["probe_cv"] = linear_probe_cv(A, doc)
        except Exception as e:
            r["probe_cv"] = None
            print(f"  L{li} probe failed: {e}")
        per_layer[li] = r
        print(f"L{li} ({li/args.n_layers:.2f}): doc_top1={r['doc_top1']:.3f} "
              f"(chance {r['chance']:.3f}, p={r['p_value']:.4f}, z={r['z']:.1f}) "
              f"probe_cv={r.get('probe_cv')}", flush=True)

    best_li, reason = select_layer(per_layer)
    top_val = max(v["doc_top1"] for k, v in per_layer.items() if k != "_n_layers")
    gate_pass = top_val >= 0.40

    result = {
        "config": {"base_model": args.base_model, "n_layers": args.n_layers,
                   "layers_swept": layers, "n_texts": len(texts),
                   "n_docs": len(set(doc)), "n_perm": args.n_perm},
        "per_layer": {str(li): v for li, v in per_layer.items() if li != "_n_layers"},
        "recommended_layer": best_li,
        "recommended_layer_reason": reason,
        "gate_pass": gate_pass,
        "gate_message": ("PASS: proceed to extraction" if gate_pass else
                         "FAIL: all doc_top1 < 0.40 — stop block, widen sweep before extraction"),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\n[sweep] wrote {OUT}")
    print(f"[sweep] RECOMMENDED LAYER: L{best_li} — {reason}")
    print(f"[sweep] GATE: {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        raise SystemExit("[sweep] Gate fail: all layers < 0.40 doc_top1. Widen sweep before extraction.")


if __name__ == "__main__":
    main()
