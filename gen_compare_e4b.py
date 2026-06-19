"""Qualitative gen-compare for E4B AV checkpoints — mode-collapse + domain-lock guard.

Fork of gen_compare.py for E4B, with:
  - Reads inject_config.json + nla_meta.json for model/injection config (§F89)
  - Generates from both the E4B eval parquet (balanced_eval_e4b) and the legacy
    indomain_eval_cmp_e4b (for cross-model comparability with E2B §F87B)
  - Mode-collapse guard: prints n_unique / n and flags if < 5 (§F87B lesson)
  - Domain-lock check: if all outputs share the same predicted domain → flag
  - Persists to results/gen_compare_e4b/<tag>.json

Usage:
    python gen_compare_e4b.py <ckpt> [--n 12] [--max-new 48] [--tag NAME]
    python gen_compare_e4b.py <ckpt> --eval-cmp  # use legacy indomain_eval_cmp_e4b
"""
from __future__ import annotations
import argparse, json, os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np, pandas as pd, torch
from pathlib import Path

HERE = Path(__file__).parent
INJ_ID  = 249568
LEFT_ID = 236813
RIGHT_ID = 954
INJ_CHAR = chr(0x3297)
AV_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation into activation "
    "vectors from a language model. Your overall task is to describe the semantic content of "
    "that activation vector.\n\nWe will pass the vector enclosed in <concept> tags into your "
    "context. You must then produce an explanation for the vector, enclosed within "
    "<explanation> tags. The explanation consists of 2-3 text snippets describing that vector."
    f"\n\nHere is the vector:\n\n<concept>{INJ_CHAR}</concept>"
)
DEFAULT_EVAL = HERE / "data/stage3_balanced/balanced_eval_e4b.parquet"
LEGACY_EVAL  = HERE / "data/stage3_v0_4_fineweb/indomain_eval_cmp_e4b.parquet"


def bnb():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval-cmp", action="store_true", help="Use legacy indomain_eval_cmp_e4b")
    a = ap.parse_args()

    ckpt = Path(a.ckpt)
    eval_path = LEGACY_EVAL if a.eval_cmp else DEFAULT_EVAL
    if not eval_path.exists():
        raise SystemExit(f"[gc-e4b] eval parquet not found: {eval_path}\n"
                         "Run stage0_reextract_e4b.py first (GPU required).")

    # Read checkpoint config
    cfg = {"inject_layer": "embed", "inject_mode": "raw"}
    for d in (ckpt, ckpt.parent):
        cf = d / "inject_config.json"
        if cf.exists():
            cfg = json.loads(cf.read_text()); break
    inj_layer = None if str(cfg["inject_layer"]) == "embed" else int(cfg["inject_layer"])

    meta = {}
    for d in (ckpt, ckpt.parent):
        mf = d / "nla_meta.json"
        if mf.exists():
            meta = json.loads(mf.read_text()); break
    base_model = meta.get("base_model", "google/gemma-4-E4B")
    d_model = meta.get("d_model")
    if d_model is None:
        raise SystemExit(f"[gc-e4b] nla_meta.json missing d_model in {ckpt}")
    inj_scale = meta.get("injection_scale", float(np.sqrt(d_model)))

    mean_vec = None
    if cfg.get("inject_mode") == "center":
        for d in (ckpt, ckpt.parent):
            mf = d / "inject_mean.npy"
            if mf.exists():
                mean_vec = np.load(mf).astype(np.float32); break

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb(),
                                                device_map={"": torch.cuda.current_device()})
    av = PeftModel.from_pretrained(base, ckpt); av.eval()
    dev = av.device
    ids = tok.encode(AV_TEMPLATE, return_tensors="pt").to(dev)

    inj_pos = None
    for p2 in range(1, ids.shape[1] - 1):
        if ids[0, p2].item() == INJ_ID and ids[0, p2-1].item() == LEFT_ID and ids[0, p2+1].item() == RIGHT_ID:
            inj_pos = p2; break
    if inj_pos is None:
        raise SystemExit("[gc-e4b] FATAL: injection token not found. Check tokenizer gate (Phase 1).")

    pend = {"vec": None}
    def hook(module, inp, out):
        is_t = isinstance(out, tuple); o = out[0] if is_t else out
        if o.shape[1] <= 1 or pend["vec"] is None:
            return out
        h = o.clone(); h[0, inj_pos] = pend["vec"][0].to(h.dtype)
        return ((h,) + tuple(out[1:])) if is_t else h

    if inj_layer is not None:
        hm = av.get_base_model().model.language_model.layers[inj_layer]
    else:
        hm = av.get_input_embeddings()
    hm.register_forward_hook(hook)

    df = pd.read_parquet(eval_path)
    n = min(a.n, len(df))

    recs = []
    for i in range(n):
        raw = np.asarray(df["activation_vector"].iloc[i], dtype=np.float32)
        if mean_vec is not None:
            raw = raw - mean_vec
        vec = raw / (np.linalg.norm(raw) + 1e-9) * inj_scale
        pend["vec"] = torch.from_numpy(vec.astype(np.float32)).to(dev).unsqueeze(0)
        src = str(df["detokenized_text_truncated"].iloc[i]) if "detokenized_text_truncated" in df.columns else ""
        with torch.no_grad():
            g = av.generate(input_ids=ids, max_new_tokens=a.max_new, min_new_tokens=6,
                            do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True).strip()
        recs.append({"i": i, "doc_id": str(df["doc_id"].iloc[i]),
                     "domain": str(df.get("domain", pd.Series(["?"])).iloc[i] if "domain" in df.columns else "?"),
                     "source": src[:220], "generation": gen})
        print(f"\n[{i}] DOC={df['doc_id'].iloc[i]} DOM={recs[-1]['domain']}\n"
              f"  SRC: {src[:160]}\n  GEN: {gen}", flush=True)

    # Mode-collapse guard
    n_unique = len(set(r["generation"] for r in recs))
    mode_collapsed = n_unique < 5
    print(f"\n[gc-e4b] n_unique={n_unique}/{n} — {'MODE COLLAPSE ⚠' if mode_collapsed else 'OK'}")

    # Domain-lock check
    doms = [r["domain"] for r in recs]
    gen_doms = [r["generation"][:30] for r in recs]
    all_same = len(set(gen_doms)) == 1
    print(f"[gc-e4b] domain-lock={'YES ⚠' if all_same else 'no'}")

    od = HERE / "results/gen_compare_e4b"; od.mkdir(parents=True, exist_ok=True)
    tag = a.tag or ckpt.parent.name + "_" + ckpt.name
    fp = od / f"gen_{tag}.json"
    fp.write_text(json.dumps({
        "checkpoint": str(ckpt), "n": n, "n_unique": n_unique,
        "mode_collapsed": mode_collapsed, "domain_lock": all_same,
        "guard_pass": not mode_collapsed and not all_same,
        "records": recs}, indent=2))
    print(f"[gc-e4b] wrote {fp}")
    if mode_collapsed:
        raise SystemExit(f"[gc-e4b] GUARD FAIL: mode-collapsed (n_unique={n_unique} < 5). "
                         "Check training: may need higher rank, more steps, or different objective.")


if __name__ == "__main__":
    main()
