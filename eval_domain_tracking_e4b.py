"""Domain-tracking eval for Gemma-4-E4B AV checkpoints.

Fork of eval_domain_tracking.py parameterized for any base model.
Reads inject_config.json from the checkpoint to match training's injection mode
exactly (§F89 train/eval-consistency contract). Includes the upstream gotcha fix:
results/domain_tracking_e4b/ is pre-created before any file redirection.

Usage:
    python eval_domain_tracking_e4b.py <ckpt> \\
        [--eval data/stage3_balanced/balanced_eval_e4b.parquet] \\
        [--train data/stage3_balanced/av_sft_balanced_e4b.parquet] \\
        [--max-new 40] [--min-proto 3] [--tag NAME]

Output: results/domain_tracking_e4b/<tag>.json
"""
from __future__ import annotations
import argparse, json, math, os
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


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))


def bnb():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--eval",  default=str(HERE / "data/stage3_balanced/balanced_eval_e4b.parquet"))
    ap.add_argument("--train", default=str(HERE / "data/stage3_balanced/av_sft_balanced_e4b.parquet"))
    ap.add_argument("--max-new",  type=int, default=40)
    ap.add_argument("--min-proto",type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ckpt = Path(a.ckpt)
    rng = np.random.RandomState(a.seed)
    tag = a.tag or ckpt.name

    # Read checkpoint config (§F89 consistency contract)
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
        raise SystemExit(f"[dt-e4b] nla_meta.json missing d_model in {ckpt}")
    inj_scale = meta.get("injection_scale", float(np.sqrt(d_model)))

    mean_vec = None
    if cfg.get("inject_mode") == "center":
        for d in (ckpt, ckpt.parent):
            mf = d / "inject_mean.npy"
            if mf.exists():
                mean_vec = np.load(mf).astype(np.float32); break
        if mean_vec is None:
            raise SystemExit(
                f"[dt-e4b] FATAL: inject_mode=center but inject_mean.npy not found in "
                f"{ckpt} or {ckpt.parent}. §F89: silent train/eval mismatch causes false FAIL."
            )

    print(f"[dt-e4b] {ckpt.name} base={base_model} d_model={d_model} "
          f"inject={cfg['inject_layer']} mode={cfg['inject_mode']}", flush=True)

    # Build prototypes from source doc texts (de-circularized — NOT from generations/labels)
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    src: dict[str, list[str]] = {}
    for path in [a.train, a.eval]:
        if not os.path.exists(path):
            continue
        d = pd.read_parquet(path)
        if "detokenized_text_truncated" not in d.columns:
            print(f"[dt-e4b] WARNING: {path} has no detokenized_text_truncated — excluded from prototypes")
            continue
        for dom, txt in zip(d["domain"], d["detokenized_text_truncated"]):
            if isinstance(txt, str) and len(txt.strip()) > 20:
                src.setdefault(str(dom), []).append(txt.strip())
    protos = {dom: emb_model.encode(txts[:200], normalize_embeddings=True).mean(0)
              for dom, txts in src.items() if len(txts) >= a.min_proto}
    for dom in protos:
        protos[dom] = protos[dom] / (np.linalg.norm(protos[dom]) + 1e-9)
    covered = sorted(protos)
    excluded = sorted(set(src) - set(protos))
    print(f"[dt-e4b] prototypes: {len(covered)} covered / {len(excluded)} excluded (<{a.min_proto} texts)", flush=True)
    if excluded:
        print(f"         excluded: {excluded}")

    # Generate verbalizations
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb(),
                                                device_map={"": torch.cuda.current_device()})
    av = PeftModel.from_pretrained(base, ckpt); av.eval()
    dev = av.device

    tmpl = tok.encode(AV_TEMPLATE, return_tensors="pt").to(dev)
    inj_pos = None
    for p2 in range(1, tmpl.shape[1] - 1):
        if tmpl[0, p2].item() == INJ_ID and tmpl[0, p2-1].item() == LEFT_ID and tmpl[0, p2+1].item() == RIGHT_ID:
            inj_pos = p2; break
    if inj_pos is None:
        raise SystemExit("[dt-e4b] FATAL: injection token not found in template. Check tokenizer gate.")

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
    handle = hm.register_forward_hook(hook)

    eval_df = pd.read_parquet(a.eval)
    # Only rows whose domain has a prototype
    eval_df = eval_df[eval_df["domain"].isin(covered)].reset_index(drop=True)
    print(f"[dt-e4b] eval rows with covered domains: {len(eval_df)}", flush=True)

    gens, true_doms = [], []
    for i in range(len(eval_df)):
        raw = np.asarray(eval_df["activation_vector"].iloc[i], dtype=np.float32)
        if mean_vec is not None:
            raw = raw - mean_vec
        vec = raw / (np.linalg.norm(raw) + 1e-9) * inj_scale
        pend["vec"] = torch.from_numpy(vec.astype(np.float32)).to(dev).unsqueeze(0)
        with torch.no_grad():
            g = av.generate(input_ids=tmpl, max_new_tokens=a.max_new, min_new_tokens=6,
                            do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(g[0][tmpl.shape[1]:], skip_special_tokens=True).strip()
        gens.append(gen)
        true_doms.append(str(eval_df["domain"].iloc[i]))
        if i % 20 == 0:
            print(f"  [{i}/{len(eval_df)}] dom={true_doms[-1]} gen={gen[:60]!r}", flush=True)
    handle.remove()

    # Classify by nearest prototype
    gen_embs = emb_model.encode(gens, normalize_embeddings=True)
    proto_names = covered
    proto_mat = np.stack([protos[d] for d in proto_names])
    sims = gen_embs @ proto_mat.T
    pred_doms = [proto_names[int(np.argmax(row))] for row in sims]

    # Accuracy + Wilson CI + permutation null
    hits = sum(p == t for p, t in zip(pred_doms, true_doms))
    n = len(true_doms)
    acc = hits / n if n else 0.0
    ci = wilson(hits, n)
    chance = 1.0 / len(covered)

    perm_accs = []
    for _ in range(2000):
        shuffled = rng.permutation(pred_doms).tolist()
        perm_accs.append(sum(p == t for p, t in zip(shuffled, true_doms)) / n)
    p_val = float((np.sum(np.array(perm_accs) >= acc) + 1) / (len(perm_accs) + 1))

    # Domain-lock check: unique generation strings
    n_unique = len(set(gens))
    domain_lock = n_unique < max(5, len(covered) // 2)

    # Within/cross-domain stratified
    within_hits = sum(p == t for p, t in zip(pred_doms, true_doms))
    by_dom = {}
    for p, t in zip(pred_doms, true_doms):
        by_dom.setdefault(t, []).append(p == t)
    per_dom_acc = {d: sum(v)/len(v) for d, v in by_dom.items()}

    print(f"\n[dt-e4b] accuracy={acc:.3f} (chance={chance:.3f}) p={p_val:.4f} CI={ci}")
    print(f"[dt-e4b] n_unique={n_unique}/{n} domain_lock={'YES ⚠' if domain_lock else 'no'}")
    print(f"[dt-e4b] per-domain accuracy: {per_dom_acc}")

    # Upstream gotcha: pre-create output dir before anything tries to redirect there
    out_dir = HERE / "results/domain_tracking_e4b"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "checkpoint": str(ckpt), "eval": a.eval, "n": n,
        "covered_domains": covered, "excluded_domains": excluded,
        "accuracy": acc, "chance": chance, "p_value": p_val,
        "wilson_ci_95": list(ci),
        "n_unique": n_unique, "n_total": n, "domain_lock": domain_lock,
        "per_domain_acc": per_dom_acc,
        "generations_sample": list(zip(true_doms[:10], gens[:10])),
    }
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[dt-e4b] wrote {out_path}")


if __name__ == "__main__":
    main()
