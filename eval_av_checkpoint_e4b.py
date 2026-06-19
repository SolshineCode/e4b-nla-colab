"""Evaluate an E4B AV checkpoint via doc-level TF-IDF + semantic retrieval.

Fork of eval_av_checkpoint.py parameterized for any base model + d_model.
Reads inject_config.json from the checkpoint to match training's injection mode exactly
(embedding vs layer; raw vs mean-centered injection) — the §F89 train/eval-consistency contract.

Usage:
    python eval_av_checkpoint_e4b.py <checkpoint_dir> [n_rows] [eval_parquet]

Defaults to data/stage3_balanced/balanced_eval_e4b.parquet (primary).
For legacy anchor comparability: pass data/stage3_v0_4_fineweb/indomain_eval_cmp_e4b.parquet.

Output: results/content_aware_eval/<tag>_e4b.json
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

HERE = Path(__file__).parent
RESDIR = HERE / "results/content_aware_eval"
DEFAULT_EVAL = HERE / "data/stage3_balanced/balanced_eval_e4b.parquet"

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

MAX_NEW = 24
N_PERM  = 5000
RNG = np.random.RandomState(0)


def bnb():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)


def doc_top1(sim, doc):
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


def perm(sim, doc):
    obs, nd = doc_top1(sim, doc)
    idx = np.arange(sim.shape[0])
    null = np.array([doc_top1(sim[RNG.permutation(idx)], doc)[0] for _ in range(N_PERM)])
    return {"doc_top1": float(obs), "n_docs": nd, "chance": 1.0 / nd,
            "p_value": float((np.sum(null >= obs) + 1) / (N_PERM + 1)),
            "z": float((obs - null.mean()) / (null.std() + 1e-12))}


def main():
    ckpt = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 144
    eval_parquet = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_EVAL

    # Read inject_config.json (§F89 consistency contract)
    cfg = {"inject_layer": "embed", "inject_mode": "raw"}
    cfg_path = ckpt / "inject_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    inject_layer = None if str(cfg["inject_layer"]) == "embed" else int(cfg["inject_layer"])

    # Read nla_meta.json for d_model + injection_scale
    meta = {}
    meta_path = ckpt / "nla_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    base_model = meta.get("base_model", "google/gemma-4-E4B")
    d_model = meta.get("d_model")
    if d_model is None:
        raise SystemExit(f"[e4b-eval] nla_meta.json missing d_model in {ckpt}")
    inj_scale = meta.get("injection_scale", float(np.sqrt(d_model)))

    # Mean-centering
    mean_vec = None
    if cfg.get("inject_mode") == "center":
        for d in (ckpt, ckpt.parent):
            mf = d / "inject_mean.npy"
            if mf.exists():
                mean_vec = np.load(mf).astype(np.float32)
                break
        if mean_vec is None:
            # Hard error — §F89: silent train/eval mismatch causes false FAIL verdicts.
            # If training used mean-center, eval MUST also center or the result is meaningless.
            raise SystemExit(
                f"[e4b-eval] FATAL: inject_mode=center in {cfg_path} but inject_mean.npy "
                f"not found in {ckpt} or {ckpt.parent}. "
                "Cannot evaluate with different injection than training used."
            )

    print(f"[e4b-eval] {ckpt.name} on {eval_parquet.name}")
    print(f"[e4b-eval] base={base_model} d_model={d_model} inject_layer={cfg['inject_layer']} "
          f"mode={cfg['inject_mode']} inj_scale={inj_scale:.2f}")

    df = pd.read_parquet(eval_parquet).head(n)
    src = list(df["detokenized_text_truncated"])
    doc = list(df["doc_id"])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb(),
                                                device_map={"": torch.cuda.current_device()})
    av = PeftModel.from_pretrained(base, ckpt)
    av.eval()

    pend = {"vec": None}
    pos_cache = {}

    def hook(module, inp, out):
        is_t = isinstance(out, tuple); o = out[0] if is_t else out
        if o.shape[1] <= 1 or pend["vec"] is None:
            return out
        pos = pos_cache.get("pos")
        if pos is None:
            return out
        h = o.clone(); h[0, pos] = pend["vec"][0].to(h.dtype)
        return ((h,) + tuple(out[1:])) if is_t else h

    ids = tok.encode(AV_TEMPLATE, return_tensors="pt").to(av.device)
    for p2 in range(1, ids.shape[1] - 1):
        if ids[0, p2].item() == INJ_ID and ids[0, p2-1].item() == LEFT_ID and ids[0, p2+1].item() == RIGHT_ID:
            pos_cache["pos"] = p2; break

    if inject_layer is not None:
        hm = av.get_base_model().model.language_model.layers[inject_layer]
    else:
        hm = av.get_input_embeddings()
    handle = hm.register_forward_hook(hook)

    outs = []
    for i in range(len(df)):
        raw = np.asarray(df["activation_vector"].iloc[i], dtype=np.float32)
        if mean_vec is not None:
            raw = raw - mean_vec
        vec = raw / (np.linalg.norm(raw) + 1e-9) * inj_scale
        pend["vec"] = torch.from_numpy(vec.astype(np.float32)).to(av.device).unsqueeze(0)
        with torch.no_grad():
            g = av.generate(input_ids=ids, max_new_tokens=MAX_NEW, min_new_tokens=4,
                            do_sample=False, pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True).strip())
    handle.remove()

    # TF-IDF retrieval
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf_mat = vect.fit_transform(outs)
    tfidf_sim = cosine_similarity(tfidf_mat).astype(np.float32)
    np.fill_diagonal(tfidf_sim, -np.inf)
    tfidf_r = perm(tfidf_sim, doc)

    # Semantic retrieval
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = emb.encode(outs, normalize_embeddings=True)
    sem_sim = (vecs @ vecs.T).astype(np.float32)
    np.fill_diagonal(sem_sim, -np.inf)
    sem_r = perm(sem_sim, doc)

    n_unique = len(set(outs))
    print(f"[e4b-eval] tfidf: {tfidf_r}")
    print(f"[e4b-eval] semantic: {sem_r}")
    print(f"[e4b-eval] n_unique: {n_unique}/{len(outs)}")

    tag = ckpt.parent.name + "_" + ckpt.name + "_on_" + eval_parquet.stem + "_e4b"
    RESDIR.mkdir(parents=True, exist_ok=True)
    out_path = RESDIR / f"{tag}_e4b.json"
    result = {"checkpoint": str(ckpt), "eval": str(eval_parquet), "n": len(df),
              "tfidf": tfidf_r, "semantic": sem_r, "n_unique": n_unique, "n_total": len(outs),
              "generations_sample": outs[:5]}
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[e4b-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
