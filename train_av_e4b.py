"""Train an Activation Verbalizer (AV) on Gemma-4-E4B — ARM A recipe port.

Forks train_prior_deviation_av.py with parameterised model id + d_model.
Carries all ARM A flags: --contrastive, --contrastive-domain-aware, --contrastive-negs,
--contrastive-beta, --contrastive-temp, --token-dropout, --paraphrase-data,
--mean-center (new: subtracts corpus mean from activation before injection).

VRAM note: 32 negatives/anchor (ARM A) cannot all run naively — this script chunks
negative forwards in batches of --neg-chunk (default 8) so peak VRAM stays bounded.
G2a memory smoke must exercise the real objective (--contrastive-domain-aware).

Key deltas from E2B ARM A:
  --base-model google/gemma-4-E4B    (replaces hard-coded E2B)
  --d-model <D_MODEL_E4B>            (drives injection_scale = sqrt(d_model))
  --mean-center                      (subtract corpus_mean_e4b.npy at injection time)
  --mean-file <path>                 (default: data/stage3_balanced/mean_activation_e4b.npy)
  --neg-chunk 8                      (chunk negative forwards to fit 4 GB VRAM)
  prints torch.cuda.max_memory_allocated() every --save-every steps

ARM A launch command for E4B (fill in LAYER_E4B and D_MODEL_E4B from Phase 1 + G1a):
    python -u train_av_e4b.py \\
        --base-model google/gemma-4-E4B \\
        --d-model <D_MODEL_E4B> \\
        --inject-layer <LAYER_E4B> \\
        --fresh --rank 8 \\
        --contrastive --contrastive-domain-aware \\
        --contrastive-negs 2 \\
        --max-label-tokens 16 \\
        --steps 1500 --save-every 50 \\
        --seed 17 --grad-accum 1 \\
        --weight-decay 0.01 --lr 5e-5 \\
        --mean-center \\
        --data data/stage3_balanced/av_sft_balanced_e4b.parquet \\
        --out checkpoints/av_e4b_stage2_A_domaware

GPU required. Launch inside the gpu-grant window ONLY.
"""
from __future__ import annotations
import argparse, json, math, os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# ---- injection token IDs (verified shared across Gemma-4 family, vocab=262144) ----
INJ_ID  = 249568   # ㊗ chr(0x3297)
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Model
    p.add_argument("--base-model", default="google/gemma-4-E4B")
    p.add_argument("--d-model", type=int, required=True, help="Residual stream width")
    p.add_argument("--inject-layer", type=int, default=-1,
                   help="Layer to inject at (default -1 = embedding layer)")
    # LoRA
    p.add_argument("--fresh", action="store_true", help="Fresh LoRA (not resumed from v0.1)")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--resume", type=Path, help="Resume from checkpoint dir")
    # Training
    p.add_argument("--data", type=Path, required=True, help="E4B activation parquet")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-label-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--optim", choices=["adamw8bit", "paged_adamw_8bit", "adamw", "sgd"],
                   default="adamw8bit")
    # Contrastive (ARM A objective)
    p.add_argument("--contrastive", action="store_true")
    p.add_argument("--contrastive-domain-aware", action="store_true",
                   help="Sample negatives from same/diff domains (ARM A recipe)")
    p.add_argument("--contrastive-negs", type=int, default=2,
                   help="Negatives per anchor (ARM A: 2 domains * 16 = total 32 but K per domain)")
    p.add_argument("--contrastive-beta", type=float, default=1.0)
    p.add_argument("--contrastive-temp", type=float, default=1.0)
    p.add_argument("--neg-chunk", type=int, default=8,
                   help="Chunk negatives in groups to fit VRAM (CRITICAL for 32 negs on 4GB)")
    p.add_argument("--token-dropout", type=float, default=0.0,
                   help="Bowman token-dropout rate (anti-posterior-collapse)")
    # Mean-centering (new lever from §F90)
    p.add_argument("--mean-center", action="store_true",
                   help="Subtract corpus mean from activation before injection")
    p.add_argument("--mean-file", type=Path, default=None,
                   help="Path to mean_activation_e4b.npy (default: data/stage3_balanced/mean_activation_e4b.npy)")
    # Paraphrase
    p.add_argument("--paraphrase-data", type=Path, default=None,
                   help="Parquet with 'paraphrases' column (4 per row) for output-side memorisation break")
    # Misc
    p.add_argument("--smoke", action="store_true", help="50 steps only")
    return p.parse_args()


def find_inj_pos(input_ids: torch.Tensor) -> int | None:
    ids = input_ids[0].tolist()
    for p in range(1, len(ids) - 1):
        if ids[p] == INJ_ID and ids[p-1] == LEFT_ID and ids[p+1] == RIGHT_ID:
            return p
    return None


def inject(h: torch.Tensor, pos: int, vec: torch.Tensor) -> torch.Tensor:
    h = h.clone()
    h[0, pos] = vec.to(h.dtype)
    return h


def make_hook(pending: dict, pos_cache: dict):
    def hook(module, inp, out):
        is_t = isinstance(out, tuple)
        o = out[0] if is_t else out
        if o.shape[1] <= 1:
            return out
        pos = pos_cache.get("pos")
        vec = pending.get("vec")
        if pos is None or vec is None:
            return out
        o2 = inject(o, pos, vec)
        return ((o2,) + tuple(out[1:])) if is_t else o2
    return hook


def infonce_loss(logits_pos: torch.Tensor, logits_negs: list[torch.Tensor],
                 temp: float) -> torch.Tensor:
    """InfoNCE: log(exp(pos/T) / (exp(pos/T) + sum(exp(neg_i/T))))."""
    pos_s = logits_pos.sum() / temp
    neg_s = torch.stack([l.sum() / temp for l in logits_negs])
    all_s = torch.cat([pos_s.unsqueeze(0), neg_s])
    return -pos_s + torch.logsumexp(all_s, dim=0)


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    import pyarrow.parquet as pq

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    inj_scale = math.sqrt(args.d_model)
    print(f"[e4b-av] base={args.base_model} d_model={args.d_model} inj_scale={inj_scale:.2f}")
    print(f"[e4b-av] inject_layer={args.inject_layer} rank={args.rank}")

    # Load corpus mean for mean-centering
    mean_vec = None
    if args.mean_center:
        mf = args.mean_file or Path(__file__).parent / "data/stage3_balanced/mean_activation_e4b.npy"
        if not mf.exists():
            raise SystemExit(f"[e4b-av] --mean-center requires {mf} (run stage0_reextract_e4b.py first)")
        mean_vec = np.load(mf).astype(np.float32)
        print(f"[e4b-av] mean-centering ON (norm={np.linalg.norm(mean_vec):.3f})")

    # Load data
    tbl = pq.read_table(args.data)
    rows = tbl.to_pylist()
    para_map: dict[str, list[str]] = {}
    if args.paraphrase_data and args.paraphrase_data.exists():
        ptbl = pq.read_table(args.paraphrase_data)
        prows = ptbl.to_pylist()
        for pr in prows:
            key = (pr.get("doc_id", ""), pr.get("response", ""))
            para_map[str(key)] = list(pr.get("paraphrases", []))
        print(f"[e4b-av] paraphrase map: {len(para_map)} entries")

    # Domain index for domain-aware negatives
    by_domain: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_domain.setdefault(str(r.get("domain", "unknown")), []).append(i)
    domains = sorted(by_domain)
    print(f"[e4b-av] {len(rows)} rows, {len(domains)} domains")
    print(f"[e4b-av] contrastive={args.contrastive} domain_aware={args.contrastive_domain_aware} "
          f"negs={args.contrastive_negs} neg_chunk={args.neg_chunk} beta={args.contrastive_beta}")

    # Load model
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb,
        device_map={"": torch.cuda.current_device()})

    # LoRA
    if args.resume:
        model = PeftModel.from_pretrained(base, args.resume, is_trainable=True)
        print(f"[e4b-av] resumed LoRA from {args.resume}")
    else:
        # Build target_modules from nla_model_params if available
        # Gemma4 vision encoder wraps proj layers in Gemma4ClippableLinear (not torch.nn.Linear)
        # so a list-style target_modules hits them and PEFT raises ValueError.
        # Using a regex with re.fullmatch() restricts LoRA to the language_model subtree only.
        target_modules = (
            r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        )
        lora_cfg = LoraConfig(
            r=args.rank, lora_alpha=args.rank * 2,
            target_modules=target_modules,
            lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base, lora_cfg)
        # Verify matched modules (Gemma4ClippableLinear false-match warning)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[e4b-av] FRESH LoRA r={args.rank} alpha={args.rank*2} trainable_params={n_trainable/1e6:.2f}M")

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.train()

    # Injection hook
    pending: dict = {}
    pos_cache: dict = {}
    if args.inject_layer >= 0:
        hook_module = base.model.language_model.layers[args.inject_layer]
    else:
        hook_module = model.get_input_embeddings()
    hook_handle = hook_module.register_forward_hook(make_hook(pending, pos_cache))
    # §F89 consistency contract: print hook module identity so eval can be verified to match.
    print(f"[e4b-av] injection hook registered on: {type(hook_module).__name__} "
          f"id={id(hook_module)} (eval must match)")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb_lib
        opt = bnb_lib.optim.AdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "paged_adamw_8bit":
        import bitsandbytes as bnb_lib
        opt = bnb_lib.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Tokenise template once
    tmpl_ids = tok.encode(AV_TEMPLATE, return_tensors="pt").to(model.device)
    inj_pos = find_inj_pos(tmpl_ids)
    if inj_pos is None:
        raise SystemExit(f"[e4b-av] FATAL: injection token {INJ_ID} not found in template. "
                         "Check tokenizer-identity gate (Phase 1 / Gate A2).")
    pos_cache["pos"] = inj_pos
    print(f"[e4b-av] injection position in template: {inj_pos} (of {tmpl_ids.shape[1]} tokens)")

    # Validate domain-aware negative pools (WARNING 3: rare domains may have <2 docs)
    if args.contrastive and args.contrastive_domain_aware:
        small_domains = {dom: idxs for dom, idxs in by_domain.items() if len(idxs) < 2}
        if small_domains:
            print(f"[e4b-av] WARNING: {len(small_domains)} domains have <2 rows "
                  f"(no same-domain negative possible): {list(small_domains.keys())}")
        print(f"[e4b-av] negative pool sizes: " +
              ", ".join(f"{dom}:{len(idxs)}" for dom, idxs in sorted(by_domain.items())))

    rng_step = np.random.RandomState(args.seed + 42)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "trainloss.csv"
    log_f = open(log_path, "a", buffering=1)
    if log_path.stat().st_size == 0:
        log_f.write("step,ce_loss,contrastive_loss,total_loss,skipped\n")

    steps = min(args.steps, 50) if args.smoke else args.steps
    accum_steps = 0
    skipped = 0
    opt.zero_grad()

    for step in range(1, steps + 1):
        idx = rng_step.randint(len(rows))
        row = rows[idx]
        raw = np.asarray(row["activation_vector"], dtype=np.float32)
        if mean_vec is not None:
            raw = raw - mean_vec
        vec = raw / (np.linalg.norm(raw) + 1e-9) * inj_scale
        vec_t = torch.from_numpy(vec.astype(np.float32)).to(model.device).unsqueeze(0)

        # Choose label (paraphrase sampling or fixed)
        label_text = row.get("response", "")
        key = str((row.get("doc_id", ""), row.get("response", "")))
        paras = para_map.get(key, [])
        if paras:
            label_text = paras[rng_step.randint(len(paras))]

        label_ids = tok.encode(label_text, add_special_tokens=False)
        if args.max_label_tokens > 0:
            label_ids = label_ids[:args.max_label_tokens]
        if not label_ids:
            skipped += 1
            continue

        # Token dropout (Bowman anti-posterior-collapse)
        if args.token_dropout > 0:
            keep = [i for i in label_ids if rng_step.random() >= args.token_dropout]
            if not keep:
                keep = label_ids[:1]
            label_ids = keep

        full_ids = torch.cat([tmpl_ids[0], torch.tensor(label_ids, device=model.device)])
        full_ids = full_ids.unsqueeze(0)
        labels = torch.full_like(full_ids, -100)
        labels[0, tmpl_ids.shape[1]:] = full_ids[0, tmpl_ids.shape[1]:]

        # CE loss (anchor)
        pending["vec"] = vec_t
        out = model(input_ids=full_ids, labels=labels)
        ce_loss = out.loss

        # Contrastive InfoNCE loss (domain-aware)
        ctr_loss = torch.tensor(0.0, device=model.device)
        if args.contrastive:
            dom = str(row.get("domain", "unknown"))
            neg_indices = _sample_neg_indices(
                idx, rows, by_domain, dom, args.contrastive_negs, rng_step,
                args.contrastive_domain_aware)
            neg_losses = []
            for chunk_start in range(0, len(neg_indices), args.neg_chunk):
                chunk = neg_indices[chunk_start:chunk_start + args.neg_chunk]
                for ni in chunk:
                    nr = rows[ni]
                    nr_raw = np.asarray(nr["activation_vector"], dtype=np.float32)
                    if mean_vec is not None:
                        nr_raw = nr_raw - mean_vec
                    nr_vec = nr_raw / (np.linalg.norm(nr_raw) + 1e-9) * inj_scale
                    pending["vec"] = torch.from_numpy(nr_vec.astype(np.float32)).to(model.device).unsqueeze(0)
                    with torch.no_grad():
                        nr_out = model(input_ids=full_ids, labels=labels)
                    neg_losses.append(nr_out.loss.detach())
            if neg_losses:
                # InfoNCE: log( exp(pos) / (exp(pos) + sum(exp(neg_i))) ) in loss-space
                pos_s = -ce_loss.detach() / args.contrastive_temp
                neg_s = torch.stack([-l / args.contrastive_temp for l in neg_losses])
                all_s = torch.cat([pos_s.unsqueeze(0), neg_s])
                ctr_loss = args.contrastive_beta * (-pos_s + torch.logsumexp(all_s, dim=0))

        total_loss = ce_loss + ctr_loss
        (total_loss / args.grad_accum).backward()
        accum_steps += 1

        if accum_steps == args.grad_accum:
            opt.step()
            opt.zero_grad()
            accum_steps = 0

        log_f.write(f"{step},{ce_loss.item():.4f},{ctr_loss.item():.4f},"
                    f"{total_loss.item():.4f},{skipped}\n")
        if step % 10 == 0:
            print(f"step {step}/{steps} ce={ce_loss.item():.4f} ctr={ctr_loss.item():.4f} "
                  f"total={total_loss.item():.4f}", flush=True)

        if step % args.save_every == 0 or step == steps:
            ckpt_dir = args.out / f"step_{step:06d}"
            model.save_pretrained(ckpt_dir)
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[e4b-av] saved {ckpt_dir}  peak_vram={peak:.2f}GB", flush=True)
            meta = {"step": step, "base_model": args.base_model,
                    "d_model": args.d_model, "inject_layer": args.inject_layer,
                    "injection_scale": inj_scale, "rank": args.rank,
                    "mean_center": args.mean_center}
            (ckpt_dir / "nla_meta.json").write_text(json.dumps(meta, indent=2))
            inject_cfg = {"inject_layer": args.inject_layer if args.inject_layer >= 0 else "embed",
                          "inject_mode": "center" if args.mean_center else "raw"}
            (ckpt_dir / "inject_config.json").write_text(json.dumps(inject_cfg))
            if args.mean_center and mean_vec is not None:
                np.save(str(ckpt_dir / "inject_mean.npy"), mean_vec)

    hook_handle.remove()
    log_f.close()
    print(f"[e4b-av] done. {steps} steps, {skipped} skipped.")


def _sample_neg_indices(anchor_idx: int, rows: list, by_domain: dict[str, list[int]],
                        anchor_dom: str, n_negs: int, rng: np.random.RandomState,
                        domain_aware: bool) -> list[int]:
    """Sample negative row indices (domain-aware if requested)."""
    if not domain_aware:
        pool = [i for i in range(len(rows)) if i != anchor_idx]
        return rng.choice(pool, size=min(n_negs, len(pool)), replace=False).tolist()

    # ARM A: sample from same-domain-different-doc AND different-domain pools
    same_dom = [i for i in by_domain.get(anchor_dom, []) if i != anchor_idx]
    diff_dom = [i for dom, idxs in by_domain.items() if dom != anchor_dom for i in idxs]

    negs = []
    half = n_negs // 2
    if same_dom:
        negs += rng.choice(same_dom, size=min(half, len(same_dom)), replace=False).tolist()
    if diff_dom:
        negs += rng.choice(diff_dom, size=min(n_negs - len(negs), len(diff_dom)),
                           replace=False).tolist()
    # pad if needed
    all_pool = [i for i in range(len(rows)) if i != anchor_idx and i not in negs]
    while len(negs) < n_negs and all_pool:
        pick = rng.choice(all_pool)
        negs.append(int(pick))
        all_pool.remove(pick)
    return negs[:n_negs]


if __name__ == "__main__":
    main()
