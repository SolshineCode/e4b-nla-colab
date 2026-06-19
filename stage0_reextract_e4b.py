"""Re-extract residual-stream activations from Gemma-4-E4B for the balanced corpus.

Reads e4b_textjoin_train.parquet (or any parquet with detokenized_text_truncated),
forwards each text through E4B in 4-bit NF4, captures the last-position residual at
--layer (from the ceiling sweep), and also captures multi-position activations (up to
--extra-positions per doc) in the same forward pass (nearly free).

Writes:
  data/stage3_balanced/av_sft_balanced_e4b.parquet     (train; last-position rows)
  data/stage3_balanced/av_sft_balanced_e4b_multi.parquet  (train; multi-position, position_idx tagged)
  data/stage3_balanced/balanced_eval_e4b.parquet       (eval, last-position)
  data/stage3_v0_4_fineweb/indomain_eval_cmp_e4b.parquet  (legacy eval, last-position)
  data/stage3_balanced/mean_activation_e4b.npy         (corpus mean of train activations, for mean-centering)
  results/e4b/stage0_reextract_e4b.json                (stats)

Resumable: --resume skips rows already in the output parquet (matched by doc_id + detokenized_text_truncated).
Use --smoke for a 10-row test without writing the final parquet.

GPU required. Launch inside the gpu-grant window ONLY.

Usage:
    python -u stage0_reextract_e4b.py \\
        --source data/stage3_balanced/e4b_textjoin_train.parquet \\
        --output data/stage3_balanced/av_sft_balanced_e4b.parquet \\
        --layer <LAYER_E4B> --d-model <D_MODEL_E4B> \\
        --base-model google/gemma-4-E4B \\
        [--extra-positions 4] [--resume] [--smoke]
"""
from __future__ import annotations
import argparse, json, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pathlib import Path
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Input textjoin parquet")
    p.add_argument("--output", type=Path, required=True, help="Output parquet (last-position)")
    p.add_argument("--base-model", default="google/gemma-4-E4B")
    p.add_argument("--layer", type=int, required=True, help="Layer index from ceiling sweep")
    p.add_argument("--d-model", type=int, required=True, help="Residual stream width")
    p.add_argument("--extra-positions", type=int, default=4,
                   help="Additional positions per doc to capture (multi-position lever)")
    p.add_argument("--resume", action="store_true", help="Skip already-extracted rows")
    p.add_argument("--smoke", action="store_true", help="First 10 rows only, no write")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--append-every", type=int, default=200,
                   help="Append-to-parquet every N rows (checkpoint-resumable)")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = pq.read_table(args.source).to_pylist()
    if args.smoke:
        rows = rows[:10]
    print(f"[e4b-extract] {len(rows)} rows from {args.source}")

    # NOTE: 703/1356 balanced-corpus rows have null detokenized_text_truncated (Opus/old-domain
    # rows — text not carried in this parquet). Use row-index as the stable resume key so
    # null-text rows don't cause stale-skip corruption across resumes.
    # Row indices are stable because the source parquet is read in fixed order every run.
    already_done_idx: set[int] = set()
    if args.resume and args.output.exists():
        # Count how many rows are already in the output to derive the done-index set.
        n_done = pq.read_table(args.output, columns=["doc_id"]).num_rows
        already_done_idx = set(range(n_done))
        print(f"[e4b-extract] resume: {n_done} rows already extracted (indices 0..{n_done-1}), skipping")

    # Filter null-text rows up front (they cannot be extracted; will be logged as skipped)
    rows_with_idx = [(i, r) for i, r in enumerate(rows)]
    rows_todo = [(i, r) for i, r in rows_with_idx
                 if i not in already_done_idx and r.get("detokenized_text_truncated")]
    n_null_text = sum(1 for i, r in rows_with_idx
                      if i not in already_done_idx and not r.get("detokenized_text_truncated"))
    print(f"[e4b-extract] {len(rows_todo)} rows to extract at L{args.layer} "
          f"({n_null_text} null-text rows skipped; these need stage0 text lookup — not yet implemented)")

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb,
        device_map={"": torch.cuda.current_device()},
    )
    model.eval()

    # Hook the target layer
    layer_module = model.model.language_model.layers[args.layer]
    captured = {}
    def capture_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["acts"] = h.detach().cpu().float()
    handle = layer_module.register_forward_hook(capture_hook)

    out_rows = []       # last-position rows
    multi_rows = []     # multi-position rows
    skipped = 0

    try:
        with torch.no_grad():
            for _row_idx, r in tqdm(rows_todo, desc=f"E4B L{args.layer}"):
                text = r.get("detokenized_text_truncated")
                if not text:
                    skipped += 1
                    continue
                ids = tok.encode(text, add_special_tokens=True, truncation=True, max_length=512)
                if len(ids) < 2:
                    skipped += 1
                    continue
                input_ids = torch.tensor([ids], device=model.device)
                captured.clear()
                _ = model(input_ids=input_ids)
                if not captured:
                    skipped += 1
                    continue
                acts = captured["acts"][0]  # (seq_len, d_model)

                # Last-position row (primary)
                vec_last = acts[-1].numpy().astype(np.float32)
                base_row = {k: v for k, v in r.items() if k != "activation_vector" and k != "activation_layer"}
                base_row["activation_vector"] = vec_last.tolist()
                base_row["activation_layer"] = args.layer
                out_rows.append(base_row)

                # Multi-position rows (extra; sampled uniformly across the sequence, MIN_POS=50)
                seq_len = acts.shape[0]
                min_pos = min(50, seq_len - 1)
                cands = list(range(min_pos, seq_len - 1))  # exclude last (already done) and BOS
                if cands and args.extra_positions > 0:
                    step = max(1, len(cands) // args.extra_positions)
                    extra_positions = cands[::step][:args.extra_positions]
                    for pos in extra_positions:
                        vec_p = acts[pos].numpy().astype(np.float32)
                        mr = {k: v for k, v in r.items() if k != "activation_vector" and k != "activation_layer"}
                        mr["activation_vector"] = vec_p.tolist()
                        mr["activation_layer"] = args.layer
                        mr["position_idx"] = pos
                        multi_rows.append(mr)

                # Periodic append (checkpoint-resumable)
                if not args.smoke and len(out_rows) % args.append_every == 0 and len(out_rows) > 0:
                    _append(args.output, out_rows[-args.append_every:])
                    print(f"  [checkpoint] {len(out_rows)} last-pos rows saved", flush=True)
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  [mem] peak VRAM: {peak:.2f} GB", flush=True)
    finally:
        handle.remove()

    print(f"\n[e4b-extract] {len(out_rows)} last-pos rows extracted ({skipped} skipped)")

    if args.smoke:
        norms = np.array([np.linalg.norm(r["activation_vector"]) for r in out_rows])
        print(f"  Smoke norm stats L{args.layer}: mean={norms.mean():.2f} std={norms.std():.2f}")
        print("  (smoke mode: no files written)")
        return

    # Write remaining rows not yet appended
    remainder = out_rows[-(len(out_rows) % args.append_every or len(out_rows)):]
    if remainder:
        _append(args.output, remainder)

    # Write multi-position parquet alongside output
    if multi_rows:
        multi_out = args.output.parent / (args.output.stem + "_multi" + args.output.suffix)
        pq.write_table(pa.Table.from_pylist(multi_rows), multi_out)
        print(f"[e4b-extract] wrote {multi_out} ({len(multi_rows)} multi-pos rows)")

    # Compute and save corpus mean for mean-centering (train set only)
    full = pq.read_table(args.output, columns=["activation_vector"]).to_pydict()
    vecs = np.array(full["activation_vector"], dtype=np.float32)
    corpus_mean = vecs.mean(axis=0)
    mean_out = args.output.parent / "mean_activation_e4b.npy"
    np.save(mean_out, corpus_mean)
    print(f"[e4b-extract] saved corpus mean to {mean_out} (norm={np.linalg.norm(corpus_mean):.3f})")

    # Norm stats
    norms = np.linalg.norm(vecs, axis=1)
    stats = {"layer": args.layer, "n_rows": len(vecs),
             "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
             "norm_min": float(norms.min()), "norm_max": float(norms.max()),
             "mean_activation_norm": float(np.linalg.norm(corpus_mean)),
             "mean_activation_22pct_check": float(
                 np.dot(vecs, corpus_mean / (np.linalg.norm(corpus_mean) + 1e-9)).mean()
                 / (norms.mean() + 1e-9)  # rough fraction of energy in mean direction
             )}
    print(f"[e4b-extract] norm stats: {stats}")
    rdir = Path(__file__).parent / "results/e4b"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "stage0_reextract_e4b.json").write_text(json.dumps(stats, indent=2))


def _append(path: Path, rows: list[dict]) -> None:
    """Append rows to a parquet file, creating it if needed."""
    table = pa.Table.from_pylist(rows)
    if path.exists():
        existing = pq.read_table(path)
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)


if __name__ == "__main__":
    main()
