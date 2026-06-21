import os, sys, subprocess, math, json, csv, random, time
os.environ["PYTHONUNBUFFERED"] = "1"

# -----------------------------------------------------------------------
# GPU cap check -- abort on P100 (sm_60 incompatible with Python 3.12)
# -----------------------------------------------------------------------
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=10)
    GPU_CAP = float(r.stdout.strip().split('\n')[0].strip())
    print(f"GPU compute capability: {GPU_CAP}")
except Exception as e:
    print(f"Could not detect GPU capability: {e}")
    GPU_CAP = 99.0

if GPU_CAP < 7.0:
    print("ABORT: P100/sm_60 assigned. Re-push to get T4.")
    sys.exit(1)

# -----------------------------------------------------------------------
# Install deps
# -----------------------------------------------------------------------
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/huggingface/transformers.git"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "peft>=0.12", "bitsandbytes>=0.43", "huggingface_hub",
    "pyarrow", "pandas", "scikit-learn"], check=False)
print("Deps installed.")

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn.functional as F

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO  = "Solshine/e2b-nla-grpo-0620"
OUT_DIR  = Path("/kaggle/working/checkpoints/e2b_grpo")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS  = Path("/kaggle/working/results")
RESULTS.mkdir(exist_ok=True)

# -----------------------------------------------------------------------
# NLA constants (E2B)
# -----------------------------------------------------------------------
INJ_ID   = 249568   # chr(0x3297)
LEFT_ID  = 236813
RIGHT_ID = 954
INJ_CHAR = chr(0x3297)
D_MODEL  = 1536
INJECT_LAYER = 23
INJ_SCALE = math.sqrt(D_MODEL)   # ~39.19

AV_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation into activation "
    "vectors from a language model. Your overall task is to describe the semantic content of "
    "that activation vector.\n\nWe will pass the vector enclosed in <concept> tags into your "
    "context. You must then produce an explanation for the vector, enclosed within "
    "<explanation> tags. The explanation consists of 2-3 text snippets describing that vector."
    "\n\nHere is the vector:\n\n<concept>" + INJ_CHAR + "</concept>"
)

# -----------------------------------------------------------------------
# GRPO hypers
# -----------------------------------------------------------------------
G            = 4       # completions per anchor
MAX_NEW_TOK  = 64
TEMPERATURE  = 0.9
STEPS        = 600
SAVE_EVERY   = 100
LR           = 3e-5
RANK         = 16
SEED         = 17
REWARD_SOFT  = 0.2    # weight for soft cosine bonus (0.8 hard + 0.2 soft)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# -----------------------------------------------------------------------
# Load corpus -- stage1/rl.parquet from GitHub
# -----------------------------------------------------------------------
print("Downloading corpus...")
import urllib.request
corpus_path = Path("/kaggle/working/rl.parquet")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/SolshineCode/e4b-nla-colab/master/data/stage1/rl.parquet",
    corpus_path
)

df_all = pd.read_parquet(corpus_path)
print(f"Raw parquet: {len(df_all)} rows, {df_all['doc_id'].nunique()} unique docs")

# Keep only last-position rows (highest n_raw_tokens per doc)
df_last = df_all.loc[df_all.groupby('doc_id')['n_raw_tokens'].idxmax()].reset_index(drop=True)
print(f"Last-position rows: {len(df_last)}")

def parse_vec(v):
    if isinstance(v, (list, np.ndarray)):
        return np.array(v, dtype=np.float32)
    return np.array(list(v), dtype=np.float32)

df_last['act_np'] = df_last['activation_vector'].apply(parse_vec)
print(f"D_MODEL check: {len(df_last['act_np'].iloc[0])} (expected {D_MODEL})")

corpus_mean = np.stack(df_last['act_np'].values).mean(axis=0)
print(f"Corpus mean norm: {np.linalg.norm(corpus_mean):.3f}")

df_last['act_centered'] = df_last['act_np'].apply(lambda v: v - corpus_mean)

doc_texts = dict(zip(df_last['doc_id'], df_last['detokenized_text_truncated']))
doc_ids   = df_last['doc_id'].tolist()
print(f"Corpus: {len(doc_ids)} documents")

# -----------------------------------------------------------------------
# TF-IDF index for retrieval reward
# -----------------------------------------------------------------------
print("Building TF-IDF index...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = [doc_texts[did] for did in doc_ids]
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)
tfidf_matrix = tfidf.fit_transform(texts)
print(f"TF-IDF matrix: {tfidf_matrix.shape}")

def tfidf_reward(generated_text, true_doc_id):
    """Returns (hard_reward, soft_reward)."""
    if not generated_text or len(generated_text.strip()) < 5:
        return 0.0, 0.0
    try:
        q_vec = tfidf.transform([generated_text])
        sims  = cosine_similarity(q_vec, tfidf_matrix)[0]
        top_idx = int(np.argmax(sims))
        true_idx = doc_ids.index(true_doc_id) if true_doc_id in doc_ids else -1
        soft = float(sims[true_idx]) if true_idx >= 0 else 0.0
        hard = 1.0 if doc_ids[top_idx] == true_doc_id else 0.0
        return hard, soft
    except Exception:
        return 0.0, 0.0

def combined_reward(generated_text, true_doc_id):
    hard, soft = tfidf_reward(generated_text, true_doc_id)
    return (1.0 - REWARD_SOFT) * hard + REWARD_SOFT * soft

# -----------------------------------------------------------------------
# Injection helpers (mirrors train_av_e4b.py)
# -----------------------------------------------------------------------
def find_inj_pos(input_ids):
    ids = input_ids[0].tolist()
    for p in range(1, len(ids) - 1):
        if ids[p] == INJ_ID and ids[p-1] == LEFT_ID and ids[p+1] == RIGHT_ID:
            return p
    return None

def inject(h, pos, vec):
    h = h.clone()
    h[0, pos] = vec.to(h.dtype)
    return h

def make_hook(pending, pos_cache):
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

def set_injection(pending, pos_cache, act_centered, device):
    v = torch.tensor(act_centered, dtype=torch.float32, device=device)
    v_norm = v / (v.norm() + 1e-8) * INJ_SCALE
    pending["vec"] = v_norm

def extract_explanation(text):
    """Extract content from <explanation>...</explanation> tags."""
    if "<explanation>" in text:
        start = text.index("<explanation>") + len("<explanation>")
        end   = text.index("</explanation>") if "</explanation>" in text else len(text)
        return text[start:end].strip()
    return text.strip()

# -----------------------------------------------------------------------
# Download E2B model
# -----------------------------------------------------------------------
from huggingface_hub import snapshot_download, HfApi

MODEL_DIR = Path("/kaggle/working/models/gemma-4-E2B")
if not MODEL_DIR.exists():
    print("Downloading google/gemma-4-E2B...")
    snapshot_download(
        repo_id="google/gemma-4-E2B",
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN,
        ignore_patterns=["*.gguf"],
    )
    print("Model downloaded.")
else:
    print("Model already cached.")

# -----------------------------------------------------------------------
# Load model + LoRA
# -----------------------------------------------------------------------
print("Loading E2B model in NF4...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    quantization_config=bnb,
    device_map={"": 0},
)

# E2B has vision encoder (Gemma4ClippableLinear) -- restrict LoRA to language_model subtree
target_modules = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
lora_cfg = LoraConfig(
    r=RANK, lora_alpha=RANK * 2,
    target_modules=target_modules,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA r={RANK} trainable params: {n_trainable/1e6:.2f}M")

model.enable_input_require_grads()
model.train()
print(f"Peak VRAM after model load: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# -----------------------------------------------------------------------
# Injection hook on layer 23
# -----------------------------------------------------------------------
pending   = {}
pos_cache = {}
base_model  = model.base_model.model
hook_module = base_model.model.language_model.layers[INJECT_LAYER]
hook_handle = hook_module.register_forward_hook(make_hook(pending, pos_cache))
print(f"Hook on layer {INJECT_LAYER}: {type(hook_module).__name__}")

device   = next(model.parameters()).device
tmpl_ids = tok.encode(AV_TEMPLATE, return_tensors="pt").to(device)
inj_pos  = find_inj_pos(tmpl_ids)
pos_cache["pos"] = inj_pos
print(f"Template tokens: {tmpl_ids.shape[1]}, INJ pos: {inj_pos}")
if inj_pos is None:
    print("ABORT: injection position not found.")
    sys.exit(1)

# -----------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------
import bitsandbytes as bnb_lib
params    = [p for p in model.parameters() if p.requires_grad]
optimizer = bnb_lib.optim.AdamW8bit(params, lr=LR, weight_decay=0.01)

# -----------------------------------------------------------------------
# Training log
# -----------------------------------------------------------------------
log_path = OUT_DIR / "grpo_log.csv"
with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow(["step", "mean_reward", "frac_nonzero", "n_unique_5", "loss"])

# -----------------------------------------------------------------------
# GRPO training loop (DAPO variant: no KL/reference model)
# Reward: 0.8 * hard_retrieval_at1 + 0.2 * soft_cosine
# Zero-variance skip: if all advantages ~0, skip update
# -----------------------------------------------------------------------
print(f"\n=== GRPO training: {STEPS} steps, G={G}, max_new_tok={MAX_NEW_TOK} ===\n")

recent_texts = []

for step in range(1, STEPS + 1):
    row          = df_last.sample(1, random_state=step + SEED).iloc[0]
    true_doc_id  = row['doc_id']
    act_centered = row['act_centered']

    set_injection(pending, pos_cache, act_centered, device)

    # --- Rollout: G completions (no gradient) ---
    completions = []
    comp_texts  = []
    with torch.no_grad():
        model.eval()
        for g in range(G):
            torch.manual_seed(step * 1000 + g)
            out = model.generate(
                tmpl_ids,
                max_new_tokens=MAX_NEW_TOK,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tok.eos_token_id,
            )
            gen_toks = out[0, tmpl_ids.shape[1]:]
            completions.append(gen_toks)
            raw = tok.decode(gen_toks, skip_special_tokens=True)
            comp_texts.append(extract_explanation(raw))
    model.train()

    # --- Rewards and advantages ---
    rewards    = [combined_reward(t, true_doc_id) for t in comp_texts]
    mean_r     = sum(rewards) / G
    advantages = [r - mean_r for r in rewards]

    recent_texts.append(comp_texts[0])
    if len(recent_texts) > 5:
        recent_texts.pop(0)
    n_unique_5 = len(set(recent_texts))

    # Skip if zero variance (no learning signal)
    if all(abs(a) < 1e-6 for a in advantages):
        if step % 10 == 0:
            print(f"step={step:4d} zero-var skip | mean_r={mean_r:.3f} | {comp_texts[0][:50]}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([step, mean_r, 0.0, n_unique_5, 0.0])
        continue

    # --- Policy gradient update ---
    total_loss = torch.tensor(0.0, device=device)
    n_nonzero  = sum(1 for a in advantages if abs(a) > 1e-6)

    set_injection(pending, pos_cache, act_centered, device)

    for gen_toks, adv in zip(completions, advantages):
        if abs(adv) < 1e-6:
            continue
        full_seq = torch.cat([tmpl_ids[0], gen_toks]).unsqueeze(0)
        labels   = full_seq.clone()
        labels[0, :tmpl_ids.shape[1]] = -100

        logits      = model(input_ids=full_seq).logits
        shift_log   = logits[0, :-1]
        shift_lab   = labels[0, 1:]
        mask        = shift_lab != -100
        if mask.sum() == 0:
            continue
        log_probs  = -F.cross_entropy(shift_log[mask], shift_lab[mask], reduction='none')
        mean_logp  = log_probs.mean()
        adv_clip   = max(min(adv, 2.0), -2.0)
        total_loss = total_loss + (-adv_clip * mean_logp)

    if n_nonzero == 0:
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([step, mean_r, 0.0, n_unique_5, 0.0])
        continue

    total_loss = total_loss / n_nonzero
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()

    loss_val    = total_loss.item()
    frac_nz     = n_nonzero / G

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([step, mean_r, frac_nz, n_unique_5, loss_val])

    if step % 10 == 0:
        print(f"step={step:4d} | r={mean_r:.3f} | frac_nz={frac_nz:.2f} | "
              f"n_uniq5={n_unique_5} | loss={loss_val:.4f} | "
              f"VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        print(f"  [{true_doc_id}] gen0: {comp_texts[0][:80]}")
        print(f"  [{true_doc_id}] gen1: {comp_texts[1][:80]}")

    if step % SAVE_EVERY == 0:
        ckpt = OUT_DIR / f"step_{step:06d}"
        model.save_pretrained(str(ckpt))
        recent_rows = list(csv.reader(open(log_path)))[-10:]
        recent_r = [float(rw[1]) for rw in recent_rows if len(rw) > 1 and rw[1] != 'mean_reward']
        meta = {
            "step": step, "mean_reward_last10": float(np.mean(recent_r)) if recent_r else 0.0,
            "grpo_g": G, "lr": LR, "rank": RANK,
            "d_model": D_MODEL, "inject_layer": INJECT_LAYER,
            "corpus": "stage1/rl.parquet (163 docs, E2B L23)",
        }
        with open(ckpt / "grpo_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[ckpt] step {step} | VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

# -----------------------------------------------------------------------
# Post-training eval (greedy, 20 samples)
# -----------------------------------------------------------------------
print("\n=== Post-training eval ===")
model.eval()
eval_rows    = df_last.sample(min(20, len(df_last)), random_state=42)
eval_results = []
n_uniq_set   = set()

with torch.no_grad():
    for _, row in eval_rows.iterrows():
        set_injection(pending, pos_cache, row['act_centered'], device)
        out = model.generate(
            tmpl_ids, max_new_tokens=MAX_NEW_TOK,
            do_sample=False, pad_token_id=tok.eos_token_id,
        )
        gen_toks = out[0, tmpl_ids.shape[1]:]
        text     = extract_explanation(tok.decode(gen_toks, skip_special_tokens=True))
        hard, soft = tfidf_reward(text, row['doc_id'])
        n_uniq_set.add(text[:60])
        eval_results.append({"doc_id": row['doc_id'], "text": text[:120],
                              "hard_reward": hard, "soft_reward": soft})

mean_hard  = float(np.mean([r['hard_reward'] for r in eval_results]))
mean_soft  = float(np.mean([r['soft_reward'] for r in eval_results]))
n_unique   = len(n_uniq_set)
verdict    = "PASS" if (mean_hard > 0.1 and n_unique >= 5) else "COLLAPSE"

print(f"Eval n={len(eval_results)}: hard={mean_hard:.3f} soft={mean_soft:.3f} "
      f"n_unique={n_unique}/{len(eval_results)} => {verdict}")
for r in eval_results[:5]:
    print(f"  [{r['doc_id']}] h={r['hard_reward']} s={r['soft_reward']:.3f}: {r['text'][:80]}")

eval_summary = {
    "n_eval": len(eval_results), "mean_hard_retrieval": mean_hard,
    "mean_soft_cosine": mean_soft, "n_unique": n_unique,
    "steps_trained": STEPS, "verdict": verdict, "results": eval_results,
}
with open(RESULTS / "grpo_eval.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

final_ckpt = OUT_DIR / "final"
model.save_pretrained(str(final_ckpt))
print(f"Final checkpoint saved.")

# -----------------------------------------------------------------------
# Upload to HF Hub
# -----------------------------------------------------------------------
print("\nUploading to HF Hub...")
api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True, private=False)
except Exception as e:
    print(f"Repo create: {e}")

api.upload_folder(folder_path="/kaggle/working/checkpoints",
                  repo_id=HF_REPO, repo_type="model", path_in_repo="checkpoints")
api.upload_folder(folder_path=str(RESULTS),
                  repo_id=HF_REPO, repo_type="model", path_in_repo="results")
api.upload_file(path_or_fileobj=str(log_path),
                path_in_repo="grpo_log.csv", repo_id=HF_REPO, repo_type="model")
print(f"DONE => https://huggingface.co/{HF_REPO}")
print(f"verdict={verdict} hard={mean_hard:.3f} n_unique={n_unique}")
