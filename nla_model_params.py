"""Shared model-parameter registry for E2B and E4B NLA siblings.

Add entries here so every sibling script imports constants from one place.
Command-line args (--base-model, --layer, --d-model) always override these defaults.
"""

# Injection token IDs (verified: shared across Gemma-4 family, vocab=262144)
INJECTION_TOKEN_ID = 249568        # ㊗ chr(0x3297)
INJECTION_LEFT_NEIGHBOR_ID = 236813
INJECTION_RIGHT_NEIGHBOR_ID = 954
INJECTION_CHAR = chr(0x3297)

AV_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation into activation "
    "vectors from a language model. Your overall task is to describe the semantic content of "
    "that activation vector.\n\nWe will pass the vector enclosed in <concept> tags into your "
    "context. You must then produce an explanation for the vector, enclosed within "
    "<explanation> tags. The explanation consists of 2-3 text snippets describing that vector."
    f"\n\nHere is the vector:\n\n<concept>{INJECTION_CHAR}</concept>"
)

# Registry keyed by model id (lowercase, hyphen-normalised).
# Values filled in as models are verified. E4B values filled during Phase 1 + G1a.
PARAMS = {
    "google/gemma-4-e2b": {
        "d_model": 1536,
        "n_layers": 35,
        "default_layer": 23,       # capture/injection layer used in v0.1 (L17 has higher ceiling)
        "injection_scale": 39.19,  # sqrt(1536)
        # LoRA target module regex (verified; Gemma4ClippableLinear can false-match -- check count)
        "target_modules_regex": r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    },
    "google/gemma-4-e4b": {
        # Verified 2026-06-15 from config.json (text_config):
        "d_model": 2560,
        "n_layers": 42,
        # default_layer: None until G1a ceiling sweep selects it.
        # Sweep candidates at fractions {0.30,0.40,0.49,0.57,0.66,0.75}*42 = [13,17,21,24,28,32]
        # 0.49-depth candidate = L21 (closest to E2B's L17/35=0.49 winning injection site)
        "default_layer": None,     # TODO: fill after G1a sweep (run eval_layer_ceiling_sweep_e4b.py)
        "injection_scale": 50.5964,  # sqrt(2560), verified
        # target_modules_regex: TODO verify module count after model load in G1a/G2a
        # (same pattern as E2B expected; Gemma4ClippableLinear false-match warning applies)
        "target_modules_regex": r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    },
}


def get(model_id: str) -> dict:
    """Return params dict for a model id (case-insensitive, accepts google/gemma-4-E4B style)."""
    key = model_id.lower()
    if key not in PARAMS:
        raise KeyError(f"Unknown model '{model_id}'. Add it to nla_model_params.PARAMS first.")
    return PARAMS[key]
