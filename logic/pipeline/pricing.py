"""
Hardcoded per-model pricing and parameter counts, keyed by Ollama model name.

Prices are USD per 1M tokens, from public API list prices. Local models are priced
at the API rate of their base model (e.g. the fine-tuned 0.6B -> Qwen3-0.6B).
Fill in the None values before running cost().
"""

PRICING = {
    # model name                    in $/1M  out $/1M  total params (B)  active params (B)
    "SFT_Extraction_Qwen3_0.6b-v4": {"in_per_M": None, "out_per_M": None, "total_params_B": 0.6, "active_params_B": 0.6,
                                     "price_source": "Qwen3-0.6B API price"},
    "qwen3:8b":                     {"in_per_M": None, "out_per_M": None, "total_params_B": 8,   "active_params_B": 8,
                                     "price_source": ""},
    "gpt-oss:120b-cloud":           {"in_per_M": None, "out_per_M": None, "total_params_B": 117, "active_params_B": 5.1,
                                     "price_source": ""},
    "gemma4:31b-cloud":             {"in_per_M": None, "out_per_M": None, "total_params_B": 31,  "active_params_B": None,
                                     "price_source": ""},
}


def cost(model: str, in_tok: int, out_tok: int) -> float:
    """USD cost of one call (or a summed set of calls) to `model`."""
    if model not in PRICING:
        raise KeyError(f"No pricing entry for model '{model}'. Add it to PRICING in pricing.py.")
    p = PRICING[model]
    if p["in_per_M"] is None or p["out_per_M"] is None:
        raise ValueError(f"Prices for '{model}' are not filled in yet (pricing.py).")
    return in_tok / 1e6 * p["in_per_M"] + out_tok / 1e6 * p["out_per_M"]
