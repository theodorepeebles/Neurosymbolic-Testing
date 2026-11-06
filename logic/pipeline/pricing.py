"""
Hardcoded per-model pricing and parameter counts, keyed by Ollama model name.

Prices are USD per 1M tokens, from public API list prices. Local models are priced
at the API rate of their base model (e.g. the fine-tuned 0.6B -> Qwen3-0.6B).
Cached-input prices are listed for reference only: cost() charges every input token at the
full input rate, because Ollama's token counts don't say which input tokens were cached.
Fill in the None values before running cost().
"""

PRICING = {
    # ── NS pipeline (local models, priced at another provider's API rate for the base model;
    #    Ollama doesn't host Qwen3. From price-tracking sites, not confirmed on the provider pages.)
    "SFT_Extraction_Qwen3_0.6b-v4": {"in_per_M": 0.10, "out_per_M": 0.10, "cached_in_per_M": None,
                                     "total_params_B": 0.6, "active_params_B": 0.6,
                                     "price_source": "Qwen3-0.6B on Fireworks (via price tracker)"},
    "qwen3:4b-q4_K_M":              {"in_per_M": 0.08, "out_per_M": 0.24, "cached_in_per_M": None,
                                     "total_params_B": 4, "active_params_B": 4,
                                     "price_source": "Qwen3-4B on Nebius (via price tracker)"},
    "qwen3:8b":                     {"in_per_M": None, "out_per_M": None, "cached_in_per_M": None,
                                     "total_params_B": 8, "active_params_B": 8,
                                     "price_source": ""},

    # ── Baselines (Ollama Cloud list prices)
    "gpt-oss:120b-cloud":           {"in_per_M": 0.15, "out_per_M": 0.60, "cached_in_per_M": 0.014,
                                     "total_params_B": 117, "active_params_B": 5.1,
                                     "price_source": "Ollama Cloud"},
    "nemotron-3-super:cloud":       {"in_per_M": 0.015, "out_per_M": 0.60, "cached_in_per_M": 0.015,
                                     "total_params_B": 120.6, "active_params_B": 12.7,
                                     "price_source": "Ollama Cloud"},
    "gemma4:31b-cloud":             {"in_per_M": 0.14, "out_per_M": 0.40, "cached_in_per_M": 0.05,
                                     "total_params_B": 31, "active_params_B": 31,
                                     "price_source": "Ollama Cloud"},
    "nemotron-3-nano:30b-cloud":    {"in_per_M": 0.06, "out_per_M": 0.24, "cached_in_per_M": None,
                                     "total_params_B": 31.6, "active_params_B": 3.2,
                                     "price_source": "Ollama Cloud"},
    "gpt-oss:20b-cloud":            {"in_per_M": 0.07, "out_per_M": 0.30, "cached_in_per_M": 0.035,
                                     "total_params_B": 21, "active_params_B": 3.6,
                                     "price_source": "Ollama Cloud"},
}


def cost(model: str, in_tok: int, out_tok: int) -> float:
    """USD cost of one call (or a summed set of calls) to `model`."""
    if model not in PRICING:
        raise KeyError(f"No pricing entry for model '{model}'. Add it to PRICING in pricing.py.")
    p = PRICING[model]
    if p["in_per_M"] is None or p["out_per_M"] is None:
        raise ValueError(f"Prices for '{model}' are not filled in yet (pricing.py).")
    return in_tok / 1e6 * p["in_per_M"] + out_tok / 1e6 * p["out_per_M"]
