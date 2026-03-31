from __future__ import annotations
import json
import random
from pathlib import Path


def load_raw_function(json_path: str) -> dict:
    """Load a function JSON file as a raw dict. No schema enforcement."""
    with open(json_path) as f:
        return json.load(f)


def raw_to_string(fn: dict) -> str:
    """Dump raw function dict to a formatted JSON string for prompt injection."""
    return json.dumps(fn, indent=2, ensure_ascii=False)


def extract_function_name(fn: dict) -> str:
    """
    Best-effort extraction of function name from raw dict.
    Tries common key names, then falls back to parsing the signature.
    """
    for key in ("function_name", "name", "func_name", "fn_name", "functionName", "funcName"):
        if key in fn and fn[key]:
            return str(fn[key]).strip()

    # Try to parse from signature
    sig = fn.get("signature", "") or fn.get("func_signature", "") or fn.get("proto", "")
    if sig:
        # "int fn_open_session(handle_t *h, ...)" → "fn_open_session"
        tokens = sig.replace("(", " ").split()
        skip = {
            "int", "void", "char", "unsigned", "signed", "long", "short",
            "float", "double", "const", "static", "extern", "inline",
            "struct", "enum", "union", "typedef",
            "uint8_t", "uint16_t", "uint32_t", "uint64_t",
            "int8_t", "int16_t", "int32_t", "int64_t",
            "size_t", "ssize_t", "ptrdiff_t", "bool", "FILE",
        }
        for t in tokens:
            t_clean = t.lstrip("*").strip()
            if t_clean and t_clean not in skip:
                return t_clean

    return "unknown_function"


def extract_params(fn: dict) -> list[dict]:
    """
    Best-effort extraction of param list from raw dict.
    Returns list of dicts as-is — preserves all fields.
    """
    for key in ("params", "parameters", "args", "arguments", "param_list", "inputs"):
        val = fn.get(key)
        if isinstance(val, list) and len(val) > 0:
            return val
    return []


# ─── Prompt builders ─────────────────────────────────────────────────────

def build_b0_input(fn: dict) -> tuple[str, str]:
    """
    b0: Full raw JSON dump in both agent input and SFT prompt.
    Returns (full_json_string, sft_prompt).
    """
    full_json = raw_to_string(fn)
    fn_name = extract_function_name(fn)

    sft_prompt = (
        f"Write a minimal C program that correctly uses the function "
        f"`{fn_name}`. Here is the complete function information:\n\n"
        f"{full_json}\n\n"
        f"The program must compile cleanly and handle all error paths."
    )

    return full_json, sft_prompt


def build_b1_input(fn: dict) -> tuple[str, str]:
    """
    b1: Function name + params as raw JSON (correct order, all fields including type).
    Returns (model_input, sft_prompt).
    """
    fn_name = extract_function_name(fn)
    params = extract_params(fn)
    params_block = json.dumps(params, indent=2, ensure_ascii=False) if params else "[]"

    model_input = (
        f"Function: {fn_name}\n\n"
        f"Parameters (correct order, types included):\n{params_block}"
    )

    sft_prompt = (
        f"Write a minimal C program that uses `{fn_name}`. "
        f"The function takes these parameters in order:\n\n"
        f"{params_block}\n\n"
        f"The program must compile cleanly."
    )

    return model_input, sft_prompt


def build_b2_input(fn: dict) -> tuple[str, str]:
    """
    b2: Function name + params with order shuffled and type fields stripped.
    Returns (model_input, sft_prompt).
    """
    fn_name = extract_function_name(fn)
    params = extract_params(fn)

    shuffled = list(params)
    random.shuffle(shuffled)

    type_keys = {"type", "param_type", "ctype", "c_type", "data_type", "dtype", "kind"}
    stripped = [
        {k: v for k, v in p.items() if k.lower() not in type_keys}
        for p in shuffled
    ]

    params_block = json.dumps(stripped, indent=2, ensure_ascii=False) if stripped else "[]"

    model_input = (
        f"Function: {fn_name}\n\n"
        f"Parameter descriptions (order is WRONG, types have been removed):\n"
        f"{params_block}"
    )

    sft_prompt = (
        f"Write a minimal C program that uses `{fn_name}`. "
        f"The function has these parameters (order is wrong, types unknown):\n\n"
        f"{params_block}\n\n"
        f"Determine the correct order and types. The program must compile cleanly."
    )

    return model_input, sft_prompt


def build_b3_input(fn: dict) -> tuple[str, str]:
    """
    b3: Only function name.
    Returns (model_input, sft_prompt).
    """
    fn_name = extract_function_name(fn)

    model_input = f"Function: {fn_name}"

    sft_prompt = (
        f"Write a minimal C program that uses the library function "
        f"`{fn_name}`. The program must compile cleanly."
    )

    return model_input, sft_prompt
