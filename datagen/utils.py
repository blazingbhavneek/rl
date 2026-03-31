from __future__ import annotations
import re


def extract_code(text: str) -> str:
    """Extract C code from model output. Handles markdown fences and raw code."""
    # ```c ... ```
    match = re.search(r"```c\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Raw code — starts with preprocessor or has main
    if "#include" in text or "int main" in text:
        return text.strip()
    return text.strip()
