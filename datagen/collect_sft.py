"""
Collects all b0-b4 SFT pairs into a single JSONL file ready for training.
Filters to compiled-only by default (clean labels).
Reports breakdown by difficulty.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


DIFFICULTY_ORDER = {"b0": 0, "b1": 1, "b2": 2, "b3": 3, "b4": 4, "b4_vague": 5}


def collect(
    input_dirs: list[str],
    output_path: str,
    include_failed: bool = False,
    min_code_len: int = 50,
) -> None:
    pairs = []

    for input_dir in input_dirs:
        for json_file in sorted(Path(input_dir).glob("*.json")):
            if "_RAW" in json_file.name:
                continue
            try:
                with open(json_file) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"SKIP {json_file.name}: {e}")
                continue

            # Must have the fields we care about
            if "prompt" not in data or "code" not in data or "compiled" not in data:
                continue

            if not include_failed and not data["compiled"]:
                continue

            if len(data.get("code", "")) < min_code_len:
                continue

            pairs.append(data)

    # Sort by difficulty then function name
    pairs.sort(key=lambda p: (
        DIFFICULTY_ORDER.get(p.get("difficulty", ""), 99),
        p.get("function_name", ""),
    ))

    # Write as chat-format JSONL
    with open(output_path, "w") as f:
        for pair in pairs:
            row = {
                "messages": [
                    {"role": "user",      "content": pair["prompt"]},
                    {"role": "assistant", "content": pair["code"]},
                ],
                "metadata": {
                    "function_name": pair.get("function_name", ""),
                    "difficulty":    pair.get("difficulty", ""),
                    "compiled":      pair.get("compiled", False),
                },
            }
            f.write(json.dumps(row) + "\n")

    # Stats
    by_diff: dict[str, int] = {}
    for p in pairs:
        d = p.get("difficulty", "unknown")
        by_diff[d] = by_diff.get(d, 0) + 1

    print(f"\nCollected {len(pairs)} SFT pairs → {output_path}")
    for d, count in sorted(by_diff.items(), key=lambda x: DIFFICULTY_ORDER.get(x[0], 99)):
        print(f"  {d:12s}: {count:4d} pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect all SFT pairs into training JSONL")
    parser.add_argument("--input-dirs",     nargs="+", required=True,
                        help="Directories containing bX_*.json files")
    parser.add_argument("--output",         required=True,
                        help="Output .jsonl path")
    parser.add_argument("--include-failed", action="store_true",
                        help="Include pairs where code did not compile (noisy labels)")
    parser.add_argument("--min-code-len",   type=int, default=50,
                        help="Skip pairs where code is shorter than this (likely garbage)")
    args = parser.parse_args()

    collect(args.input_dirs, args.output, args.include_failed, args.min_code_len)
