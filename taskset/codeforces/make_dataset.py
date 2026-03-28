"""One-time dataset builder for tasksets/codeforces/data bucket files."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .dataset import BUCKET_RANGES, CURRICULUM_BUCKET_FILES, encode_tcs


def ngrams(text: str, n: int = 8) -> Set[Tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def build_eval_ngrams(eval_problems: List[Dict], n: int = 8) -> Set[Tuple[str, ...]]:
    out: Set[Tuple[str, ...]] = set()
    for p in eval_problems:
        out |= ngrams(p.get("question_content", ""), n=n)
    return out


def contaminated(text: str, eval_ngrams: Set[Tuple[str, ...]], n: int = 8, threshold: int = 3) -> bool:
    return len(ngrams(text, n=n) & eval_ngrams) >= threshold


def cf_bucket_index(rating: Optional[int]) -> Optional[int]:
    if rating is None:
        return None
    for idx, (lo, hi, name) in enumerate(BUCKET_RANGES):
        if name == "b0_unrated":
            continue
        if lo <= rating < hi:
            return idx - 1
    if rating >= 2400:
        return 8
    return None


def load_eval_set() -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("nuprl/Ag-LiveCodeBench-X", split="test")
    out = []
    for row in ds:
        out.append(
            {
                "question_id": row["question_id"],
                "question_content": row["question_content"],
            }
        )
    return out


def build(
    out_dir: Path,
    max_cf: int,
    min_cf_rating: int,
    max_cf_rating: int,
    no_leetcode: bool,
    decontam: bool,
    decontam_n: int,
    decontam_threshold: int,
) -> None:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        b: (out_dir / f"{b}.jsonl").open("w", encoding="utf-8")
        for b in CURRICULUM_BUCKET_FILES
    }

    eval_rows = load_eval_set()
    eval_ids = {row["question_id"] for row in eval_rows}
    eval_grams = build_eval_ngrams(eval_rows, n=decontam_n) if decontam else set()

    counts = {b: 0 for b in CURRICULUM_BUCKET_FILES}
    dropped = 0

    ds = load_dataset("open-r1/codeforces", split="train", streaming=True)
    seen = 0
    for row in ds:
        rating = row.get("rating")
        if rating is None:
            continue
        if not (min_cf_rating <= rating <= max_cf_rating):
            continue
        if not row.get("executable", False):
            continue

        statement = " ".join(
            filter(
                None,
                [
                    row.get("title", ""),
                    row.get("description", ""),
                    row.get("input_format", ""),
                    row.get("output_format", ""),
                ],
            )
        ).strip()
        if len(statement) < 50:
            continue

        examples = row.get("examples", []) or []
        test_cases = [
            {"input": t.get("input", ""), "output": t.get("output", "")}
            for t in examples
            if t.get("input") and t.get("output")
        ]
        if not test_cases:
            continue

        contest_id = str(row.get("contestId", "")).strip()
        index = str(row.get("index", "")).strip()
        pid = f"{contest_id}_{index}" if contest_id and index else str(row.get("id", ""))
        if not pid or pid in eval_ids:
            continue

        if decontam and contaminated(statement, eval_grams, n=decontam_n, threshold=decontam_threshold):
            dropped += 1
            continue

        bidx = cf_bucket_index(rating)
        if bidx is None:
            continue
        bname = CURRICULUM_BUCKET_FILES[bidx]

        item = {
            "question_id": pid,
            "question_content": statement,
            "private_test_cases": encode_tcs(test_cases),
            "source": "codeforces",
            "difficulty": str(rating),
        }
        files[bname].write(json.dumps(item) + "\n")
        counts[bname] += 1
        seen += 1
        if max_cf and seen >= max_cf:
            break

    for f in files.values():
        f.close()

    print("Bucket size summary:")
    for b in CURRICULUM_BUCKET_FILES:
        print(f"  {b}: {counts[b]}")
    print(f"Dropped by decontamination: {dropped}")

    if not no_leetcode:
        lc_path = out_dir / "lc_train.jsonl"
        lc_count = 0
        lc_ds = load_dataset("newfacade/LeetCodeDataset", split="train", streaming=True)
        with lc_path.open("w", encoding="utf-8") as f:
            for row in lc_ds:
                statement = row.get("problem_description", "")
                tests = row.get("input_output", []) or []
                if len(statement) < 50 or len(tests) < 1:
                    continue
                slug = row.get("slug", str(row.get("question_id", "")))
                pid = f"lc_{slug}"
                item = {
                    "question_id": pid,
                    "question_content": statement,
                    "private_test_cases": encode_tcs(tests),
                    "source": "leetcode",
                    "difficulty": row.get("difficulty", "unknown"),
                    "entry_point": row.get("entry_point", ""),
                    "starter_code": row.get("starter_code", ""),
                }
                f.write(json.dumps(item) + "\n")
                lc_count += 1
        print(f"LeetCode side split written: {lc_count} rows -> {lc_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bucketed Codeforces dataset for tasksets")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--max-cf", type=int, default=5000)
    parser.add_argument("--min-cf-rating", type=int, default=800)
    parser.add_argument("--max-cf-rating", type=int, default=2500)
    parser.add_argument("--no-leetcode", action="store_true")
    parser.add_argument("--decontam", action="store_true")
    parser.add_argument("--decontam-n", type=int, default=8)
    parser.add_argument("--decontam-threshold", type=int, default=3)
    args = parser.parse_args()

    build(
        out_dir=Path(args.out_dir),
        max_cf=args.max_cf,
        min_cf_rating=args.min_cf_rating,
        max_cf_rating=args.max_cf_rating,
        no_leetcode=args.no_leetcode,
        decontam=args.decontam,
        decontam_n=args.decontam_n,
        decontam_threshold=args.decontam_threshold,
    )


if __name__ == "__main__":
    main()
