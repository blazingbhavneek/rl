import base64
import hashlib
import json
import pickle
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..base import Problem


BUCKET_RANGES = [
    (0, 800, "b0_unrated"),
    (800, 1000, "b1"),
    (1000, 1200, "b2"),
    (1200, 1400, "b3"),
    (1400, 1600, "b4"),
    (1600, 1800, "b5"),
    (1800, 2000, "b6"),
    (2000, 2200, "b7"),
    (2200, 2400, "b8"),
    (2400, 9999, "b9"),
]

CURRICULUM_BUCKET_FILES = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9"]


def encode_tcs(tcs: List[Dict]) -> str:
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tcs)))).decode("utf-8")


def decode_tcs(raw: str) -> List[Dict]:
    try:
        obj = pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        if isinstance(obj, (str, bytes)):
            obj = json.loads(obj)
        out = []
        for item in obj:
            if isinstance(item, dict) and "input" in item and "output" in item:
                out.append({"input": str(item["input"]), "output": str(item["output"])})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append({"input": str(item[0]), "output": str(item[1])})
        return out
    except Exception:
        return []


def _ngrams(text: str, n: int = 8) -> Set[Tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


class CodeforcesDataset:
    def __init__(
        self,
        data_dir: str,
        decontaminate: bool = False,
        livecodebench_path: Optional[str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.decontaminate = decontaminate
        self.livecodebench_path = Path(livecodebench_path) if livecodebench_path else None

        self._bucket_cache: Dict[int, List[Problem]] = {}
        self._problem_to_bucket: Dict[str, int] = {}
        self._eval_ngrams: Optional[Set[Tuple[str, ...]]] = None

        self.audit_path = self.data_dir.parent / "checkpoint" / "decontam_audit.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def n_buckets(self) -> int:
        return len(CURRICULUM_BUCKET_FILES)

    def get_bucket(self, idx: int) -> List[Problem]:
        if idx in self._bucket_cache:
            return self._bucket_cache[idx]

        if idx < 0 or idx >= self.n_buckets():
            raise IndexError(f"bucket idx out of range: {idx}")

        file_name = CURRICULUM_BUCKET_FILES[idx] + ".jsonl"
        path = self.data_dir / file_name
        problems: List[Problem] = []

        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    pid = row.get("question_id") or row.get("id")
                    statement = row.get("question_content") or row.get("statement", "")
                    encoded = row.get("private_test_cases", "")
                    test_cases = decode_tcs(encoded) if encoded else row.get("test_cases", [])
                    metadata = dict(row)
                    metadata["test_cases"] = test_cases

                    p = Problem(
                        id=pid,
                        statement=statement,
                        bucket=idx,
                        difficulty_label=CURRICULUM_BUCKET_FILES[idx],
                        metadata=metadata,
                    )
                    problems.append(p)
                    self._problem_to_bucket[pid] = idx

        if self.decontaminate:
            problems = self._decontaminate(problems)

        self._bucket_cache[idx] = problems
        return problems

    def get_unrated(self) -> List[Problem]:
        path = self.data_dir / "b0_unrated.jsonl"
        out: List[Problem] = []
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pid = row.get("question_id") or row.get("id")
                statement = row.get("question_content") or row.get("statement", "")
                encoded = row.get("private_test_cases", "")
                test_cases = decode_tcs(encoded) if encoded else row.get("test_cases", [])
                metadata = dict(row)
                metadata["test_cases"] = test_cases
                out.append(
                    Problem(
                        id=pid,
                        statement=statement,
                        bucket=-1,
                        difficulty_label="b0_unrated",
                        metadata=metadata,
                    )
                )
        return out

    def get_problem_bucket(self, problem_id: str) -> Optional[int]:
        if problem_id in self._problem_to_bucket:
            return self._problem_to_bucket[problem_id]
        for idx in range(self.n_buckets()):
            _ = self.get_bucket(idx)
            if problem_id in self._problem_to_bucket:
                return self._problem_to_bucket[problem_id]
        return None

    def _load_eval_ngrams(self) -> Set[Tuple[str, ...]]:
        if self._eval_ngrams is not None:
            return self._eval_ngrams
        if not self.livecodebench_path or not self.livecodebench_path.exists():
            self._eval_ngrams = set()
            return self._eval_ngrams

        grams: Set[Tuple[str, ...]] = set()
        with self.livecodebench_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("question_content", "")
                grams |= _ngrams(text, n=8)
        self._eval_ngrams = grams
        return self._eval_ngrams

    def _decontaminate(self, problems: List[Problem]) -> List[Problem]:
        eval_ngrams = self._load_eval_ngrams()
        if not eval_ngrams:
            return problems

        kept: List[Problem] = []
        dropped: List[Dict] = []
        for p in problems:
            overlap = len(_ngrams(p.statement, n=8) & eval_ngrams)
            if overlap >= 3:
                dropped.append({"id": p.id, "bucket": p.bucket, "overlap": overlap})
            else:
                kept.append(p)

        if dropped:
            with self.audit_path.open("a", encoding="utf-8") as f:
                for row in dropped:
                    f.write(json.dumps(row) + "\n")
        return kept

    def _hash_statement(self, statement: str) -> str:
        return hashlib.sha256(statement.strip().lower().encode("utf-8")).hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CodeforcesDataset smoke test")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--decontam", action="store_true")
    parser.add_argument("--livecodebench-path", type=str, default=None)
    args = parser.parse_args()

    ds = CodeforcesDataset(
        data_dir=args.data_dir,
        decontaminate=args.decontam,
        livecodebench_path=args.livecodebench_path,
    )

    for i in range(ds.n_buckets()):
        bucket = ds.get_bucket(i)
        print(f"bucket={i} size={len(bucket)}")

    for idx in (0, 4):
        b = ds.get_bucket(idx)
        if not b:
            continue
        p = b[0]
        tc_count = len(p.metadata.get("test_cases", []))
        print(f"sample bucket={idx} id={p.id} statement_chars={len(p.statement)} tests={tc_count}")
