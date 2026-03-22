from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from inference.base import SamplingParams


class QAAugmenter:
    def __init__(self, engine, config) -> None:
        self.engine = engine
        self.cfg = config

    def run(self, qa_path: str) -> str:
        out_path = Path(self.cfg.GENERATED_DIR) / "augmented_pairs.jsonl"
        if self.cfg.SKIP_EXISTING_AUG and out_path.exists() and out_path.stat().st_size > 0:
            return str(out_path)

        qa_rows = self._load_jsonl(qa_path)
        augmented: List[Dict] = []
        for row in qa_rows:
            question = row["messages"][0]["content"].strip()
            answer = row["messages"][1]["content"].strip()
            variants = self._generate_question_variants(question)
            for q in variants:
                if q == question:
                    continue
                augmented.append(
                    {
                        "messages": [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": answer},
                        ],
                        "source": row.get("source", "unknown"),
                        "chunk_heading": row.get("chunk_heading", ""),
                        "augmented_from": question,
                    }
                )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in augmented:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(out_path)

    def _generate_question_variants(self, question: str) -> List[str]:
        n = max(1, int(self.cfg.EXAMPLES_PER_FUNCTION))
        prompt = (
            "Rewrite the following developer question into different, valid phrasings.\n"
            "Keep the technical meaning identical.\n"
            "Return one rewritten question per line.\n"
            f"Generate {n} rewrites.\n\n"
            f"Question: {question}"
        )
        out = self.engine.generate(
            prompt,
            SamplingParams(
                max_new_tokens=self.cfg.MAX_COMPLETION_TOKENS,
                temperature=0.6,
                n=1,
                top_p=0.95,
            ),
        )
        text = out[0].text if out else ""
        lines = []
        seen = {question}
        for raw in text.splitlines():
            s = re.sub(r"^\s*[-*\d\.)]+\s*", "", raw.strip())
            if not s or s in seen:
                continue
            seen.add(s)
            lines.append(s)
            if len(lines) >= n:
                break
        if not lines:
            lines.append(f"Can you explain: {question}")
        return lines

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        rows: List[Dict] = []
        p = Path(path)
        if not p.exists():
            return rows
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
