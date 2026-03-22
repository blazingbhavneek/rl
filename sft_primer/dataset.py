from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from inference.base import SamplingParams

from .augment import QAAugmenter


class SFTPrimerDataset:
    def __init__(self, docs_folder, tokenizer, engine, config):
        self.docs_folder = Path(docs_folder)
        self.tokenizer = tokenizer
        self.engine = engine
        self.cfg = config
        self.qa_path = Path(self.cfg.GENERATED_DIR) / "qa_pairs.jsonl"

    def run(self) -> Tuple[str, str]:
        """
        Main entry point.
        1. Load and chunk all .md files
        2. For each chunk: two-pass QA generation
        3. Optional augmentation
        4. Build final train/val datasets
        """
        self.cfg.ensure_dirs()
        chunks = self._load_and_chunk()
        existing_hashes = self._existing_chunk_hashes() if self.cfg.SKIP_EXISTING_QA else set()

        self.qa_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.qa_path.exists() and self.qa_path.stat().st_size > 0 else "w"
        with self.qa_path.open(mode, encoding="utf-8") as f:
            for chunk in chunks:
                if chunk["chunk_hash"] in existing_hashes:
                    continue
                qa_rows = self._generate_qa_for_chunk(chunk)
                for row in qa_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        augmented_path: Optional[str] = None
        if self.cfg.USE_AUGMENTATION:
            augmenter = QAAugmenter(self.engine, self.cfg)
            augmented_path = augmenter.run(str(self.qa_path))

        return self.build_final_dataset(augmented_path)

    def _load_and_chunk(self) -> List[Dict]:
        """
        Walk DOCS_FOLDER recursively for .md files.
        Chunk each with CHUNK_SIZE/CHUNK_OVERLAP using tokenizer ids.
        Prepend file path + nearest heading to each chunk as context header.
        """
        if not self.docs_folder.exists():
            raise FileNotFoundError(f"Docs folder not found: {self.docs_folder}")

        chunks: List[Dict] = []
        md_files = sorted(self.docs_folder.rglob("*.md"))

        for md in md_files:
            text = md.read_text(encoding="utf-8", errors="ignore")
            sections = self._split_sections(text)
            rel_source = str(md)
            for heading, section_text in sections:
                ids = self.tokenizer.encode(section_text, add_special_tokens=False)
                if not ids:
                    continue

                stride = max(1, int(self.cfg.CHUNK_SIZE - self.cfg.CHUNK_OVERLAP))
                for start in range(0, len(ids), stride):
                    window = ids[start : start + int(self.cfg.CHUNK_SIZE)]
                    if len(window) < int(self.cfg.MIN_CHUNK_TOKENS):
                        continue
                    body = self.tokenizer.decode(window)
                    header = f"Source: {rel_source}\nHeading: {heading}\n\n"
                    chunk_text = header + body
                    chunk_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()[:16]
                    chunks.append(
                        {
                            "text": chunk_text,
                            "source": rel_source,
                            "heading": heading,
                            "chunk_hash": chunk_hash,
                        }
                    )
                    if start + int(self.cfg.CHUNK_SIZE) >= len(ids):
                        break

        return chunks

    def _generate_qa_for_chunk(self, chunk: Dict) -> List[Dict]:
        questions = self._pass1_questions(chunk["text"])
        out: List[Dict] = []
        for q in questions:
            answer = self._pass2_answer(chunk["text"], q)
            if not answer.strip():
                continue
            out.append(
                {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": answer.strip()},
                    ],
                    "source": chunk["source"],
                    "chunk_heading": chunk["heading"],
                    "chunk_hash": chunk["chunk_hash"],
                }
            )
        return out

    def _pass1_questions(self, chunk_text: str) -> List[str]:
        prompt = (
            "You are reading technical documentation for a C library.\n"
            "Read this documentation chunk carefully:\n\n"
            f"{chunk_text}\n\n"
            "Generate every question that a developer using this library might ask\n"
            "whose complete answer is contained in this chunk.\n"
            "Be exhaustive. Include questions about:\n"
            "- Function signatures and parameters\n"
            "- Return values and error conditions\n"
            "- Memory ownership and lifecycle rules\n"
            "- Correct usage patterns\n"
            "- Constraints and preconditions\n"
            "- What happens in edge cases\n\n"
            "Output one question per line. No answers."
        )
        out = self.engine.generate(
            prompt,
            SamplingParams(
                max_new_tokens=self.cfg.MAX_COMPLETION_TOKENS,
                temperature=self.cfg.TEMPERATURE_QUESTIONS,
                n=1,
                top_p=0.95,
            ),
        )
        text = out[0].text if out else ""
        return self._parse_questions(text)[: int(self.cfg.MAX_QUESTIONS_PER_CHUNK)]

    def _pass2_answer(self, chunk_text: str, question: str) -> str:
        prompt = (
            "You are answering a question about a C library.\n"
            "Here is the relevant documentation:\n\n"
            f"{chunk_text}\n\n"
            f"Question: {question}\n\n"
            "Answer this question using ONLY information from the documentation above.\n"
            "Quote the relevant parts verbatim where possible.\n"
            "Do not add any information not present in the documentation.\n"
            "Do not paraphrase if the documentation states something precisely.\n"
            "If the documentation uses specific terms, types, or values, use them exactly."
        )
        out = self.engine.generate(
            prompt,
            SamplingParams(
                max_new_tokens=self.cfg.MAX_COMPLETION_TOKENS,
                temperature=self.cfg.TEMPERATURE_ANSWERS,
                n=1,
                top_p=1.0,
            ),
        )
        return out[0].text if out else ""

    def build_final_dataset(self, augmented_path: Optional[str]) -> Tuple[str, str]:
        qa_rows = self._load_jsonl(str(self.qa_path))
        aug_rows = self._load_jsonl(augmented_path) if augmented_path else []

        weighted_rows = []
        weighted_rows.extend(self._apply_weight(qa_rows, float(self.cfg.QA_WEIGHT)))
        weighted_rows.extend(self._apply_weight(aug_rows, float(self.cfg.AUGMENTED_WEIGHT)))

        if not weighted_rows:
            raise RuntimeError("No rows found to build dataset.")

        samples = [self._to_training_row(r) for r in weighted_rows]
        rng = random.Random(int(self.cfg.SEED))
        rng.shuffle(samples)

        n_total = len(samples)
        n_val = max(1, int(round(n_total * float(self.cfg.EVAL_SPLIT))))
        n_val = min(n_val, max(1, n_total - 1))
        val = samples[:n_val]
        train = samples[n_val:]

        dataset_dir = Path(self.cfg.DATASET_DIR)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        train_path = dataset_dir / "train.jsonl"
        val_path = dataset_dir / "val.jsonl"
        self._write_jsonl(train_path, train)
        self._write_jsonl(val_path, val)

        print(
            f"[sft_primer] dataset built: train={len(train)} val={len(val)} "
            f"(qa={len(qa_rows)}, augmented={len(aug_rows)})",
            flush=True,
        )

        return str(train_path), str(val_path)

    def _to_training_row(self, row: Dict) -> Dict:
        user = row["messages"][0]["content"].strip()
        assistant = row["messages"][1]["content"].strip()

        prompt_text = f"User: {user}\nAssistant:"
        completion_text = f" {assistant}"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)

        if not completion_ids:
            completion_ids = self.tokenizer.encode(" ", add_special_tokens=False)

        return {
            "messages": row["messages"],
            "prompt": prompt_text,
            "completion": completion_text,
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "completion_mask": [1] * len(completion_ids),
            "source": row.get("source", "unknown"),
            "chunk_heading": row.get("chunk_heading", ""),
        }

    def _existing_chunk_hashes(self) -> set:
        seen = set()
        for row in self._load_jsonl(str(self.qa_path)):
            chunk_hash = row.get("chunk_hash")
            if chunk_hash:
                seen.add(chunk_hash)
        return seen

    @staticmethod
    def _split_sections(text: str) -> List[Tuple[str, str]]:
        lines = text.splitlines()
        sections: List[Tuple[str, List[str]]] = []
        current_heading = "Document"
        current_lines: List[str] = []

        for line in lines:
            if re.match(r"^#{1,6}\s+", line.strip()):
                if current_lines:
                    sections.append((current_heading, current_lines))
                current_heading = re.sub(r"^#{1,6}\s+", "", line.strip())
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, current_lines))

        out = []
        for heading, body_lines in sections:
            body = "\n".join(body_lines).strip()
            if body:
                out.append((heading, body))
        return out

    @staticmethod
    def _parse_questions(text: str) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in text.splitlines():
            q = raw.strip()
            q = re.sub(r"^\s*[-*\d\.)]+\s*", "", q)
            if not q:
                continue
            if not q.endswith("?"):
                q = q + "?"
            if q in seen:
                continue
            seen.add(q)
            out.append(q)
        return out

    @staticmethod
    def _apply_weight(rows: List[Dict], weight: float) -> List[Dict]:
        if weight <= 0:
            return []
        whole = int(weight)
        frac = weight - whole
        out: List[Dict] = []
        out.extend(rows * max(0, whole))
        if frac > 0:
            threshold = int(round(frac * 1000))
            for row in rows:
                key = row["messages"][0]["content"] + "\n" + row["messages"][1]["content"]
                h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 1000
                if h < threshold:
                    out.append(row)
        return out

    @staticmethod
    def _load_jsonl(path: Optional[str]) -> List[Dict]:
        if not path:
            return []
        p = Path(path)
        if not p.exists():
            return []
        rows: List[Dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _write_jsonl(path: Path, rows: List[Dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
