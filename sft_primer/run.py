from __future__ import annotations

import argparse
from pathlib import Path

from inference import SGLangOfflineEngine

from .config import build_config
from .dataset import SFTPrimerDataset
from .train import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT primer pipeline")
    p.add_argument("--library", type=str, default=None, help="Library name for output namespace")
    p.add_argument("--skip-generation", action="store_true", help="Use existing train/val datasets")
    p.add_argument("--train-only", action="store_true", help="Alias for skip generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args.library)

    dataset_path = str(Path(cfg.DATASET_DIR) / "train.jsonl")
    val_path = str(Path(cfg.DATASET_DIR) / "val.jsonl")

    if not args.skip_generation and not args.train_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)
        engine = SGLangOfflineEngine(
            cfg.MODEL_PATH,
            engine_kwargs={
                "mem_fraction_static": cfg.SGLANG_MEM_FRAC,
                "context_length": cfg.MAX_SEQ_LEN,
            },
        )
        try:
            dataset = SFTPrimerDataset(cfg.DOCS_FOLDER, tokenizer, engine, cfg)
            dataset_path, val_path = dataset.run()
        finally:
            engine.shutdown()

    train(dataset_path, val_path, config=cfg)

    print("\nDone. Load into RL pipeline:", flush=True)
    print(f"  SFT_PRIMER_CHECKPOINT = '{cfg.FINAL_LORA_DIR}'", flush=True)


if __name__ == "__main__":
    main()
