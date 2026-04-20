import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert custom QA JSONL to the required qa_pair.jsonl format"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input JSONL file (e.g. your claude-opus dataset)",
    )
    parser.add_argument(
        "--output", required=True, type=str, help="Output file (e.g. qa_pair.jsonl)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "messages" in data and isinstance(data["messages"], list):
                question = ""
                answer = ""
                reasoning = ""

                for msg in data["messages"]:
                    if msg.get("role") == "user":
                        question = msg.get("content", "")
                    elif msg.get("role") == "assistant":
                        answer = msg.get("content", "")
                        reasoning = msg.get("reasoning", "")

                if question and answer:
                    out_row = {
                        "question": question.strip(),
                        "answer": answer.strip(),
                        "reasoning": reasoning.strip(),
                    }
                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    count += 1

    print(f"Successfully converted {count} rows to {output_path}")


if __name__ == "__main__":
    main()
