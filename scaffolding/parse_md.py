from __future__ import annotations

import re
from pathlib import Path


def parse_scada_md(path: str | Path) -> list[dict]:
    """Parse scada.md and return a list of function entries.

    Each entry:
        {
            "name": str,
            "description": str,
            "interacts_with": list[str],
        }
    two_process is intentionally absent — the LLM decides in generate_code.
    """
    text = Path(path).read_text(encoding="utf-8")
    entries = []

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue

        body = line[2:]

        # name — rest  (em-dash U+2014, en-dash U+2013, or spaced hyphen)
        match = re.match(r"^(\S+)\s+[—–-]+\s+(.+)$", body)
        if not match:
            continue

        name = match.group(1)
        rest = match.group(2)

        description = rest
        interacts_with: list[str] = []

        # Parse trailing (Usage: ...; Interacts with: ...)
        paren_match = re.search(r"\((.+)\)\s*$", rest)
        if paren_match:
            inner = paren_match.group(1)
            description = rest[: paren_match.start()].strip()

            iw_match = re.search(r"Interacts with:\s*([^;)]+)", inner)
            if iw_match:
                interacts_with = [
                    x.strip() for x in iw_match.group(1).split(",") if x.strip()
                ]

        entries.append(
            {
                "name": name,
                "description": description,
                "interacts_with": interacts_with,
            }
        )

    return entries


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "scada.md"
    for entry in parse_scada_md(path):
        print(json.dumps(entry, indent=2))
