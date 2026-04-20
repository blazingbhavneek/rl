import json
from pathlib import Path
from typing import Dict, List


# What this class does:
# - Writes per-step curriculum stats to a JSONL file.
# - Can also read the full saved history back.
class StatsWriter:

    # What this does:
    # - Prepare stats output path and make sure directory exists.
    #
    # Parameter meanings:
    # - checkpoint_dir: folder where stats.jsonl will be stored.
    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.checkpoint_dir / "stats.jsonl"

    # What this does:
    # - Append one stats record for a single training step.
    #
    # Parameter meanings:
    # - step: current training step index.
    # - distribution: current bucket probability list.
    # - solve_rates: per-bucket solve-rate summary.
    # - mean_score: mean score for this step's evaluated problems.
    # - promoted_this_step: number of problems promoted this step.
    # - total_problems_seen: number of unique tracked problems so far.
    def write(
        self,
        step: int,
        distribution: List[float],
        solve_rates: Dict[int, float],
        mean_score: float,
        promoted_this_step: int,
        total_problems_seen: int,
    ) -> None:
        row = {
            "step": int(step),
            "distribution": distribution,
            "solve_rates": {str(k): v for k, v in solve_rates.items()},
            "mean_score": float(mean_score),
            "promoted_this_step": int(promoted_this_step),
            "total_problems_seen": int(total_problems_seen),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # What this does:
    # - Read all saved stats rows from stats.jsonl.
    #
    # Parameter meanings:
    # - No input parameters.
    #
    # Return:
    # - List of parsed JSON rows in file order.
    def read_history(self) -> List[Dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
