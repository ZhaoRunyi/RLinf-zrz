#!/usr/bin/env python3
"""Summarize GPT RoboChallenge score reconciliation outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--expected-total", type=int, default=None)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.results_jsonl))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    retry_rows = [r for r in rows if int(r.get("total_retry_count") or 0) > 0 or r.get("retry_events")]
    mismatch_rows = [r for r in rows if not bool(r.get("score_matches_known"))]

    (out / "retry_candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in retry_rows),
        encoding="utf-8",
    )
    (out / "score_mismatches.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mismatch_rows),
        encoding="utf-8",
    )

    score_dist = Counter(str(float(r.get("known_rollout_score", 0.0))) for r in rows)
    retry_dist = Counter(str(int(r.get("total_retry_count") or 0)) for r in rows)
    stage_retry_dist: Counter[str] = Counter()
    for r in retry_rows:
        for stage in r.get("stage_scores") or []:
            count = int(stage.get("retry_count") or 0)
            if count > 0:
                stage_retry_dist[str(stage.get("stage_id"))] += count

    summary = {
        "results_jsonl": str(Path(args.results_jsonl)),
        "expected_total": args.expected_total,
        "completed": len(rows),
        "progress": None if not args.expected_total else len(rows) / args.expected_total,
        "retry_candidate_count": len(retry_rows),
        "mismatch_count": len(mismatch_rows),
        "known_score_distribution": dict(sorted(score_dist.items(), key=lambda kv: float(kv[0]))),
        "retry_count_distribution": dict(sorted(retry_dist.items(), key=lambda kv: int(kv[0]))),
        "stage_retry_count_distribution": dict(sorted(stage_retry_dist.items())),
        "retry_candidate_groups": [r.get("episode_group") for r in retry_rows],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
