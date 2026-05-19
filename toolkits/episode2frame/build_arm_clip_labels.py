#!/usr/bin/env python3
"""Build clip advantage labels from scored RoboChallenge episodes.

This script is fully local: it consumes the GPT/RoboChallenge score reconciliation
JSONL and does not call any model API.

Internal labeling semantics:
- Stage score is converted into event-sparse score increments anchored near the
  stage achievement time.
- Retry penalties are included as negative score increments, but retry does not
  hard-flip a clip label. If net score rises during retry, the clip is positive.
- Clip score gain uses absolute score gain:
    local_gain > margin   -> positive
    local_gain < -margin  -> negative
    otherwise             -> unclear

Training-oriented π0.6/RECAP semantics:
- ``pi06_advantage_label`` is always binary: positive / negative.
- Weak-gain clips can be kept as negative or masked with training_weight=0,
  depending on ``--pi06-unclear-policy``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

INTERVAL_LABEL_TO_ID = {"regressing": -1, "stagnant": 0, "progressing": 1}
ADVANTAGE_LABEL_TO_ID = {"negative": -1, "unclear": 0, "positive": 1}
PI06_ADVANTAGE_LABEL_TO_ID = {"negative": 0, "positive": 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample-fps", type=float, default=1.0)
    p.add_argument("--source-fps", type=float, default=5.0)
    p.add_argument("--window-size", type=int, default=5, help="Observation samples per clip window.")
    p.add_argument("--stride", type=int, default=1, help="Stride in sampled observations.")
    p.add_argument("--advantage-margin", type=float, default=0.03, help="0-10 score gain margin for clip labels.")
    p.add_argument(
        "--pi06-unclear-policy",
        choices=["discard", "negative"],
        default="discard",
        help=(
            "How to convert weak-gain unclear clips for π0.6-style binary advantage. "
            "'discard' keeps a binary label but sets pi06_training_weight=0; "
            "'negative' trains them as Advantage: negative."
        ),
    )
    p.add_argument("--interval-margin", type=float, default=0.03, help="0-10 score gain margin for interval labels.")
    p.add_argument("--retry-penalty", type=float, default=0.5)
    p.add_argument("--credit-pre-window-sec", type=float, default=6.0)
    p.add_argument("--completion-mass", type=float, default=0.8)
    p.add_argument("--approach-mass", type=float, default=0.2)
    p.add_argument("--interval-positive-ratio-per-stage", type=float, default=0.25)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def episode_duration_sec(result: dict[str, Any], source_fps: float) -> float:
    frame_counts = result.get("frame_counts") or {}
    if frame_counts:
        return max(float(v) for v in frame_counts.values()) / source_fps
    stage_times = []
    for stage in result.get("stage_scores") or []:
        for key in ["stable_confirmation_time_sec", "first_achievement_time_sec", "completion_time_sec"]:
            value = stage.get(key)
            if isinstance(value, int | float):
                stage_times.append(float(value))
    return max(stage_times + [0.0])


def stage_time(stage: dict[str, Any]) -> float | None:
    for key in ["first_achievement_time_sec", "completion_time_sec"]:
        value = stage.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def normalized_times(duration: float, sample_fps: float) -> np.ndarray:
    if duration <= 0:
        return np.asarray([0.0], dtype=np.float32)
    step = 1.0 / sample_fps
    count = int(math.floor(duration / step)) + 1
    times = np.asarray([i * step for i in range(count)], dtype=np.float32)
    if float(times[-1]) < duration:
        times = np.concatenate([times, np.asarray([duration], dtype=np.float32)])
    return times


def add_weighted_credit(
    increments: np.ndarray,
    times: np.ndarray,
    start: float,
    end: float,
    amount: float,
    *,
    weight_power: float,
) -> None:
    if amount <= 0 or end <= start or len(increments) == 0:
        return
    weights = np.zeros_like(increments, dtype=np.float32)
    for i in range(len(increments)):
        t0 = float(times[i])
        t1 = float(times[i + 1])
        ov = overlap(t0, t1, start, end)
        if ov <= 0:
            continue
        center = (max(t0, start) + min(t1, end)) * 0.5
        rel = (center - start) / max(end - start, 1e-6)
        weights[i] = ov * max(rel, 1e-3) ** weight_power
    total = float(weights.sum())
    if total > 1e-8:
        increments += amount * weights / total


def build_score_curve(
    result: dict[str, Any],
    times: np.ndarray,
    *,
    retry_penalty: float,
    credit_pre_window_sec: float,
    completion_mass: float,
    approach_mass: float,
    interval_positive_ratio_per_stage: float,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]], dict[str, float]]:
    """Build sparse score increments and cumulative score curve."""
    increments = np.zeros(max(0, len(times) - 1), dtype=np.float32)
    stage_ids = ["none"] * len(times)
    interval_stage_ids = ["none"] * max(0, len(times) - 1)
    events: list[dict[str, Any]] = []
    stages = result.get("stage_scores") or []

    prev_time = 0.0
    for i, stage in enumerate(stages):
        done_time = stage_time(stage)
        points = float(stage.get("base_points_awarded", 0.0))
        if not stage.get("completed") or done_time is None or points <= 0:
            continue

        start = max(0.0, prev_time)
        end = max(start + 1e-6, done_time)
        stage_id = str(stage.get("stage_id") or f"stage_{i}")
        win_start = max(start, end - credit_pre_window_sec)

        add_weighted_credit(
            increments,
            times,
            win_start,
            end,
            points * completion_mass,
            weight_power=1.5,
        )
        if approach_mass > 0 and win_start > start:
            add_weighted_credit(
                increments,
                times,
                start,
                win_start,
                points * approach_mass,
                weight_power=2.0,
            )

        for k in range(len(times) - 1):
            if overlap(float(times[k]), float(times[k + 1]), start, end) > 0:
                interval_stage_ids[k] = stage_id
        for k, t in enumerate(times):
            if start <= float(t) <= end:
                stage_ids[k] = stage_id

        events.append(
            {
                "type": "stage_sparse_credit",
                "stage_id": stage_id,
                "start_time_sec": start,
                "end_time_sec": end,
                "completion_window_start_sec": win_start,
                "points": points,
                "completion_mass": completion_mass,
                "approach_mass": approach_mass,
            }
        )
        prev_time = end

    for retry in result.get("retry_events") or []:
        rng = retry.get("time_range_sec") or []
        if len(rng) < 2:
            continue
        r0 = float(rng[0])
        r1 = max(r0 + 1e-6, float(rng[1]))
        weights = np.zeros_like(increments, dtype=np.float32)
        for k in range(len(increments)):
            weights[k] = overlap(float(times[k]), float(times[k + 1]), r0, r1)
        total = float(weights.sum())
        if total > 1e-8:
            increments -= retry_penalty * weights / total
        events.append(
            {
                "type": "retry_penalty",
                "stage_id": retry.get("stage_id"),
                "start_time_sec": r0,
                "end_time_sec": r1,
                "points": -retry_penalty,
                "reason": retry.get("reason", ""),
            }
        )

    known = float(result.get("known_rollout_score", result.get("computed_score", 0.0)) or 0.0)
    correction = known - float(increments.sum())
    if len(increments) and abs(correction) > 1e-3:
        positive_indices = np.flatnonzero(increments > 1e-8)
        target_idx = int(positive_indices[-1]) if len(positive_indices) else len(increments) - 1
        increments[target_idx] += correction
        events.append({"type": "endpoint_event_correction", "interval_index": target_idx, "points": correction})

    curve = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(increments, dtype=np.float32)])
    curve = curve[: len(times)]

    stage_thresholds: dict[str, float] = {}
    q = max(0.0, min(1.0, 1.0 - interval_positive_ratio_per_stage))
    for stage_id in sorted(set(interval_stage_ids)):
        if stage_id == "none":
            continue
        vals = np.asarray([increments[i] for i, sid in enumerate(interval_stage_ids) if sid == stage_id and increments[i] > 0])
        if len(vals) == 0:
            continue
        max_points = next((float(s.get("max_points", 0.0)) for s in stages if str(s.get("stage_id")) == stage_id), 0.0)
        stage_thresholds[stage_id] = max(0.05 * max_points, float(np.quantile(vals, q)))
    return curve.astype(np.float32), increments.astype(np.float32), stage_ids, events, stage_thresholds


def retry_overlap_info(result: dict[str, Any], start: float, end: float) -> tuple[bool, list[dict[str, Any]]]:
    hits = []
    for retry in result.get("retry_events") or []:
        rng = retry.get("time_range_sec") or []
        if len(rng) < 2:
            continue
        ov = overlap(start, end, float(rng[0]), float(rng[1]))
        if ov > 0:
            hits.append(
                {
                    "stage_id": retry.get("stage_id"),
                    "overlap_sec": ov,
                    "time_range_sec": [float(rng[0]), float(rng[1])],
                    "reason": retry.get("reason", ""),
                }
            )
    return bool(hits), hits


def interval_label(delta: float, threshold: float, margin: float) -> str:
    if delta < -margin:
        return "regressing"
    if delta >= max(margin, threshold):
        return "progressing"
    return "stagnant"


def advantage_label(local_gain: float, margin: float) -> str:
    if local_gain > margin:
        return "positive"
    if local_gain < -margin:
        return "negative"
    return "unclear"


def pi06_advantage(local_gain: float, margin: float, unclear_policy: str) -> tuple[str, int, float]:
    """Convert local score gain into π0.6/RECAP-style binary advantage."""

    if local_gain > margin:
        return "positive", PI06_ADVANTAGE_LABEL_TO_ID["positive"], 1.0
    if local_gain < -margin:
        return "negative", PI06_ADVANTAGE_LABEL_TO_ID["negative"], 1.0
    if unclear_policy == "negative":
        return "negative", PI06_ADVANTAGE_LABEL_TO_ID["negative"], 1.0
    return "negative", PI06_ADVANTAGE_LABEL_TO_ID["negative"], 0.0


def build_records(result: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    duration = episode_duration_sec(result, args.source_fps)
    times = normalized_times(duration, args.sample_fps)
    curve, increments, stage_ids, events, stage_thresholds = build_score_curve(
        result,
        times,
        retry_penalty=float(args.retry_penalty),
        credit_pre_window_sec=float(args.credit_pre_window_sec),
        completion_mass=float(args.completion_mass),
        approach_mass=float(args.approach_mass),
        interval_positive_ratio_per_stage=float(args.interval_positive_ratio_per_stage),
    )

    intervals: list[dict[str, Any]] = []
    for i in range(len(times) - 1):
        start = float(times[i])
        end = float(times[i + 1])
        retry_active, retry_hits = retry_overlap_info(result, start, end)
        delta = float(increments[i])
        stage_id = str(stage_ids[i])
        threshold = float(stage_thresholds.get(stage_id, args.interval_margin))
        label = interval_label(delta, threshold, float(args.interval_margin))
        intervals.append(
            {
                "episode_group": result.get("episode_group"),
                "episode_id": result.get("episode_id"),
                "combined_video_path": result.get("combined_video_path"),
                "known_rollout_score": float(result.get("known_rollout_score", 0.0)),
                "interval_index": i,
                "start_sec": start,
                "end_sec": end,
                "source_frame_start": int(round(start * args.source_fps)),
                "source_frame_end": int(round(end * args.source_fps)),
                "stage_id": stage_id,
                "score_before": float(curve[i]),
                "score_after": float(curve[i + 1]),
                "score_delta": delta,
                "score_increment": delta,
                "stage_positive_threshold": threshold,
                "retry_active": retry_active,
                "retry_overlaps": retry_hits,
                "label": label,
                "label_id": INTERVAL_LABEL_TO_ID[label],
            }
        )

    clips: list[dict[str, Any]] = []
    window_size = int(args.window_size)
    stride = max(1, int(args.stride))
    if window_size < 2:
        raise ValueError("--window-size must be >= 2")

    for start_idx in range(0, max(0, len(times) - window_size + 1), stride):
        end_idx = start_idx + window_size - 1
        clip_intervals = intervals[start_idx:end_idx]
        if len(clip_intervals) != window_size - 1:
            continue
        label_names = [r["label"] for r in clip_intervals]
        label_ids = [r["label_id"] for r in clip_intervals]
        counts = Counter(label_names)
        local_gain = float(increments[start_idx:end_idx].sum())
        adv = advantage_label(local_gain, float(args.advantage_margin))
        pi06_label, pi06_label_id, pi06_weight = pi06_advantage(
            local_gain,
            float(args.advantage_margin),
            str(args.pi06_unclear_policy),
        )

        weighted_stage_gain: Counter[str] = Counter()
        stage_counter: Counter[str] = Counter()
        for r in clip_intervals:
            sid = str(r.get("stage_id", "none"))
            if sid == "none":
                continue
            stage_counter[sid] += 1
            weighted_stage_gain[sid] += max(0.0, float(r.get("score_increment", 0.0)))
        if weighted_stage_gain:
            primary_stage_id = weighted_stage_gain.most_common(1)[0][0]
        elif stage_counter:
            primary_stage_id = stage_counter.most_common(1)[0][0]
        else:
            primary_stage_id = "none"

        clips.append(
            {
                "episode_group": result.get("episode_group"),
                "episode_id": result.get("episode_id"),
                "combined_video_path": result.get("combined_video_path"),
                "known_rollout_score": float(result.get("known_rollout_score", 0.0)),
                "clip_index": len(clips),
                "sample_fps": float(args.sample_fps),
                "window_size": window_size,
                "clip_start_sec": float(times[start_idx]),
                "clip_end_sec": float(times[start_idx + window_size - 1]),
                "times_sec": [float(x) for x in times[start_idx : start_idx + window_size]],
                "source_frames": [int(round(float(x) * args.source_fps)) for x in times[start_idx : start_idx + window_size]],
                "stage_ids": [stage_ids[i] for i in range(start_idx, start_idx + window_size)],
                "score_values": [float(x) for x in curve[start_idx : start_idx + window_size]],
                "interval_labels": label_names,
                "interval_label_ids": label_ids,
                "majority_label": counts.most_common(1)[0][0] if counts else "stagnant",
                "has_retry": any(r["retry_active"] for r in clip_intervals),
                "has_regression": any(r["label"] == "regressing" for r in clip_intervals),
                "local_gain": local_gain,
                "primary_stage_id": primary_stage_id,
                "advantage_margin": float(args.advantage_margin),
                "advantage_label": adv,
                "advantage_label_id": ADVANTAGE_LABEL_TO_ID[adv],
                "pi06_advantage_label": pi06_label,
                "pi06_advantage_label_id": pi06_label_id,
                "pi06_training_weight": pi06_weight,
                "pi06_prompt_condition": f"Advantage: {pi06_label}",
                "pi06_unclear_policy": str(args.pi06_unclear_policy),
                "intervals": clip_intervals,
            }
        )

    dense = {
        "episode_group": result.get("episode_group"),
        "episode_id": result.get("episode_id"),
        "combined_video_path": result.get("combined_video_path"),
        "known_rollout_score": float(result.get("known_rollout_score", 0.0)),
        "duration_sec": duration,
        "sample_fps": float(args.sample_fps),
        "times_sec": [float(x) for x in times],
        "source_frames": [int(round(float(x) * args.source_fps)) for x in times],
        "stage_ids": stage_ids,
        "score_curve": [float(x) for x in curve],
        "score_increment": [float(x) for x in increments],
        "stage_positive_thresholds": stage_thresholds,
        "events": events,
    }
    return intervals, clips, dense


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    interval_path = out / "arm_interval_labels.jsonl"
    clip_path = out / "arm_clip_labels.jsonl"
    dense_path = out / "arm_dense_score_curves.jsonl"
    summary_path = out / "summary.json"
    if not args.overwrite:
        existing = [p for p in [interval_path, clip_path, dense_path, summary_path] if p.exists()]
        if existing:
            raise FileExistsError(f"Output exists; pass --overwrite: {existing[0]}")

    results = read_jsonl(Path(args.results_jsonl))
    total_intervals = 0
    total_clips = 0
    interval_label_counts: Counter[str] = Counter()
    clip_majority_counts: Counter[str] = Counter()
    advantage_counts: Counter[str] = Counter()
    pi06_advantage_counts: Counter[str] = Counter()
    pi06_trainable_counts: Counter[str] = Counter()
    score_counts: Counter[str] = Counter()
    stage_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    positive_increments: list[float] = []
    completion_window_increment = 0.0
    total_positive_credit = 0.0
    retry_clip_count = 0

    with interval_path.open("w", encoding="utf-8") as f_interval, clip_path.open("w", encoding="utf-8") as f_clip, dense_path.open(
        "w", encoding="utf-8"
    ) as f_dense:
        for idx, result in enumerate(results):
            intervals, clips, dense = build_records(result, args)
            for rec in intervals:
                f_interval.write(json.dumps(rec, ensure_ascii=False) + "\n")
                interval_label_counts[str(rec["label"])] += 1
                stage_label_counts[str(rec["stage_id"])][str(rec["label"])] += 1
                inc = float(rec.get("score_increment", 0.0))
                if inc > 0:
                    positive_increments.append(inc)
            for rec in clips:
                f_clip.write(json.dumps(rec, ensure_ascii=False) + "\n")
                clip_majority_counts[str(rec["majority_label"])] += 1
                advantage_counts[str(rec["advantage_label"])] += 1
                pi06_advantage_counts[str(rec["pi06_advantage_label"])] += 1
                if float(rec.get("pi06_training_weight", 0.0)) > 0:
                    pi06_trainable_counts[str(rec["pi06_advantage_label"])] += 1
                retry_clip_count += int(bool(rec.get("has_retry")))
            f_dense.write(json.dumps(dense, ensure_ascii=False) + "\n")

            times = [float(x) for x in dense.get("times_sec") or []]
            increments = [float(x) for x in dense.get("score_increment") or []]
            for event in dense.get("events") or []:
                if event.get("type") != "stage_sparse_credit":
                    continue
                total_positive_credit += max(0.0, float(event.get("points", 0.0)))
                win_start = float(event.get("completion_window_start_sec", event.get("start_time_sec", 0.0)))
                win_end = float(event.get("end_time_sec", win_start))
                for i, inc in enumerate(increments):
                    if inc <= 0 or i + 1 >= len(times):
                        continue
                    if overlap(times[i], times[i + 1], win_start, win_end) > 0:
                        completion_window_increment += inc

            total_intervals += len(intervals)
            total_clips += len(clips)
            score_counts[str(float(result.get("known_rollout_score", 0.0)))] += 1
            if (idx + 1) % 100 == 0:
                print(f"[done] episodes={idx + 1}/{len(results)} intervals={total_intervals} clips={total_clips}", flush=True)

    summary = {
        "results_jsonl": str(Path(args.results_jsonl)),
        "episodes": len(results),
        "sample_fps": float(args.sample_fps),
        "source_fps": float(args.source_fps),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "advantage_margin": float(args.advantage_margin),
        "pi06_unclear_policy": str(args.pi06_unclear_policy),
        "interval_margin": float(args.interval_margin),
        "credit_pre_window_sec": float(args.credit_pre_window_sec),
        "completion_mass": float(args.completion_mass),
        "approach_mass": float(args.approach_mass),
        "retry_penalty": float(args.retry_penalty),
        "intervals": total_intervals,
        "clips": total_clips,
        "interval_label_counts": dict(sorted(interval_label_counts.items())),
        "clip_majority_label_counts": dict(sorted(clip_majority_counts.items())),
        "clip_advantage_label_counts": dict(sorted(advantage_counts.items())),
        "pi06_advantage_label_counts": dict(sorted(pi06_advantage_counts.items())),
        "pi06_trainable_label_counts": dict(sorted(pi06_trainable_counts.items())),
        "retry_clip_count": retry_clip_count,
        "positive_increment_quantiles": (
            {}
            if not positive_increments
            else {
                "q50": float(np.quantile(positive_increments, 0.50)),
                "q70": float(np.quantile(positive_increments, 0.70)),
                "q90": float(np.quantile(positive_increments, 0.90)),
            }
        ),
        "completion_window_positive_increment_fraction": (
            None if total_positive_credit <= 1e-8 else completion_window_increment / total_positive_credit
        ),
        "known_score_distribution": dict(sorted(score_counts.items(), key=lambda kv: float(kv[0]))),
        "stage_label_counts": {k: dict(sorted(v.items())) for k, v in sorted(stage_label_counts.items())},
        "outputs": {
            "interval_labels": str(interval_path),
            "clip_labels": str(clip_path),
            "dense_score_curves": str(dense_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
