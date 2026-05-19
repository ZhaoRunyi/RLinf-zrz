#!/usr/bin/env python3
"""Official RoboChallenge rubric scoring with GPT-5.5 and score reconciliation."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from toolkits.episode2frame.gpt55_robochallenge_annotate_combined import (
    build_grid_contact_sheets,
    call_model,
    get_client,
    image_to_data_url,
    load_records,
)

RUBRIC = {
    "task_name": "put_cup_on_coaster",
    "prompt": "place the cup on the coaster",
    "max_score": 10.0,
    "scoring": [
        {"stage_id": "s1_pick_up_cup", "stage": "Pick up the cup", "points": 4.0, "critical": True},
        {"stage_id": "s2_move_to_destination", "stage": "Move the cup to the destination", "points": 2.0, "critical": True},
        {"stage_id": "s3_place_on_coaster", "stage": "Place the cup correctly on the coaster", "points": 3.0, "critical": True},
        {"stage_id": "s4_reset_arm", "stage": "Reset the robotic arm", "points": 1.0, "critical": False},
    ],
    "retry_penalty": "For each visible retry after a failed attempt in a stage, subtract 0.5 points from that stage. If a stage score drops below 0 or consecutive failures exceed 4, the rollout may terminate.",
}

SYSTEM = """You are a strict RoboChallenge official-score reconciler.
Use only visible evidence from the attached time-labeled multi-view contact sheets.
Evaluate the rollout with the provided official RoboChallenge scoring rubric.

Scoring protocol:
- Award each stage's points only when the stage is visibly completed.
- For completed stages, report the earliest visible achievement time, not the later
  time when the completion is merely confirmed or remains stable.
- A retry means a visible failed attempt followed by another attempt in the same stage; each retry subtracts 0.5 points from that stage.
- Do not count harmless waiting/static frames as retries.
- "Reset the robotic arm" is a scoring stage; it is different from retry count.
- Compute the score as sum(max(stage_points_if_completed - 0.5 * retry_count, 0) for each stage).
- The known rollout score is an official constraint. If your visual estimate disagrees, first state the visual estimate, then provide a reconciled retry/stage allocation whose computed_score matches the known score as closely as possible without inventing impossible evidence.
- Return valid JSON only.
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "episode_group": {"type": "string"},
        "episode_id": {"type": "string"},
        "known_rollout_score": {"type": "number"},
        "visual_estimated_score_before_reconcile": {"type": "number"},
        "computed_score": {"type": "number"},
        "score_matches_known": {"type": "boolean"},
        "score_difference": {"type": "number"},
        "total_retry_count": {"type": "integer"},
        "reset_arm_stage_completed": {"type": "boolean"},
        "success_prediction": {"type": "boolean"},
        "stage_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "stage_id": {"type": "string"},
                    "stage_name": {"type": "string"},
                    "max_points": {"type": "number"},
                    "completed": {"type": "boolean"},
                    "first_achievement_time_sec": {"type": ["number", "null"]},
                    "completion_time_sec": {"type": ["number", "null"]},
                    "stable_confirmation_time_sec": {"type": ["number", "null"]},
                    "base_points_awarded": {"type": "number"},
                    "retry_count": {"type": "integer"},
                    "retry_penalty": {"type": "number"},
                    "final_points": {"type": "number"},
                    "evidence": {"type": "string"},
                    "uncertainty": {"type": "string"},
                },
                "required": [
                    "stage_id",
                    "stage_name",
                    "max_points",
                    "completed",
                    "first_achievement_time_sec",
                    "completion_time_sec",
                    "stable_confirmation_time_sec",
                    "base_points_awarded",
                    "retry_count",
                    "retry_penalty",
                    "final_points",
                    "evidence",
                    "uncertainty",
                ],
            },
        },
        "retry_events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "stage_id": {"type": "string"},
                    "time_range_sec": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "reason": {"type": "string"},
                },
                "required": ["stage_id", "time_range_sec", "reason"],
            },
        },
        "reconciliation_notes": {"type": "string"},
    },
    "required": [
        "episode_group",
        "episode_id",
        "known_rollout_score",
        "visual_estimated_score_before_reconcile",
        "computed_score",
        "score_matches_known",
        "score_difference",
        "total_retry_count",
        "reset_arm_stage_completed",
        "success_prediction",
        "stage_scores",
        "retry_events",
        "reconciliation_notes",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined-dir", default="logs/episode2frame/put_cup_on_coaster_combined_videos")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task", default="put_cup_on_coaster")
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    p.add_argument("--api-workers", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=512)
    p.add_argument("--frames-per-image", type=int, default=64)
    p.add_argument("--grid-cols", type=int, default=8)
    p.add_argument("--tile-width", type=int, default=240)
    p.add_argument("--tile-label-font-size", type=int, default=13)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--selection", choices=["sample", "all"], default="sample")
    p.add_argument("--limit-episodes", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--episode-id", action="append", default=None)
    p.add_argument("--score-min", type=float, default=None)
    p.add_argument("--score-max", type=float, default=None)
    p.add_argument("--sort-by", choices=["path", "score", "duration_desc"], default="path")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:180]


def unique_records(records: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    records = [r for r in records if str(r.get("task")) == task]
    unique = []
    seen_paths = set()
    for r in records:
        if r.get("path") in seen_paths:
            continue
        seen_paths.add(r.get("path"))
        unique.append(r)
    return unique


def episode_group(record: dict[str, Any]) -> str:
    # The same episode id can appear in multiple exports. The combined-video
    # directory name is unique and stable, e.g. 000123_<uuid>.
    return safe_stem(Path(record["path"]).parent.name)


def choose_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    unique = unique_records(records, args.task)
    if args.episode_id:
        wanted = set(args.episode_id)
        unique = [r for r in unique if str(r.get("episode_id")) in wanted or episode_group(r) in wanted]
    if args.score_min is not None:
        unique = [r for r in unique if float(r.get("score", 0.0)) >= float(args.score_min)]
    if args.score_max is not None:
        unique = [r for r in unique if float(r.get("score", 0.0)) <= float(args.score_max)]

    if args.sort_by == "score":
        unique.sort(key=lambda r: (float(r.get("score", 0.0)), episode_group(r)))
    elif args.sort_by == "duration_desc":
        unique.sort(key=lambda r: (-max((r.get("frame_counts") or {}).values() or [0]), episode_group(r)))
    else:
        unique.sort(key=lambda r: episode_group(r))

    if args.selection == "all":
        start = max(0, int(args.start_index))
        selected = unique[start:]
        if args.limit_episodes is not None:
            selected = selected[: max(0, int(args.limit_episodes))]
        return [(episode_group(r), r) for r in selected]

    picks = []
    for name, target in [("high", 10.0), ("mid", 5.0), ("low", 0.0)]:
        picks.append(
            (
                name,
                min(
                    unique,
                    key=lambda r: (
                        abs(float(r.get("score", 0.0)) - target),
                        -max((r.get("frame_counts") or {}).values() or [0]),
                    ),
                ),
            )
        )
    return picks


def ensure_images(group: str, record: dict[str, Any], args: argparse.Namespace, media_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    prefix = f"{group}_{record['episode_id']}"
    paths = sorted(media_dir.glob(f"{prefix}.page*.jpg"))
    meta_path = media_dir / f"{prefix}.frames.json"
    if paths and meta_path.exists():
        return paths, json.loads(meta_path.read_text(encoding="utf-8"))
    sheets, frame_meta = build_grid_contact_sheets(
        Path(record["path"]),
        max_frames=args.max_frames,
        frames_per_image=args.frames_per_image,
        grid_cols=args.grid_cols,
        tile_width=args.tile_width,
        label_font_size=args.tile_label_font_size,
    )
    paths = []
    for i, sheet in enumerate(sheets):
        path = media_dir / f"{prefix}.page{i:02d}.jpg"
        sheet.save(path, quality=int(args.jpeg_quality))
        paths.append(path)
    meta_path.write_text(json.dumps(frame_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths, frame_meta


def request_payload(model: str, prompt: str, images: list[Path]) -> dict[str, Any]:
    content = [{"type": "input_text", "text": prompt}]
    content.extend({"type": "input_image", "image_url": image_to_data_url(p)} for p in images)
    return {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM}]},
            {"role": "user", "content": content},
        ],
        "text": {"format": {"type": "json_schema", "name": "official_score_reconcile", "schema": SCHEMA, "strict": True}},
    }


def redacted(payload: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(payload))
    for item in data.get("input", []):
        for part in item.get("content", []):
            if part.get("type") == "input_image":
                part["image_url"] = "<base64 image data omitted>"
    return data


def build_prompt(group: str, record: dict[str, Any], frame_meta: list[dict[str, Any]]) -> str:
    body = {
        "episode_group": group,
        "episode_id": record.get("episode_id"),
        "known_rollout_score": float(record.get("score", 0.0)),
        "task": RUBRIC,
        "video_metadata": {
            "layout": "Each attached image is a 512-sample grid page. Each tile contains horizontally stacked multi-view frames. Tile labels show #sample_id, source frame, and t=seconds.",
            "frame_counts": record.get("frame_counts"),
            "sampled_frames": frame_meta,
        },
        "required_output_behavior": [
            "First decide which official scoring stages are visibly completed.",
            "For each completed stage, first_achievement_time_sec is the earliest labeled sampled time at which the stage's scoring criterion is visibly true.",
            "completion_time_sec must equal first_achievement_time_sec for backward compatibility.",
            "stable_confirmation_time_sec is the later time when the achieved state is confirmed stable; use null if no separate confirmation is needed.",
            "Do not use the last static/waiting frame as a completion time unless the scoring criterion first becomes visible only there.",
            "If the evidence is a range like t=a-b, choose the earliest time in the range where the criterion itself is already visible, not the end of the range.",
            "Count visible retries per stage. Do not count static waiting as retry.",
            "Compute base_points_awarded, retry_penalty = 0.5 * retry_count, and final_points for each stage.",
            "Make computed_score equal the official known_rollout_score if the video evidence can reasonably support it; otherwise make the closest score and explain the mismatch.",
        ],
    }
    return json.dumps(body, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    media_dir = out / "media"
    req_dir = out / "requests"
    resp_dir = out / "responses"
    for d in [media_dir, req_dir, resp_dir]:
        d.mkdir(parents=True, exist_ok=True)
    summary = out / "official_score_reconcile.jsonl"
    done = set()
    if args.resume and summary.exists():
        for line in summary.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done.add(d.get("episode_group"))
    elif summary.exists():
        summary.unlink()

    jobs = []
    for group, record in choose_records(load_records(Path(args.combined_dir)), args):
        if group in done:
            continue
        jobs.append((group, record))
    print(json.dumps({"jobs": len(jobs), "workers": args.api_workers, "output_dir": str(out)}, ensure_ascii=False), flush=True)
    if not jobs:
        print(f"[write] {summary}", flush=True)
        return
    client = get_client(args.api_key_env, os.environ.get(args.base_url_env))
    lock = threading.Lock()
    progress = {"done": 0, "total": len(jobs)}

    def worker(job: tuple[Any, ...]) -> dict[str, Any]:
        group, record = job
        images, frame_meta = ensure_images(group, record, args, media_dir)
        prompt = build_prompt(group, record, frame_meta)
        payload = request_payload(args.model, prompt, images)
        stem = safe_stem(f"{group}_{record['episode_id']}")
        (req_dir / f"{stem}.request.json").write_text(json.dumps(redacted(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        parsed, raw = call_model(client, payload)
        (resp_dir / f"{stem}.raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {
            **parsed,
            "episode_group": group,
            "episode_id": str(record["episode_id"]),
            "known_rollout_score": float(record.get("score", 0.0)),
            "combined_video_path": record["path"],
            "source_metadata_path": str(Path(record["path"]).parent / "metadata.json"),
            "frame_counts": record.get("frame_counts"),
        }
        (resp_dir / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        with lock:
            progress["done"] += 1
            with summary.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[done] {progress['done']}/{progress['total']} {group} known={result.get('known_rollout_score')} "
                f"computed={result.get('computed_score')} "
                f"match={result.get('score_matches_known')} retries={result.get('total_retry_count')}",
                flush=True,
            )
        return result

    with ThreadPoolExecutor(max_workers=max(1, int(args.api_workers))) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
    print(f"[write] {summary}", flush=True)


if __name__ == "__main__":
    main()
