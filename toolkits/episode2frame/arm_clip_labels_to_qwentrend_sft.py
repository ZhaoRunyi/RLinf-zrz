#!/usr/bin/env python3
"""Convert arm_clip_labels.jsonl into QwenTrendProgressSFTDataset data."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.reward.preprocess_qwentrend_reward_dataset import (
    _build_messages,
    _build_prompt,
    _to_uint8_rgb,
    balance_and_split_by_episode,
)

LABELS = ("positive", "negative", "unclear")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_hstack_frame(frame_bgr: Any, num_views: int) -> list[Any]:
    width = frame_bgr.shape[1]
    view_width = width // num_views
    return [frame_bgr[:, index * view_width : (index + 1) * view_width] for index in range(num_views)]


def read_clip_views(
    video_path: Path,
    frame_indices: list[int],
    *,
    num_views: int,
    main_view_index: int,
    extra_view_index: int,
) -> tuple[list[Any], list[Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last_frame = max(0, frame_count - 1)
    main_frames = []
    extra_view_frames = []
    for frame_index in frame_indices:
        safe_index = min(max(0, int(frame_index)), last_frame)
        capture.set(cv2.CAP_PROP_POS_FRAMES, safe_index)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {safe_index} from {video_path}")
        views = split_hstack_frame(frame_bgr, num_views)
        main_frames.append(cv2.cvtColor(views[main_view_index], cv2.COLOR_BGR2RGB))
        extra_view_frames.append(cv2.cvtColor(views[extra_view_index], cv2.COLOR_BGR2RGB))
    capture.release()
    return main_frames, extra_view_frames


def sample_label(clip: dict[str, Any], label_field: str) -> str:
    label = str(clip.get(label_field, ""))
    if label not in LABELS:
        raise ValueError(f"Unsupported label {label!r} from field {label_field}")
    return label


def build_prompt(task: str, window_size: int, cot_format: str) -> str:
    prompt = _build_prompt(task, window_size)
    if cot_format == "clean-cot":
        return prompt.replace(
            "Answer with exactly one word: positive, negative, or unclear.",
            "Use <think></think> tags to show a brief reasoning process, "
            "then provide the final label as `Answer: positive`, "
            "`Answer: negative`, or `Answer: unclear`.",
        )
    return prompt


def build_answer(label: str, clip: dict[str, Any], cot_format: str) -> str:
    if cot_format == "none":
        return label
    stage = str(clip.get("primary_stage_id") or "the current stage")
    gain = float(clip.get("local_gain", 0.0))
    if label == "positive":
        thought = f"The score curve increases over this window near {stage}, so task progress improves."
    elif label == "negative":
        if float(clip.get("known_rollout_score", 0.0)) <= 0.0:
            thought = "The episode is failed by ground-truth outcome, so this window is treated as negative reward evidence."
        else:
            thought = f"The score curve decreases over this window near {stage}, so task progress worsens."
    else:
        thought = f"The score change is small ({gain:.4f}) or ambiguous, so the progress direction is unclear."
    return f"<think>{thought}</think>\nAnswer: {label}"


def build_episodes(args: argparse.Namespace) -> list[dict[str, Any]]:
    prompt = build_prompt(args.task_description, args.window_size, args.cot_format)
    clips = read_jsonl(Path(args.clip_labels))
    grouped_clips = {}
    for clip in clips:
        if args.trainable_only and float(clip.get("pi06_training_weight", 1.0)) <= 0.0:
            continue
        grouped_clips.setdefault(str(clip["combined_video_path"]), []).append(clip)

    episodes = []
    for video_path_text, episode_clips in grouped_clips.items():
        video_path = Path(video_path_text)
        samples = []
        for clip in episode_clips:
            label = sample_label(clip, args.label_field)
            if args.fail_windows_negative and float(clip.get("known_rollout_score", 0.0)) <= 0.0:
                label = "negative"
            answer = build_answer(label, clip, args.cot_format)
            frame_indices = [int(frame) for frame in clip["source_frames"]]
            main_frames, extra_view_frames = read_clip_views(
                video_path,
                frame_indices,
                num_views=args.num_views,
                main_view_index=args.main_view_index,
                extra_view_index=args.extra_view_index,
            )
            episode_id = str(clip["episode_id"])
            samples.append(
                {
                    "sample_id": f"{episode_id}_clip_{int(clip['clip_index']):06d}",
                    "task": args.task_description,
                    "prompt": prompt,
                    "answer": answer,
                    "label": label,
                    "score": float(clip["local_gain"]),
                    "main_frames": main_frames,
                    "extra_view_frames": extra_view_frames,
                    "source_episode_path": str(video_path),
                    "episode_id": episode_id,
                    "env_idx": None,
                    "success": float(clip.get("known_rollout_score", 0.0)) > 0.0,
                    "start_idx": frame_indices[0],
                    "end_idx": frame_indices[-1],
                    "augmentation": None,
                    "supervision": {
                        "label": label,
                        "score": float(clip["local_gain"]),
                        "score_name": "arm_clip_local_gain",
                        "score_source": args.label_field,
                        "advantage_label": clip.get("advantage_label"),
                        "pi06_advantage_label": clip.get("pi06_advantage_label"),
                        "pi06_training_weight": clip.get("pi06_training_weight"),
                        "primary_stage_id": clip.get("primary_stage_id"),
                        "times_sec": clip.get("times_sec"),
                        "score_values": clip.get("score_values"),
                        "stage_ids": clip.get("stage_ids"),
                    },
                }
            )
        if samples:
            episodes.append({"samples": samples, "source_episode_path": str(video_path), "episode_key": str(video_path.resolve())})
    return episodes


def save_split(samples: list[dict[str, Any]], split_dir: Path, window_size: int) -> tuple[str, dict[str, int]]:
    pkl_dir = split_dir / "pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        pkl_path = (pkl_dir / f"{sample['label']}_{sample['sample_id']}.pkl").resolve()
        with pkl_path.open("wb") as file:
            pickle.dump(
                {
                    "main_frames": [_to_uint8_rgb(frame) for frame in sample["main_frames"]],
                    "extra_view_frames": [_to_uint8_rgb(frame) for frame in sample["extra_view_frames"]],
                    "label": sample["label"],
                    "score": sample["score"],
                    "source_episode_path": sample["source_episode_path"],
                    "start_idx": sample["start_idx"],
                    "end_idx": sample["end_idx"],
                    "augmentation": sample["augmentation"],
                },
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        row = {
            "task": sample["task"],
            "prompt": sample["prompt"],
            "question": sample["prompt"],
            "answer": sample["answer"],
            "pkl_path": str(pkl_path),
            "messages": _build_messages(sample["prompt"], sample["answer"]),
            "source_episode_path": sample["source_episode_path"],
            "segment_metadata": {
                "start_step": sample["start_idx"],
                "end_step": sample["end_idx"],
                "window_size": window_size,
                "episode_id": sample["episode_id"],
                "env_idx": sample["env_idx"],
                "success": sample["success"],
                "augmentation": sample["augmentation"],
                "views": ["hstack_view_0", "hstack_view_1"],
            },
            "supervision": sample["supervision"],
        }
        rows.append(row)
    manifest_path = split_dir / "segments.jsonl"
    write_jsonl(manifest_path, rows)
    return str(manifest_path), dict(Counter(row["supervision"]["label"] for row in rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-description", default="fold towel")
    parser.add_argument("--label-field", choices=["advantage_label", "pi06_advantage_label"], default="advantage_label")
    parser.add_argument("--trainable-only", action="store_true")
    parser.add_argument("--fail-windows-negative", action="store_true")
    parser.add_argument("--cot-format", choices=["none", "clean-cot"], default="none")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-labels", dest="balance_labels", action="store_true", default=True)
    parser.add_argument("--no-balance-labels", dest="balance_labels", action="store_false")
    parser.add_argument("--max-samples-per-label", type=int, default=None)
    parser.add_argument("--eval-max-samples-per-label", type=int, default=None)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--main-view-index", type=int, default=0)
    parser.add_argument("--extra-view-index", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = build_episodes(args)
    train_samples, eval_samples = balance_and_split_by_episode(
        episodes,
        val_split=args.val_split,
        balance_labels=args.balance_labels,
        max_samples_per_label=args.max_samples_per_label,
        eval_max_samples_per_label=args.eval_max_samples_per_label,
        reverse_positive_as_negative=False,
        random_seed=args.seed,
    )
    train_manifest, train_counts = save_split(train_samples, output_dir / "train", args.window_size)
    eval_manifest, eval_counts = save_split(eval_samples, output_dir / "eval", args.window_size)
    metadata = {
        "clip_labels": args.clip_labels,
        "output_dir": str(output_dir),
        "task_description": args.task_description,
        "label_field": args.label_field,
        "trainable_only": args.trainable_only,
        "fail_windows_negative": args.fail_windows_negative,
        "cot_format": args.cot_format,
        "cot_reference": "RLinf clean-cot uses <think></think> reasoning tags in VLM SFT prompts; QwenTrend itself still stores answer as a raw text field.",
        "export_format": "pkl",
        "num_source_episodes": len(episodes),
        "num_train_samples": len(train_samples),
        "num_eval_samples": len(eval_samples),
        "train_manifest": train_manifest,
        "eval_manifest": eval_manifest,
        "train_label_counts": train_counts,
        "eval_label_counts": eval_counts,
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
