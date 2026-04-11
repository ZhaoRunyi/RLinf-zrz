#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

from rlinf.models.embodiment.reward import get_reward_model_class
from rlinf.models.embodiment.reward.vlm_reward_utils.debug_video import (
    render_debug_video,
)


DEFAULT_CONFIG = REPO_ROOT / "examples/embodiment/config/maniskill_ppo_mlp_qwen3vl4b_robochallenge_reward_useoutput0.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--episode-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_episode(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_step_observations(episode: dict[str, Any]) -> list[dict[str, Any]]:
    observations = list(episode.get("observations", []))
    actions = list(episode.get("actions", []))
    if len(observations) >= len(actions) + 1:
        observations = observations[1 : len(actions) + 1]
    return [obs for obs in observations if isinstance(obs, dict)]


def get_step_rewards(episode: dict[str, Any], num_steps: int) -> list[float]:
    rewards = list(episode.get("rewards", []))
    if len(rewards) >= num_steps + 1:
        rewards = rewards[1 : num_steps + 1]
    rewards = rewards[:num_steps]
    return [float(r) for r in rewards]


def get_task_description(observations: list[Any]) -> str:
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        if "task_descriptions" not in obs:
            continue
        task_description = obs["task_descriptions"]
        if isinstance(task_description, (list, tuple)):
            if not task_description:
                continue
            task_description = task_description[0]
        if task_description is None:
            continue
        task_description = str(task_description).strip()
        if task_description:
            return task_description
    return ""


def get_main_image(obs: dict[str, Any]) -> Any | None:
    if "main_images" not in obs:
        return None
    image = obs["main_images"]
    if isinstance(image, (list, tuple)) and image:
        image = image[0]
    return image


def get_history_frames(
    step_observations: list[dict[str, Any]], start: int, end: int
) -> list[Any]:
    frames: list[Any] = []
    for obs in step_observations[start:end]:
        image = get_main_image(obs)
        if image is not None:
            frames.append(image)
    return frames


def build_sample_ranges(
    num_steps: int,
    history_window_size: int,
    interval: int,
    input_on_done: bool,
) -> list[tuple[int, int]]:
    if history_window_size <= 0 or interval <= 0 or num_steps <= 0:
        return []

    end_steps = list(range(history_window_size, num_steps + 1, interval))
    if input_on_done and num_steps % interval != 0:
        end_steps.append(num_steps)

    ranges: list[tuple[int, int]] = []
    for end_step in end_steps:
        if end_step % interval == 0:
            start_step = max(0, end_step - history_window_size)
        else:
            remainder = end_step % interval
            start_step = max(0, end_step - remainder)
        ranges.append((start_step, end_step))
    return ranges


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def inspect_simple_output(text: str) -> dict[str, Any]:
    obj = extract_json_object(text)
    judgement: str | None = None

    if obj is not None:
        for key in ("judgement", "answer", "label"):
            value = obj.get(key)
            if value is not None:
                judgement = str(value).strip().lower()
                break

    if judgement is None:
        matches = re.findall(r"\b(positive|negative)\b", text.strip().lower())
        if matches:
            judgement = matches[-1]

    return {
        "parse_ok": judgement in {"positive", "negative"},
        "judgement": judgement,
    }


def compute_proxy_gt(
    step_rewards: list[float],
    segment_start: int,
    end_step: int,
    success: bool,
) -> dict[str, float]:
    segment_rewards = step_rewards[segment_start:end_step]
    prefix_rewards = step_rewards[:end_step]
    segment_return = float(sum(segment_rewards))
    prefix_return = float(sum(prefix_rewards))
    return {
        "gt_reward": float(segment_return > 0.0),
        "segment_return": segment_return,
        "prefix_return": prefix_return,
        "episode_success": float(success),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / (var_x**0.5 * var_y**0.5)


def build_model(config_path: str, device: str):
    full_cfg = OmegaConf.load(config_path)
    model_cfg = full_cfg.reward.model
    with open_dict(model_cfg):
        model_cfg.num_envs = 1
    reward_cls = get_reward_model_class(model_cfg.model_type)
    model = reward_cls(model_cfg)
    model = model.to(device)
    model.eval()
    return model, full_cfg


def build_history_input(
    history_buffer_names: list[str],
    history_window_frames: list[Any],
) -> dict[str, dict[str, list[list[Any]]]]:
    if set(history_buffer_names) != {"history_window"}:
        raise ValueError(
            "test_4b.py assumes the 4B simple config only uses history_window"
        )
    return {"history_window": {"main_images": [history_window_frames]}}


def run_single_sample(
    model,
    task_description: str,
    history_window_frames: list[Any],
) -> dict[str, Any]:
    observations = {"task_descriptions": [task_description]}
    history_input = build_history_input(
        model.history_buffer_names,
        history_window_frames=history_window_frames,
    )

    with torch.inference_mode():
        batched_inputs, valid_input_ids = model.input_builder.build_inputs(
            observations,
            model._model.device,
            history_input,
        )
        if len(valid_input_ids) == 0:
            return {
                "pred_reward": 0.0,
                "raw_output": "",
                "parse_ok": False,
                "judgement": None,
                "skip_reason": "invalid_input",
            }

        prompt_length = batched_inputs["input_ids"].shape[-1]
        output_ids = model._model.generate(**batched_inputs, **model.gen_kwargs)
        outputs = model._processor.batch_decode(
            output_ids[..., prompt_length:],
            skip_special_tokens=True,
        )
        pred_rewards = model.reward_parser.parse_rewards(outputs)

    raw_output = outputs[0]
    parsed = inspect_simple_output(raw_output)
    return {
        "pred_reward": float(pred_rewards[0].item()),
        "raw_output": raw_output,
        "parse_ok": bool(parsed["parse_ok"]),
        "judgement": parsed["judgement"],
        "skip_reason": None,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = 4

    model, cfg = build_model(args.config, args.device)

    history_buffers = cfg.reward.model.history_buffers
    history_window_cfg = history_buffers.history_window
    history_window_size = int(history_window_cfg.history_size)
    interval = int(history_window_cfg.input_interval)
    input_on_done = bool(history_window_cfg.get("input_on_done", False))

    episode_paths = sorted(glob.glob(args.episode_glob))[: args.max_episodes]
    rows: list[dict[str, Any]] = []

    for episode_path in episode_paths:
        episode = load_episode(episode_path)
        step_observations = get_step_observations(episode)
        sample_ranges = build_sample_ranges(
            num_steps=len(step_observations),
            history_window_size=history_window_size,
            interval=interval,
            input_on_done=input_on_done,
        )
        if not sample_ranges:
            continue

        task_description = get_task_description(step_observations)
        if not task_description:
            continue

        step_rewards = get_step_rewards(episode, len(step_observations))
        success = bool(episode.get("success", False))
        rank = int(episode.get("rank", -1))
        env_idx = int(episode.get("env_idx", -1))
        episode_id = int(episode.get("episode_id", -1))

        for sample_idx, (segment_start, end_step) in enumerate(sample_ranges):
            history_window_frames = get_history_frames(
                step_observations, segment_start, end_step
            )
            if not history_window_frames:
                continue

            pred = run_single_sample(
                model,
                task_description,
                history_window_frames=history_window_frames,
            )
            gt = compute_proxy_gt(step_rewards, segment_start, end_step, success)

            raw_output = pred["raw_output"].strip() or "<empty>"
            footer_lines = [
                f"predict_reward: {pred['pred_reward']:.4f} | gt_reward: {gt['gt_reward']:.4f}",
                f"segment_return: {gt['segment_return']:.4f} | prefix_return: {gt['prefix_return']:.4f} | success: {int(success)}",
                f"task: {task_description}",
                f"raw_output: {raw_output}",
            ]

            video_rel_path = Path(
                f"videos/rank_{rank}_env_{env_idx}_episode_{episode_id}/sample_{sample_idx:04d}_end_{end_step}.mp4"
            )
            render_debug_video(
                history_window_frames=history_window_frames,
                footer_lines=footer_lines,
                output_path=output_dir / video_rel_path,
                fps=fps,
            )

            rows.append(
                {
                    "episode_path": episode_path,
                    "rank": rank,
                    "env_idx": env_idx,
                    "episode_id": episode_id,
                    "success": success,
                    "sample_idx": sample_idx,
                    "segment_start": segment_start,
                    "end_step": end_step,
                    "history_window_len": len(history_window_frames),
                    "task_description": task_description,
                    "video_path": str(video_rel_path),
                    **gt,
                    **pred,
                }
            )

    rows_path = output_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    parse_failures = [row for row in rows if not row["parse_ok"]]
    with (output_dir / "top_parse_failures.jsonl").open("w", encoding="utf-8") as f:
        for row in parse_failures[:100]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pred_rewards = [row["pred_reward"] for row in rows]
    gt_rewards = [row["gt_reward"] for row in rows]
    segment_returns = [row["segment_return"] for row in rows]
    prefix_returns = [row["prefix_return"] for row in rows]
    success_labels = [row["episode_success"] for row in rows]

    summary = {
        "config": args.config,
        "num_episodes": len(episode_paths),
        "num_rows": len(rows),
        "num_videos": len(rows),
        "parse_ok_rate": None
        if not rows
        else sum(row["parse_ok"] for row in rows) / len(rows),
        "pred_reward_mean": None
        if not rows
        else sum(pred_rewards) / len(pred_rewards),
        "gt_reward_mean": None if not rows else sum(gt_rewards) / len(gt_rewards),
        "corr_pred_vs_gt_reward": pearson(pred_rewards, gt_rewards),
        "corr_pred_vs_segment_return": pearson(pred_rewards, segment_returns),
        "corr_pred_vs_prefix_return": pearson(pred_rewards, prefix_returns),
        "corr_pred_vs_episode_success": pearson(pred_rewards, success_labels),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"videos saved to: {output_dir / 'videos'}")


if __name__ == "__main__":
    main()
