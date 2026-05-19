#!/usr/bin/env python3
"""Parallel GPT-5.5 annotation from pre-combined multi-view MP4 files."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CLIENT_LOCK = threading.Lock()
_CLIENT = None

DEFAULT_SYSTEM_PROMPT = """You are a strict RoboChallenge video grader.
Use only visible evidence from the attached time-labeled multi-view contact sheets.
Return valid JSON only. If evidence is unclear, mark it uncertain instead of guessing.
"""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "episode_id": {"type": "string"},
        "known_rollout_score": {"type": "number"},
        "visual_estimated_score": {"type": "number"},
        "video_quality_label": {"type": "string"},
        "stage_results": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
        "retry_events": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
        "notes": {"type": "string"},
    },
    "required": [
        "episode_id",
        "known_rollout_score",
        "visual_estimated_score",
        "video_quality_label",
        "stage_results",
        "retry_events",
        "notes",
    ],
}


def load_rubric(path: str | None) -> dict[str, Any]:
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "task_name": "put_cup_on_coaster",
        "prompt": "place the cup on the coaster",
        "max_score": 10.0,
        "scoring": [
            {"stage_id": "s1_pick_up_cup", "stage": "Pick up the cup", "points": 4.0, "critical": True},
            {"stage_id": "s2_move_to_destination", "stage": "Move the cup to the destination", "points": 2.0, "critical": True},
            {"stage_id": "s3_place_on_coaster", "stage": "Place the cup correctly on the coaster", "points": 3.0, "critical": True},
            {"stage_id": "s4_reset_arm", "stage": "Reset the robotic arm", "points": 1.0, "critical": False},
        ],
        "retry_penalty": 0.5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined-dir",
        default="logs/episode2frame/put_cup_on_coaster_combined_videos",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="put_cup_on_coaster")
    parser.add_argument("--task-instruction", default=None)
    parser.add_argument("--rubric-json", default=None)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--episode-id", action="append", default=None)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--contact-layout", choices=["vertical", "grid"], default="vertical")
    parser.add_argument("--frames-per-image", type=int, default=64)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--tile-width", type=int, default=240)
    parser.add_argument("--tile-label-font-size", type=int, default=13)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--api-workers", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    return parser.parse_args()


def get_client(api_key_env: str, base_url: str | None):
    global _CLIENT  # noqa: PLW0603
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {api_key_env}")
        kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _CLIENT = OpenAI(**kwargs)
        return _CLIENT


def extract_json_from_response(response: Any) -> dict[str, Any]:
    text = getattr(response, "output_text", None)
    if not text:
        data = response.model_dump() if hasattr(response, "model_dump") else response
        text = json.dumps(data)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_json_from_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def request_payload(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_urls: list[str],
) -> dict[str, Any]:
    user_content = [{"type": "input_text", "text": user_prompt}]
    user_content.extend(
        {"type": "input_image", "image_url": image_data_url}
        for image_data_url in image_data_urls
    )
    return {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "robochallenge_rollout_annotation",
                "schema": OUTPUT_SCHEMA,
                "strict": True,
            }
        },
    }


def redacted_payload_for_disk(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(payload))
    for item in redacted.get("input", []):
        for part in item.get("content", []):
            if part.get("type") == "input_image" and "image_url" in part:
                part["image_url"] = "<base64 image data omitted>"
            if part.get("type") == "input_file" and "file_data" in part:
                part["file_data"] = "<base64 file data omitted>"
    return redacted


def chat_messages_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in payload["input"]:
        content = []
        for part in item["content"]:
            if part["type"] == "input_text":
                content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "input_image":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": part["image_url"]},
                    }
                )
            elif part["type"] == "input_file":
                content.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": part.get("filename", "input_file"),
                            "file_data": part.get("file_data"),
                        },
                    }
                )
        messages.append({"role": item["role"], "content": content})
    return messages


def call_model(client: Any, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        response = client.responses.create(**payload)
        raw = response.model_dump()
        return extract_json_from_response(response), raw
    except Exception as responses_error:  # noqa: BLE001
        # Some OpenAI-compatible gateways expose chat/completions but not the
        # newer Responses API. Keep the same visual prompt and request JSON.
        response = client.chat.completions.create(
            model=payload["model"],
            messages=chat_messages_from_payload(payload),
            response_format={"type": "json_object"},
        )
        raw = response.model_dump()
        raw["_responses_api_error"] = str(responses_error)
        text = response.choices[0].message.content or "{}"
        return extract_json_from_text(text), raw


def image_to_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def uniform_indices(length: int, max_frames: int) -> list[int]:
    if length <= 0:
        return []
    if length <= max_frames:
        return list(range(length))
    return np.linspace(0, length - 1, num=max_frames).round().astype(int).tolist()


def draw_label(image: Image.Image, text: str, font_size: int = 20) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((0, 0, bbox[2] + 12, bbox[3] + 10), fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255), font=font)


def decode_sampled_frames(video_path: Path, max_frames: int) -> tuple[list[Image.Image], float, int]:
    frames: list[Image.Image] = []
    with av.open(str(video_path), mode="r") as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 5.0
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    frame_count = len(decoded)
    for index in uniform_indices(frame_count, max_frames):
        frames.append(Image.fromarray(decoded[index]).convert("RGB"))
    return frames, fps, frame_count


def build_contact_sheet(
    video_path: Path,
    *,
    max_frames: int,
    image_width: int,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    frames, fps, frame_count = decode_sampled_frames(video_path, max_frames)
    indices = uniform_indices(frame_count, max_frames)
    if len(indices) != len(frames):
        indices = indices[: len(frames)]
    rows = []
    metadata = []
    target_w = int(image_width)
    for image, frame_index in zip(frames, indices, strict=False):
        time_sec = frame_index / max(fps, 1e-6)
        image = image.copy()
        image.thumbnail((target_w, target_w), Image.Resampling.LANCZOS)
        row = Image.new("RGB", (target_w, image.height), (10, 10, 10))
        row.paste(image, ((target_w - image.width) // 2, 0))
        draw_label(row, f"combined_3view | f={frame_index} | t={time_sec:.1f}s")
        rows.append(row)
        metadata.append({"frame_index": int(frame_index), "time_sec": float(time_sec)})
    sheet = Image.new("RGB", (target_w, sum(row.height for row in rows)), (10, 10, 10))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet, metadata


def build_grid_contact_sheets(
    video_path: Path,
    *,
    max_frames: int,
    frames_per_image: int,
    grid_cols: int,
    tile_width: int,
    label_font_size: int,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    frames, fps, frame_count = decode_sampled_frames(video_path, max_frames)
    indices = uniform_indices(frame_count, max_frames)
    if len(indices) != len(frames):
        indices = indices[: len(frames)]

    metadata = []
    tiles: list[Image.Image] = []
    for sample_id, (image, frame_index) in enumerate(zip(frames, indices, strict=False)):
        time_sec = frame_index / max(fps, 1e-6)
        image = image.copy()
        image.thumbnail((int(tile_width), int(tile_width)), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (int(tile_width), image.height), (10, 10, 10))
        tile.paste(image, ((int(tile_width) - image.width) // 2, 0))
        draw_label(
            tile,
            f"#{sample_id:03d} f={frame_index} t={time_sec:.1f}s",
            font_size=int(label_font_size),
        )
        tiles.append(tile)
        metadata.append(
            {
                "sample_id": int(sample_id),
                "frame_index": int(frame_index),
                "time_sec": float(time_sec),
            }
        )

    sheets = []
    frames_per_image = max(1, int(frames_per_image))
    grid_cols = max(1, int(grid_cols))
    for page_id in range(0, len(tiles), frames_per_image):
        page_tiles = tiles[page_id : page_id + frames_per_image]
        tile_h = max(tile.height for tile in page_tiles) if page_tiles else 1
        rows = int(np.ceil(len(page_tiles) / grid_cols))
        sheet = Image.new(
            "RGB",
            (int(tile_width) * grid_cols, tile_h * rows),
            (10, 10, 10),
        )
        for local_i, tile in enumerate(page_tiles):
            x = (local_i % grid_cols) * int(tile_width)
            y = (local_i // grid_cols) * tile_h
            sheet.paste(tile, (x, y))
        sheets.append(sheet)
    return sheets, metadata


def load_records(combined_dir: Path) -> list[dict[str, Any]]:
    records = []
    for metadata_path in sorted(combined_dir.glob("[0-9]*_*/metadata.json")):
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        path = Path(record["path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            continue
        record["path"] = str(path)
        record["metadata_path"] = str(metadata_path)
        records.append(record)
    return records


def build_user_prompt(
    *,
    record: dict[str, Any],
    task: str,
    task_instruction: str,
    rubric: dict[str, Any],
    frame_metadata: list[dict[str, Any]],
) -> str:
    payload = {
        "task_name": task,
        "task_instruction": task_instruction,
        "episode_id": record.get("episode_id"),
        "known_rollout_score": float(record.get("score", 0.0)),
        "max_score": 10.0,
        "video_metadata": {
            "views": record.get("views", []),
            "layout": (
                "Attached images are time-labeled contact sheets. Each tile is "
                "a horizontally stacked 3-view frame. Use the visible sample id "
                "#NNN and t=...s labels to localize stages."
            ),
            "frame_counts": record.get("frame_counts"),
            "sampled_frames": frame_metadata,
            "sampling_note": "All temporal boundaries must be returned in seconds using the t=... labels. Do not return frame indices unless needed.",
        },
        "rollout_metadata": record.get("source", {}),
        "robochallenge_scoring_rubric": rubric,
    }
    return (
        "Evaluate this rollout from the attached time-labeled multi-view contact sheet.\n"
        "Return valid JSON matching the requested schema.\n"
        "Use approximate seconds from the labels for completion_time_sec and time_range_sec.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def annotate_one(
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
    rubric: dict[str, Any],
    task_instruction: str,
    output_dir: Path,
    media_dir: Path,
    request_dir: Path,
    response_dir: Path,
    base_url: str | None,
) -> dict[str, Any]:
    episode_id = str(record["episode_id"])
    out_json = response_dir / f"{episode_id}.json"
    if args.resume and out_json.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))

    image_paths = []
    request_json = request_dir / f"{episode_id}.request.json"
    if args.contact_layout == "grid":
        sheets, frame_metadata = build_grid_contact_sheets(
            Path(record["path"]),
            max_frames=args.max_frames,
            frames_per_image=args.frames_per_image,
            grid_cols=args.grid_cols,
            tile_width=args.tile_width,
            label_font_size=args.tile_label_font_size,
        )
        for page_id, sheet in enumerate(sheets):
            image_path = media_dir / f"{episode_id}.page{page_id:02d}.jpg"
            sheet.save(image_path, quality=int(args.jpeg_quality))
            image_paths.append(image_path)
    else:
        sheet, frame_metadata = build_contact_sheet(
            Path(record["path"]),
            max_frames=args.max_frames,
            image_width=args.image_width,
        )
        image_path = media_dir / f"{episode_id}.jpg"
        sheet.save(image_path, quality=int(args.jpeg_quality))
        image_paths.append(image_path)
    user_prompt = build_user_prompt(
        record=record,
        task=args.task,
        task_instruction=task_instruction,
        rubric=rubric,
        frame_metadata=frame_metadata,
    )
    payload = request_payload(
        model=args.model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        image_data_urls=[image_to_data_url(image_path) for image_path in image_paths],
    )
    request_json.write_text(
        json.dumps(
            {
                **redacted_payload_for_disk(payload),
                "episode_id": episode_id,
                "local_media_paths": [str(image_path) for image_path in image_paths],
                "combined_video_path": record["path"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        result = {
            "episode_id": episode_id,
            "task_name": args.task,
            "known_rollout_score": float(record.get("score", 0.0)),
            "dry_run": True,
            "request_path": str(request_json),
            "media_paths": [str(image_path) for image_path in image_paths],
            "combined_video_path": record["path"],
        }
    else:
        client = get_client(args.api_key_env, base_url)
        parsed, raw_response = call_model(client, payload)
        raw_path = response_dir / f"{episode_id}.raw.json"
        raw_path.write_text(
            json.dumps(raw_response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result = {
            **parsed,
            "episode_id": episode_id,
            "known_rollout_score": float(record.get("score", 0.0)),
            "request_path": str(request_json),
            "media_paths": [str(image_path) for image_path in image_paths],
            "combined_video_path": record["path"],
            "raw_response_path": str(raw_path),
        }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.sleep_sec > 0:
        time.sleep(float(args.sleep_sec))
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    media_dir = output_dir / "media"
    request_dir = output_dir / "requests"
    response_dir = output_dir / "responses"
    for directory in (media_dir, request_dir, response_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url or os.environ.get(args.base_url_env) or None
    rubric = load_rubric(args.rubric_json)
    task_instruction = (
        args.task_instruction
        or rubric.get("prompt")
        or f"Complete the RoboChallenge task: {args.task}."
    )
    records = load_records(Path(args.combined_dir))
    records = [r for r in records if str(r.get("task")) == args.task]
    if args.episode_id:
        wanted = set(args.episode_id)
        records = [r for r in records if str(r.get("episode_id")) in wanted]
    records = [
        r
        for i, r in enumerate(records)
        if i >= args.start_index and i % int(args.num_shards) == int(args.shard_id)
    ]
    if args.limit_episodes is not None:
        records = records[: args.limit_episodes]

    summary_path = output_dir / "annotations.jsonl"
    if not args.resume and summary_path.exists():
        summary_path.unlink()
    write_lock = threading.Lock()
    completed: set[str] = set()
    if args.resume and summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(str(json.loads(line).get("episode_id")))
    records = [r for r in records if str(r.get("episode_id")) not in completed]

    print(
        json.dumps(
            {
                "records_to_run": len(records),
                "api_workers": int(args.api_workers),
                "dry_run": bool(args.dry_run),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    def worker(record: dict[str, Any]) -> dict[str, Any]:
        try:
            result = annotate_one(
                record,
                args=args,
                rubric=rubric,
                task_instruction=task_instruction,
                output_dir=output_dir,
                media_dir=media_dir,
                request_dir=request_dir,
                response_dir=response_dir,
                base_url=base_url,
            )
        except Exception as exc:  # noqa: BLE001
            episode_id = str(record.get("episode_id"))
            error_path = response_dir / f"{episode_id}.error.json"
            result = {
                "episode_id": episode_id,
                "known_rollout_score": float(record.get("score", 0.0)),
                "error": type(exc).__name__,
                "error_message": str(exc),
                "combined_video_path": record.get("path"),
                "error_path": str(error_path),
            }
            error_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        with write_lock:
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            done = sum(1 for _ in summary_path.open(encoding="utf-8"))
            label = result.get("video_quality_label", result.get("dry_run"))
            status = "error" if "error" in result else "done"
            print(f"[{status}] {done} {result.get('episode_id')} label={label}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=max(1, int(args.api_workers))) as pool:
        futures = [pool.submit(worker, record) for record in records]
        for future in as_completed(futures):
            future.result()

    print(f"[write] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
