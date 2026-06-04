#!/usr/bin/env python3
"""Prepare FoldTowel episodes as GPT-readable multi-view videos.

This mirrors prepare_robochallenge_combined_videos.py's output contract:
for each selected episode, write views_hstack.mp4 and metadata.json, then write
an episodes.jsonl manifest and prepare_summary.json. The only FoldTowel-specific
part is discovering local episode folders that already contain face/left/right
MP4 files instead of decoding RoboChallenge RRD streams.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PREFERRED_VIEW_ORDER = ("face", "left", "right")
VIEW_FILES = {
    "face": "faceImg.mp4",
    "left": "leftImg.mp4",
    "right": "rightImg.mp4",
}


@dataclass(frozen=True)
class EpisodeSource:
    """One FoldTowel episode plus the metadata needed downstream."""

    index: int
    episode_dir: Path
    episode_id: str
    task: str
    score: float | None
    source: dict[str, Any]


def load_json(path: Path, default: Any) -> Any:
    """Read JSON if it exists, otherwise return ``default``."""

    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    """Write stable UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def safe_name(value: Any, *, max_len: int = 120) -> str:
    """Return a filesystem-safe identifier."""

    text = str(value)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return (safe or "unknown")[:max_len]


def draw_view_label(frame_rgb: np.ndarray, label: str) -> np.ndarray:
    """Draw a compact view name in the top-left corner."""

    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, 120, 28), fill=(0, 0, 0))
    draw.text((8, 5), label, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def resize_rgb_to_height(frame_rgb: np.ndarray, height: int) -> np.ndarray:
    """Resize an RGB frame to ``height`` while preserving aspect ratio."""

    source_height, source_width = frame_rgb.shape[:2]
    if source_height == height:
        return frame_rgb
    width = max(2, int(round(source_width * (height / source_height))))
    width += width % 2
    image = Image.fromarray(frame_rgb)
    return np.asarray(image.resize((width, height), Image.Resampling.LANCZOS))


def classify_score(episode_dir: Path) -> float | None:
    """Best-effort FoldTowel final-score extraction from the folder layout."""

    lower_path = str(episode_dir).lower()
    lower_name = episode_dir.name.lower()
    if "tele" in lower_path or "dagger" in lower_path:
        return 10.0
    if "success" in lower_name:
        return 10.0
    if "fail" in lower_name:
        return 0.0
    return None


def discover_sources(data_dir: Path) -> list[EpisodeSource]:
    """Discover FoldTowel episode directories containing face/left/right videos."""

    sources = []
    for episode_dir in sorted(path for path in data_dir.rglob("*") if path.is_dir()):
        if not all((episode_dir / filename).is_file() for filename in VIEW_FILES.values()):
            continue
        json_paths = sorted(
            path
            for path in episode_dir.glob("*.json")
            if not path.name.endswith(".bak.json")
        )
        if not json_paths:
            continue
        sources.append(
            EpisodeSource(
                index=len(sources),
                episode_dir=episode_dir,
                episode_id=episode_dir.name,
                task="fold_towel",
                score=classify_score(episode_dir),
                source={
                    "data_dir": str(data_dir),
                    "episode_dir": str(episode_dir),
                    "json_path": str(json_paths[0]),
                    "layout": "foldtowel_mp4_triplet",
                },
            )
        )
    return sources


def read_video_frames(
    video_path: Path,
    *,
    fps: float,
    height: int,
    max_frames: int | None,
) -> tuple[list[np.ndarray], int, float]:
    """Read and sample one FoldTowel MP4 view."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps)
    source_frames_seen = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(source_fps / fps))) if fps > 0 else 1
    frames = []
    frame_index = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            break
        if frame_index % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(resize_rgb_to_height(frame_rgb, height))
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_index += 1
    capture.release()
    return frames, source_frames_seen, source_fps


def make_combined_frames(decoded: dict[str, list[np.ndarray]]) -> tuple[list[np.ndarray], dict[str, int]]:
    """Horizontally stack synchronized sampled frames."""

    ordered_views = [view for view in PREFERRED_VIEW_ORDER if decoded.get(view)]
    if not ordered_views:
        raise RuntimeError("No decoded frames")
    min_len = min(len(decoded[view]) for view in ordered_views)
    if min_len <= 0:
        raise RuntimeError("At least one view decoded zero frames")

    combined_frames = []
    for frame_index in range(min_len):
        parts = []
        for view_name in ordered_views:
            parts.append(draw_view_label(decoded[view_name][frame_index], view_name))
        combined_frames.append(np.concatenate(parts, axis=1))
    return combined_frames, {view: len(decoded[view]) for view in ordered_views}


def write_h264_mp4(frames_rgb: list[np.ndarray], output_path: Path, *, fps: float) -> None:
    """Write browser/VSCode-compatible H.264 baseline MP4."""

    if not frames_rgb:
        raise RuntimeError("No frames to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    width -= width % 2
    height -= height % 2
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_rgb in frames_rgb:
            cropped = np.ascontiguousarray(frame_rgb[:height, :width])
            process.stdin.write(cropped.tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def process_one(
    source: EpisodeSource,
    *,
    output_dir: Path,
    fps: float,
    height: int,
    max_frames_per_view: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Write ``views_hstack.mp4`` plus metadata for one FoldTowel episode."""

    episode_dir = output_dir / f"{source.index:06d}_{safe_name(source.episode_id)}"
    video_path = episode_dir / "views_hstack.mp4"
    metadata_path = episode_dir / "metadata.json"
    if not overwrite and video_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    decoded = {}
    source_frame_counts = {}
    source_fps_by_view = {}
    for view_name in PREFERRED_VIEW_ORDER:
        frames, source_frames_seen, source_fps = read_video_frames(
            source.episode_dir / VIEW_FILES[view_name],
            fps=fps,
            height=height,
            max_frames=max_frames_per_view,
        )
        decoded[view_name] = frames
        source_frame_counts[view_name] = source_frames_seen
        source_fps_by_view[view_name] = source_fps

    frames, frame_counts = make_combined_frames(decoded)
    write_h264_mp4(frames, video_path, fps=fps)
    ordered_views = [view for view in PREFERRED_VIEW_ORDER if view in decoded and decoded[view]]
    metadata = {
        "episode_group": episode_dir.name,
        "episode_id": source.episode_id,
        "task": source.task,
        "score": source.score,
        "path": str(video_path.resolve()),
        "views": ordered_views,
        "fps": fps,
        "height": height,
        "frame_counts": {**frame_counts, "combined": len(frames)},
        "duration_sec": len(frames) / fps if fps > 0 else None,
        "foldtowel_episode_dir": str(source.episode_dir.resolve()),
        "source_frame_counts": source_frame_counts,
        "source_fps_by_view": source_fps_by_view,
        "video_entities": {view: VIEW_FILES[view] for view in ordered_views},
        "source": source.source,
    }
    write_json(metadata_path, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--max-frames-per-view", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_sources(sources: list[EpisodeSource], args: argparse.Namespace) -> list[EpisodeSource]:
    """Apply index filters and re-index selected sources."""

    selected = sources[max(0, int(args.start_index)) :]
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]
    return [
        EpisodeSource(
            index=index,
            episode_dir=source.episode_dir,
            episode_id=source.episode_id,
            task=source.task,
            score=source.score,
            source=source.source,
        )
        for index, source in enumerate(selected)
    ]


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(data_dir)
    selected = select_sources(sources, args)
    print(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "output_dir": str(output_dir),
                "discovered": len(sources),
                "selected": len(selected),
                "fps": args.fps,
                "height": args.height,
                "workers": args.num_workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not selected:
        return

    rows = []
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.num_workers))) as pool:
        futures = {
            pool.submit(
                process_one,
                source,
                output_dir=output_dir,
                fps=float(args.fps),
                height=int(args.height),
                max_frames_per_view=args.max_frames_per_view,
                overwrite=bool(args.overwrite),
            ): source
            for source in selected
        }
        for done_count, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            try:
                row = future.result()
                rows.append(row)
                print(
                    f"[done] {done_count}/{len(selected)} {row['episode_group']} "
                    f"task={row.get('task')} score={row.get('score')} "
                    f"frames={row.get('frame_counts', {}).get('combined')}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "episode_dir": str(source.episode_dir),
                        "episode_id": source.episode_id,
                        "error": str(exc),
                    }
                )
                print(f"[fail] {done_count}/{len(selected)} {source.episode_dir}: {exc}", flush=True)

    rows.sort(key=lambda row: row.get("episode_group", ""))
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_dir / "prepare_summary.json", {"completed": len(rows), "failed": len(failures), "failures": failures})
    print(f"[write] {output_dir / 'episodes.jsonl'}", flush=True)
    if failures:
        print(f"[warn] {len(failures)} failures; see {output_dir / 'prepare_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
