#!/usr/bin/env python3
"""Prepare RoboChallenge RRD rollouts as GPT-readable multi-view videos.

This is the first pipeline step for raw RoboChallenge data downloaded in the
layout produced by ``toolkits/download_robochallenge_runs.py``:

    robochallenge/
      manifest.json
      run_000000_<run_id>/
        run.json
        rollouts.json
        rollout_00_<rollout_id>.rrd

For every selected rollout this script decodes the three camera streams from
the RRD file, horizontally stacks the views, writes ``views_hstack.mp4``, and
writes the ``metadata.json`` consumed by the GPT annotation scripts.
"""

# ruff: noqa: E402,I001

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

# Video decode/encode is already parallelized at the process level.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import av
import numpy as np
import rerun_bindings as rb
from PIL import Image, ImageDraw, ImageFont


VIDEO_ENTITY_GROUPS = [
    {
        "front": "/videos_front",
        "left": "/videos_left",
        "right": "/videos_right",
    },
    {
        "front": "/videos_1",
        "left": "/videos_2",
        "right": "/videos_3",
    },
]
PREFERRED_VIEW_ORDER = ("front", "left", "right")


@dataclass(frozen=True)
class EpisodeSource:
    """One RRD rollout plus the metadata needed downstream."""

    index: int
    rrd_path: Path
    episode_id: str
    task: str
    score: float | None
    source: dict[str, Any]


def load_json(path: Path, default: Any) -> Any:
    """Read JSON if it exists, otherwise return ``default``."""

    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    """Write stable UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def safe_name(value: Any, *, max_len: int = 120) -> str:
    """Return a filesystem-safe identifier."""

    text = str(value)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return (safe or "unknown")[:max_len]


def first_present(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """Return the first non-empty value from ``mapping``."""

    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return default


def timestamp_ns(value: Any) -> int:
    """Convert Rerun/Arrow timestamp values to integer nanoseconds."""

    if hasattr(value, "value"):
        return int(value.value)
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[ns]").astype(np.int64))
    if hasattr(value, "timestamp"):
        return int(value.timestamp() * 1_000_000_000)
    return int(value)


def sample_packets(sample: Any) -> list[bytes]:
    """Normalize a Rerun VideoStream sample into one or more H.264 packets."""

    if isinstance(sample, list) and sample and isinstance(sample[0], list):
        return [bytes(packet) for packet in sample]
    return [bytes(sample)]


def timestamp_index_column(recording: Any) -> Any:
    """Find the timestamp index column in a Rerun recording."""

    for col in recording.schema().index_columns():
        if "timeline:timestamp" in str(col):
            return col
    for col in recording.schema().index_columns():
        if "timestamp" in str(col):
            return col
    raise RuntimeError("No timestamp index column found in RRD")


def detect_video_entities(recording: Any) -> dict[str, str]:
    """Detect the available front/left/right video entities."""

    schema = recording.schema()
    for group in VIDEO_ENTITY_GROUPS:
        entities: dict[str, str] = {}
        for view_name, entity_path in group.items():
            try:
                schema.column_for_selector(f"{entity_path}:VideoStream:sample")
            except Exception:  # noqa: BLE001
                continue
            entities[view_name] = entity_path
        if entities:
            return entities
    return {}


def iter_video_samples(recording: Any, entity_path: str) -> Any:
    """Yield ``(timestamp, sample)`` rows for one video stream."""

    if not hasattr(recording, "view") and hasattr(recording, "chunks"):
        for chunk in recording.chunks():
            if getattr(chunk, "is_static", False):
                continue
            if str(chunk.entity_path) != entity_path:
                continue
            data = chunk.to_record_batch().to_pydict()
            timestamps = data.get("timestamp", [])
            samples = data.get("VideoStream:sample", [])
            for sample_index, sample in enumerate(samples):
                ts = timestamps[sample_index] if sample_index < len(timestamps) else None
                yield ts, sample
        return

    schema = recording.schema()
    index_col = timestamp_index_column(recording)
    sample_col = schema.column_for_selector(f"{entity_path}:VideoStream:sample")
    reader = (
        recording.view(index="timestamp", contents=entity_path)
        .filter_is_not_null(sample_col)
        .select(index_col, sample_col)
    )
    sample_name = f"{entity_path}:VideoStream:sample"
    while True:
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            break
        data = batch.to_pydict()
        timestamps = data.get("timestamp", [])
        samples = data.get(sample_name, [])
        for sample_index, sample in enumerate(samples):
            ts = timestamps[sample_index] if sample_index < len(timestamps) else None
            yield ts, sample


def resize_rgb_to_height(frame_rgb: np.ndarray, height: int) -> np.ndarray:
    """Resize an RGB frame to ``height`` while preserving aspect ratio."""

    src_h, src_w = frame_rgb.shape[:2]
    if src_h == height:
        return frame_rgb
    width = max(2, int(round(src_w * (height / src_h))))
    width += width % 2
    image = Image.fromarray(frame_rgb)
    return np.asarray(image.resize((width, height), Image.Resampling.LANCZOS))


def decode_view_pyav(
    recording: Any,
    entity_path: str,
    *,
    fps: float,
    height: int,
    max_frames: int | None,
) -> dict[str, Any]:
    """Decode and sample one view from a Rerun H.264 stream."""

    codec = av.CodecContext.create("h264", "r")
    codec.thread_count = 1
    interval_ns = int(1_000_000_000 / fps) if fps > 0 else None
    next_keep_ns: int | None = None
    frames: list[np.ndarray] = []
    timestamps: list[int | None] = []
    source_indices: list[int] = []
    source_seen = 0

    for sample_ts_raw, sample in iter_video_samples(recording, entity_path):
        if max_frames is not None and len(frames) >= max_frames:
            break
        source_index = source_seen
        source_seen += 1
        sample_ts = timestamp_ns(sample_ts_raw) if sample_ts_raw is not None else None
        try:
            for packet in sample_packets(sample):
                for frame in codec.decode(av.packet.Packet(packet)):
                    keep = True
                    if interval_ns is not None and sample_ts is not None:
                        keep = next_keep_ns is None or sample_ts >= next_keep_ns
                    if not keep:
                        continue
                    frame_rgb = frame.to_ndarray(format="rgb24")
                    frames.append(resize_rgb_to_height(frame_rgb, height))
                    timestamps.append(sample_ts)
                    source_indices.append(source_index)
                    if interval_ns is not None and sample_ts is not None:
                        next_keep_ns = sample_ts + interval_ns
                    if max_frames is not None and len(frames) >= max_frames:
                        break
                if max_frames is not None and len(frames) >= max_frames:
                    break
        except av.error.InvalidDataError:
            continue

    return {
        "frames": frames,
        "timestamps_ns": timestamps,
        "source_sample_indices": source_indices,
        "source_samples_seen": source_seen,
    }


def decode_rrd_views(
    rrd_path: Path,
    *,
    fps: float,
    height: int,
    max_frames_per_view: int | None,
) -> dict[str, dict[str, Any]]:
    """Decode all available views from one RRD file."""

    recording = rb.load_recording(str(rrd_path))
    entities = detect_video_entities(recording)
    if not entities:
        raise RuntimeError(f"No supported video entities found in {rrd_path}")

    decoded: dict[str, dict[str, Any]] = {}
    for view_name in PREFERRED_VIEW_ORDER:
        entity_path = entities.get(view_name)
        if entity_path is None:
            continue
        decoded[view_name] = decode_view_pyav(
            recording,
            entity_path,
            fps=fps,
            height=height,
            max_frames=max_frames_per_view,
        )
        decoded[view_name]["entity_path"] = entity_path
    return decoded


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


def make_combined_frames(decoded: dict[str, dict[str, Any]]) -> tuple[list[np.ndarray], dict[str, int]]:
    """Horizontally stack synchronized sampled frames."""

    ordered_views = [v for v in PREFERRED_VIEW_ORDER if v in decoded and decoded[v]["frames"]]
    if not ordered_views:
        raise RuntimeError("No decoded frames")
    min_len = min(len(decoded[v]["frames"]) for v in ordered_views)
    if min_len <= 0:
        raise RuntimeError("At least one view decoded zero frames")

    combined_frames: list[np.ndarray] = []
    for frame_index in range(min_len):
        parts = []
        for view_name in ordered_views:
            frame = decoded[view_name]["frames"][frame_index]
            parts.append(draw_view_label(frame, view_name))
        combined_frames.append(np.concatenate(parts, axis=1))

    return combined_frames, {v: len(decoded[v]["frames"]) for v in ordered_views}


def write_h264_mp4(frames_rgb: list[np.ndarray], output_path: Path, *, fps: float) -> None:
    """Write browser/VSCode-compatible H.264 baseline MP4."""

    if not frames_rgb:
        raise RuntimeError("No frames to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    width -= width % 2
    height -= height % 2
    cmd = [
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
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame_rgb in frames_rgb:
            cropped = np.ascontiguousarray(frame_rgb[:height, :width])
            proc.stdin.write(cropped.tobytes())
    finally:
        proc.stdin.close()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def normalize_task_name(run_info: dict[str, Any], rollout: dict[str, Any], fallback: str) -> str:
    """Best-effort task name extraction across RoboChallenge metadata variants."""

    value = first_present(
        rollout,
        ("task", "task_name", "task_id", "scenario", "task_tag"),
        default=None,
    )
    if value is None:
        value = first_present(
            run_info,
            ("task", "task_name", "task_id", "scenario", "task_tag", "benchmark_task"),
            default=fallback,
        )
    return str(value)


def rollout_score(run_info: dict[str, Any], rollout: dict[str, Any]) -> float | None:
    """Best-effort per-rollout 0-10 score extraction."""

    value = first_present(rollout, ("score", "progress_score", "rollout_score"), default=None)
    if value is None:
        value = first_present(run_info, ("rollout_score",), default=None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sources_from_manifest(data_dir: Path) -> list[EpisodeSource]:
    """Load rollout sources from ``manifest.json``."""

    manifest_path = data_dir / "manifest.json"
    manifest = load_json(manifest_path, default=None)
    if not isinstance(manifest, list):
        return []
    runs_list = load_json(data_dir / "runs_list.json", default=[])
    runs_by_id = {
        str(run.get("run_id")): run
        for run in runs_list
        if isinstance(run, dict) and run.get("run_id") is not None
    }

    sources: list[EpisodeSource] = []
    for run_record in manifest:
        if not isinstance(run_record, dict):
            continue
        run_dir = data_dir / str(run_record.get("run_dir", ""))
        packed_run_dir = data_dir / "rrd_run_dirs" / str(run_record.get("run_dir", ""))
        actual_run_dir = packed_run_dir if packed_run_dir.is_dir() else run_dir
        run_info = load_json(actual_run_dir / "run.json", default={})
        if not run_info:
            run_info = runs_by_id.get(str(run_record.get("run_id")), {})
        rollouts = run_record.get("rollouts") or []
        for rollout in rollouts:
            if not isinstance(rollout, dict):
                continue
            file_value = rollout.get("file") or rollout.get("path") or rollout.get("rrd_path")
            if not file_value:
                continue
            rrd_path = Path(file_value)
            if not rrd_path.is_absolute():
                rrd_path = actual_run_dir / rrd_path
            rollout_index = int(rollout.get("rollout_index", len(sources)))
            rollout_id = str(rollout.get("rollout_id") or rrd_path.stem)
            task = normalize_task_name(run_info, rollout, fallback=str(run_record.get("run_id", "unknown_task")))
            score = rollout_score(run_info, rollout)
            sources.append(
                EpisodeSource(
                    index=len(sources),
                    rrd_path=rrd_path,
                    episode_id=rollout_id,
                    task=task,
                    score=score,
                    source={
                        "data_dir": str(data_dir),
                        "manifest_path": str(manifest_path),
                        "run_dir": str(actual_run_dir),
                        "run_id": run_record.get("run_id"),
                        "source_index": run_record.get("source_index"),
                        "local_index": run_record.get("local_index"),
                        "rollout_index": rollout_index,
                        "rollout": rollout,
                        "run": run_info,
                    },
                )
            )
    return sources


def sources_from_flat_rrds(data_dir: Path) -> list[EpisodeSource]:
    """Fallback for directories containing raw ``*.rrd`` files without manifest."""

    sources = []
    for rrd_path in sorted(data_dir.rglob("*.rrd")):
        sources.append(
            EpisodeSource(
                index=len(sources),
                rrd_path=rrd_path,
                episode_id=rrd_path.stem,
                task=rrd_path.parent.name,
                score=None,
                source={"data_dir": str(data_dir), "rrd_path": str(rrd_path), "layout": "flat_rrd_fallback"},
            )
        )
    return sources


def discover_sources(data_dir: Path) -> list[EpisodeSource]:
    """Discover RRD rollouts in a RoboChallenge data directory."""

    sources = sources_from_manifest(data_dir)
    if sources:
        return sources
    return sources_from_flat_rrds(data_dir)


def process_one(
    source: EpisodeSource,
    *,
    output_dir: Path,
    fps: float,
    height: int,
    max_frames_per_view: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Decode one RRD and write ``views_hstack.mp4`` plus metadata."""

    episode_dir = output_dir / f"{source.index:06d}_{safe_name(source.episode_id)}"
    video_path = episode_dir / "views_hstack.mp4"
    metadata_path = episode_dir / "metadata.json"
    if not overwrite and video_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if not source.rrd_path.exists():
        raise FileNotFoundError(f"RRD file not found: {source.rrd_path}")

    decoded = decode_rrd_views(
        source.rrd_path,
        fps=fps,
        height=height,
        max_frames_per_view=max_frames_per_view,
    )
    frames, frame_counts = make_combined_frames(decoded)
    write_h264_mp4(frames, video_path, fps=fps)

    ordered_views = [v for v in PREFERRED_VIEW_ORDER if v in decoded and decoded[v]["frames"]]
    timestamps_by_view = {
        view: decoded[view]["timestamps_ns"][: len(frames)]
        for view in ordered_views
    }
    metadata = {
        "episode_group": episode_dir.name,
        "episode_id": source.episode_id,
        "task": source.task,
        "score": source.score if source.score is not None else 0.0,
        "path": str(video_path.resolve()),
        "views": ordered_views,
        "fps": fps,
        "height": height,
        "frame_counts": {**frame_counts, "combined": len(frames)},
        "duration_sec": len(frames) / fps if fps > 0 else None,
        "rrd_path": str(source.rrd_path.resolve()),
        "timestamps_ns_by_view": timestamps_by_view,
        "video_entities": {view: decoded[view].get("entity_path") for view in ordered_views},
        "source": source.source,
    }
    write_json(metadata_path, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Raw RoboChallenge directory containing manifest.json/run_*/rollout_*.rrd.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write per-episode combined videos and metadata.")
    parser.add_argument("--task", default=None, help="Optional exact task filter from run/rollout metadata.")
    parser.add_argument("--task-regex", default=None, help="Optional regex task filter.")
    parser.add_argument("--fps", type=float, default=5.0, help="Target sampled FPS for GPT videos.")
    parser.add_argument("--height", type=int, default=360, help="Per-view output height before horizontal stacking.")
    parser.add_argument("--max-frames-per-view", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_sources(sources: list[EpisodeSource], args: argparse.Namespace) -> list[EpisodeSource]:
    """Apply task/index filters and re-index selected sources."""

    selected = sources
    if args.task:
        selected = [s for s in selected if s.task == args.task]
    if args.task_regex:
        pattern = re.compile(args.task_regex)
        selected = [s for s in selected if pattern.search(s.task)]
    selected = selected[max(0, int(args.start_index)) :]
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]
    return [
        EpisodeSource(
            index=i,
            rrd_path=s.rrd_path,
            episode_id=s.episode_id,
            task=s.task,
            score=s.score,
            source=s.source,
        )
        for i, s in enumerate(selected)
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

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
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
                    f"task={row.get('task')} score={row.get('score')} frames={row.get('frame_counts', {}).get('combined')}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"rrd_path": str(source.rrd_path), "episode_id": source.episode_id, "error": str(exc)})
                print(f"[fail] {done_count}/{len(selected)} {source.rrd_path}: {exc}", flush=True)

    rows.sort(key=lambda r: r.get("episode_group", ""))
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_dir / "prepare_summary.json", {"completed": len(rows), "failed": len(failures), "failures": failures})
    print(f"[write] {output_dir / 'episodes.jsonl'}", flush=True)
    if failures:
        print(f"[warn] {len(failures)} failures; see {output_dir / 'prepare_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
