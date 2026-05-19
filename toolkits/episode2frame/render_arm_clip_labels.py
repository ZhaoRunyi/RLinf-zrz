#!/usr/bin/env python3
"""Render ARM-style clip/transition labels onto combined multi-view videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LABEL_COLORS = {
    "progressing": (70, 220, 120),
    "stagnant": (105, 112, 124),
    "regressing": (50, 70, 235),
}
ADV_COLORS = {"positive": (70, 220, 120), "unclear": (75, 145, 245), "negative": (55, 80, 220)}
STAGE_COLORS = {
    "s1_pick_up_cup": (69, 188, 255),
    "s2_move_to_destination": (90, 220, 120),
    "s3_place_on_coaster": (255, 176, 80),
    "s4_reset_arm": (210, 130, 255),
    "none": (75, 80, 92),
}
STAGE_NAMES = {
    "s1_pick_up_cup": "Pick up cup",
    "s2_move_to_destination": "Move to coaster",
    "s3_place_on_coaster": "Place/release",
    "s4_reset_arm": "Reset arm",
    "none": "No active scoring stage",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dense-jsonl", required=True)
    p.add_argument("--clip-jsonl", required=True)
    p.add_argument("--score-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episode-group", action="append", default=None)
    p.add_argument("--select-defaults", action="store_true", help="Render high/mid/low/retry examples.")
    p.add_argument("--target-fps", type=float, default=12.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.5,
    color: tuple[int, int, int] = (245, 248, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def load_jsonl_by_group(path: Path) -> dict[str, Any]:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        group = str(row.get("episode_group"))
        data[group] = row
    return data


def load_clips(path: Path, groups: set[str]) -> dict[str, list[dict[str, Any]]]:
    clips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        group = str(row.get("episode_group"))
        if group in groups:
            clips[group].append(row)
    for group in clips:
        clips[group].sort(key=lambda r: float(r.get("clip_start_sec", 0.0)))
    return clips


def choose_defaults(score_rows: dict[str, Any]) -> list[tuple[str, str]]:
    rows = list(score_rows.values())
    picks: list[tuple[str, str]] = []
    high = sorted([r for r in rows if float(r.get("known_rollout_score", 0.0)) == 10.0], key=lambda r: r["episode_group"])
    mid = sorted(
        [r for r in rows if 4.0 <= float(r.get("known_rollout_score", 0.0)) <= 6.0],
        key=lambda r: (abs(float(r.get("known_rollout_score", 0.0)) - 5.0), r["episode_group"]),
    )
    low = sorted([r for r in rows if float(r.get("known_rollout_score", 0.0)) == 0.0], key=lambda r: r["episode_group"])
    retry = sorted([r for r in rows if int(r.get("total_retry_count") or 0) > 0], key=lambda r: r["episode_group"])
    for name, cand in [("high", high), ("mid", mid), ("low", low), ("retry", retry)]:
        if cand:
            picks.append((name, str(cand[0]["episode_group"])))
    return picks


def interp_curve(times: list[float], values: list[float], t: float) -> float:
    if not times:
        return 0.0
    if t <= times[0]:
        return float(values[0])
    if t >= times[-1]:
        return float(values[-1])
    return float(np.interp(t, np.asarray(times, dtype=np.float32), np.asarray(values, dtype=np.float32)))


def interval_at(dense: dict[str, Any], t: float) -> tuple[str, float]:
    times = dense.get("times_sec") or []
    stage_ids = dense.get("stage_ids") or []
    if not times or not stage_ids:
        return "none", 0.0
    idx = int(np.searchsorted(np.asarray(times), t, side="right") - 1)
    idx = max(0, min(idx, len(stage_ids) - 1))
    score = interp_curve(times, dense.get("score_curve") or [], t)
    return str(stage_ids[idx]), score


def compress_stage_segments(dense: dict[str, Any]) -> list[tuple[float, float, str]]:
    times = [float(x) for x in dense.get("times_sec") or []]
    stage_ids = [str(x) for x in dense.get("stage_ids") or []]
    if len(times) < 2:
        return []
    segments = []
    start = times[0]
    cur = stage_ids[0]
    for i in range(1, len(times)):
        if stage_ids[i] != cur:
            segments.append((start, times[i], cur))
            start = times[i]
            cur = stage_ids[i]
    segments.append((start, times[-1], cur))
    return segments


def retry_events(score: dict[str, Any]) -> list[tuple[float, float, str]]:
    out = []
    for event in score.get("retry_events") or []:
        rng = event.get("time_range_sec") or []
        if len(rng) >= 2:
            out.append((float(rng[0]), float(rng[1]), str(event.get("stage_id", ""))))
    return out


def draw_panel(
    image: np.ndarray,
    *,
    dense: dict[str, Any],
    clips: list[dict[str, Any]],
    score: dict[str, Any],
    t_sec: float,
    duration: float,
) -> None:
    h, w = image.shape[:2]
    panel_h = 220
    y0 = h - panel_h
    cv2.rectangle(image, (0, y0), (w, h), (13, 17, 23), -1)

    group = str(dense.get("episode_group", "episode"))
    known = float(dense.get("known_rollout_score", 0.0))
    stage_id, score_now = interval_at(dense, t_sec)
    state = STAGE_NAMES.get(stage_id, stage_id)
    draw_text(
        image,
        f"ARM clip labels | {group} | known={known:.1f} | score(t)={score_now:.2f} | state={state}",
        (24, y0 + 25),
        scale=0.58,
        color=(255, 230, 120),
    )
    draw_text(
        image,
        "green=progressing  gray=stagnant  red=regressing  lower bar: advantage positive/unclear/negative",
        (24, y0 + 48),
        scale=0.43,
        color=(210, 220, 235),
    )

    bar_x = 28
    bar_w = w - 56
    stage_y = y0 + 66
    int_y = y0 + 98
    adv_y = y0 + 126
    curve_y = y0 + 166
    curve_h = 38
    inc_y = y0 + 146
    inc_h = 14

    def x_at(t: float) -> int:
        if duration <= 0:
            return bar_x
        return int(round(bar_x + bar_w * max(0.0, min(1.0, t / duration))))

    # Active stage timeline.
    for start, end, sid in compress_stage_segments(dense):
        x0, x1 = x_at(start), max(x_at(end), x_at(start) + 1)
        color = STAGE_COLORS.get(sid, STAGE_COLORS["none"])
        cv2.rectangle(image, (x0, stage_y), (x1, stage_y + 18), color, -1)
    cv2.rectangle(image, (bar_x, stage_y), (bar_x + bar_w, stage_y + 18), (235, 240, 250), 1)
    draw_text(image, "active stage", (bar_x, stage_y - 5), scale=0.36, color=(220, 228, 240))

    # Interval and clip labels.
    for clip in clips:
        times = [float(x) for x in clip.get("times_sec") or []]
        labels = clip.get("interval_labels") or []
        for i, label in enumerate(labels):
            if i + 1 >= len(times):
                continue
            x0, x1 = x_at(times[i]), max(x_at(times[i + 1]), x_at(times[i]) + 1)
            cv2.rectangle(image, (x0, int_y), (x1, int_y + 18), LABEL_COLORS.get(label, (120, 120, 120)), -1)
        x0, x1 = x_at(float(clip.get("clip_start_sec", 0.0))), x_at(float(clip.get("clip_end_sec", 0.0)))
        adv = str(clip.get("advantage_label", "negative"))
        cv2.rectangle(image, (x0, adv_y), (max(x1, x0 + 1), adv_y + 14), ADV_COLORS.get(adv, ADV_COLORS["negative"]), -1)
    cv2.rectangle(image, (bar_x, int_y), (bar_x + bar_w, int_y + 18), (235, 240, 250), 1)
    cv2.rectangle(image, (bar_x, adv_y), (bar_x + bar_w, adv_y + 14), (235, 240, 250), 1)
    draw_text(image, "ARM interval", (bar_x, int_y - 5), scale=0.36, color=(220, 228, 240))
    draw_text(image, "advantage", (bar_x, adv_y - 5), scale=0.36, color=(220, 228, 240))

    for start, end, sid in retry_events(score):
        x0, x1 = x_at(start), max(x_at(end), x_at(start) + 2)
        cv2.rectangle(image, (x0, adv_y + 20), (x1, adv_y + 32), (30, 30, 230), -1)
        draw_text(image, "RETRY", (x0 + 3, adv_y + 49), scale=0.32, color=(85, 120, 255))

    # Sparse increment spikes: this is the actual ARM label source.
    incs = [float(x) for x in dense.get("score_increment") or []]
    times = [float(x) for x in dense.get("times_sec") or []]
    max_abs_inc = max([abs(x) for x in incs] + [1e-6])
    cv2.rectangle(image, (bar_x, inc_y), (bar_x + bar_w, inc_y + inc_h), (22, 28, 38), -1)
    for i, inc in enumerate(incs):
        if i + 1 >= len(times) or abs(inc) < 1e-8:
            continue
        x0, x1 = x_at(times[i]), max(x_at(times[i + 1]), x_at(times[i]) + 1)
        frac = min(1.0, abs(inc) / max_abs_inc)
        if inc >= 0:
            y_top = int(round(inc_y + inc_h * (1.0 - frac)))
            cv2.rectangle(image, (x0, y_top), (x1, inc_y + inc_h), (90, 220, 255), -1)
        else:
            cv2.rectangle(image, (x0, inc_y), (x1, int(round(inc_y + inc_h * frac))), (60, 80, 235), -1)
    cv2.rectangle(image, (bar_x, inc_y), (bar_x + bar_w, inc_y + inc_h), (140, 150, 170), 1)
    draw_text(image, "score increment", (bar_x, inc_y - 5), scale=0.34, color=(220, 228, 240))

    # Score curve.
    cv2.rectangle(image, (bar_x, curve_y), (bar_x + bar_w, curve_y + curve_h), (22, 28, 38), -1)
    cv2.rectangle(image, (bar_x, curve_y), (bar_x + bar_w, curve_y + curve_h), (140, 150, 170), 1)
    vals = [float(x) for x in dense.get("score_curve") or []]
    pts = []
    for t, v in zip(times, vals, strict=False):
        x = x_at(t)
        y = int(round(curve_y + curve_h - curve_h * max(0.0, min(10.0, v)) / 10.0))
        pts.append((x, y))
    for p0, p1 in zip(pts, pts[1:], strict=False):
        cv2.line(image, p0, p1, (90, 220, 255), 2, cv2.LINE_AA)
    draw_text(image, "score curve 0-10", (bar_x, curve_y - 6), scale=0.36, color=(220, 228, 240))

    now_x = x_at(t_sec)
    cv2.line(image, (now_x, y0 + 58), (now_x, h - 12), (80, 255, 255), 1, cv2.LINE_AA)
    draw_text(image, f"t={t_sec:.1f}s", (max(5, min(w - 100, now_x + 5)), h - 12), scale=0.42, color=(80, 255, 255))


def encode_h264(input_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ],
        check=True,
    )


def render_one(
    name: str,
    group: str,
    dense: dict[str, Any],
    clips: list[dict[str, Any]],
    score: dict[str, Any],
    out_dir: Path,
    target_fps: float,
    overwrite: bool,
) -> dict[str, Any]:
    src = Path(str(dense["combined_video_path"]))
    out_name = f"arm_{name}_{group}_score{float(dense.get('known_rollout_score', 0.0)):g}.mp4"
    out_path = out_dir / out_name
    if out_path.exists() and not overwrite:
        return {"name": name, "group": group, "video": out_name, "score": dense.get("known_rollout_score"), "dense": dense}

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else float(dense.get("duration_sec", 0.0))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_h = 220
    out_w, out_h = src_w, src_h + panel_h
    step = max(1, int(round(fps / target_fps))) if target_fps > 0 else 1
    write_fps = fps / step

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), write_fps, (out_w, out_h))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                canvas[:src_h, :src_w] = frame
                draw_panel(canvas, dense=dense, clips=clips, score=score, t_sec=idx / fps if fps > 0 else 0.0, duration=duration)
                writer.write(canvas)
            idx += 1
        cap.release()
        writer.release()
        encode_h264(raw, out_path)
    return {"name": name, "group": group, "video": out_name, "score": dense.get("known_rollout_score"), "dense": dense}


def write_index(items: list[dict[str, Any]], out_dir: Path) -> None:
    def esc(s: Any) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    cards = []
    for item in items:
        dense = item["dense"]
        cards.append(
            f'<div class="card"><h2>{esc(item["name"])} · {esc(item["group"])} · score={float(item["score"]):.1f}</h2>'
            f'<video controls preload="metadata" src="{esc(item["video"])}"></video>'
            f'<pre>{esc(json.dumps({"episode_group": item["group"], "known_score": item["score"], "duration_sec": dense.get("duration_sec")}, ensure_ascii=False, indent=2))}</pre></div>'
        )
    html = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>ARM Clip Labels</title><style>body{margin:0;background:#0d1117;color:#f3f6ff;font-family:ui-sans-serif,system-ui}"
        "main{max-width:1500px;margin:0 auto;padding:24px}h1{font-size:34px;margin:0 0 8px}.sub{color:#9aa7b8;margin-bottom:20px}"
        ".card{border:1px solid rgba(255,255,255,.14);border-radius:18px;overflow:hidden;margin:20px 0;background:rgba(255,255,255,.04)}"
        "h2{font-size:18px;padding:12px 16px;margin:0;border-bottom:1px solid rgba(255,255,255,.12)}video{display:block;width:100%;background:#000}"
        "pre{white-space:pre-wrap;margin:0;padding:12px 16px;color:#b8c4d6;background:rgba(0,0,0,.22)}</style></head><body><main>"
        "<h1>ARM-style Clip Labels</h1><div class='sub'>green=progressing, gray=stagnant, red=regressing. "
        "Lower advantage bar is positive/unclear/negative. Red RETRY blocks are GPT-derived retry penalties.</div>"
        + "".join(cards)
        + "</main></body></html>"
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dense_rows = load_jsonl_by_group(Path(args.dense_jsonl))
    score_rows = load_jsonl_by_group(Path(args.score_jsonl))

    picks: list[tuple[str, str]] = []
    if args.select_defaults:
        picks.extend(choose_defaults(score_rows))
    if args.episode_group:
        offset = len(picks)
        picks.extend((f"sample{offset + i + 1:02d}", g) for i, g in enumerate(args.episode_group))
    if not picks:
        raise ValueError("No episodes selected. Use --select-defaults or --episode-group.")

    groups = {group for _, group in picks}
    clips_by_group = load_clips(Path(args.clip_jsonl), groups)
    rendered = []
    for name, group in picks:
        rendered.append(
            render_one(
                name,
                group,
                dense_rows[group],
                clips_by_group.get(group, []),
                score_rows[group],
                out_dir,
                args.target_fps,
                args.overwrite,
            )
        )
    (out_dir / "arm_clip_render_summary.json").write_text(json.dumps(rendered, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index(rendered, out_dir)
    print(json.dumps({"videos": len(rendered), "output_dir": str(out_dir), "videos_out": [r["video"] for r in rendered]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
