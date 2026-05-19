#!/usr/bin/env python3
"""Render official RoboChallenge scoring bars onto combined multi-view videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STAGE_COLORS = [
    (69, 188, 255),
    (90, 220, 120),
    (255, 176, 80),
    (210, 130, 255),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--target-fps", type=float, default=12.0)
    return p.parse_args()


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (245, 248, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def stage_time(stage: dict[str, Any]) -> float | None:
    for key in ["first_achievement_time_sec", "completion_time_sec"]:
        value = stage.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def stable_time(stage: dict[str, Any]) -> float | None:
    value = stage.get("stable_confirmation_time_sec")
    if isinstance(value, int | float):
        return float(value)
    return None


def current_state(result: dict[str, Any], t_sec: float) -> str:
    stages = result.get("stage_scores") or []
    intervals = [
        stage
        for stage in stages
        if stage.get("state_interval_start_sec") is not None and stage.get("state_interval_end_sec") is not None
    ]
    for stage in intervals:
        start = float(stage["state_interval_start_sec"])
        end = float(stage["state_interval_end_sec"])
        if start <= t_sec < end:
            current_index = stages.index(stage)
            score_now = sum(float(prev.get("final_points", 0.0)) for prev in stages[:current_index] if prev.get("completed"))
            state_label = stage.get("state_label") or f"{stage.get('stage_name')} not completed"
            return f"STATE: {state_label} | score_now={score_now:.1f}"

    completed = [stage for stage in stages if stage.get("completed") and stage_time(stage) is not None]
    completed.sort(key=lambda stage: stage_time(stage) or 0.0)
    achieved = [stage for stage in completed if (stage_time(stage) or 0.0) <= t_sec]
    score_now = sum(float(stage.get("final_points", 0.0)) for stage in achieved)
    for stage in completed:
        first = stage_time(stage)
        if first is not None and t_sec < first:
            name = str(stage.get("stage_name", "stage"))
            if str(stage.get("stage_id")) == "s4_reset_arm":
                return f"STATE: reset not completed - returning to home/rest | score_now={score_now:.1f}"
            return f"STATE: {name} not completed | score_now={score_now:.1f}"

    if not completed:
        return "STATE: no official stage completed | score_now=0.0"
    last = completed[-1]
    if str(last.get("stage_id")) == "s4_reset_arm":
        stable = stable_time(last)
        if stable is not None and t_sec < stable:
            return f"STATE: reset achieved, confirming home/rest stability | score_now={score_now:.1f}"
        return f"STATE: reset/home completed | score_now={score_now:.1f}"
    return f"STATE: all credited stages completed | score_now={score_now:.1f}"


def draw_hatched_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    cv2.rectangle(image, (x0, y0), (x1, y1), (70, 74, 84), -1)
    for x in range(x0 - (y1 - y0), x1 + 20, 12):
        cv2.line(image, (x, y1), (x + y1 - y0, y0), (105, 110, 124), 1, cv2.LINE_AA)


def draw_score_bar(image: np.ndarray, result: dict[str, Any], t_sec: float, duration: float) -> None:
    h, w = image.shape[:2]
    panel_h = 156
    y0 = h - panel_h
    cv2.rectangle(image, (0, y0), (w, h), (13, 17, 23), -1)

    stage_scores = result.get("stage_scores") or []
    total = sum(float(s.get("max_points", 0.0)) for s in stage_scores) or 10.0
    bar_x = 28
    bar_y = y0 + 72
    bar_w = w - 56
    bar_h = 34

    draw_text(
        image,
        current_state(result, t_sec)[:120],
        (28, y0 + 25),
        scale=0.66,
        color=(255, 230, 120),
    )
    draw_text(
        image,
        (
            f"{result.get('episode_group', '').upper()}  "
            f"known={float(result.get('known_rollout_score', 0.0)):.1f}  "
            f"computed={float(result.get('computed_score', 0.0)):.1f}  "
            "marker=first achievement"
        ),
        (28, y0 + 51),
        scale=0.55,
    )

    x = bar_x
    for i, stage in enumerate(stage_scores):
        points = float(stage.get("max_points", 0.0))
        seg_w = int(round(bar_w * points / total))
        x1 = bar_x + bar_w if i == len(stage_scores) - 1 else min(bar_x + bar_w, x + seg_w)
        completed = bool(stage.get("completed"))
        color = STAGE_COLORS[i % len(STAGE_COLORS)]
        if completed:
            cv2.rectangle(image, (x, bar_y), (x1, bar_y + bar_h), color, -1)
        else:
            draw_hatched_rect(image, x, bar_y, x1, bar_y + bar_h)
        cv2.rectangle(image, (x, bar_y), (x1, bar_y + bar_h), (235, 240, 250), 1)

        name = str(stage.get("stage_name", "")).replace("The ", "")
        label = f"{points:g}p {name}"
        draw_text(image, label[:34], (x + 5, bar_y - 7), scale=0.42, color=(220, 228, 240))
        t = stage_time(stage)
        if t is not None and duration > 0:
            tx = int(round(bar_x + bar_w * max(0.0, min(1.0, t / duration))))
            cv2.line(image, (tx, bar_y - 17), (tx, bar_y + bar_h + 28), (255, 255, 255), 2, cv2.LINE_AA)
            draw_text(image, f"{t:.1f}s", (tx + 4, bar_y + bar_h + 22), scale=0.42)
        st = stable_time(stage)
        if st is not None and t is not None and abs(st - t) > 1e-3 and duration > 0:
            sx = int(round(bar_x + bar_w * max(0.0, min(1.0, st / duration))))
            cv2.line(image, (sx, bar_y - 7), (sx, bar_y + bar_h + 9), (170, 190, 210), 1, cv2.LINE_AA)
        x = x1

    for boundary in result.get("transition_boundaries", []) or []:
        t = boundary.get("time_sec")
        if not isinstance(t, int | float) or duration <= 0:
            continue
        bx = int(round(bar_x + bar_w * max(0.0, min(1.0, float(t) / duration))))
        cv2.line(image, (bx, bar_y - 28), (bx, bar_y + bar_h + 38), (0, 255, 255), 1, cv2.LINE_AA)
        name = str(boundary.get("boundary_id", "boundary")).replace("_", " ")
        draw_text(image, name, (bx + 3, bar_y - 23), scale=0.34, color=(0, 255, 255))

    retry_events = list(result.get("retry_events_detected", []) or [])
    for retry in result.get("retry_events", []) or []:
        if isinstance(retry.get("time_range_sec"), list) and len(retry["time_range_sec"]) >= 2:
            retry_events.append(
                {
                    "start_time_sec": retry["time_range_sec"][0],
                    "end_time_sec": retry["time_range_sec"][1],
                    "reason": retry.get("reason", ""),
                    "stage_id": retry.get("stage_id", ""),
                }
            )

    for retry in retry_events:
        start = retry.get("start_time_sec")
        end = retry.get("end_time_sec")
        if not isinstance(start, int | float) or not isinstance(end, int | float) or duration <= 0:
            continue
        rx0 = int(round(bar_x + bar_w * max(0.0, min(1.0, float(start) / duration))))
        rx1 = int(round(bar_x + bar_w * max(0.0, min(1.0, float(end) / duration))))
        if rx1 <= rx0:
            rx1 = rx0 + 2
        cv2.rectangle(image, (rx0, bar_y + bar_h + 40), (rx1, bar_y + bar_h + 52), (30, 30, 230), -1)
        cv2.rectangle(image, (rx0, bar_y + bar_h + 40), (rx1, bar_y + bar_h + 52), (255, 255, 255), 1)
        draw_text(image, "RETRY", (rx0 + 3, bar_y + bar_h + 68), scale=0.34, color=(80, 120, 255))

    if duration > 0:
        now_x = int(round(bar_x + bar_w * max(0.0, min(1.0, t_sec / duration))))
        cv2.line(image, (now_x, y0 + 6), (now_x, h - 10), (80, 255, 255), 1, cv2.LINE_AA)
        draw_text(image, f"t={t_sec:.1f}s", (max(4, min(w - 110, now_x + 5)), h - 12), scale=0.42, color=(80, 255, 255))


def encode_h264(input_path: Path, output_path: Path) -> None:
    cmd = [
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
    ]
    subprocess.run(cmd, check=True)


def render_one(result: dict[str, Any], out_dir: Path, target_fps: float, overwrite: bool) -> dict[str, Any]:
    src = Path(result["combined_video_path"])
    group = str(result.get("episode_group", "episode"))
    score = float(result.get("known_rollout_score", 0.0))
    eid = str(result.get("episode_id", "unknown"))
    out_name = f"{group}_score{score:g}_{eid}.official_score.mp4"
    out_path = out_dir / out_name
    if out_path.exists() and not overwrite:
        return {"group": group, "score": score, "episode_id": eid, "video": out_name, "result": result}

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = src_w, src_h + 156

    step = max(1, int(round(fps / target_fps))) if target_fps > 0 else 1
    write_fps = fps / step
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), write_fps, (out_w, out_h))
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                canvas[:src_h, :src_w] = frame
                draw_score_bar(canvas, result, frame_idx / fps if fps > 0 else 0.0, duration)
                writer.write(canvas)
            frame_idx += 1
        writer.release()
        cap.release()
        encode_h264(tmp_path, out_path)
    return {"group": group, "score": score, "episode_id": eid, "video": out_name, "result": result}


def write_index(summary: list[dict[str, Any]], out_dir: Path) -> None:
    def esc_html(s: Any) -> str:
        text = str(s)
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def esc_attr(s: Any) -> str:
        return esc_html(s).replace("'", "&#39;")

    cards = []
    for item in sorted(summary, key=lambda x: {"high": 0, "mid": 1, "low": 2}.get(x["group"], 9)):
        result_json = json.dumps(item["result"], ensure_ascii=False, indent=2)
        cards.append(
            f'<div class="card"><div class="head"><div class="title">{esc_html(item["group"].upper())} '
            f'known={item["score"]:.1f} computed={float(item["result"].get("computed_score", 0.0)):.1f}</div>'
            f'<div class="meta">retry={item["result"].get("total_retry_count")} · '
            f'reset={item["result"].get("reset_arm_stage_completed")}</div></div>'
            f'<video controls preload="metadata" src="{esc_attr(item["video"])}"></video>'
            f"<pre>{esc_html(result_json)}</pre></div>"
        )
    html = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Official RoboChallenge Score</title>"
        "<style>body{margin:0;background:#0d1117;color:#f3f6ff;font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif}"
        "main{max-width:1500px;margin:0 auto;padding:24px}h1{margin:0 0 6px;font-size:36px;letter-spacing:-.04em}"
        ".sub{color:#9aa7b8;margin-bottom:22px;line-height:1.5}.card{background:linear-gradient(180deg,rgba(255,255,255,.07),rgba(255,255,255,.025));"
        "border:1px solid rgba(255,255,255,.14);border-radius:20px;overflow:hidden;margin:22px 0}.head{display:flex;justify-content:space-between;"
        "padding:14px 16px;border-bottom:1px solid rgba(255,255,255,.12)}.title{font-weight:800}.meta{color:#9aa7b8}video{display:block;width:100%;background:#000}"
        "pre{white-space:pre-wrap;color:#b8c4d6;padding:14px 16px;margin:0;background:rgba(0,0,0,.22);font-size:12px;max-height:360px;overflow:auto}</style></head>"
        '<body><main><h1>官方 RoboChallenge 4 项打分对齐</h1>'
        '<div class="sub">底部第一行显示当前状态和当前累计小分；下面大色块是官方 4 个 scoring stage 的分数预算。'
        '彩色表示完成，斜纹灰表示未完成。白色竖线表示 <b>first_achievement_time_sec：首次可见达成时间</b>；'
        '浅灰细线如果出现，表示 stable_confirmation_time_sec：后续稳定确认时间。'
        '这里的 reset 严格按“机械臂回到 home/rest pose”理解，不再把单纯离开杯子区域算作 reset。</div>'
        + "".join(cards)
        + "</main></body></html>"
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for line in Path(args.results_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            results.append(json.loads(line))
    summary = [render_one(r, out_dir, args.target_fps, args.overwrite) for r in results]
    (out_dir / "official_score_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index(summary, out_dir)
    print(json.dumps({"videos": len(summary), "output_dir": str(out_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
