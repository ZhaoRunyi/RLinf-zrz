from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def normalize_frame(frame: Any) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if isinstance(frame, Image.Image):
        frame = np.asarray(frame)
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[0] in {1, 3} and frame.shape[-1] not in {1, 3}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def resize_frame(frame: np.ndarray, scale: int = 3) -> np.ndarray:
    image = Image.fromarray(frame)
    width, height = image.size
    return np.asarray(
        image.resize((width * scale, height * scale), resample=Image.BILINEAR)
    )


def get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build_footer(width: int, lines: list[str]) -> np.ndarray:
    font = get_font(18)
    wrapped_lines: list[str] = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=84) or [""])
    line_height = 28
    footer_height = 24 + line_height * len(wrapped_lines) + 16
    image = Image.new("RGB", (width, footer_height), "white")
    draw = ImageDraw.Draw(image)
    y = 16
    for line in wrapped_lines:
        draw.text((16, y), line, fill="black", font=font)
        y += line_height
    return np.asarray(image)


def render_debug_video(
    history_frames: list[Any],
    footer_lines: list[str],
    output_path: Path,
    fps: int,
) -> None:
    if not history_frames:
        return
    sample_frame = resize_frame(normalize_frame(history_frames[0]), scale=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for history_frame in history_frames:
            video_frame = resize_frame(normalize_frame(history_frame), scale=3)
            footer = build_footer(sample_frame.shape[1], footer_lines)
            writer.append_data(np.concatenate([video_frame, footer], axis=0))
