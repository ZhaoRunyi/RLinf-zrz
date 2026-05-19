# ARM-Style Clip Labeling Pipeline

This document describes the final `put_cup_on_coaster` clip-labeling pipeline used in `toolkits/episode2frame`.

The goal is to convert RoboChallenge rollout-level / stage-level scoring into short-clip advantage labels:

```text
positive / unclear / negative
```

The final rule is based on **absolute score gain**, not top-k ranking and not retry hard-gating.

## Final Label Semantics

For each 5-frame clip sampled at 1 Hz, compute:

```text
local_gain = score(t + H) - score(t)
```

Then assign:

```text
local_gain >  0.03  -> positive
local_gain < -0.03  -> negative
otherwise           -> unclear
```

Retry is preserved as metadata only:

```text
has_retry / retry_overlaps
```

Retry does **not** flip the label. If score rises during retry, the clip is positive.

## High-Level Flow

```text
combined multi-view videos
  -> GPT official score reconciliation
  -> stage_scores + retry_events
  -> event-sparse score_increment
  -> cumulative score_curve
  -> 5-frame ARM-style clips
  -> positive / unclear / negative advantage labels
  -> MP4 visualization
```

## Main Scripts

### 1. Official score reconciliation

```text
toolkits/episode2frame/gpt55_official_score_reconcile.py
```

Purpose:
- Sends sampled multi-view contact sheets to GPT.
- Reconstructs how each official RoboChallenge score was earned.
- Outputs per-stage completion, retry events, and computed score.

Main output:

```text
logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/official_score_reconcile.jsonl
```

Each row contains:

```json
{
  "episode_group": "...",
  "known_rollout_score": 4.5,
  "computed_score": 4.5,
  "total_retry_count": 3,
  "stage_scores": [...],
  "retry_events": [...],
  "combined_video_path": "..."
}
```

### 2. Build ARM clip labels

```text
toolkits/episode2frame/build_arm_clip_labels.py
```

Purpose:
- Runs locally.
- Does not call any API.
- Converts `stage_scores` and `retry_events` into event-sparse score increments.
- Generates interval labels and 5-frame clip advantage labels.

Final output directory:

```text
logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/arm_clip_labels_final_1hz_w5
```

Outputs:

```text
arm_interval_labels.jsonl
arm_clip_labels.jsonl
arm_dense_score_curves.jsonl
summary.json
```

### 3. Render visualization videos

```text
toolkits/episode2frame/render_arm_clip_labels.py
```

Purpose:
- Renders multi-view video with bottom overlays.
- Shows active stage, interval labels, advantage labels, score increment, cumulative score, and retry regions.
- Encodes MP4 as H.264 baseline + yuv420p for browser / VSCode compatibility.

Page directory:

```text
logs/episode2frame/gpt55_put_cup_segments_512/overlay_videos_named
```

Served at:

```text
http://127.0.0.1:6011/
```

## Step-By-Step Usage

### Step 0: Inputs

Combined videos should exist at:

```text
logs/episode2frame/put_cup_on_coaster_combined_videos
```

Each episode directory should contain:

```text
views_hstack.mp4
metadata.json
```

### Step 1: Run GPT official score reconciliation

Only this step uses API.

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."

python toolkits/episode2frame/gpt55_official_score_reconcile.py \
  --selection all \
  --combined-dir logs/episode2frame/put_cup_on_coaster_combined_videos \
  --output-dir logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile \
  --api-workers 32 \
  --max-frames 512 \
  --frames-per-image 64 \
  --grid-cols 8 \
  --tile-width 240 \
  --tile-label-font-size 13 \
  --jpeg-quality 80 \
  --resume
```

Expected output:

```text
official_score_reconcile.jsonl
media/*.jpg
requests/*.request.json
responses/*.json
```

Check progress / summary:

```bash
python toolkits/episode2frame/summarize_gpt55_score_reconcile.py \
  --results-jsonl logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/official_score_reconcile.jsonl \
  --output-dir logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile \
  --expected-total 1389
```

Final expected count for this run:

```text
1389 / 1389
```

### Step 2: Build final ARM clip labels

This step is local only and does not use API.

```bash
python toolkits/episode2frame/build_arm_clip_labels.py \
  --results-jsonl logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/official_score_reconcile.jsonl \
  --output-dir logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/arm_clip_labels_final_1hz_w5 \
  --sample-fps 1 \
  --source-fps 5 \
  --window-size 5 \
  --stride 1 \
  --advantage-margin 0.03 \
  --interval-margin 0.03 \
  --credit-pre-window-sec 6 \
  --completion-mass 0.8 \
  --approach-mass 0.2 \
  --retry-penalty 0.5 \
  --overwrite
```

Final observed distribution:

```text
clips:    120536
positive: 18215
negative: 279
unclear:  102042
```

### Step 3: Render review MP4s

Example: render 20 review samples.

```bash
python toolkits/episode2frame/render_arm_clip_labels.py \
  --dense-jsonl logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/arm_clip_labels_final_1hz_w5/arm_dense_score_curves.jsonl \
  --clip-jsonl logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/arm_clip_labels_final_1hz_w5/arm_clip_labels.jsonl \
  --score-jsonl logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/official_score_reconcile.jsonl \
  --output-dir logs/episode2frame/gpt55_put_cup_segments_512/overlay_videos_named \
  --target-fps 12 \
  --overwrite \
  --episode-group 000301_42621043-f106-4c2c-a5ac-cca21737f985 \
  --episode-group 000682_7ea689bc-e3e6-44c2-8401-ad407a167683 \
  --episode-group 001198_4df79111-c6c4-47e4-80e5-0ec4a809d713 \
  --episode-group 001162_6e55b7f8-9424-4637-9bcc-2f20c98b50a5 \
  --episode-group 001172_6e55b7f8-9424-4637-9bcc-2f20c98b50a5 \
  --episode-group 000321_1e3cc94f-214b-4a9f-a347-edc94a68d237 \
  --episode-group 000000_18134816-9a53-4beb-90cf-313239e41062 \
  --episode-group 000001_bfaae1c8-bf41-4385-ab1c-de2c70d85496 \
  --episode-group 000002_98c4bf37-c31a-45f1-9c79-5c2e03cffe65 \
  --episode-group 000020_866b2120-3499-499b-83c4-d48538f50b4a \
  --episode-group 000021_6b5d9260-b928-4886-9abb-fff58f2a8e0a \
  --episode-group 000008_4318ef71-cd4f-4f72-8023-122d7b76a65e \
  --episode-group 000055_5bfce511-6da4-45e7-9fa4-8791a22d505a \
  --episode-group 001107_70fdaaed-00ab-4db9-8045-b968b76e4073 \
  --episode-group 001117_70fdaaed-00ab-4db9-8045-b968b76e4073 \
  --episode-group 000029_7017dd6c-e912-4293-8e84-5067f73d36c4 \
  --episode-group 000032_67aefb3e-0350-4017-908e-6cc3278a2bee \
  --episode-group 000363_686ae88e-a09c-429b-ad7c-d96c57e08bbf \
  --episode-group 000194_8220cdc9-d4c4-4978-b0c2-823797063559 \
  --episode-group 000010_09c6f3ed-cb64-4b9e-8a52-31568375b96f
```

If the server is not already running:

```bash
python -m http.server 6011 \
  --bind 0.0.0.0 \
  --directory logs/episode2frame/gpt55_put_cup_segments_512/overlay_videos_named
```

Open:

```text
http://127.0.0.1:6011/
```

## Output Schemas

### `arm_dense_score_curves.jsonl`

One row per episode.

Important fields:

```json
{
  "episode_group": "...",
  "known_rollout_score": 4.5,
  "duration_sec": 115.8,
  "times_sec": [0.0, 1.0, 2.0],
  "source_frames": [0, 5, 10],
  "stage_ids": ["s1_pick_up_cup", "s1_pick_up_cup"],
  "score_curve": [0.0, 0.1, 0.2],
  "score_increment": [0.1, 0.1],
  "events": [...]
}
```

### `arm_interval_labels.jsonl`

One row per adjacent sampled interval.

```json
{
  "episode_group": "...",
  "start_sec": 31.0,
  "end_sec": 32.0,
  "stage_id": "s1_pick_up_cup",
  "score_before": 1.05,
  "score_after": 1.33,
  "score_increment": 0.28,
  "retry_active": true,
  "label": "stagnant | progressing | regressing"
}
```

Interval labels are for analysis. The training-oriented label is the clip-level `advantage_label`.

### `arm_clip_labels.jsonl`

One row per 5-frame window.

```json
{
  "episode_group": "...",
  "clip_start_sec": 31.0,
  "clip_end_sec": 35.0,
  "times_sec": [31, 32, 33, 34, 35],
  "source_frames": [155, 160, 165, 170, 175],
  "local_gain": 2.44,
  "has_retry": true,
  "retry_overlaps": [...],
  "advantage_label": "positive | unclear | negative",
  "advantage_label_id": 1
}
```

Final training recommendation:

```text
positive: use as good action chunk
negative: use as bad action chunk
unclear: discard from binary training loss or use as unlabeled/low-weight
```

## Visualization Meaning

The rendered MP4 bottom panel contains:

```text
active stage bar     : current scoring stage
ARM interval bar     : local interval transition label
advantage bar        : final clip-level positive/unclear/negative
RETRY blocks         : GPT-derived retry windows
score increment      : sparse event credit used for labels
score curve          : cumulative official score reconstruction
```

Color convention:

```text
positive / progressing : green
unclear                : orange/blue
negative / regressing  : red
stagnant               : gray
```

## Notes and Assumptions

- `score_curve` is not globally smoothed.
- Stage credit is not linearly spread across the whole stage.
- `completion_mass=0.8` places 80% of stage credit near the stage achievement event.
- `approach_mass=0.2` allows limited approach credit before the completion window.
- Retry penalties are included in `score_increment` but do not hard-flip labels.
- The final score curve endpoint is corrected to match the official known rollout score.
- The current task-specific constants are tuned for `put_cup_on_coaster` at 5 FPS input and 1 FPS clip labeling.
