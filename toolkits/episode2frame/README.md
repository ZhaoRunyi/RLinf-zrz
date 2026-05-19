# RoboChallenge Episode-to-Frame Clip Labeling

This directory contains a narrow pipeline for converting RoboChallenge rollout-level scores into short-clip π0.6/RECAP-style binary advantage labels, while keeping ARM-style diagnostics for inspection.

The current target task is `put_cup_on_coaster`, but the raw-data preparation script can read any task from the RoboChallenge dump as long as the task name exists in `runs_list.json`.

## What This Pipeline Produces

For every rollout, the pipeline produces three levels of data:

```text
1. combined multi-view video
   views_hstack.mp4 + metadata.json

2. GPT official score reconciliation
   stage_scores + retry_events + computed_score

3. π0.6-style short-clip advantage labels
   Advantage: positive / Advantage: negative

   The file also keeps an internal diagnostic label:
   positive / unclear / negative
```

The final clip labels are for short transitions, not isolated frames:

```text
(o_t, ..., o_{t+H}) -> Advantage: positive | Advantage: negative
```

Default clip setting:

```text
sample_fps = 1
window_size = 5
stride = 1
```

So each label describes roughly a 5-second local transition.

## Full Flow

```text
raw RoboChallenge RRD directory
  -> decode 3-view RRD videos
  -> horizontally stacked MP4 + metadata.json
  -> GPT score/stage/retry reconciliation
  -> event-sparse score_increment
  -> cumulative score_curve
  -> π0.6-style binary clip advantages
  -> visualization MP4s
```

Only the GPT reconciliation step calls an API. Everything else is local.

## Expected Raw Data Layout

The raw RoboChallenge directory can look like this:

```text
/vepfs-mlp2/c20250301/240403026/robochallenge/
  manifest.json
  runs_list.json
  rrd_run_dirs/
    run_000007_<run_id>/
      run.json
      rollouts.json
      rollout_00_<rollout_id>.rrd
      rollout_01_<rollout_id>.rrd
      ...
```

The script also supports the older layout where RRDs live directly under:

```text
robochallenge/run_000007_<run_id>/rollout_00_<rollout_id>.rrd
```

For the current VEPFS dump, the heavy RRD files are under `rrd_run_dirs/run_*/`.

## Environment Note

Use the Rerun-compatible Python environment for decoding the current RRD dump:

```bash
/vepfs-mlp2/c20250301/240403026/robochallenge/rrd2lerobot_venv/bin/python
```

Reason: the current RRD files were written by Rerun `0.24.x`. Some default RLinf environments have older `rerun_bindings`, which may list schemas but fail when decoding the actual recording.

## Main Scripts

### `prepare_robochallenge_combined_videos.py`

Reads raw RoboChallenge `.rrd` files and writes GPT-readable combined videos.

Input:

```text
manifest.json + runs_list.json + rollout_*.rrd
```

Output:

```text
logs/episode2frame/put_cup_on_coaster_combined_videos/
  000000_<rollout_id>/
    views_hstack.mp4
    metadata.json
  episodes.jsonl
  prepare_summary.json
```

### `gpt55_official_score_reconcile.py`

Sends sampled multi-view contact sheets to GPT and asks it to reconstruct how the official RoboChallenge rollout score was obtained.

Output:

```text
official_score_reconcile.jsonl
media/*.jpg
requests/*.request.json
responses/*.json
```

The request JSON redacts image payloads before saving to disk.

### `summarize_gpt55_score_reconcile.py`

Checks annotation coverage and score consistency.

### `build_arm_clip_labels.py`

Builds event-sparse score curves and short-clip labels locally. No API call.

Output:

```text
arm_interval_labels.jsonl
arm_clip_labels.jsonl
arm_dense_score_curves.jsonl
summary.json
```

### `render_arm_clip_labels.py`

Renders MP4 visualizations with score curves, labels, stage bars, and retry regions.

## Step 0: Prepare Combined Videos From Raw RRD

Full run for `put_cup_on_coaster`:

```bash
cd /c20250301/zhoutianxing/RLinf-arm-clip-labeling-only

/vepfs-mlp2/c20250301/240403026/robochallenge/rrd2lerobot_venv/bin/python \
  toolkits/episode2frame/prepare_robochallenge_combined_videos.py \
  --data-dir /vepfs-mlp2/c20250301/240403026/robochallenge \
  --output-dir logs/episode2frame/put_cup_on_coaster_combined_videos \
  --task put_cup_on_coaster \
  --fps 5 \
  --height 360 \
  --num-workers 8
```

Smoke test with 3 rollouts:

```bash
/vepfs-mlp2/c20250301/240403026/robochallenge/rrd2lerobot_venv/bin/python \
  toolkits/episode2frame/prepare_robochallenge_combined_videos.py \
  --data-dir /vepfs-mlp2/c20250301/240403026/robochallenge \
  --output-dir logs/episode2frame/put_cup_on_coaster_combined_videos_smoke \
  --task put_cup_on_coaster \
  --fps 5 \
  --height 360 \
  --limit 3 \
  --num-workers 2 \
  --overwrite
```

Useful filters:

```bash
--task put_cup_on_coaster
--task-regex 'cup|coaster'
--start-index 100
--limit 20
--overwrite
```

Output `metadata.json` contains fields like:

```json
{
  "episode_group": "000000_<rollout_id>",
  "episode_id": "<rollout_id>",
  "task": "put_cup_on_coaster",
  "score": 10.0,
  "path": "/abs/path/views_hstack.mp4",
  "views": ["front", "left", "right"],
  "fps": 5.0,
  "height": 360,
  "frame_counts": {
    "front": 512,
    "left": 512,
    "right": 512,
    "combined": 512
  },
  "rrd_path": "/abs/path/rollout.rrd",
  "source": {
    "run_id": "...",
    "rollout_index": 0,
    "rollout": {...},
    "run": {...}
  }
}
```

## Step 1: GPT Official Score Reconciliation

This is the only API-consuming step.

Set API environment variables outside the repository:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."
```

Run all prepared videos:

```bash
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

Run a small test first:

```bash
python toolkits/episode2frame/gpt55_official_score_reconcile.py \
  --selection all \
  --combined-dir logs/episode2frame/put_cup_on_coaster_combined_videos_smoke \
  --output-dir logs/episode2frame/gpt55_put_cup_segments_512/smoke_official_score_reconcile \
  --api-workers 2 \
  --max-frames 128 \
  --frames-per-image 64 \
  --resume
```

Important output:

```text
logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/official_score_reconcile.jsonl
```

Each row contains:

```json
{
  "episode_group": "000000_<rollout_id>",
  "episode_id": "<rollout_id>",
  "known_rollout_score": 4.5,
  "visual_estimated_score_before_reconcile": 5.0,
  "computed_score": 4.5,
  "score_matches_known": true,
  "total_retry_count": 3,
  "stage_scores": [
    {
      "stage_id": "s1_pick_up_cup",
      "stage_name": "Pick up the cup",
      "completed": true,
      "first_achievement_time_sec": 12.4,
      "final_points": 4.0,
      "retry_count": 0
    }
  ],
  "retry_events": [
    {
      "stage_id": "s2_move_to_destination",
      "time_range_sec": [20.0, 24.0],
      "reason": "..."
    }
  ],
  "combined_video_path": "/abs/path/views_hstack.mp4"
}
```

Check progress and summary:

```bash
python toolkits/episode2frame/summarize_gpt55_score_reconcile.py \
  --results-jsonl logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/official_score_reconcile.jsonl \
  --output-dir logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile \
  --expected-total 1240
```

For the current raw dump, `put_cup_on_coaster` discovery found 1240 rollouts.

## Step 2: Build π0.6-Style Clip Advantage Labels

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
  --pi06-unclear-policy discard \
  --interval-margin 0.03 \
  --credit-pre-window-sec 6 \
  --completion-mass 0.8 \
  --approach-mass 0.2 \
  --retry-penalty 0.5 \
  --overwrite
```

Final π0.6/RECAP-style rule:

```text
local_gain = score(t + H) - score(t)

local_gain >  0.03 -> pi06_advantage_label = positive
local_gain < -0.03 -> pi06_advantage_label = negative
otherwise          -> pi06_advantage_label = negative, pi06_training_weight = 0
```

So the exported training condition is always binary:

```text
Advantage: positive
Advantage: negative
```

The old `advantage_label = positive / unclear / negative` is now only a diagnostic field. For π0.6-style training, use `pi06_advantage_label`, `pi06_advantage_label_id`, `pi06_training_weight`, and `pi06_prompt_condition`.

If you want stagnant/weak clips to train as negative instead of being masked, set:

```bash
--pi06-unclear-policy negative
```

Retry is metadata, not a hard label override. If a retry interval still increases score, it can be positive.

## Step 3: Render MP4 Visualizations

Render a few examples:

```bash
python toolkits/episode2frame/render_arm_clip_labels.py \
  --clip-labels-jsonl logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/arm_clip_labels_final_1hz_w5/arm_clip_labels.jsonl \
  --dense-curves-jsonl logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/arm_clip_labels_final_1hz_w5/arm_dense_score_curves.jsonl \
  --output-dir logs/episode2frame/gpt55_put_cup_segments_512/overlay_videos_named \
  --limit 20 \
  --overwrite
```

The renderer encodes browser/VSCode-compatible videos:

```text
H.264 baseline + yuv420p + faststart
```

Serve the page on the fixed port:

```bash
cd logs/episode2frame/gpt55_put_cup_segments_512/overlay_videos_named
python -m http.server 6011
```

Open:

```text
http://127.0.0.1:6011/
```

## Label Semantics

### π0.6 training fields

Use these fields for policy / VLA conditioning:

```json
{
  "pi06_advantage_label": "positive | negative",
  "pi06_advantage_label_id": 1,
  "pi06_training_weight": 1.0,
  "pi06_prompt_condition": "Advantage: positive"
}
```

`pi06_training_weight = 0` means the clip is too weak/unclear and should be skipped or downweighted in binary advantage training.

### diagnostic positive

The short clip makes visible task progress under the reconstructed score curve.

Examples:

```text
cup becomes grasped and lifted
cup moves closer to coaster while still controlled
gripper releases and cup remains stable on coaster
arm returns home for reset stage
```

### diagnostic unclear

The short clip does not show enough score-changing evidence.

Examples:

```text
minor approach motion
camera jitter
brief pause
small motion with no clear task effect
```

### diagnostic negative

The short clip decreases reconstructed progress.

Examples:

```text
object is dropped
object moves away from goal
stage progress is undone
retry/regression produces negative score gain
```

## Why Event-Sparse Curves

Do not linearly spread a stage score over the whole stage duration. That makes every frame weakly positive and creates flat, uninformative labels.

Instead, this pipeline uses event-sparse credit:

```text
most stage credit is concentrated near first achievement
approach gets limited credit
retry/regression is preserved as metadata and can contribute negative intervals
```

This is closer to ARM-style local transition labeling and avoids turning long stagnant intervals into fake positive clips.

## API Safety

Do not put API keys in committed files.

Use environment variables:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."
```

Saved request JSON files redact image payloads:

```text
requests/*.request.json
```

## Common Checks

Count prepared videos:

```bash
find logs/episode2frame/put_cup_on_coaster_combined_videos -name metadata.json | wc -l
```

Check discovered tasks from the raw dump:

```bash
python - <<'PY'
from pathlib import Path
from collections import Counter
from toolkits.episode2frame.prepare_robochallenge_combined_videos import discover_sources
root = Path('/vepfs-mlp2/c20250301/240403026/robochallenge')
sources = discover_sources(root)
for task, count in Counter(s.task for s in sources).most_common(30):
    print(count, task)
PY
```

Check GPT annotation count:

```bash
wc -l logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/official_score_reconcile.jsonl
```

Check final label distribution:

```bash
python -m json.tool \
  logs/episode2frame/gpt55_put_cup_segments_512/full_official_score_reconcile/arm_clip_labels_final_1hz_w5/summary.json
```

## Recovery / Resume

Preparation step:

```text
Existing episode directories are skipped unless --overwrite is set.
```

GPT step:

```text
Use --resume to skip already-written episode_group rows in official_score_reconcile.jsonl.
```

ARM label build:

```text
Use --overwrite when intentionally rebuilding labels with different thresholds.
```

## Current Branch

This toolkit lives on:

```text
feature/arm-clip-labeling
```

This README is local documentation. Do not push local edits unless explicitly requested.
