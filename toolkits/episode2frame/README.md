# episode2frame clip labeling

This folder is a narrow RoboChallenge clip-labeling toolkit. It keeps only the
pieces needed to turn rollout videos and official scores into ARM/RECAP-style
short-clip advantage labels.

Pipeline:

```text
combined multi-view videos
  -> GPT official score reconciliation
  -> stage_scores + retry_events
  -> event-sparse score curve
  -> 1 Hz, 5-observation clip labels
  -> positive / unclear / negative
```

The GPT step is explicit and isolated in `gpt55_official_score_reconcile.py`.
The clip-label construction step, `build_arm_clip_labels.py`, is fully local and
does not call any API.

Main files:

- `gpt55_robochallenge_annotate_combined.py`: shared OpenAI-compatible client,
  contact-sheet construction, and combined-video record loading.
- `gpt55_official_score_reconcile.py`: calls GPT to infer stage completion,
  retry events, and score reconciliation from RoboChallenge rubric.
- `summarize_gpt55_score_reconcile.py`: checks annotation coverage and score
  consistency.
- `build_arm_clip_labels.py`: builds event-sparse score curves and clip
  advantage labels.
- `render_gpt55_official_score_videos.py`: visualizes GPT stage/retry scoring.
- `render_arm_clip_labels.py`: visualizes final clip labels.

See `ARM_CLIP_LABELING.md` for exact commands and label semantics.
