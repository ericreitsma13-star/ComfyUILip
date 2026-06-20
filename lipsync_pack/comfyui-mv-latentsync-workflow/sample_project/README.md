# Sample Project

Validate the LatentSync pipeline end-to-end before committing to a real project.

## What to add

```
sample_project/
├── lyrics_example.txt          ← already here
├── singer_ref_images/          ← drop 1 singer portrait here (front-facing, head+shoulders)
├── broll_ref_images/           ← drop 3-5 B-roll images here
└── output/                     ← pipeline output lands here
```

## Quick test run (30-second clip)

1. **Get a 30-second Suno clip** → save as `sample_project/song.mp3`
2. **Add ONE singer portrait** to `singer_ref_images/` (e.g. `singer_01.png`)
   - Front-facing, head+shoulders visible
   - Neutral expression, even lighting
   - Resolution ≥ 768×768
3. **Add 3 B-roll images** to `broll_ref_images/`
4. **Set env vars**:
   ```bash
   export ZAI_API_KEY="your_key"
   export COMFYUI_INPUT_DIR="/path/to/ComfyUI/input"
   ```
5. **Run**:
   ```bash
   cd /path/to/comfyui-mv-latentsync-workflow
   
   python scripts/vocal_isolation.py \
       --audio sample_project/song.mp3 \
       --output-dir sample_project/output/separated/
   
   python scripts/audio_analysis.py \
       --audio sample_project/song.mp3 \
       --vocals sample_project/output/separated/vocals.wav \
       --output sample_project/output/beatmap.json
   
   python scripts/llm_shotlist_latentsync.py \
       --beatmap sample_project/output/beatmap.json \
       --lyrics sample_project/lyrics_example.txt \
       --singer-dir sample_project/singer_ref_images/ \
       --broll-dir sample_project/broll_ref_images/ \
       --output sample_project/output/shotlist.json \
       --send-images
   
   python scripts/batch_render_latentsync.py \
       --shotlist sample_project/output/shotlist.json \
       --singer-dir sample_project/singer_ref_images/ \
       --broll-dir sample_project/broll_ref_images/ \
       --vocals sample_project/output/separated/vocals.wav \
       --output-dir sample_project/output/scenes/ \
       --post-fx
   
   python scripts/assemble_mv.py \
       --shotlist sample_project/output/shotlist.json \
       --render-log sample_project/output/scenes/render_log.json \
       --audio sample_project/song.mp3 \
       --output sample_project/output/music_video_final.mp4 \
       --title "LatentSync Test" \
       --artist "Me"
   ```

6. **Check** `sample_project/output/music_video_final.mp4`

For a 30-second test clip with 1-2 singer scenes + 2-3 broll scenes, the full pipeline should complete in 15-25 minutes.

## What to verify

- ✅ `separated/vocals.wav` is clean (only vocals, no instruments)
- ✅ `shotlist.json` has 1-2 `singer_latentsync` scenes + 2-3 `broll` scenes
- ✅ Each singer scene produced 3 files: `base_singer_NNN.mp4`, `lipsync_latentsync_NNN.mp4`, `scene_NNN_final.mp4`
- ✅ `lipsync_latentsync_NNN.mp4` shows mouth synced to vocals
- ✅ `scene_NNN_final.mp4` has applied CodeFormer + color grading + grain
- ✅ Final video has synced audio (lip movements match vocals during verses)
- ✅ B-roll scenes play during choruses and instrumental breaks

## Debugging

For each singer scene, the 3 intermediate files are kept for debugging:
- `base_singer_NNN.mp4` — base video BEFORE LatentSync (singer's mouth should be neutral)
- `lipsync_latentsync_NNN.mp4` — AFTER LatentSync (mouth synced to vocals)
- `scene_NNN_final.mp4` — AFTER post-FX (final color graded version)

If the final video looks wrong, inspect each intermediate to find which stage failed:
- Mouth not moving? Check `lipsync_latentsync_NNN.mp4` (LatentSync issue)
- Face looks wrong? Check `base_singer_NNN.mp4` (IP-Adapter / LTX issue)
- Colors wrong? Check `scene_NNN_final.mp4` (LUT / post-FX issue)
