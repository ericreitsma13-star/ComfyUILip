# Sample Project

Use this directory to validate the pipeline end-to-end before committing to a real project.

## What to add

```
sample_project/
├── lyrics_example.txt          ← already here (sample lyrics with verse/chorus/bridge)
├── singer_ref_images/          ← drop 1 singer portrait here (front-facing, head+shoulders)
├── broll_ref_images/           ← drop 3-5 B-roll images here (landscapes, objects, abstract)
└── output/                     ← pipeline output lands here
```

## Quick test run

1. **Get a 30-second Suno clip** (don't use a full song for the test — faster iteration)
2. **Save as `sample_project/song.mp3`**
3. **Add ONE singer portrait** to `singer_ref_images/` (e.g. `singer_01.png`)
4. **Add 3 B-roll images** to `broll_ref_images/` (e.g. `city_night.png`, `rain_window.png`, `neon_sign.png`)
5. **Set env vars**:
   ```bash
   export ZAI_API_KEY="your_key"
   export COMFYUI_INPUT_DIR="/path/to/ComfyUI/input"
   ```
6. **Run the pipeline**:
   ```bash
   cd /path/to/comfyui-mv-lipsync-workflow
   
   python scripts/vocal_isolation.py \
       --audio sample_project/song.mp3 \
       --output-dir sample_project/output/separated/
   
   python scripts/audio_analysis.py \
       --audio sample_project/song.mp3 \
       --vocals sample_project/output/separated/vocals.wav \
       --output sample_project/output/beatmap.json
   
   python scripts/llm_shotlist_lipsync.py \
       --beatmap sample_project/output/beatmap.json \
       --lyrics sample_project/lyrics_example.txt \
       --singer-dir sample_project/singer_ref_images/ \
       --broll-dir sample_project/broll_ref_images/ \
       --output sample_project/output/shotlist.json \
       --send-images
   
   python scripts/batch_render_lipsync.py \
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
       --title "Test Run" \
       --artist "Me"
   ```

7. **Check the result** at `sample_project/output/music_video_final.mp4`

For a 30-second test clip, the full pipeline should complete in 15-20 minutes on your RTX 4090 mobile.

## What to verify

- ✅ `separated/vocals.wav` sounds clean (only singing, no instruments)
- ✅ `beatmap.json` shows reasonable BPM and section labels
- ✅ `shotlist.json` has a mix of `lipsync` and `broll` scenes (typically 1-2 lipsync + 2-3 broll for a 30s clip)
- ✅ Each scene MP4 plays correctly with the right duration
- ✅ Final video has synced audio (lip movements match vocals)
- ✅ Social variants exist in the output directory

If any of these fail, check `reference_docs/setup_guide.docx` chapter 11 (Troubleshooting).
