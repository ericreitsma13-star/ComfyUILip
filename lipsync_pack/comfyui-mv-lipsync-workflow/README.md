# ComfyUI Lip-Sync Music Video Workflow

End-to-end pipeline for making **lip-synced** music videos from a Suno-generated song, one singer reference image, and a folder of B-roll reference images. Targets a 16GB-VRAM laptop GPU (RTX 4090 mobile / i9-13980HX / 64GB RAM).

## Pipeline at a glance

```
Suno song (.mp3)
        │
        ▼
vocal_isolation.py            (Demucs htdemucs)
        │ separated/vocals.wav + instrumental.wav
        ▼
audio_analysis.py             (librosa + vocal-active detection)
        │ beatmap.json (with vocal_active flags per section)
        ▼
llm_shotlist_lipsync.py       ◄── LLM (GLM-4V / GPT-4o / Llama)
        │ shotlist.json (scene_type: "lipsync" or "broll" per scene)
        ▼
batch_render_lipsync.py  ────► ComfyUI API ────► per-scene MP4s
        │                       │
        │            ┌──────────┴───────────┐
        │            ▼                      ▼
        │     01_lipsync_render_Sonic   02_broll_render_LTX
        │     (verses — singer on cam)   03_broll_render_Wan
        │            │                      (choruses, intro, bridge, outro)
        │            ▼                      │
        │     04_face_restore_postfx  ◄─────┴── 05_standard_postfx
        │     (CodeFormer + RIFE + upscale + LUT + grain)
        ▼
        render_log.json
        │
        ▼
assemble_mv.py  ───────► ffmpeg ───────► final master MP4
        │                              + 1080p / square / vertical variants
        ▼
   music_video_final.mp4
```

## Files in this pack

| Path | Purpose |
|------|---------|
| `workflows/01_lipsync_render_Sonic.json` | ComfyUI workflow — Sonic audio-driven portrait (lip-sync) |
| `workflows/02_broll_render_LTX.json`     | ComfyUI workflow — LTX-Video I2V for fast B-roll |
| `workflows/03_broll_render_Wan.json`     | ComfyUI workflow — Wan 2.1 I2V 1.3B GGUF for hero B-roll |
| `workflows/04_face_restore_postfx.json`  | ComfyUI workflow — CodeFormer + RIFE + upscale + LUT + grain (for lip-sync scenes) |
| `workflows/05_standard_postfx.json`      | ComfyUI workflow — RIFE + upscale + LUT + grain (for B-roll scenes) |
| `scripts/vocal_isolation.py`             | Demucs htdemucs wrapper → vocals.wav + instrumental.wav |
| `scripts/audio_analysis.py`              | librosa beat/section/BPM + vocal_active detection → beatmap.json |
| `scripts/llm_shotlist_lipsync.py`        | LLM-directed shot list with lipsync/broll scene types |
| `scripts/batch_render_lipsync.py`        | Orchestrates ComfyUI API to render both lipsync + broll scenes |
| `scripts/assemble_mv.py`                 | ffmpeg: concat scenes + mux Suno audio + social variants |
| `scripts/run_full_pipeline.sh`           | One-shot end-to-end runner |
| `config/config.example.json`             | Editable config — paths, ComfyUI URL, LLM provider |
| `reference_docs/setup_guide.docx`        | Full setup guide with installation, model links, troubleshooting |
| `sample_project/`                        | Sample lyrics + placeholder dirs for refs |

## Quick start (TL;DR)

```bash
# 1) Activate ComfyUI env
conda activate comfyui

# 2) Install deps (vocal isolation needs demucs; rest is same as B-roll pack)
pip install demucs librosa==0.10.2.post1 soundfile numpy scipy requests torch torchaudio

# 3) Start ComfyUI (separate terminal)
cd /path/to/ComfyUI
python main.py --enable-cors-header --lowvram --use-pytorch-cross-attention --cache-none

# 4) Convert the 5 workflow JSONs to API format:
#    Open each in ComfyUI UI → "Save (API Format)" → save as *_api.json

# 5) Set env vars
export ZAI_API_KEY="your_zai_key"
export COMFYUI_INPUT_DIR="/path/to/ComfyUI/input"

# 6) Run pipeline
python scripts/vocal_isolation.py --audio song.mp3 --output-dir separated/
python scripts/audio_analysis.py --audio song.mp3 --vocals separated/vocals.wav --output beatmap.json
python scripts/llm_shotlist_lipsync.py --beatmap beatmap.json --lyrics lyrics.txt --singer-dir singer_ref_images/ --broll-dir broll_ref_images/ --output shotlist.json --send-images
python scripts/batch_render_lipsync.py --shotlist shotlist.json --singer-dir singer_ref_images/ --broll-dir broll_ref_images/ --vocals separated/vocals.wav --output-dir output/scenes/ --post-fx
python scripts/assemble_mv.py --shotlist shotlist.json --render-log output/scenes/render_log.json --audio song.mp3 --output output/music_video_final.mp4 --title "My Song" --artist "Me"
```

For full installation steps, model download links, custom-node list, VRAM tuning, and troubleshooting, **read `reference_docs/setup_guide.docx`**.

## Hardware notes (RTX 4090 mobile / 16GB VRAM)

| Stage | VRAM peak | Time per scene (~10s clip) |
|-------|-----------|----------------------------|
| Demucs vocal isolation | ~3 GB | ~30s per song (one-time) |
| Sonic lip-sync render | ~8 GB (512×512) | ~2 min per 10s |
| LTX B-roll render | ~9 GB (768×432) | ~45s per 5s |
| Wan B-roll render | ~13 GB (832×480) | ~12-15 min per 5s |
| Post-FX (CodeFormer + RIFE + 4x upscale) | ~10 GB | ~30-45s per scene |
| Final ffmpeg assembly | CPU only | ~5 min total |

For a typical 12-scene music video (~5 lipsync + ~7 broll), expect:
- Vocal isolation: ~30s (one-time)
- LTX B-roll pass (5 shots): ~5 minutes
- Wan B-roll pass (2 hero shots): ~30 minutes
- Sonic lip-sync (5 verses): ~10-15 minutes
- Post-FX (all 12 scenes): ~10-15 minutes
- Final assembly: ~5 minutes
- **TOTAL: ~1-1.5 hours** for a full lip-sync MV

## License

MIT-licensed Python scripts. Model weights follow their respective licenses:
- Sonic: Tencent Apache 2.0
- Demucs: MIT
- LTX-Video: Lightricks
- Wan 2.1: Alibaba Apache 2.0
- CodeFormer: NTU S-Lab
