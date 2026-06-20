# ComfyUI LatentSync Music Video Workflow

End-to-end pipeline for **lip-synced music videos** using **LatentSync 1.5** (diffusion-based lip-sync) instead of Sonic. LatentSync is a post-processing lip-sync model — it needs an existing base video as input. This pack generates the base singer video with LTX/Wan + IP-Adapter FaceID (locks singer identity from your reference image), then LatentSync re-renders just the mouth region to sync with isolated vocals.

## Pipeline at a glance

```
Suno song (.mp3)
        │
        ▼
vocal_isolation.py            (Demucs htdemucs)
        │ separated/vocals.wav
        ▼
audio_analysis.py             (librosa + vocal_active detection)
        │ beatmap.json
        ▼
llm_shotlist_latentsync.py    ◄── LLM (GLM-4V / GPT-4o / Llama)
        │ shotlist.json (scene_type: "singer_latentsync" or "broll")
        ▼
batch_render_latentsync.py  ──► ComfyUI API ──► per-scene MP4s
        │                       │
        │            ┌──────────┴───────────┐
        │            ▼                      ▼
        │   SINGER scenes:            BROLL scenes:
        │   1. 01_base_singer_LTX/Wan  03_broll_render_LTX
        │      + IP-Adapter FaceID     04_broll_render_Wan
        │   2. 05_latentsync_apply          
        │      (re-renders mouth)           
        │   3. 06_face_restore_postfx  07_standard_postfx
        │      (CodeFormer + RIFE +         
        │       upscale + LUT + grain)      
        ▼
        render_log.json
        │
        ▼
assemble_mv.py  ────► ffmpeg ────► music_video_final.mp4
                              + 1080p / square / vertical variants
```

## Files in this pack

| Path | Purpose |
|------|---------|
| `workflows/01_base_singer_LTX_IPAdapter.json` | LTX + IP-Adapter FaceID for fast base singer shots |
| `workflows/02_base_singer_Wan_IPAdapter.json` | Wan 2.1 + IP-Adapter FaceID for final base singer shots |
| `workflows/03_broll_render_LTX.json`          | LTX-Video I2V for fast B-roll |
| `workflows/04_broll_render_Wan.json`          | Wan 2.1 I2V 1.3B GGUF for hero B-roll |
| `workflows/05_latentsync_apply.json`          | LatentSync 1.5 lip-sync on a base singer video |
| `workflows/06_face_restore_postfx.json`       | CodeFormer + RIFE + upscale + LUT + grain (singer scenes) |
| `workflows/07_standard_postfx.json`           | RIFE + upscale + LUT + grain (B-roll scenes) |
| `scripts/vocal_isolation.py`                  | Demucs htdemucs → vocals.wav + instrumental.wav |
| `scripts/audio_analysis.py`                   | librosa + vocal_active detection → beatmap.json |
| `scripts/llm_shotlist_latentsync.py`          | LLM-directed shot list with singer_latentsync/broll scene types |
| `scripts/batch_render_latentsync.py`          | ComfyUI orchestrator (3-stage singer + 2-stage broll) |
| `scripts/assemble_mv.py`                      | ffmpeg concat + Suno audio mux + social variants |
| `scripts/run_full_pipeline.sh`                | One-shot end-to-end runner |
| `config/config.example.json`                  | Editable config template |
| `reference_docs/setup_guide.docx`             | Full setup guide |
| `sample_project/`                             | Sample lyrics + placeholder ref dirs |

## Quick start (TL;DR)

```bash
# 1) Activate ComfyUI env
conda activate comfyui

# 2) Install deps
pip install demucs librosa==0.10.2.post1 soundfile numpy scipy requests torch torchaudio

# 3) Start ComfyUI
cd /path/to/ComfyUI
python main.py --enable-cors-header --lowvram --use-pytorch-cross-attention --cache-none

# 4) Convert the 7 workflow JSONs to API format (Save (API Format) in ComfyUI UI)

# 5) Set env vars
export ZAI_API_KEY="your_zai_key"
export COMFYUI_INPUT_DIR="/path/to/ComfyUI/input"

# 6) Run pipeline
python scripts/vocal_isolation.py --audio song.mp3 --output-dir separated/
python scripts/audio_analysis.py --audio song.mp3 --vocals separated/vocals.wav --output beatmap.json
python scripts/llm_shotlist_latentsync.py --beatmap beatmap.json --lyrics lyrics.txt --singer-dir singer_ref_images/ --broll-dir broll_ref_images/ --output shotlist.json --send-images
python scripts/batch_render_latentsync.py --shotlist shotlist.json --singer-dir singer_ref_images/ --broll-dir broll_ref_images/ --vocals separated/vocals.wav --output-dir output/scenes/ --post-fx
python scripts/assemble_mv.py --shotlist shotlist.json --render-log output/scenes/render_log.json --audio song.mp3 --output output/music_video_final.mp4 --title "My Song" --artist "Me"
```

For full installation steps, model download links, custom-node list, VRAM tuning, and troubleshooting, **read `reference_docs/setup_guide.docx`**.

## Why LatentSync vs Sonic?

| Aspect | Sonic | LatentSync |
|--------|-------|------------|
| Approach | Audio-driven portrait generation (1-stage) | Base video + audio→mouth re-rendering (2-stage) |
| Identity preservation | Can drift over long clips | Excellent (mouth-only modification) |
| Teeth/tongue rendering | Sometimes muddy | Cleaner |
| Base video quality | Generated by Sonic itself | Decoupled — use LTX/Wan/real footage |
| Speed (5s clip) | ~2 min | ~30-45s base + ~1 min LatentSync = ~1.5 min |
| VRAM | ~8GB (512²) | ~10-14GB (base + LatentSync) |
| Setup complexity | Simpler (1 model) | More complex (IP-Adapter + LatentSync + CodeFormer) |
| Best for | Quick lip-sync from a single portrait | High-quality lip-sync where you control the base video |

## Hardware notes (RTX 4090 mobile / 16GB VRAM)

| Stage | VRAM peak | Time per scene |
|-------|-----------|----------------|
| Demucs vocal isolation | ~3 GB | ~30s per song (one-time) |
| Base singer render (LTX + IP-Adapter) | ~10 GB | ~30-45s per 5s clip |
| Base singer render (Wan + IP-Adapter) | ~14 GB | ~12-15 min per 5s clip |
| LatentSync 1.5 apply | ~10 GB | ~1 min per 5s clip |
| CodeFormer + RIFE + upscale post-FX | ~10 GB | ~30-45s per scene |
| LTX B-roll | ~9 GB | ~45s per 5s clip |
| Wan B-roll | ~13 GB | ~12-15 min per 5s clip |
| Final ffmpeg assembly | CPU only | ~5 min total |

For a typical 12-scene MV (5 singer + 7 broll, hybrid LTX+Wan):
- **TOTAL: ~1.5-2 hours** for a full lip-sync MV with LatentSync

## License

MIT-licensed Python scripts. Models retain their respective licenses:
- LatentSync: ByteDance Apache 2.0
- IP-Adapter FaceID: Tencent CC-BY-NC (non-commercial)
- Demucs: MIT
- LTX-Video: Lightricks
- Wan 2.1: Alibaba Apache 2.0
- CodeFormer: NTU S-Lab non-commercial

⚠ **Note**: IP-Adapter FaceID has a non-commercial license. For commercial use, swap to the commercial-license IP-Adapter variants or use real footage as base video.
