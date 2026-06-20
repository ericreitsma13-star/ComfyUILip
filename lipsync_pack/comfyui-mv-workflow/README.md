# ComfyUI Music Video Workflow

End-to-end pipeline for making cinematic music videos from a Suno-generated song + a folder of reference images, on a 16GB-VRAM laptop GPU (RTX 4090 mobile, i9-13980HX, 64GB RAM).

## Pipeline at a glance

```
Suno song (.mp3)              Reference images (.png/.jpg)
        │                              │
        ▼                              │
audio_analysis.py                      │
        │ beatmap.json                 │
        ▼                              ▼
llm_shotlist.py  ◄─────── LLM (GLM-4V / GPT-4o / Llama)
        │ shotlist.json
        ▼
batch_render.py  ───────► ComfyUI API ───────► per-scene MP4s
        │                    │
        │              workflows/01_scene_render_LTX.json  (fast)
        │              workflows/02_scene_render_Wan.json   (final)
        │              workflows/03_post_fx_pipeline.json   (RIFE + upscale + LUT)
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
| `workflows/01_scene_render_LTX.json` | ComfyUI workflow — LTX-Video 2.3 I2V (fast prototype renderer) |
| `workflows/02_scene_render_Wan.json` | ComfyUI workflow — Wan 2.1 I2V 1.3B GGUF (final quality) |
| `workflows/03_post_fx_pipeline.json` | ComfyUI workflow — RIFE + deflicker + 4x upscale + LUT + grain |
| `scripts/audio_analysis.py`          | librosa beat/section/BPM analysis → beatmap.json |
| `scripts/llm_shotlist.py`            | LLM-directed shot list (calls Z.ai / OpenAI / Ollama) |
| `scripts/batch_render.py`            | Orchestrates ComfyUI API to render every scene |
| `scripts/assemble_mv.py`             | ffmpeg: concat scenes + mux Suno audio + social variants |
| `config/config.example.json`         | Editable config — paths, ComfyUI URL, LLM provider |
| `reference_docs/setup_guide.docx`    | Full setup guide with installation, model links, troubleshooting |
| `sample_project/`                    | Tiny sample to validate the pipeline end-to-end |

## Quick start (TL;DR)

```bash
# 1) Activate your ComfyUI Python env
conda activate comfyui

# 2) Install Python deps
pip install librosa==0.10.2.post1 soundfile numpy scipy requests

# 3) Start ComfyUI (in another terminal)
cd /path/to/ComfyUI
python main.py --enable-cors-header --lowvram --use-pytorch-cross-attention

# 4) Convert the 3 workflow JSONs to API format:
#    - Open each workflow in ComfyUI UI
#    - Right-click → "Save (API Format)" 
#    - Save as 01_scene_render_LTX_api.json, 02_scene_render_Wan_api.json, 03_post_fx_pipeline_api.json

# 5) Set LLM API key (choose one)
export ZAI_API_KEY="your_zai_key"           # for GLM-4V
# OR
export OPENAI_API_KEY="your_openai_key"     # for GPT-4o

# 6) Tell batch_render.py where ComfyUI's input folder is (for ref images)
export COMFYUI_INPUT_DIR="/path/to/ComfyUI/input"

# 7) Run the pipeline (from this folder)
python scripts/audio_analysis.py --audio song.mp3 --output beatmap.json
python scripts/llm_shotlist.py --beatmap beatmap.json --lyrics lyrics.txt --ref-dir ref_images/ --output shotlist.json --send-images
python scripts/batch_render.py --shotlist shotlist.json --ref-dir ref_images/ --output-dir output/scenes/ --post-fx
python scripts/assemble_mv.py --shotlist shotlist.json --render-log output/scenes/render_log.json --audio song.mp3 --output output/music_video_final.mp4 --title "My Song" --artist "Me"
```

For full installation steps, model download links, custom-node list, VRAM tuning, and troubleshooting, **read `reference_docs/setup_guide.docx`**.

## Hardware notes (RTX 4090 mobile / 16GB VRAM)

| Setting | LTX-Video | Wan 2.1 (1.3B Q5) | Wan 2.1 (14B Q4) |
|---------|-----------|-------------------|------------------|
| Max safe resolution | 768×432 | 832×480 | 704×384 |
| Max safe length (frames @ 24fps) | 97 (~4s) | 81 (~3.4s) | 49 (~2s) |
| VRAM peak | ~9GB | ~13GB | ~15GB (risky) |
| Render time per 5s clip | ~30-45s | ~12-15 min | ~25-30 min |

For full 3-4 minute music videos (12-18 scenes), expect:
- All-LTX prototyping pass: ~15-20 minutes total
- All-Wan final pass: ~3-4 hours overnight
- Post-FX pass: ~30-45 minutes total

## License & attribution

The Python scripts in this pack are MIT-licensed — modify freely. Model weights follow their respective licenses (LTX-Video = Lightricks, Wan 2.1 = Alibaba, LUTs = their respective creators).
