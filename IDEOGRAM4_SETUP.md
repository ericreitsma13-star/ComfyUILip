# ComfyUI Ideogram 4 Setup — Session Summary (2026-06-11)

## What Was Built

### 1. Flash Attention Installation
- Installed flash-attn 2.8.3 by copying compiled `.so` from `flash` conda env to ComfyUI venv
- Added `--use-flash-attention` to comfyui.service

### 2. Ollama + Local LLM Setup
- Installed ollama at `~/.local/bin/ollama`
- Pulled qwen2.5:7b and ideogram-qwen3-14b models
- Created `~/ComfyUI/prompt_builder.py` — standalone CLI prompt builder
- Model: `opencode-go/deepseek-v4-flash` (via OpenCode Go API)
- Default: `opencode-go/mimo-v2.5`

### 3. Ideogram 4 Prompt Builder Node (Custom)
- **Location:** `~/ComfyUI/custom_nodes/ComfyUI-IdeogramPrompt/`
- **4 inputs:** preset (30 options), model (4 options), style (13 options), prompt (text)
- **Optional:** avoid (text field)
- Presets: Photorealistic Clean, Cinematic Drama, Editorial Fashion, Documentary Raw, Film Noir, Vintage Kodachrome, 70s Film, 90s Grunge, VHS Camcorder, Polaroid, Super 8, Epic Landscape, Intimate Portrait, Horror Tension, Wes Anderson, Street Photography, Night City, Studio Product, War Zone, Dreamy Ethereal, Retro 80s, B&W High Contrast, Food Photography, Automotive, Aerial Drone, Underwater, Infrared, Tilt Shift Miniature, Trash Polka, Magazine Cover
- Uses OpenCode Go API: `~/.opencode/bin/opencode run -m <model>`
- System prompt trained on Ideogram 4's JSON schema (high_level_description, style_description, compositional_deconstruction with bboxes)

### 4. KJNodes Ideogram 4 Prompt Builder
- **Pulled latest ComfyUI-KJNodes** which includes `ideogram4_nodes.py`
- **Visual bbox canvas** for drawing composition regions directly
- Replaces our custom node for better composition control
- Outputs: prompt JSON, preview image, bboxes, width/height
- Has import_json input for chaining with our custom node

### 5. GGUF Workflow (Separate)
- **Replaced ComfyUI-GGUF** with molbal fork (supports Ideogram 4 GGUF)
- Downloaded GGUF models to `~/ComfyUI/models/diffusion_models/`:
  - `ideogram4-Q4_0.gguf` (5.4 GB)
  - `ideogram4_uncond-Q4_0.gguf` (5.4 GB)
- Downloaded CLIP GGUF to `~/ComfyUI/models/clip/`:
  - `Qwen3VL-8B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` (4.8 GB)
- Downloaded LoRA to `~/ComfyUI/models/loras/ideogram/`:
  - `Realism_Engine_Ideogram4_V1.safetensors` (1 GB)
- Workflow saved: `~/ComfyUI/user/default/workflows/ideogram4_gguf.json`
- **Note:** GGUF workflow is slower than fp8, not recommended for regular use

### 6. Ideogram 4 Models (Already Installed)
- `ideogram4_fp8_scaled.safetensors` — main diffusion model
- `ideogram4_unconditional_fp8_scaled.safetensors` — unconditional model
- `qwen3vl_8b_fp8_scaled.safetensors` — text encoder
- `gemma4_e4b_it_fp8_scaled.safetensors` — text encoder
- `flux2-vae.safetensors` — VAE

## Ideogram 4 JSON Schema (Training Format)
```json
{
  "aspect_ratio": "W:H",
  "high_level_description": "scene summary",
  "style_description": {
    "aesthetics": "visual style",
    "lighting": "light sources",
    "photo": "film/camera qualities",
    "medium": "the medium",
    "color_palette": ["#hex"]
  },
  "compositional_deconstruction": {
    "background": "scene with lighting",
    "elements": [
      {
        "type": "obj"|"text",
        "bbox": [y1, x1, y2, x2],
        "desc": "detailed description",
        "color_palette": ["#hex"]
      }
    ]
  }
}
```

## Quality Presets
| Preset | Steps | Mu | Std |
|--------|-------|-----|-----|
| Quality | 48 | 0.0 | 1.5 |
| Default | 20 | 0.0 | 1.75 |
| Turbo | 12 | 0.5 | 1.75 |

## Key Learnings
- Ideogram 4 responds best to **simple, descriptive prompts** — complex scenes with multiple interacting elements confuse it
- The model was trained on **style_description** blocks (aesthetics, lighting, photo, medium, color_palette) — we were missing this initially
- **Bboxes are critical** for composition — KJNodes visual editor is the best way to control placement
- Photographic realism requires explicit camera/lens/film references in the prompt
- Presets need to override user prompt details to work correctly
- The model splits complex descriptions into separate elements — keep prompts focused on ONE clear subject
- No negative prompts in Ideogram 4 — use "avoid" directives instead

## Service Management
```bash
systemctl --user start|stop|restart comfyui.service
```
Runs on port 8188 with `--lowvram` flag and `--use-flash-attention`.
