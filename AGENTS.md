# Agent Memory — ComfyUI Ideogram + SUPIR

## Workflow Files

| File | Description |
|------|-------------|
| `/home/ericr/Downloads/c0a2671af7c4_supir.json` | KJNodes-based Ideogram v4 + native SUPIR upscaling (active) |
| `/home/ericr/Downloads/c0a2671af7c4.json` | Original download (LLM prompt template version) |
| `/home/ericr/ComfyUI/user/default/workflows/c0a2671af7c4.json` | Base KJNodes workflow (source for KJNodes version) |

## Key Architecture

### Ideogram v4 Pipeline
- UNET: `ideogram4_fp8_scaled.safetensors` (diffusion_models/)
- Uncond UNET: `ideogram4_unconditional_fp8_scaled.safetensors` (diffusion_models/)
- CLIP: `qwen3vl_8b_fp8_scaled.safetensors` (text_encoders/)
- VAE: `flux2-vae.safetensors` (vae/)
- Prompt source: KJNodes `Ideogram4PromptBuilderKJ` (visual bbox editor)

### SUPIR Upscaling Branch
- SDXL checkpoint: `sd_xl_base_1.0.safetensors` (checkpoints/)
- SUPIR model: `SUPIR-v0Q_fp16.safetensors` (model_patches/, downloaded via aria2c)
- VAE: `ae.safetensors` (vae/, SDXL VAE from checkpoint)
- Nodes: CheckpointLoaderSimple → ModelPatchLoader → SUPIRApply → KSampler → VAEDecode → SaveImage

## CFG Tweaks
- **Recommended: Turbo preset, CFG 3.0** — best balance of quality/speed across all subjects
- Turbo cfg3 is clean and natural-looking on landscapes, portraits, animals, and still life
- Higher CFG (5, 7) overcooks portraits — too pronounced, harsh contrast/artifacts
- **Absurd/surreal prompts: CFG 5 wins** — more prompt adherence helps bizarre concepts, turbo cfg5 also acceptable
- Default preset (20 steps) marginal quality gain over Turbo (12 steps) not worth 1.7x time
- DualModelGuider CFG: 7 (unchanged, internal to Ideogram 4)

## Runtime Notes
- Launched with: `--lowvram --use-flash-attention` (required for 16GB VRAM)
- Flash attention confirmed active
- comfy-aimdo + DynamicVRAM detected

## Resolution
- KJNodes: 896×1152 (portrait, 5:7.2 ratio)
- EmptyLatentImage: matches at 896×1152 (same-res SUPIR)
- For upscaling: add ImageScale node before SUPIRApply, set both it and EmptyLatentImage to target

## Dimension Error Fix
- Error: `tensor a (192) != tensor b (112) at dim 3` — hint latent spatial dims must match target
- Hint latent encodes at VAE 8x compression: image_px / 8 = latent_dim
- Target latent matches EmptyLatentImage / 8
- Fix: ensure both hint image and empty latent are the SAME pixel resolution

## Models
- `SUPIR-v0Q_fp16.safetensors` (2.5GB, from Kijai/SUPIR_pruned via aria2c)
- `sd_xl_base_1.0.safetensors` (6.5GB, pre-existing)
- `ae.safetensors` (320MB, pre-existing SDXL VAE)

## Ideogram 4.0 Parameter Sweep Experiment

### Overview
Automated experiment that runs 36 prompts × 4 settings = **144 image generations** through the Ideogram 4 workflow, then builds a contact sheet for comparison.

### Pipeline
```
Natural language prompt
  → LLM (OpenCode mimo-v2.5) → Ideogram 4 structured JSON
  → ComfyUI API → Image generation
  → Contact sheet (PNG + HTML)
```

### Files

| File | Purpose |
|------|---------|
| `~/ComfyUI/experiment.py` | Main experiment script |
| `~/ComfyUI/prompt_builder.py` | LLM prompt→JSON transform (reusable) |
| `~/ComfyUI/experiment_template.json` | ComfyUI prompt format template (from history) |
| `~/ComfyUI/output/experiment/prompts.json` | Cached LLM transforms (raw + structured JSON) |
| `~/ComfyUI/output/experiment/*.png` | Generated images |
| `~/ComfyUI/output/experiment/contact_sheet.png` | PNG contact sheet |
| `~/ComfyUI/output/experiment/contact_sheet.html` | HTML contact sheet (click-to-copy prompts) |

### Prompt Format (Ideogram 4 JSON Schema)
```json
{
  "high_level_description": "one or two sentence summary",
  "style_description": {
    "aesthetics": "style keywords",
    "lighting": "lighting description",
    "photo": "camera/lens details",
    "medium": "photograph",
    "color_palette": ["#RRGGBB"]
  },
  "compositional_deconstruction": {
    "background": "scene shell description",
    "elements": [
      {"type": "obj", "bbox": [y_min, x_min, y_max, x_max], "desc": "element description"}
    ]
  }
}
```
- `compositional_deconstruction` is **required**
- Bbox: normalized 0-1000, `[y_min, x_min, y_max, x_max]`, origin top-left
- Style key order: `aesthetics`, `lighting`, `photo`/`art_style`, `medium`, `color_palette`
- Reference: https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md

### Experiment Grid

**6 subjects × 6 variations = 36 prompts:**

| Subject | Variations |
|---------|-----------|
| landscape | ocean, mountain, desert, forest, arctic, urban |
| portrait | studio, outdoor, elderly, child, group, profile |
| animals | wildlife, pet, bird, marine, insect, fantasy |
| absurd | surreal, scale, mashup, physics, time, dreamscape |
| still_life | food, flowers, vintage, organic, tech, mineral |
| architecture | modern, ancient, interior, exterior, ruin, futuristic |

**4 settings:**

| Name | Preset | CFG | Steps | Mu | Std |
|------|--------|-----|-------|----|-----|
| turbo_cfg3 | Turbo | 3.0 | 12 | 0.5 | 1.75 |
| turbo_cfg5 | Turbo | 5.0 | 12 | 0.5 | 1.75 |
| default_cfg3 | Default | 3.0 | 20 | 0.0 | 1.75 |
| default_cfg7 | Default | 7.0 | 20 | 0.0 | 1.75 |

### Key ComfyUI Nodes (Prompt Format)

| Node ID | Type | What it controls |
|---------|------|-----------------|
| `209` | Ideogram4PromptBuilderKJ | HLD, style, background, elements, bboxes |
| `134:115` | PrimitiveStringMultiline | Raw user prompt (clear when using HLD directly) |
| `98:157` | CFGOverride | CFG value |
| `98:156` | CustomCombo | Preset (Quality/Default/Turbo) |
| `98:17` | Ideogram4Scheduler | Steps, mu, std (wired from preset via JSON extraction) |
| `98:18` | RandomNoise | Seed |
| `98:24` | CLIPTextEncode | Positive prompt ( fed from node 209) |
| `158` | SaveImage | Output |

### Workflow Format
ComfyUI API `/prompt` expects **prompt format** (flat dict), NOT workflow format:
```json
{
  "node_id": {
    "inputs": {"key": value, "key2": ["other_node_id", slot]},
    "class_type": "NodeType",
    "_meta": {"title": "Node Title"}
  }
}
```
Subgraph nodes use `"subgraph_id:node_id"` notation (e.g., `"98:157"`).

### Running
```bash
# Use conda flash env (has flash-attn 2.8.3)
conda run -n flash python3 experiment.py
```
- Conda env at: `/home/ericr/miniconda3/envs/flash/` (PyTorch 2.12.0+cu130, flash-attn 2.8.3)
- First run: LLM transforms all 36 prompts, saves to `prompts.json`
- Subsequent runs: loads cached prompts, fills missing via LLM, skips existing images
- To re-transform: delete `prompts.json`
- Sequential processing: one generation at a time (~30s each, ~72 min total)

### Headless Single Image
```python
import json, sys
sys.path.insert(0, "/home/ericr/ComfyUI")
from prompt_builder import query_opencode, validate_and_fix

raw = query_opencode("A golden retriever on a skateboard")
caption = validate_and_fix(json.loads(raw))
# Patch experiment_template.json node "209" with caption, POST to /prompt
```

### Safety Filter Notes
- Ideogram 4 has built-in safety filter (not a ComfyUI node)
- False positives higher with non-JSON prompts
- Using structured JSON reduces false positive rate
- Reference: prompting.md safety section

## Ideogram 4.0 Text Rendering Experiment

### Overview
Tests literal text rendering in images — posters, banners, signs, packaging, covers, and real-life scenes with text. Single setting: **Turbo, CFG 3.0**. 40 prompts total.

### Files

| File | Purpose |
|------|---------|
| `~/ComfyUI/experiment_text.py` | Main experiment script |
| `~/ComfyUI/experiment_template.json` | ComfyUI prompt format template (shared) |
| `~/ComfyUI/output/experiment_text/prompts.json` | Cached LLM transforms |
| `~/ComfyUI/output/experiment_text/*.png` | Generated images |
| `~/ComfyUI/output/experiment_text/contact_sheet.png` | PNG contact sheet |
| `~/ComfyUI/output/experiment_text/contact_sheet.html` | HTML contact sheet (click-to-copy prompts) |

### Running
```bash
/home/ericr/miniconda3/envs/flash/bin/python3 ~/ComfyUI/experiment_text.py
```
- Same conda flash env as parameter sweep experiment
- Cached prompts: loads existing, transforms missing, skips existing images
- Sequential processing (~30s per image, ~20 min for 10 new prompts)
- ComfyUI must be running: `main.py --lowvram --use-flash-attention`

### Text Rendering in Ideogram 4 JSON Schema

Text uses `"type": "text"` elements with a `"text"` field for **literal text** to render:

```json
{
  "compositional_deconstruction": {
    "elements": [
      {
        "type": "text",
        "bbox": [50, 100, 150, 900],
        "text": "THE EXACT TEXT TO RENDER",
        "desc": "bold white sans-serif font, centered, large size",
        "color_palette": ["#FFFFFF", "#000000"]
      }
    ]
  }
}
```

Key rules:
- `"type": "text"` — tells the model this is text, not an object
- `"text"` field — the literal string rendered on the image (spell exactly right)
- `"desc"` — describes font style, color, material, size
- `"bbox"` — normalized 0-1000, `[y_min, x_min, y_max, x_max]`
- `"color_palette"` — per-element hex colors for precise text color control
- **Use `"medium": "graphic_design"`** for posters/banners/signs (not `"photograph"`)
- **Use `"art_style"` instead of `"photo"`** for non-photographic text layouts
- Combine text elements with `"type": "obj"` elements for background imagery

### Prompt Categories (40 total)

| Category | Count | Examples |
|----------|-------|---------|
| Movie/event posters | 5 | sci-fi, horror, comedy, concert, festival |
| Book/album covers | 5 | thriller novel, fantasy novel, hip-hop album, electronic album, magazine |
| Signs/neon | 5 | bar sign, diner sign, bakery, street sign, protest sign |
| Packaging/products | 5 | wine label, coffee bag, soda can, cereal box, tech box |
| Tickets/passes/cards | 4 | concert ticket, boarding pass, business card, movie stub |
| Banners/signage | 3 | grand opening, marathon finish, welcome mat |
| Digital/UI | 3 | app icon, game cover, t-shirt graphic |
| **Real-life scenes** | **10** | Tokyo street, diner, construction site, farmer's market, record store, bus stop, gym, laundromat, bookstore, airport gate |

### Findings
- **Turbo CFG 3 works well for text** — clean rendering, good prompt adherence
- **Simple text (1-3 words) renders reliably** — signs, titles, single phrases
- **Long text (4+ lines)** works but occasionally garbles characters
- **Graphic design medium** is key — using `"medium": "graphic_design"` + `"art_style"` for posters/banners/signs
- **Real-life scenes** are more compelling than isolated product shots — text appears naturally in context
- **Horror prompts** tend to trigger safety filter (false positives)
- **Kanji/CJK characters** render correctly when specified in the `"text"` field
- **Multiple text elements** in one image work — the model handles spatial layout

### LLM Transform for Text
The text experiment uses a specialized `text_render_transform()` function with a system prompt that:
- Instructs the LLM to use `"type": "text"` elements (not just `"obj"`)
- Emphasizes `"medium": "graphic_design"` for non-photo text layouts
- Specifies key order: `aesthetics`, `lighting`, `medium`, `art_style`, `color_palette`
- Returns raw JSON string for direct patching of node 209

### Known Bugs & Fixes

**DynamicCombo API format (node 209):**
- `Ideogram4PromptBuilderKJ.style` is a v3 `DynamicCombo` input
- API format: `"style": "photo"` (plain string) + `"style.photo": "85mm f/1.4"` (dotted sub-input key)
- NOT a dict — passing `{"style": "photo", "photo": "..."}` fails validation
- The API framework wraps string values into dicts via `build_nested_inputs()` using `dynamic_paths`

**LLM JSON parsing failures (text experiment):**
- Some LLM outputs fail JSON parse: `Expecting ',' delimiter` at high char positions
- This happens when the LLM produces complex nested JSON with many text elements
- Fallback: sends raw prompt text as `high_level_description` (no structured elements)
- Affects ~30% of prompts on first try; re-running deletes `prompts.json` to re-transform
- The raw prompt fallback still works but loses text-specific element control

**SUPIR template contamination:**
- `experiment_template.json` originally contained a separate SUPIR upscaling branch (nodes 200-208)
- SUPIR had a hardcoded prompt (`"high quality, detailed, sharp, 8k, masterpiece"`)
- The experiment's image download loop grabbed SUPIR output (node 208) instead of Ideogram 4 (node 158)
- Fix: removed SUPIR nodes, explicitly target node 158 in download logic

## LTX 2.3 GGUF Lip-Sync Pipeline

### Overview
Lip-synced video generation using LTX 2.3 + IC-LoRA on 16GB VRAM. Reference image + vocals audio → 4s lip-synced video at 704×1280.

### Files

| File | Purpose |
|------|---------|
| `~/ComfyUI/ltx_lipsync_fixed.py` | Main lip-sync script (CLI tool, current active) |
| `~/ComfyUI/ltx_lipsync_test.py` | Original script (GGUF-only, no IC-LoRA, deprecated) |
| `~/ComfyUI/ltx_fp8_test.py` | FP8 checkpoint test script |
| `~/ComfyUI/ltx_fullckpt_test.py` | Full checkpoint test script |
| `~/ComfyUI/input/reference_pop_singer_v2.png` | Reference portrait (704×1280) |
| `~/ComfyUI/input/sunburned_smile_vocals_8s.wav` | Extracted vocals (8s) |
| `~/ComfyUI/output/ltx_lipsync_*.mp4` | Generated videos (12+ iterations) |

### Running
```bash
/home/ericr/miniconda3/envs/flash/bin/python3 ~/ComfyUI/ltx_lipsync_fixed.py \
  --image reference_pop_singer_v2.png \
  --audio sunburned_smile_vocals_8s.wav \
  --duration 4 --seed 53 --lora 0.2 --i2v 0.5
```

### Key Parameters
| Param | Current | Notes |
|-------|---------|-------|
| `--width` | 704 | Must be divisible by 32 |
| `--height` | 1280 | Must be divisible by 32; height/32 must be even for IC-LoRA |
| `--duration` | 4 | 4s best for coherence; longer = end frame degrades |
| `--lora` | 0.2 | Distilled LoRA strength; lower = less face artifacts |
| `--i2v` | 0.5 | Image-to-video strength; lower = more motion freedom |
| `--seed` | varies | Random seed for reproducibility |

### Model

| Component | File | Size | VRAM |
|-----------|------|------|------|
| UNET (Q4_K_M) | `LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf` | 13 GB | ~9 GB |
| UNET (Q6_K, fallback) | `LTX-2.3-22B-distilled-1.1-Q6_K.gguf` | 20 GB | ~13 GB |
| Text Encoder | `gemma_3_12B_it_fp8_e4m3fn.safetensors` | 13 GB | offloaded |
| Text Projection | `ltx-2-3-22b-text_encoder.safetensors` | 2.2 GB | — |
| IC-LoRA | `ltx-2.3-22b-ic-lora-lipdub.safetensors` | 70 MB | — |
| Distilled LoRA | `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.1 GB | patched |
| Audio VAE | `ltx-2.3-22b-distilled_audio_vae.safetensors` | 348 MB | ~0.3 GB |
| Video VAE | `ltx-2.3-22b-distilled_video_vae.safetensors` | 1.4 GB | ~0.3 GB |
| Latent Upscaler | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 950 MB | not used yet |

### Workflow Architecture (ltx_lipsync_fixed.py)

```
UnetLoaderGGUF (Q4_K_M) → LoraLoaderModelOnly (distilled) → LTXICLoRALoaderModelOnly
LTXAVTextEncoderLoader → clip
VAELoader → video_vae
LTXVAudioVAELoader → audio_vae

LoadImage → LTXVPreprocess → image
LoadAudio → LTXVAudioVAEEncode → audio_latent
EmptyLTXVLatentVideo → latent (352×640 = 704×1280/2)
LTXVImgToVideoInplace (strength=0.5) → video_latent

CLIPTextEncode(pos + neg) → IC-LoRA guide(cond + latent) → conditioned

LTXVConcatAVLatent(video + audio) → av_latent
LTXVSetAudioVideoMaskByTime (mask_video:1, mask_audio:0, SOTAI) → masked_av

LTXVConditioning → conditioning
RandomNoise + KSamplerSelect(euler_ancestral_cfg_pp) + ManualSigmas(8-step) + CFGGuider
SamplerCustomAdvanced → sampled_av

LTXVSeparateAVLatent → video + audio latents
LTXVSpatioTemporalTiledVAEDecode(spatial_tiles=2, temporal_tile_length=16, last_frame_fix)
LTXVAudioVAEDecode → audio
CreateVideo + LTXMotionSaveVideo → .mp4
```

### Architecture Fixes (vs original ltx_lipsync_test.py)

| Fix | Problem | Solution |
|-----|---------|----------|
| Resolution | 720×1280 not divisible by 32 | 704×1280 (704/32=22) |
| Audio mask | `SolidMask+SetLatentNoiseMask` wrong shape for 4D audio | `LTXVSetAudioVideoMaskByTime` (SOTAI: mask_audio=False, preserve signal) |
| Character drift | No identity preservation | IC-LoRA guide with lipdub LoRA |
| Sampling | Wrong sigmas/sampler | `euler_ancestral_cfg_pp` + 8-step schedule |
| VAE decode | STD VAEDecodeTiled had boundary artifacts | `LTXVSpatioTemporalTiledVAEDecode` with proper normalization |
| End frame | Coherence decay at tail | `last_frame_fix=True`, temporal_overlap=2 |

### Experiment Summary (12 iterations)

| # | Model | I2V | IC-LoRA | LoRA | Duration | VAE | Result |
|---|-------|-----|---------|------|----------|-----|--------|
| 01 | Q6_K | 0.6 | frame0@1.0 | 0.3 | 6s | VAEDecodeTiled | OOM |
| 02 | Q6_K | 0.6 | frame0@1.0 | 0.3 | 5s | SpatioTemporal | end weird, lip sync good |
| 03 | Q6_K | 0.6 | dual(f0@1.0+f80@0.5) | 0.3 | 5s | SpatioTemporal | static, too constrained |
| 04 | Q6_K | 0.6 | dual(f0@1.0+f100@0.3) | 0.3 | 5s | SpatioTemporal | end still off, static |
| 05 | Q6_K | 0.6 | single f0@1.0 | 0.3 | 4s | VAEDecodeTiled | **best so far** |
| 06 | Q6_K | 0.6 | single f0@1.0 | 0.3 | 4s | VAEDecodeTiled | 4-step refine sigmas = trash |
| 07 | Q6_K | 0.6 | single f0@1.0 | 0.3 | 4s | VAEDecodeTiled | good, end slightly off |
| 08 | **Q4_K_M** | 0.6 | single f0@1.0 | 0.3 | 4s | VAEDecodeTiled | better, "few only now" |
| 09 | Q4_K_M | 0.6 | dual(f0@0.7+f88@0.2) | 0.3 | 4s | VAEDecodeTiled | stiff |
| 10 | Q4_K_M | **0.5** | single f0@**0.5** | **0.2** | 4s | VAEDecodeTiled | expressive, end "like negative" |
| 11 | Q4_K_M | 0.5 | single f0@0.5 | 0.2 | 4s | **SpatioTemporal** | expressive, end still bad |
| 12 | Q4_K_M | 0.5 | **single f88@0.35** (end only) | 0.2 | 4s | SpatioTemporal | ? (latest) |

### Remaining Issues

- **End frame weirdness**: Last 1-3 frames have artifacts (color inversion/"negative" or static/distorted). Present in all iterations. Suspected causes:
  1. LTX temporal coherence budget exhausted at ~4s
  2. SOTAI mask boundary at end_time=4.0s exceeding actual fc=89 frames (3.708s)
  3. VAE decode tile boundary at sequence end

- **IC-LoRA motion trade-off**: Higher strength = identity preserved but motion constrained. Lower strength = expressive but identity drifts. End-only guide may help but not fully tested.

### Untried Approaches
- **Two-stage latent upscale**: Official ComfyUI workflow approach (half-res gen → latent upscaler 2x → refinement pass). Would need restructuring, requires ~7GB extra VRAM (now available with Q4)
- **Post-process last frame**: Crop last 2 frames and blend/duplicate previous frame
- **Q5_K_M model**: Available from unsloth/LTX-2.3-GGUF, about 16GB file
- **LTX 2.3 native workflows**: Pre-built templates at ComfyUI → Workflow Templates → filter "LTX 2.3"
- **MSR LoRA approach**: liconstudio/ComfyUI-Licon-MSR for multi-subject reference with MSR LoRA

### ComfyUI Launch
```bash
systemctl --user start comfyui
# Flags: --lowvram --use-flash-attention (systemd service)
```
