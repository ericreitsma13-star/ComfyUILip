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

## Music Video Pipeline (LTX 2.3 I2V Only) — Heatwave

> **Active build plan: `plan_heatwave_music_video.md`**
> Lyric-driven broll spec, 12 unique Ideogram 4 refs, per-clip LTX params.
> Step 0+ not yet executed. Read that file first if you are picking up this work.

### Goal
Longform music video for Heatwave.wav (180s, 123 BPM, single singer, realism).
Use the **same proven LTX 2.3 + IC-LoRA pipeline** for both singer and broll
scenes (no SVI / Video-Infinity — see "Long-Video Path Abandoned" below).

### Current Architecture (as of Jun 24 2026)
- **All scenes use LTX 2.3 Q6_K** with the same prompt/LoRA/encoder stack as
  the 00026 winner
- Singer scenes: `ltx_lipsync_fixed.py` (audio-driven lip sync) — **singer is good, do not touch**
- B-roll scenes: `broll_generate.py` (same model, no audio path) — **being replaced by lyric-driven v2 (see plan)**
- Driver scripts: `singer_heatwave.py` and `broll_heatwave.py` orchestrators
- Workflow template exported from the 00026 MP4: `workflow_ltx_00026_*.json`

**Active working pipeline (laptop, NVIDIA RTX 4090 16GB):**
- Singer scenes: `ltx_lipsync_fixed.py` → LTX 2.3 Q6_K + distilled LoRA + audio VAE
  - 7.5s clips at 960×544, LoRA 0.8, I2V 0.6, CFG 3.5, euler, linear_quadratic, 15 steps
  - Ref: `heatwave_refs/portrait_v2.png` (Z-Image natural-skin)
  - Audio: `segment_NNN.wav` (7.5s chunks)
  - Outputs: `output/ltx_lipsync_00018-audio.mp4` through `00055-audio.mp4` (~7.4s each, ~1-2.5MB)
  - 00026 = best result (used segment_010.wav, seed 50)
- B-roll scenes: `broll_generate.py` → same model, no audio path
  - Refs: `broll_refs/{city_aerial,neon_bokeh,dance_floor,feet_puddle,skyline}.png`
  - 7.5s clips at 960×544, seed varies per clip
  - Driven by `broll_heatwave_serial.py` (one clip at a time, restart ComfyUI between)
  - 12 clips total at ~5 min each = ~70-90 min

### Long-Video Path Abandoned
- **SVI (Stable Video Infinity)** — error-recycling LoRA, 22B Q6_K OOMs on 16GB
- **Video-Infinity** (Yuanshi9815) — multi-GPU distributed, won't work on single GPU
- **Wan 2.2 I2V** — too large for laptop, only on cloud
- Strix Halo (96GB unified) could run SVI but user is on RTX 4090
- **Conclusion: long-video generation methods aren't viable for this hardware.
  Stick with the proven LTX 2.3 I2V + stitching approach.**

### Critical Pattern: Restart ComfyUI Between Clips
- 22B Q6_K model OOMs on 16GB if ComfyUI runs back-to-back clips (state leak)
- **Solution**: Restart ComfyUI between every broll render:
  ```bash
  systemctl --user restart comfyui
  # wait for /system_stats to return
  ```
- Built into `broll_heatwave_serial.py` (--no-restart flag for testing)
- Without restart, all clips after the first OOM with "Allocation on device 0 would exceed allowed memory"

### Files (Final, Jun 24 2026)

| File | Purpose |
|------|---------|
| `ltx_lipsync_fixed.py` | **Singer scenes** — proven 00026 winner, audio-driven lip sync |
| `singer_heatwave.py` | Singer scene orchestrator (18 clips via ltx_lipsync_fixed.py) |
| `broll_generate.py` | **B-roll scenes** — LTX 2.3 I2V, no audio path |
| `broll_heatwave.py` | B-roll orchestrator (batch mode, 12 clips in queue) |
| `broll_heatwave_serial.py` | **B-roll serial mode** — one clip at a time with ComfyUI restart |
| `beat_stitch.py` | Final assembly with crossfades and color matching |
| `workflows_export.py` | Export 00026 winning workflow as reusable API JSON templates |
| `workflow_ltx_00026_lipsync.json` | Exported 00026 winning workflow (24 nodes, audio) |
| `workflow_ltx_00026_broll.json` | Exported broll variant (17 nodes, audio stripped) |

**Removed (SVI/abandoned):** `audio_analyzer.py`, `lyrics_parser.py`, `svi_workflow.py`,
`svi_runner.py`, `broll_director.py`, `music_video_director.py`, `test_svi_chorus.py`,
`prompt_stream_generator.py`

### Models (all in /home/ericr/ComfyUI/models/)

| Model | Path | Size | Used by |
|-------|------|------|---------|
| `LTX-2.3-22B-distilled-1.1-Q6_K.gguf` | `diffusion_models/` | ~13GB | Singer + B-roll UNet |
| `ltx-2.3-22b-distilled-1.1.safetensors` | `diffusion_models/` | ~20GB | Singer audio VAE |
| `ltx-2.3-22b-distilled_video_vae.safetensors` | `vae/` | ~250MB | Singer + B-roll video VAE |
| `gemma_3_12B_it_fp8_e4m3fn.safetensors` | `text_encoders/` | ~6GB | Text encoder (T5) |
| `ltx-2-3-22b-text_encoder.safetensors` | `text_encoders/` | ~2GB | Text projection |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | `loras/` | ~7GB | Distilled LoRA |
| `wan2.1-i2v-14b-480p-Q4_K_M.gguf` | `diffusion_models/` | ~7GB | SVI (unused, kept) |
| `Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors` | `diffusion_models/` | 17GB | SVI (unused, kept) |
| `svi_*.safetensors` | `loras/svi_wan21/version-1.0/` | 2.4GB each | SVI (unused, kept) |

### Music Video Plan for Heatwave (180s, 123 BPM)

| Scene | Section | Type | Duration | Model | Ref |
|-------|---------|------|----------|-------|-----|
| 0 | intro | broll | 12s | LTX 2.3 I2V | city_aerial |
| 1 | verse1 | singer | 24s | LTX 2.3 + IC-LoRA | singer_01 (portrait_v2) |
| 2 | prechorus1 | singer | 12s | LTX 2.3 + IC-LoRA | singer_01 |
| 3 | chorus1 | broll | 24s | LTX 2.3 I2V | feet_puddle |
| 4 | verse2 | singer | 24s | LTX 2.3 + IC-LoRA | singer_01 |
| 5 | prechorus2 | singer | 12s | LTX 2.3 + IC-LoRA | singer_01 |
| 6 | chorus2 | broll | 24s | LTX 2.3 I2V | dance_floor |
| 7 | bridge | broll | 18s | LTX 2.3 I2V | skyline |
| 8 | solo | broll | 6s | LTX 2.3 I2V | neon_bokeh |
| 9 | chorus3 | singer | 18s | LTX 2.3 + IC-LoRA | singer_01 |
| 10 | outro | broll | 6.3s | LTX 2.3 I2V | city_aerial |

### Broll Concept Bank

| Concept | Mood | Motion | File |
|---------|------|--------|------|
| city_aerial | atmospheric, mysterious | slow aerial push-in over neon city | `broll_refs/city_aerial.png` |
| neon_bokeh | intimate, moody | shallow DOF pan over bokeh | `broll_refs/neon_bokeh.png` |
| dance_floor | rising tension | camera pushes toward dance floor | `broll_refs/dance_floor.png` |
| feet_puddle | triumphant, kinetic | low-angle feet walking through neon puddle | `broll_refs/feet_puddle.png` |
| skyline | expansive, contemplative | slow cinematic reveal | `broll_refs/skyline.png` |

### Settings (SVI on Wan 2.1 I2V 480P)

| Param | Value | Notes |
|-------|-------|-------|
| `width` | 832 | Native SVI training res |
| `height` | 480 | Native SVI training res |
| `num_frames` | 81 | 3.24s per clip @ 25fps |
| `cfg` | 5.0 | Per SVI FAQ Q3 |
| `lora_strength` | 1.0 | Full SVI strength |
| `vram_blocks_to_swap` | 20 | For 16GB; reduce to 5-10 for 48GB |
| `motion_latent_count` | 5 (Film) / 1 (Shot) | Per SVI FAQ Q5 |
| `steps` | 30 | Default |
| `sampler` | euler / uni_pc | Both work |
| `scheduler` | normal | |
| `flow_shift` | 5.0 | For 480p |
| `seed` | 42 + i*137 | Different per clip (CRITICAL) |

### Test Commands

```bash
# Smoke test: render SVI-Film on chorus 1 (dry run)
python test_svi_chorus.py --dry-run

# Real render (after Wan 2.1 I2V download completes)
python test_svi_chorus.py

# Full MV orchestration
python music_video_director.py \
  --audio input/Heatwave.wav \
  --lyrics input/Heatwave_lyrics.txt \
  --singer-ref input/heatwave_singer.png \
  --broll-refs-dir output/broll_refs \
  --output-dir output/heatwave_v2 \
  --url http://<cloud-url>:8188

# Just broll, skip singer (faster iteration)
python music_video_director.py \
  --audio input/Heatwave.wav --lyrics input/Heatwave_lyrics.txt \
  --skip-singer --output-dir output/heatwave_broll_test
```

### Test Results (Jun 23 2026)

✅ **Pipeline works end-to-end** on laptop with Strix Halo 16GB VRAM:
- `audio_analyzer.py` — BPM 123.05, 16 sections detected (after noise-section filtering)
- `lyrics_parser.py` — clean parsing of [Section] tags
- `svi_workflow.py` — valid ComfyUI workflow with VACE + I2V + SVI LoRA pattern
- `svi_runner.py` — queues to ComfyUI, runs SVI-Film, downloads output
- `broll_director.py` — scene plan + prompt stream generation
- `music_video_director.py` — full orchestration (LTX singer + SVI broll)

**Hardware notes:**
- Wan 2.1 14B I2V fp8 (17GB) **OOMs on 16GB VRAM** even with full block swap
- **Use the GGUF version** `wan2.1-i2v-14b-480p-Q4_K_M.gguf` (~7GB) for 16GB VRAM
- On cloud 48GB VRAM: use fp8 safetensors (17GB) for better quality, or GGUF for faster
- **Laptop test (Strix Halo)**: ~25-30 min per 81-frame clip with GGUF + 20 block swap
- **Cloud estimate**: 3-5 min per clip on 48GB VRAM

**Key workflow fix (svi_workflow.py):**
- Use `WanVideoModelLoader` (not `UnetLoaderGGUF`) — auto-detects format
- For GGUF: `quantization: "disabled"` (not `fp8_e4m3fn`)
- Pattern: `LoadImage → WanVideoVACEStartToEndFrame → WanVideoImageToVideoEncode → WanVideoSampler`
- Required sampler field: `riflex_freq_index: 0`
- Valid schedulers: `euler`, `unipc`, `dpm++`, `lcm`, etc. (NOT `normal`)

### SVI LoRA Strengths Found

Tested with `svi_wan21/version-1.0/svi-film-opt-10212025.safetensors`:
- Strength 1.0 (full SVI) — model dominates, strong reference-image lock
- Need to enable LoRA via `WanVideoLoraSelect` node before `WanVideoSetBlockSwap`

### Critical Notes

- **SVI LoRAs cannot use vanilla Wan 2.1 I2V workflow** — needs `WanVideoSVIProEmbeds` for chaining
- **Different seed per clip is mandatory** (community confirmed)
- **Use fp16 LoRA not quantized** (per SVI FAQ Issue #51)
- **480×832 horizontal is the only stable resolution** (training match)
- **No 121-frame clips — must use 81** (causes color degradation per community)
- **End frame weirdness** (LTX 2.3 IC-LoRA) is a base-model limit, not SVI — SVI's error recycling mitigates it
- SVI-Talk is for **speaking**, not singing. For singing: keep LTX 2.3 IC-LoRA, or wait for community SVI-Sing fine-tune

### Future Improvements (not yet built)

- Auto LLM prompt generation per scene (currently static bank)
- Color consistency pass between scenes (deflicker node exists in custom_nodes)
- Beat-aligned cuts (currently time-aligned to section boundaries)
- SVI-Sing: singing-specific LoRA (community project, not in SVI repo yet)

## Working Lip-Sync Pipeline (Jun 20-23 2026)

### What Works
Z-Image reference → LTX 2.3 (audio-conditioned) → video with motion + lip sync.

### Winning Settings (output 00026)
```bash
python3 ltx_lipsync_fixed.py \
  --image heatwave_refs/portrait_v2.png \
  --audio segment_010.wav \
  --prompt "female singer on neon-lit city street, subtle movement, dramatic neon lighting, cinematic portrait, photorealistic, detailed face, natural skin, 35mm film" \
  --duration 7.5 --seed 50 --lora 0.8 --i2v 0.6 --cfg 3.5 \
  --width 960 --height 544
```

| Param | Value | Notes |
|-------|-------|-------|
| Model | Q6_K | 20GB, better than Q4 for quality |
| LoRA strength | 0.8 | Distilled LoRA |
| I2V strength | 0.6 | Lower = preserves reference face, less distortion |
| CFG | 3.5 | Balance of prompt adherence + quality |
| Sampler | euler | Simple, fast |
| Scheduler | linear_quadratic | 15 steps |
| Resolution | 960x544 | Matches pro_clip format |
| Duration | 7.5s | 180 frames |
| Reference | `heatwave_refs/portrait_v2.png` | Z-Image natural-skin ref, LoRA 0.3/0.2 |
| Seed | 50 (base) | + `scene_index * 100 + clip_index * 13` for variation |
| Audio | `segment_NNN.wav` | 7.5s chunks from `input/` |

### Singer Scene Orchestration (`singer_heatwave.py`)

The 5 singer scenes (verse1, prechorus1, verse2, prechorus2, chorus3) need
~18 lip-sync clips total (3-4 per scene, depending on duration). Use:

```bash
# Show the plan first
python singer_heatwave.py --plan

# Render all singer scenes (sequential, ~5-6 min each = ~90-108 min)
python singer_heatwave.py --submit

# Check progress
python singer_heatwave.py --status
cat /home/ericr/ComfyUI/output/heatwave_singers/submit_log.json | jq

# Stitch when done
python singer_heatwave.py --stitch
# Output: output/heatwave_singers/singer_scenes_full.mp4
```

Segment-to-scene mapping (7.5s segments @ 123 BPM, 24fps):
- **verse1** (12-36s, 24s) → segment_001 (cut) + 002, 003, 004 (cut) = 4 clips
- **prechorus1** (36-48s, 12s) → segment_004 (cut) + 005, 006 (cut) = 3 clips
- **verse2** (72-96s, 24s) → segment_009 (cut) + 010, 011, 012 (cut) = 4 clips
- **prechorus2** (96-108s, 12s) → segment_012 (cut) + 013, 014 (cut) = 3 clips
- **chorus3** (156-174s, 18s) → segment_020 (cut) + 021, 022 + 023 (cut) = 4 clips

`segment_010.wav` (verse2, clip 1) is what produced the 00026 winner.

### Reference Image Generation (Z-Image)
```python
# Z-Image settings for reference images
UNET: z_image_turbo_bf16.safetensors
CLIP: qwen_3_4b.safetensors (type: "qwen_image")
VAE: ae.safetensors  # CRITICAL: NOT flux2-vae.safetensors
LoRAs: DarkB ZIT lora (0.3) + REDZ15_DetailDaemon (0.2)
Steps: 4, CFG: 1.5, Sampler: euler
Resolution: 960x544 for singer refs, 1920x1088 for b-roll
```

**Key finding: Z-Image VAE is `ae.safetensors`, NOT `flux2-vae.safetensors`**
Using the wrong VAE causes tensor dimension mismatch (16 vs 128 channels).

### Reference Image Style (portrait_v2 — Natural Skin)
```
positive: "close-up portrait of a young woman singing, visible skin pores,
  natural skin texture, film grain, warm golden skin, curly dark hair,
  neon pink and blue light on face, candid moment, 35mm film photography,
  photorealistic, natural imperfections"
negative: "ugly, deformed, blurry, low quality, plastic skin, airbrushed,
  smooth skin, doll-like, cartoon"
LoRA strengths: DarkB 0.3, DetailDaemon 0.2 (lower = more natural)
```

### Shot Plan (Heatwave)
- **Singer scenes**: studio_front.png / studio_34.png references
- **B-roll**: Z-Image or Ideogram 4 references at 1920x1088
- **B-roll audio**: instrumental track (not vocals) drives LTX motion
- **Final assembly**: overlay full song audio in ffmpeg

### Wav2Lip (Installed but Degrades Quality)
- Node: `ComfyUI_wav2lip` (patched: soundfile.write instead of torchaudio.save)
- Model: `wav2lip_gan.pth` (415MB) in `custom_nodes/ComfyUI_wav2lip/Wav2Lip/checkpoints/`
- Result: adds lip sync but causes eye glitches and face artifacts
- **Verdict: LTX alone produces better output than LTX + Wav2Lip**

### What Doesn't Work
- **IC-LoRA**: lip sync not convincing, 4s limit, end-frame artifacts
- **LatentSync**: post-processing mouth-mover, not singing
- **Wav2Lip**: eye glitches, face artifacts
- **Empty prompt**: static video with no motion
- **"mouth wide open" prompt**: causes face distortion
- **Wrong VAE**: `flux2-vae.safetensors` breaks Z-Image (use `ae.safetensors`)

### Full Song Assembly Formula (Working as of Jun 24 2026)

**The proven pipeline: render individual clips → concat → overlay full song audio.**

#### Step 1: Split vocals into 7.5s segments
```bash
python3 scripts/split_audio.py \
  --input output/heatwave/separated/vocals.wav \
  --output-dir input/ --segment-sec 7.5
```
⚠ **Critical**: re-split from `vocals.wav` (NOT instrumental). Instrumental
segments produce silent/bass-only clips with no lip sync motion.

#### Step 2: Render all segments with ltx_lipsync_fixed.py
```bash
# Render each segment (alternate studio_front/studio_34 for variety)
for idx in $(seq 0 23); do
  ref="heatwave_refs/studio_front.png"
  [ $((idx % 2)) -eq 1 ] && ref="heatwave_refs/studio_34.png"
  python3 ltx_lipsync_fixed.py \
    --image "$ref" --audio "segment_$(printf '%03d' $idx).wav" \
    --prompt "female singer performing in studio, singing into microphone, dramatic warm lighting, cinematic, photorealistic, detailed face, natural skin, 35mm film" \
    --duration 7.5 --seed $((50 + idx)) --lora 0.8 --i2v 0.6 --cfg 3.5 \
    --width 960 --height 544 --output output/
done
```

**⚠ Restart ComfyUI between renders** to avoid Q6_K OOM:
```bash
systemctl --user restart comfyui && sleep 30
```

#### Step 3: Concat video + overlay full song
```bash
# Create concat list (24 clips × 7.5s = 180s, matches song length)
for i in $(seq 61 84); do
  echo "file 'output/ltx_lipsync_000${i}.mp4'"
  echo "duration 7.5"
done > /tmp/concat.txt
echo "file 'output/ltx_lipsync_00084.mp4'" >> /tmp/concat.txt

# Concat + force 7.5s per clip via fps filter (fixes 177-frame drift)
ffmpeg -y -f concat -safe 0 -i /tmp/concat.txt \
  -vf "fps=24" -an -c:v libx264 -crf 18 /tmp/video.mp4

# Overlay full Heatwave.wav audio
ffmpeg -y -i /tmp/video.mp4 -i input/Heatwave.wav \
  -c:v copy -c:a aac -b:a 192k -map 0:v -map 1:a -shortest \
  output/heatwave_studio_full.mp4
```

**Key details:**
- LTX generates 177 frames (7.375s) per clip, but audio segments are 7.5s
- The `fps=24` filter stretches each clip to exactly 7.5s, fixing the 0.125s/clip drift
- Without this, sync drifts ~0.5s after 4 clips
- 24 clips × 7.5s = 180s (matches 180.28s song length)
- Strip per-clip audio (`-an`) and overlay full song — individual clip audio causes desync

#### Reference images used
- **Studio front**: `heatwave_refs/studio_front.png` (Z-Image, LoRA 0.3/0.2, 960x544)
- **Studio 3/4**: `heatwave_refs/studio_34.png` (same settings, different angle)
- Alternate refs generated with lower LoRA strength for natural skin texture

#### What was tried and abandoned
- **Wav2Lip** post-processing: adds lip sync but causes eye glitches
- **B-roll with Z-Image refs**: too generic, not lyric-literal
- **B-roll with blank reference + text prompts**: random/cartoon output
- **IC-LoRA + OmniNFT stacking**: no improvement over base pipeline
- **Two-pass GAP-style sampling**: higher VRAM, no quality gain
- **LatentSync post-processing**: mouth moves but doesn't sing

#### Remaining improvements (from plan_heatwave_music_video.md)
- Lyric-literal b-roll (12 entries with Ideogram 4 refs)
- Snap to 8n+1 frames (not 7.5s which is off-grid)
- 8-section prompt structure for better quality
- Singer reference alternates (close-up, wide) per scene type
