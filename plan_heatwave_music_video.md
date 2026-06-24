# Heatwave Music Video — Lyric-Driven Build Plan

**Status:** approved, ready to execute
**Created:** 2026-06-24
**Owner:** Eric R / another agent to pick up
**Hardware:** Rog Strix G634JY, NVIDIA RTX 4090 16GB VRAM, Linux

---

## Goal

Build a coherent ~3-minute music video for `Heatwave.wav` (180.28s, 123 BPM,
single studio singer, realism). Replace the current generic mood-board b-roll
with **lyric-literal b-roll** that follows the song's actual words and imagery.
Singer scenes stay as-is (the 00026 winning settings work).

## Why this plan exists

Current broll (`broll_heatwave.py` + `input/broll_refs/*.png`) is hardcoded
to abstract concepts (city_aerial, feet_puddle, dance_floor, neon_bokeh,
skyline). None of it parses the lyrics. The result is random b-roll that
doesn't match the words. The user explicitly said: *"we have lyrics already
and a studio singer. We need to have b-roll in line (somewhat) with the
lyrics. Not some random b-roll. Outcome should be a coherent videoclip."*

This plan rewires the broll to be **literal** — a "hand on a glowing fader"
for "Yeah. Turn it up.", an ice cube under neon light for "melting the ice",
dice rolling for "rolling the rhythm like rolling the dice", etc.

## What we already have (do not rebuild)

- `ltx_lipsync_fixed.py` — proven singer pipeline. **Do not touch.**
- `workflow_ltx_00026_lipsync.json` — 24-node template with audio path. **Do not touch.**
- `portrait_v2.png` (in `heatwave_refs/`) — singer reference. **Do not touch.**
- `Heatwave.wav`, `Heatwave_lyrics.txt`, `output/heatwave/shotlist.json`,
  `output/heatwave/beatmap.json` — source of truth.
- `ltx_lipsync_00018-audio.mp4` through `00055-audio.mp4` — prior singer renders.
- `output/ltx_lipsync_00026-audio.mp4` — the **00026 winner** (settings to copy for new singer renders).

## What we lifted from KupkaProd (and what we did not)

The `Matticusnicholas/KupkaProd-Music-Video-Pipeline` repo was reviewed.
It targets a different problem (Windows GUI, performance video, no audio
conditioning, Z-Image Turbo keyframes). We are not replacing our flow with it.
We lifted 4 features:

1. **Snap-to-LTX-grid** (8n+1 frames @ 24fps) — fixes a real bug. Our 7.5s
   clips = 180 frames = 8×22+4, NOT 8n+1. Off-grid durations cause audio
   drift in final assembly. Source: `timeline_planner._snap_to_ltx_durations`.
2. **8-section camera-direction prompt structure** — shot, subject, action,
   environment, lighting, color+materials, style, quality. Forces explicit
   material words so output doesn't look plastic. Source: `scene_prompt_gen._ZIMAGE_TURBO_RULES`.
3. **Creative Director pattern** — LLM generates 3 treatments, scores, picks
   best. Deferred to next song. Source: `creative_director.py` is the reference.
4. **Extend Scene mode** — within a scene, render N×4s chunks chained via
   last-frame extraction. Reduces clip count, less drift. Source:
   `ia2v_handoff._extract_last_frame` + `_plan_chunks` + `_concat_video_parts`.

We did NOT lift: their 4-candidate keyframe generation (too expensive at
cloud), their continuous_video_mode (loses our lyric-literal control), their
project_manager/resume_manager (overkill for 12 clips), their film-school
RAG (heavy deps), their Whisper transcription (we have lyrics already),
their Windows GUI/start.bat.

---

## File operations

### Move (not delete) — 5 existing broll refs

```
input/broll_refs/{city_aerial,feet_puddle,dance_floor,neon_bokeh,skyline}.png
input/broll_refs/{ideo_*,z_*}.png
  → input/broll_refs_fallback/
```

### Create — 5 new files

| File | Purpose |
|---|---|
| `output/heatwave/broll_lyric_spec.json` | 12-entry master spec, one per broll clip |
| `broll_spec.py` | Spec loader + snap-to-LTX-grid (inline 5-line snap function) |
| `broll_refs_generate.py` | Per-entry Ideogram 4 ref still generation |
| `broll_heatwave_v2.py` | Per-scene LTX 2.3 I2V render with extend-scene mode |
| `workflow_ltx_00026_broll_v2.json` | Re-exported template: broll negative, I2V strength lift to patched |

### Modify — 2 existing files

| File | Change |
|---|---|
| `broll_generate.py` | Add `i2v_strength`, `cfg`, `steps` to `patch_workflow()` signature; accept alternate template path |
| `AGENTS.md` | Add a pointer at the top of "Music Video Pipeline" section → this file |

### Rename (not delete) — 1 file

| File | Change |
|---|---|
| `broll_heatwave.py` | Rename to `broll_heatwave_v1.py.bak`. The new orchestrator is `broll_heatwave_v2.py`. |

### Untouched

- `ltx_lipsync_fixed.py` — singer is good
- `singer_heatwave.py` — singer is good
- `workflow_ltx_00026_lipsync.json` — singer template
- `heatwave_refs/portrait_v2.png` — singer ref
- `beat_stitch.py` — works for final assembly
- AGENTS.md sections before "Music Video Pipeline"

---

## `broll_lyric_spec.json` schema

```json
{
  "intro_0": {
    "section": "intro",
    "clip_index": 0,
    "time_start": 0.0,
    "time_end": 7.292,
    "duration_seconds": 7.292,
    "duration_frames": 175,
    "lyric": "Yeah.",
    "lyric_concept": "intimate, tactile, anticipation",

    "ref_prompt": {
      "high_level_description": "...",
      "style_description": {
        "aesthetics": "cinematic, moody, intimate",
        "lighting": "single warm key light, neon rim",
        "photo": "85mm f/1.4 macro, shallow DOF",
        "medium": "photograph",
        "color_palette": ["#FF1A8C", "#00D9FF", "#0A0A0A"]
      },
      "compositional_deconstruction": {
        "background": "dark studio with rack of glowing equipment, soft bokeh",
        "elements": [
          {"type": "obj", "bbox": [200, 300, 800, 700], "desc": "fingers on a metal fader with glowing red light"}
        ]
      }
    },

    "ltx_prompt": "hand pushing a glowing metal fader upward, dark studio, neon pink rim light on fingers, 85mm macro, shallow DOF, cinematic 4k, photorealistic",

    "prompt_text": {
      "shot_framing": "extreme close-up, locked tripod, no camera motion",
      "subject": "a hand on a metal fader, fingernails short, no rings",
      "action": "index finger pushes the fader upward in a slow deliberate motion",
      "environment": "dark recording studio, out-of-focus rack of glowing equipment in background",
      "lighting": "single warm key light from upper right, soft blue rim from monitor glow",
      "color_materials": "matte black fader, brushed aluminum housing, neon orange LED indicator, dark walnut wood desk",
      "style_medium": "cinematic photorealistic, 85mm macro, shallow depth of field",
      "quality_boosters": "sharp focus, crisp detail on skin pores, natural film grain, balanced composition"
    },

    "negative": "ugly, deformed, blurry, low quality, cartoon, watermark, text, person, face, headshot",
    "i2v_strength": 0.85,
    "cfg": 4.0,
    "steps": 15,
    "seed": 42000,
    "chunk_duration": null
  }
}
```

`chunk_duration` (optional): if set (e.g. 4.0), the scene is rendered as N
chained 4s clips via last-frame extraction. If null, one render at
`duration_seconds`. Use 4.0 for long chorus scenes (24s) to avoid OOM.

---

## 12-entry lyric spec (the actual content)

| Key | Section | Lyric | Concept | Ref | I2V | CFG | Steps | Chunk |
|---|---|---|---|---|---|---|---|---|
| `intro_0` | intro (0-7.3s) | "Yeah." | Hand on a glowing fader, dark studio | fader | 0.85 | 4.0 | 15 | — |
| `intro_1` | intro (7.3-14.6s) | "Turn it up." | Amp cone pulsing, low-angle | amp | 0.80 | 4.0 | 15 | — |
| `chorus1_0` | chorus1 (48-55.3s) | "Caught in the heatwave" | Macro: ice cube under neon, heat distortion | ice | 0.90 | 4.5 | 20 | — |
| `chorus1_1` | chorus1 (55.3-62.5s) | "melting the ice" | Macro: ice cube dripping, melting | ice_drip | 0.85 | 4.0 | 20 | — |
| `chorus1_2` | chorus1 (62.5-69.8s) | "rolling the rhythm / dice" | Dice tumbling on glossy black table | dice | 0.70 | 4.0 | 15 | — |
| `chorus2_0` | chorus2 (108-115.3s) | "heatwave" (var) | Heat shimmer on pavement | heatwave | 0.90 | 4.5 | 20 | — |
| `chorus2_1` | chorus2 (115.3-122.5s) | "rolling dice / midnight" | Clock striking 12, dark room | clock | 0.75 | 4.0 | 15 | — |
| `chorus2_2` | chorus2 (122.5-129.8s) | "own the tempo / taking the night" | Hands in air, crowd silhouette | hands | 0.70 | 4.0 | 15 | — |
| `bridge_0` | bridge (132-139.3s) | "Step back. Breathe in." | Figure stepping back from camera, exhaling fog | stepback | 0.65 | 3.5 | 15 | — |
| `bridge_1` | bridge (139.3-146.5s) | "heavy bass kick in" | Subwoofer cone pulsing, low-angle | bass | 0.85 | 4.0 | 15 | — |
| `bridge_2` | bridge (146.5-153.8s) | "Watch the whole world spin" | Top-down spinning city, vertigo | spin | 0.85 | 4.0 | 20 | — |
| `solo_0` | solo (150-157.3s) | "Guitar or Synth Solo" | Fingers on fretboard, neon rim | fretboard | 0.75 | 4.0 | 15 | — |
| `outro_0` | outro (174-181.3s) | "Taking the night / melt the ice" | Time-lapse ice fully melted, fade to black | ice_melted | 0.95 | 4.5 | 20 | — |

**Note:** the table shows 13 entries but `intro_1` and `solo_0` overlap on
time. Use `intro_0` (0-7.3s) and `intro_1` (7.3-14.6s) for intro; the solo
6s window fits `solo_0` at 150-157.3s and outro_0 at 174-181.3s. Adjust
times in the actual spec to match `shotlist.json` exactly. Authoritative
time ranges: see `shotlist.json`.

**Note on durations:** 7.292s = 175 frames @ 24fps (8×21+7 = 175). Use
`snap_ltx_duration()` in `broll_spec.py` to compute. Acceptable alternatives:
7.708s (185 frames) if you need longer. Never use 7.5s (180 frames — off-grid).

---

## Singer variation (kept tight)

**5 singer scenes, same face, varied framing per section.**
Prompts already in `shotlist.json` (lines 30, 46, 76, 92, 150). No changes.

**Ref strategy:**
- Primary: `heatwave_refs/portrait_v2.png` for all 18 singer clips
- Alt 1: `heatwave_refs/portrait_v2_alt_close.png` — tighter framing, verse close-ups
- Alt 2: `heatwave_refs/portrait_v2_alt_wide.png` — wider studio frame, chorus energy

Per-scene ref selection (suggested):
- verse1, verse2 → `portrait_v2.png` (primary)
- prechorus1, prechorus2 → `portrait_v2_alt_close.png` (intimate)
- chorus3 → `portrait_v2_alt_wide.png` (wider)

Generate the 2 alternates via Z-Image Turbo or Ideogram 4 before the singer
batch. They must keep the same face, hair, lighting direction. Use the same
singer positive prompt and 2-3 different framing modifiers.

**Singer settings (unchanged, copy from 00026):**
- Model: `LTX-2.3-22B-distilled-1.1-Q6_K.gguf`
- LoRA 0.8, I2V 0.6, CFG 3.5, euler, linear_quadratic, 15 steps
- Resolution: 960×544, 7.5s
- Audio: `segment_NNN.wav` from `input/`

---

## Execution order (test-first, no batched runs)

### Step 0 — Read-only audit + move fallback refs
- Verify `singer_heatwave.py` reads from `shotlist.json` (don't modify, just confirm)
- Move 5 fallback refs to `input/broll_refs_fallback/`
- Confirm ComfyUI is up: `systemctl --user status comfyui`

**Gate:** no output rendered, but you can show "moved 5 refs" log.

### Step 1 — Author `output/heatwave/broll_lyric_spec.json` (12 entries)
- Use the 12-entry table above as the starting point
- Each entry gets a `ref_prompt` (Ideogram 4 structured JSON), `ltx_prompt`,
  and `prompt_text` (8 sections)
- Durations already in the table; do not edit unless `shotlist.json` differs
- Run `python3 broll_spec.py --validate` to confirm it loads + snaps

**Gate:** user eyeballs the JSON. No rendering.

### Step 2 — Render 1 test ref (`intro_0` only)
- `python3 broll_refs_generate.py --only intro_0 --url http://127.0.0.1:8188`
- Output: `input/broll_refs_lyric/intro_0.png`

**Gate:** user eyeballs the still: "does this look like a fader turn?"
- If yes → continue
- If no → tweak `ref_prompt`, re-render

### Step 3 — Render 1 test broll clip (`intro_0` ref)
- `python3 broll_heatwave_v2.py --only intro_0 --url http://127.0.0.1:8188`
- Output: `output/heatwave_broll_lyric/intro_0.mp4`
- Restart ComfyUI after the render (state leak prevention per AGENTS.md)

**Gate:** user eyeballs the motion: "does the fader actually move up?"
- If yes → continue
- If no → tweak `ltx_prompt` / `i2v_strength` / `seed`

### Step 4 — Render 1 clip per section (6 clips, ~30 min)
- `python3 broll_heatwave_v2.py --first-of-each-section --url ...`
- Clips: `intro_0`, `chorus1_0`, `chorus2_0`, `bridge_0`, `solo_0`, `outro_0`
- Restart ComfyUI between each clip

**Gate:** user reviews the 6 outputs for coherence and lyric-literal alignment.
- If good → continue
- If specific clips fail → iterate on those entries' prompts/params

### Step 5 — Render remaining 6 clips (~30 min)
- `python3 broll_heatwave_v2.py --start 1 --url ...`
- All 12 broll clips should now exist

### Step 6 — Stitch per-section then full broll track
- `python3 beat_stitch.py --broll-only --output output/heatwave_broll_lyric/broll_full.mp4`
- Verify durations sum correctly (use snapped frame counts)

### Step 7 — Singer batch
- Confirm `singer_heatwave.py` reads `shotlist.json` prompts
- Confirm alternates generated (or skip and use primary for all)
- `python3 singer_heatwave.py --submit`
- ~18 clips, ~90 min on RTX 4090

### Step 8 — Final assembly
- Stitch singer + broll per `shotlist.json` order
- Overlay `Heatwave.wav` audio
- `output/heatwave/heatwave_full.mp4`

---

## Critical rules

1. **Snap to 8n+1 frames @ 24fps.** Never use 7.5s for new clips. 7.292s (175)
   or 7.708s (185). The snap function in `broll_spec.py` enforces this.

2. **Per-clip broll negative prompt.** Default: `"ugly, deformed, blurry, low
   quality, cartoon, watermark, text, person, face, headshot"`. The current
   `workflow_ltx_00026_broll.json` has singer-tuned negatives — must change.

3. **I2V strength 0.85 baseline for broll.** Singer uses 0.6 (face lock).
   Broll wants 0.85 (motion divergence from ref). Adjust per-clip ±0.1.

4. **Restart ComfyUI between renders.** 22B Q6_K OOMs on 16GB if state leaks.
   `broll_heatwave_v2.py` does this automatically between clips.

5. **Don't touch the singer pipeline.** `ltx_lipsync_fixed.py`,
   `singer_heatwave.py`, `workflow_ltx_00026_lipsync.json`, `portrait_v2.png`
   are proven. If a singer clip looks off, regenerate the audio segment
   (Demucs + manual edit) before touching the model code.

6. **8-section prompt structure is mandatory for `ltx_prompt`.** The current
   broll prompts are 30-40 words. Bump to 120-220 words covering all 8 sections.
   Material words are required (e.g. "brushed aluminum", "wet cobblestone") to
   avoid the "plastic" look.

7. **Don't delete the 5 fallback refs.** Move to `input/broll_refs_fallback/`
   only. They might be needed for re-renders or alt compositions.

8. **Test-first, no batched runs.** Steps 2-5 each have a user eyeball gate.
   Don't queue all 12 clips at once.

9. **Track every change in `AGENTS.md` "Working" section** as the work
   progresses. Update with: what was built, what worked, what didn't, new
   findings. This file is the historical record.

---

## Reference: first reads for the other agent

Read in this order before writing any code:

1. `/home/ericr/ComfyUI/AGENTS.md` (the "Music Video Pipeline" section, lines ~413+)
2. `/home/ericr/ComfyUI/plan_heatwave_music_video.md` (this file)
3. `/home/ericr/ComfyUI/workflow_ltx_00026_broll.json` (the current broll template — what we're replacing)
4. `/home/ericr/ComfyUI/Heatwave_lyrics.txt` (the source of truth for what each clip should literally show)
5. `/home/ericr/ComfyUI/output/heatwave/shotlist.json` (time ranges, prompts, section structure)
6. `/home/ericr/ComfyUI/broll_heatwave.py` (the v1 to replace — read for context, then rename to .bak)
7. `/home/ericr/ComfyUI/broll_generate.py` (the helper to extend)

Do NOT read (out of scope, will distract):
- SVI history, SUPIR experiments, Ideogram 4 parameter sweep results
- Old ltx_lipsync_test_*.mp4 outputs (before 00026)
- lipsync_pack/ examples

---

## What "done" looks like

`output/heatwave/heatwave_full.mp4` exists, ~180s, plays the original
`Heatwave.wav` audio, has the studio singer in close-up during verses and
chorus, has lyric-literal b-roll during intro/chorus1/chorus2/bridge/solo/outro,
and the visuals match the words being sung.
