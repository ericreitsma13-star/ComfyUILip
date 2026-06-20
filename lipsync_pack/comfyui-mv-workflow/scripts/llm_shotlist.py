#!/usr/bin/env python3
"""
llm_shotlist.py - LLM-directed shot list generator.

Reads:
  - beatmap.json  (from audio_analysis.py)
  - lyrics.txt    (Suno lyrics, plain text)
  - reference_images/  (directory of ref images)

Calls an LLM (OpenAI-compatible API or local) to produce a shot list that:
  - Aligns scene cuts to song sections (intro / verse1 / chorus1 / verse2 / ...)
  - Pairs each scene with a reference image (LLM picks from available images)
  - Generates a cinematic prompt per scene
  - Suggests model choice (LTX for prototyping, Wan for finals)
  - Suggests motion strength + duration per scene

USAGE:
    # Using Z.ai (GLM) — set ZAI_API_KEY env var
    export ZAI_API_KEY="your_key_here"
    python llm_shotlist.py \
        --beatmap beatmap.json \
        --lyrics lyrics.txt \
        --ref-dir reference_images/ \
        --output shotlist.json

    # Using OpenAI — set OPENAI_API_KEY
    python llm_shotlist.py --provider openai --model gpt-4o ... 

    # Using local Ollama
    python llm_shotlist.py --provider ollama --model llama3.1:70b ...

OUTPUT SCHEMA (shotlist.json):
{
  "song_title": "...",
  "bpm": 124.0,
  "total_duration": 215.4,
  "scenes": [
    {
      "index": 0,
      "start_time": 0.0,
      "end_time": 12.5,
      "section": "intro",
      "ref_image": "city_skyline_night.png",
      "prompt": "cinematic aerial shot, neon-lit city skyline at night, slow camera push-in, anamorphic lens flare, rain on lens, teal and orange color grade, 35mm film, dramatic atmosphere",
      "negative_prompt": "blurry, low quality, cartoon, anime, watermark, text, deformed, distorted, oversaturated",
      "model": "ltx",                       # or "wan"
      "motion_strength": 0.4,
      "duration_sec": 12.5,
      "notes": "Long establishing shot, low motion. Sets the mood."
    },
    ...
  ]
}
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: missing 'requests'. Run: pip install requests")
    sys.exit(1)


# -------- LLM provider adapters --------

def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_mime(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")


def call_zai(prompt: str, images: list[str] | None, model: str, temperature: float, max_tokens: int) -> str:
    """Call Z.ai GLM-4V or compatible endpoint."""
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise RuntimeError("ZAI_API_KEY env var not set.")
    url = os.environ.get("ZAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
    content = [{"type": "text", "text": prompt}]
    if images:
        for img_path in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{_image_mime(img_path)};base64,{_encode_image_b64(img_path)}"},
            })
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a music video director with expertise in cinematic shot design and beat-synced editing. You output strictly valid JSON when asked."},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_openai(prompt: str, images: list[str] | None, model: str, temperature: float, max_tokens: int) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set.")
    url = "https://api.openai.com/v1/chat/completions"
    content = [{"type": "text", "text": prompt}]
    if images:
        for img_path in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{_image_mime(img_path)};base64,{_encode_image_b64(img_path)}"},
            })
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a music video director with expertise in cinematic shot design and beat-synced editing. You output strictly valid JSON when asked."},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_ollama(prompt: str, images: list[str] | None, model: str, temperature: float, max_tokens: int) -> str:
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{base}/api/chat"
    images_b64 = [_encode_image_b64(p) for p in (images or [])]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a music video director with expertise in cinematic shot design and beat-synced editing. You output strictly valid JSON when asked."},
            {"role": "user", "content": prompt, "images": images_b64},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


PROVIDERS = {
    "zai":     (call_zai,     "glm-4v-plus"),
    "openai":  (call_openai,  "gpt-4o"),
    "ollama":  (call_ollama,  "llama3.1:70b"),
}


# -------- Prompt construction --------

DIRECTOR_PROMPT_TEMPLATE = """\
You are directing a music video for the song below. The visuals must be cinematic —
think Blade Runner 2049, The Weeknd "Blinding Lights", or Villeneuve's Dune.
Photorealistic, shallow depth of field, anamorphic lens flares, subtle film grain.

== SONG METADATA ==
- Title: {title}
- BPM: {bpm}
- Duration: {duration:.1f}s
- Time signature: {ts}

== SONG SECTIONS (already segmented by audio analysis) ==
{sections_block}

== LYRICS ==
{lyrics}

== AVAILABLE REFERENCE IMAGES (LLM picks from this list; do NOT invent new filenames) ==
{ref_listing}

== TASK ==
Design a shot list for the full music video. Each scene MUST:
1. Start and end on a section boundary from above (use exact start/end times from sections).
2. Be assigned ONE reference image from the list above (use exact filename).
3. Have a detailed cinematic prompt (50-120 words) that describes:
   - Subject / action (what's happening on screen)
   - Camera (aerial, dolly, push-in, tracking, static, handheld)
   - Lighting (volumetric, neon, golden hour, moonlight, etc.)
   - Mood / atmosphere
   - Specific cinematic touches (lens flare, grain, color grade direction)
4. Specify motion_strength: 0.1-0.3 for static shots, 0.4-0.6 for gentle motion, 0.7-1.0 for dynamic.
5. Specify model: "ltx" for fast prototyping scenes (brief shots, transitions, B-roll),
                  "wan" for hero shots (key visual moments, choruses, emotional peaks).
6. Add brief "notes" explaining the creative intent.

RULES:
- Use ALL the song's sections (no skipping).
- Chorus sections should get the most visually striking, "wan"-rendered shots.
- Verse sections can be more static / intimate ("ltx" is fine for speed).
- Bridge should feel different — change of location / color palette / mood.
- Spread reference image usage — try to use each image at least once (if more scenes than images, that's fine to reuse).
- Total scene count: between {min_scenes} and {max_scenes}.

== OUTPUT FORMAT ==
Output a SINGLE valid JSON object (no markdown, no commentary) with this exact schema:
{{
  "song_title": "...",
  "bpm": {bpm},
  "total_duration": {duration},
  "scenes": [
    {{
      "index": 0,
      "start_time": 0.0,
      "end_time": 12.5,
      "section": "intro",
      "ref_image": "filename.png",
      "prompt": "...",
      "negative_prompt": "blurry, low quality, cartoon, anime, watermark, text, deformed, distorted, oversaturated, plastic skin",
      "model": "ltx",
      "motion_strength": 0.4,
      "duration_sec": 12.5,
      "notes": "..."
    }}
  ]
}}
"""


def build_prompt(beatmap, lyrics, ref_images, song_title):
    sections_block = "\n".join(
        f"  - {i}. {s['label']:10s}  {s['start']:6.1f}s - {s['end']:6.1f}s  (len {s['end']-s['start']:.1f}s)"
        for i, s in enumerate(beatmap["sections"])
    )
    ref_listing = "\n".join(f"  - {Path(p).name}" for p in ref_images)
    n_sections = len(beatmap["sections"])
    min_scenes = max(n_sections, 8)
    max_scenes = min(n_sections * 2, 24)
    prompt = DIRECTOR_PROMPT_TEMPLATE.format(
        title=song_title,
        bpm=beatmap["bpm"],
        duration=beatmap["duration_sec"],
        ts=beatmap["time_signature"],
        sections_block=sections_block,
        lyrics=lyrics or "(no lyrics provided — instrumental)",
        ref_listing=ref_listing,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
    )
    return prompt


# -------- Main --------

def main():
    p = argparse.ArgumentParser(description="Generate an LLM-directed shot list from beatmap + lyrics + ref images.")
    p.add_argument("--beatmap", required=True, help="beatmap.json from audio_analysis.py")
    p.add_argument("--lyrics", required=True, help="Plain-text lyrics file (UTF-8).")
    p.add_argument("--ref-dir", required=True, help="Directory containing reference images.")
    p.add_argument("--output", default="shotlist.json", help="Output JSON path.")
    p.add_argument("--title", default="Untitled", help="Song title (for the shot list metadata).")
    p.add_argument("--provider", choices=list(PROVIDERS.keys()), default="zai",
                   help="LLM provider (default: zai)")
    p.add_argument("--model", default=None, help="Model override (else provider default).")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--send-images", action="store_true",
                   help="Send reference images to the LLM (multimodal). Recommended if model supports vision.")
    args = p.parse_args()

    # Load beatmap
    with open(args.beatmap) as f:
        beatmap = json.load(f)

    # Load lyrics
    with open(args.lyrics, encoding="utf-8") as f:
        lyrics = f.read().strip()

    # Discover ref images
    ref_exts = {".png", ".jpg", ".jpeg", ".webp"}
    ref_dir = Path(args.ref_dir)
    ref_images = sorted(
        [str(p) for p in ref_dir.iterdir() if p.suffix.lower() in ref_exts]
    )
    if not ref_images:
        print(f"ERROR: no reference images found in {ref_dir}")
        sys.exit(1)
    print(f"[llm_shotlist] Found {len(ref_images)} reference images in {ref_dir}")

    # Build prompt
    prompt = build_prompt(beatmap, lyrics, ref_images, args.title)
    print(f"[llm_shotlist] Prompt length: {len(prompt)} chars")

    # Pick provider + model
    caller, default_model = PROVIDERS[args.provider]
    model = args.model or default_model
    print(f"[llm_shotlist] Calling {args.provider} / {model} ...")

    # Optionally send images to multimodal LLM (so it can "see" the reference images and match scenes to them)
    images_to_send = ref_images if args.send_images else None
    if images_to_send:
        print(f"[llm_shotlist] Sending {len(images_to_send)} reference images as vision input")

    raw = caller(prompt, images_to_send, model, args.temperature, args.max_tokens)

    # Strip markdown code fences if present
    raw_stripped = raw.strip()
    if raw_stripped.startswith("```"):
        raw_stripped = raw_stripped.split("```", 2)[1] if raw_stripped.count("```") >= 2 else raw_stripped
        if raw_stripped.startswith("json"):
            raw_stripped = raw_stripped[4:]
        raw_stripped = raw_stripped.strip("`\n ")

    try:
        shotlist = json.loads(raw_stripped)
    except json.JSONDecodeError as e:
        print(f"[llm_shotlist] ERROR: LLM output was not valid JSON: {e}")
        print("[llm_shotlist] Raw output saved to shotlist_raw.txt for inspection.")
        with open("shotlist_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        sys.exit(1)

    # Validate / sanitize
    if "scenes" not in shotlist or not isinstance(shotlist["scenes"], list):
        print(f"[llm_shotlist] ERROR: shot list missing 'scenes' array")
        sys.exit(1)

    # Ensure each scene's ref_image exists; if missing, fall back to first image
    ref_basenames = {Path(p).name: p for p in ref_images}
    for sc in shotlist["scenes"]:
        if sc.get("ref_image") not in ref_basenames:
            print(f"[llm_shotlist] WARN: scene {sc.get('index')} ref_image '{sc.get('ref_image')}' not in ref-dir; using first available")
            sc["ref_image"] = next(iter(ref_basenames.keys()))
        if "negative_prompt" not in sc:
            sc["negative_prompt"] = "blurry, low quality, cartoon, anime, watermark, text, deformed, distorted, oversaturated, plastic skin"
        if "motion_strength" not in sc or not (0.0 <= float(sc.get("motion_strength", 0.5)) <= 1.0):
            sc["motion_strength"] = 0.5
        if sc.get("model") not in ("ltx", "wan"):
            sc["model"] = "wan"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(shotlist, f, indent=2)
    print(f"[llm_shotlist] Wrote shot list to {args.output}")
    print(f"[llm_shotlist] Total scenes: {len(shotlist['scenes'])}")
    for sc in shotlist["scenes"]:
        print(f"  Scene {sc['index']:2d}  {sc['start_time']:6.1f}-{sc['end_time']:6.1f}s  [{sc['section']:10s}]  "
              f"model={sc['model']:3s}  motion={sc['motion_strength']:.2f}  ref={sc['ref_image']}")


if __name__ == "__main__":
    main()
