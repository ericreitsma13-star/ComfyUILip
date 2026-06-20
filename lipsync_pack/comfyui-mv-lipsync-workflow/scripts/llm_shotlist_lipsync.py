#!/usr/bin/env python3
"""
llm_shotlist_lipsync.py - LLM-directed shot list with lipsync + broll scene types.

Reads:
  - beatmap.json     (from audio_analysis.py — includes vocal_active flags per section)
  - lyrics.txt       (Suno lyrics, plain text)
  - singer_ref_images/  (1 or more reference images of the singer — used for lip-sync scenes)
  - broll_ref_images/   (reference images for B-roll scenes — landscapes, objects, abstract)

Calls an LLM to produce a shot list where each scene is tagged as either:
  - "lipsync"  → singer on camera, mouth synced to vocals (uses workflow 01 + 04)
  - "broll"    → cinematic B-roll, no singer (uses workflow 02/03 + 05)

DECISION LOGIC (LLM is given these rules):
  • Verses (vocal_active=true, label starts with "verse") → lipsync scenes
  • Choruses (vocal_active=true, label starts with "chorus") → broll scenes
    (choruses work better as B-roll — more visual energy, less "talking head" feel)
  • Intro / outro / instrumental breaks (vocal_active=false) → broll scenes
  • Bridge → broll scenes

USAGE:
    export ZAI_API_KEY="..."
    python llm_shotlist_lipsync.py \\
        --beatmap beatmap.json \\
        --lyrics lyrics.txt \\
        --singer-dir singer_ref_images/ \\
        --broll-dir broll_ref_images/ \\
        --output shotlist.json \\
        --send-images

OUTPUT SCHEMA (shotlist.json):
{
  "song_title": "...",
  "bpm": 124.0,
  "total_duration": 215.4,
  "singer_ref_image": "singer_01.png",          # single image, used for all lipsync scenes
  "scenes": [
    {
      "index": 0,
      "start_time": 0.0,
      "end_time": 12.5,
      "section": "intro",
      "scene_type": "broll",                     # ← "lipsync" or "broll"
      "ref_image": "city_skyline_night.png",
      "prompt": "cinematic aerial shot, neon-lit city skyline at night, ...",
      "negative_prompt": "blurry, low quality, ...",
      "model": "ltx",                            # for broll: "ltx" or "wan"
      "motion_strength": 0.4,
      "duration_sec": 12.5,
      "notes": "Long establishing shot..."
    },
    {
      "index": 1,
      "start_time": 12.5,
      "end_time": 36.8,
      "section": "verse1",
      "scene_type": "lipsync",                   # singer on camera
      "ref_image": "singer_01.png",              # the singer's reference image
      "vocal_audio_segment": "vocal_segment_001.wav",  # trimmed from isolated vocals
      "prompt": "intimate close-up of singer, warm rim lighting, ...",
      "negative_prompt": "blurry, distorted face, deformed, ...",
      "motion_strength": 0.6,                    # Sonic motion_strength widget
      "lip_sync_strength": 0.7,
      "duration_sec": 24.3,
      "notes": "Singer delivering verse 1, slow head movement"
    }
  ]
}
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path

# Reuse the LLM provider adapters from the B-roll pack's llm_shotlist.py
# (duplicated here for self-containment)
try:
    import requests
except ImportError:
    print("ERROR: missing 'requests'. Run: pip install requests")
    sys.exit(1)


def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_mime(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")


def call_zai(prompt: str, images: list[str] | None, model: str, temperature: float, max_tokens: int) -> str:
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
            {"role": "system", "content": "You are a music video director with expertise in cinematic shot design, beat-synced editing, and lip-sync performance direction. You output strictly valid JSON when asked."},
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
            {"role": "system", "content": "You are a music video director with expertise in cinematic shot design, beat-synced editing, and lip-sync performance direction. You output strictly valid JSON when asked."},
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
            {"role": "system", "content": "You are a music video director with expertise in cinematic shot design, beat-synced editing, and lip-sync performance direction. You output strictly valid JSON when asked."},
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
You are directing a music video for the song below. The video MIXES two kinds of scenes:
  - LIP-SYNC shots: the singer is on camera, head + shoulders, mouth synced to vocals
  - B-ROLL shots: cinematic environments, NO singer visible (landscapes, objects, abstract)

The visual style is cinematic — think Blade Runner 2049, The Weeknd "Blinding Lights",
Villeneuve's Dune. Photorealistic, shallow depth of field, anamorphic lens flares,
subtle film grain.

== SONG METADATA ==
- Title: {title}
- BPM: {bpm}
- Duration: {duration:.1f}s
- Time signature: {ts}

== SONG SECTIONS (with vocal-active flags) ==
{sections_block}

== LYRICS ==
{lyrics}

== AVAILABLE SINGER REFERENCE IMAGES (for lip-sync scenes) ==
{singer_listing}

== AVAILABLE B-ROLL REFERENCE IMAGES (for B-roll scenes) ==
{broll_listing}

== TASK ==
Design a shot list for the full music video. Each scene MUST:
1. Start and end on a section boundary from above (use exact start/end times).
2. Be tagged scene_type = "lipsync" OR "broll" based on these rules:
   • Sections with vocal_active=true AND label starts with "verse"   → "lipsync"
   • Sections with vocal_active=true AND label starts with "chorus"  → "broll" (choruses work better as B-roll)
   • Sections with vocal_active=false (intro, outro, instrumental)   → "broll"
   • "bridge" sections                                                → "broll"
3. For LIP-SYNC scenes:
   - ref_image = ONE singer image from the singer list (use exact filename)
   - prompt describes the singer's appearance, lighting, mood (focus on the face)
   - motion_strength: 0.4-0.7 (gentle head motion)
   - lip_sync_strength: 0.7 (tight vocal sync)
   - model field: ignore for lipsync (Sonic handles all lip-sync)
   - notes describe the singer's performance / expression for this section
4. For B-ROLL scenes:
   - ref_image = ONE broll image from the broll list (use exact filename)
   - prompt describes environment, camera, lighting (NO singer mentioned)
   - model = "ltx" for fast shots (intro, transitions) OR "wan" for hero shots (choruses)
   - motion_strength: 0.3-0.8 (vary for visual interest)
   - notes describe the scene's role in pacing

RULES:
- Use ALL the song's sections (no skipping).
- Singer image should be the SAME one across all lipsync scenes (consistency).
- Spread broll image usage — try to use each broll image at least once.
- Total scene count: between {min_scenes} and {max_scenes}.

== OUTPUT FORMAT ==
Output a SINGLE valid JSON object (no markdown, no commentary) with this exact schema:
{{
  "song_title": "...",
  "bpm": {bpm},
  "total_duration": {duration},
  "singer_ref_image": "<filename from singer list>",
  "scenes": [
    {{
      "index": 0,
      "start_time": 0.0,
      "end_time": 12.5,
      "section": "intro",
      "scene_type": "broll",
      "ref_image": "<broll filename>",
      "prompt": "...",
      "negative_prompt": "blurry, low quality, ...",
      "model": "ltx",
      "motion_strength": 0.4,
      "duration_sec": 12.5,
      "notes": "..."
    }},
    {{
      "index": 1,
      "start_time": 12.5,
      "end_time": 36.8,
      "section": "verse1",
      "scene_type": "lipsync",
      "ref_image": "<singer filename>",
      "vocal_audio_segment": "vocal_segment_001.wav",
      "prompt": "...",
      "negative_prompt": "blurry, distorted face, deformed, ...",
      "motion_strength": 0.6,
      "lip_sync_strength": 0.7,
      "duration_sec": 24.3,
      "notes": "..."
    }}
  ]
}}

For LIP-SYNC scenes, include the fields: motion_strength, lip_sync_strength, vocal_audio_segment.
For B-ROLL scenes, include the field: model ("ltx" or "wan").
"""


def build_prompt(beatmap, lyrics, singer_images, broll_images, song_title):
    sections_block = "\n".join(
        f"  - {i}. {s['label']:10s}  {s['start']:6.1f}s - {s['end']:6.1f}s  "
        f"vocal_active={s.get('vocal_active', 'unknown')}"
        for i, s in enumerate(beatmap["sections"])
    )
    singer_listing = "\n".join(f"  - {Path(p).name}" for p in singer_images) or "  (none — please add singer ref images)"
    broll_listing = "\n".join(f"  - {Path(p).name}" for p in broll_images) or "  (none — please add broll ref images)"
    n_sections = len(beatmap["sections"])
    min_scenes = max(n_sections, 6)
    max_scenes = min(n_sections * 2, 20)
    prompt = DIRECTOR_PROMPT_TEMPLATE.format(
        title=song_title,
        bpm=beatmap["bpm"],
        duration=beatmap["duration_sec"],
        ts=beatmap["time_signature"],
        sections_block=sections_block,
        lyrics=lyrics or "(no lyrics provided — instrumental)",
        singer_listing=singer_listing,
        broll_listing=broll_listing,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
    )
    return prompt


def main():
    p = argparse.ArgumentParser(description="Generate an LLM-directed shot list with lipsync + broll scene types.")
    p.add_argument("--beatmap", required=True, help="beatmap.json from audio_analysis.py")
    p.add_argument("--lyrics", required=True, help="Plain-text lyrics file (UTF-8).")
    p.add_argument("--singer-dir", required=True, help="Directory containing singer reference images (for lip-sync scenes).")
    p.add_argument("--broll-dir", required=True, help="Directory containing B-roll reference images (for non-singer scenes).")
    p.add_argument("--output", default="shotlist.json", help="Output JSON path.")
    p.add_argument("--title", default="Untitled", help="Song title.")
    p.add_argument("--provider", choices=list(PROVIDERS.keys()), default="zai")
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--send-images", action="store_true",
                   help="Send both singer + broll images as vision input to the LLM.")
    args = p.parse_args()

    with open(args.beatmap) as f:
        beatmap = json.load(f)
    with open(args.lyrics, encoding="utf-8") as f:
        lyrics = f.read().strip()

    ref_exts = {".png", ".jpg", ".jpeg", ".webp"}
    singer_dir = Path(args.singer_dir)
    broll_dir = Path(args.broll_dir)
    singer_images = sorted([str(p) for p in singer_dir.iterdir() if p.suffix.lower() in ref_exts])
    broll_images = sorted([str(p) for p in broll_dir.iterdir() if p.suffix.lower() in ref_exts])

    if not singer_images:
        print(f"ERROR: no singer reference images found in {singer_dir}")
        print("       Lip-sync scenes require at least one singer reference image.")
        sys.exit(1)
    if not broll_images:
        print(f"WARN: no B-roll reference images found in {broll_dir}")
        print("      All scenes will be lipsync — consider adding B-roll refs for variety.")

    print(f"[llm_shotlist_lipsync] Singer images: {len(singer_images)}")
    print(f"[llm_shotlist_lipsync] B-roll images: {len(broll_images)}")

    prompt = build_prompt(beatmap, lyrics, singer_images, broll_images, args.title)
    print(f"[llm_shotlist_lipsync] Prompt length: {len(prompt)} chars")

    caller, default_model = PROVIDERS[args.provider]
    model = args.model or default_model
    print(f"[llm_shotlist_lipsync] Calling {args.provider} / {model} ...")

    images_to_send = None
    if args.send_images:
        images_to_send = singer_images + broll_images
        print(f"[llm_shotlist_lipsync] Sending {len(images_to_send)} reference images as vision input")

    raw = caller(prompt, images_to_send, model, args.temperature, args.max_tokens)

    # Strip markdown code fences
    raw_stripped = raw.strip()
    if raw_stripped.startswith("```"):
        parts = raw_stripped.split("```")
        if len(parts) >= 3:
            raw_stripped = parts[1]
            if raw_stripped.startswith("json"):
                raw_stripped = raw_stripped[4:]
            raw_stripped = raw_stripped.strip("`\n ")

    try:
        shotlist = json.loads(raw_stripped)
    except json.JSONDecodeError as e:
        print(f"[llm_shotlist_lipsync] ERROR: LLM output was not valid JSON: {e}")
        with open("shotlist_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        print("[llm_shotlist_lipsync] Raw output saved to shotlist_raw.txt for inspection.")
        sys.exit(1)

    if "scenes" not in shotlist or not isinstance(shotlist["scenes"], list):
        print("[llm_shotlist_lipsync] ERROR: shot list missing 'scenes' array")
        sys.exit(1)

    # Validate / sanitize
    singer_basenames = {Path(p).name: p for p in singer_images}
    broll_basenames = {Path(p).name: p for p in broll_images}
    singer_ref_default = next(iter(singer_basenames.keys()))

    if "singer_ref_image" not in shotlist or shotlist["singer_ref_image"] not in singer_basenames:
        shotlist["singer_ref_image"] = singer_ref_default

    lipsync_count = 0
    broll_count = 0
    for i, sc in enumerate(shotlist["scenes"]):
        sc["index"] = i
        scene_type = sc.get("scene_type", "broll")
        if scene_type not in ("lipsync", "broll"):
            print(f"[llm_shotlist_lipsync] WARN: scene {i} has unknown scene_type '{scene_type}', defaulting to 'broll'")
            sc["scene_type"] = "broll"
            scene_type = "broll"

        if scene_type == "lipsync":
            lipsync_count += 1
            # Lipsync scenes always use the singer_ref_image
            sc["ref_image"] = shotlist["singer_ref_image"]
            # Assign a vocal segment filename for batch_render_lipsync.py to populate
            sc["vocal_audio_segment"] = f"vocal_segment_{i:03d}.wav"
            sc.setdefault("motion_strength", 0.6)
            sc.setdefault("lip_sync_strength", 0.7)
            sc.setdefault("negative_prompt",
                          "blurry, distorted face, deformed, asymmetric eyes, plastic skin, "
                          "extra teeth, melting face, bad anatomy, watermark, text")
        else:
            broll_count += 1
            # B-roll scenes must use a broll image
            if sc.get("ref_image") not in broll_basenames:
                if broll_basenames:
                    sc["ref_image"] = next(iter(broll_basenames.keys()))
                else:
                    sc["ref_image"] = shotlist["singer_ref_image"]
            if sc.get("model") not in ("ltx", "wan"):
                sc["model"] = "wan"
            sc.setdefault("motion_strength", 0.5)
            sc.setdefault("negative_prompt",
                          "blurry, low quality, cartoon, anime, watermark, text, deformed, "
                          "distorted, bad anatomy, jpeg artifacts, oversaturated, plastic skin")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(shotlist, f, indent=2)

    print(f"\n[llm_shotlist_lipsync] Wrote shot list to {args.output}")
    print(f"[llm_shotlist_lipsync] Total scenes: {len(shotlist['scenes'])}")
    print(f"[llm_shotlist_lipsync]   Lip-sync: {lipsync_count}")
    print(f"[llm_shotlist_lipsync]   B-roll:   {broll_count}")
    print(f"[llm_shotlist_lipsync] Singer ref: {shotlist['singer_ref_image']}")
    print()
    for sc in shotlist["scenes"]:
        st = "🎤 LIP" if sc["scene_type"] == "lipsync" else "  BROLL"
        print(f"  Scene {sc['index']:2d}  {sc['start_time']:6.1f}-{sc['end_time']:6.1f}s  "
              f"[{sc['section']:10s}]  {st}  ref={sc['ref_image']}")


if __name__ == "__main__":
    main()
