#!/usr/bin/env python3
"""Generate Sunset Cruise assets using experiment_template.json + prompt_builder.py."""

import json, requests, time, sys, os, re

sys.path.insert(0, "/home/ericr/ComfyUI")
from prompt_builder import query_opencode, validate_and_fix

API = "http://127.0.0.1:8188"

def extract_json(text):
    """Robust JSON extraction from LLM output."""
    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass
    # Try to find outermost { }
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i+1])
    raise ValueError(f"Cannot extract JSON from: {text[:200]}")


# Resolutions we want
# (prefix, aspect_ratio, megapixels, resolution_label, WxH for node 209)
ASSETS = [
    ("skybox_sunset",
     "A seamless equirectangular 360-degree HDRI panorama of a golden hour sunset sky over calm ocean, layered pink orange purple clouds, cinematic atmospheric glow, photorealistic 8K HDR sky dome, no land visible",
     "2:1", "2880x1440", 2880, 1440),

    ("miami_skyline",
     "A wide panoramic shot of Miami Beach art deco skyline at golden hour sunset, pastel pink white blue buildings, palm trees along the shore, calm turquoise ocean with golden reflections, 1980s Miami Vice postcard aesthetic",
     "2:1", "2880x1440", 2880, 1440),

    ("asphalt_road",
     "Top-down macro photograph of worn coastal highway asphalt road texture, cracks and faded yellow lane markings, weathered dark gray pavement, warm golden hour sunlight, seamless tiling PBR surface, 8K detail",
     "1:1", "2048x2048", 2048, 2048),

    ("sand_beach",
     "Top-down macro photograph of golden beach sand texture, fine grain, subtle wind ripples, warm sunset light, slight wet sheen near edge, seamless tiling PBR surface, 8K extreme detail",
     "1:1", "2048x2048", 2048, 2048),

    ("lut_reference",
     "A classic white Lamborghini Countach convertible driving on a coastal highway at golden hour sunset, Miami beach backdrop, 1980s OutRun retro aesthetic, teal and orange color grade, anamorphic lens flare, cinematic automotive editorial photography",
     "2:1", "2880x1440", 2880, 1440),

    ("palm_trees",
     "Silhouettes of multiple tropical palm trees against a golden pink sunset sky, viewed from beach level looking up, high contrast dark silhouettes with distinct frond shapes, calm ocean horizon below, natural rim light on edges",
     "1:1", "2048x2048", 2048, 2048),
]

with open("/home/ericr/ComfyUI/experiment_template.json") as f:
    template = json.load(f)

for prefix, prompt, ar, res, W, H in ASSETS:
    print(f"\n{'='*60}")
    print(f"Generating: {prefix} ({W}x{H})")

    # 1. Build structured prompt via OpenCode LLM (with retry)
    caption = None
    for attempt in range(3):
        print(f"  Building prompt{' (retry ' + str(attempt) + ')' if attempt else ''}...")
        raw_json = query_opencode(prompt, ar)
        try:
            caption = validate_and_fix(extract_json(raw_json))
            break
        except Exception as e:
            print(f"  Parse failed: {e}")
            if attempt == 2:
                raise
    if caption is None:
        raise RuntimeError("Failed after 3 attempts")

    # 2. Patch the template
    p = dict(template)  # shallow copy

    # Node 209: set prompt data
    p["209"]["inputs"]["width"] = W
    p["209"]["inputs"]["height"] = H
    p["209"]["inputs"]["high_level_description"] = caption.get("high_level_description", prompt)
    bg = caption.get("compositional_deconstruction", {}).get("background", "")
    p["209"]["inputs"]["background"] = bg
    p["209"]["inputs"]["style"] = "photorealistic"
    p["209"]["inputs"]["aesthetics"] = "photorealistic, cinematic, sharp detail"
    p["209"]["inputs"]["lighting"] = "golden hour warm natural light"
    p["209"]["inputs"]["medium"] = "photograph"

    # Elements: convert from Ideogram 4 bbox [y1,x1,y2,x2] to KJNodes x/y/w/h format
    elements = caption.get("compositional_deconstruction", {}).get("elements", [])
    if elements:
        kj_els = []
        for el in elements:
            bbox = el.get("bbox", [0, 0, 1000, 1000])
            y1, x1, y2, x2 = [v/1000.0 for v in bbox]
            kj_els.append({
                "x": x1, "y": y1,
                "w": x2 - x1, "h": y2 - y1,
                "type": el.get("type", "obj"),
                "desc": el.get("desc", ""),
                "text": el.get("text", ""),
                "palette": el.get("color_palette", [])
            })
        p["209"]["inputs"]["elements_data"] = json.dumps(kj_els)

    # Palette
    style_palette = caption.get("style_description", {}).get("color_palette", [])
    if not style_palette:
        style_palette = caption.get("compositional_deconstruction", {}).get("color_palette", [])
    p["209"]["inputs"]["style_palette_data"] = json.dumps(style_palette)

    # Node 98:157 CFG = 4.0 (already default in template)
    # Node 98:156 choice = "Default" (already default)

    # Node 158: set output prefix
    p["158"]["inputs"]["filename_prefix"] = f"sunset_cruise/{prefix}"

    # Node 134:169 (aspect_ratio for the LLM prompt builder part)
    if "134:165" in p:
        p["134:165"]["inputs"]["aspect_ratio"] = ar
        p["134:165"]["inputs"]["resolution"] = res

    # 3. Submit
    print(f"  Submitting to ComfyUI...")
    r = requests.post(f"{API}/prompt", json={"prompt": p}, timeout=30)
    if r.status_code != 200:
        print(f"  Failed: {r.status_code} {r.text[:200]}")
        continue
    resp = r.json()
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        print(f"  No prompt_id: {resp}")
        continue
    print(f"  Prompt ID: {prompt_id}")

    # 4. Poll until complete
    print(f"  Generating...", end="", flush=True)
    while True:
        time.sleep(5)
        h = requests.get(f"{API}/history/{prompt_id}", timeout=10).json()
        if prompt_id in h:
            history = h[prompt_id]
            status = history.get("status", {})
            if status.get("completed", False):
                print(" done!")
                break
            if status.get("status_str") == "error":
                print(f" ERROR: {json.dumps(status)[:200]}")
                break
        print(".", end="", flush=True)

print(f"\n\nAll done. Output: ~/ComfyUI/output/sunset_cruise/")
