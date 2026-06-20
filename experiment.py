#!/usr/bin/env python3
"""
Ideogram 4.0 Parameter Sweep Experiment
Runs prompts through ComfyUI API with varying settings, outputs contact sheet.
"""

import json
import time
import urllib.request
import urllib.parse
import uuid
import random
import re
import os
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

COMFYUI_URL = "http://127.0.0.1:8188"
TEMPLATE_PATH = Path.home() / "ComfyUI" / "experiment_template.json"
OUTPUT_DIR = Path.home() / "ComfyUI" / "output" / "experiment"
PROMPTS_CACHE = OUTPUT_DIR / "prompts.json"
CONTACT_SHEET_PATH = OUTPUT_DIR / "contact_sheet.png"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OPENCODE_BIN = Path.home() / ".opencode" / "bin" / "opencode"
OPENCODE_MODEL = "opencode-go/mimo-v2.5"

# ── DeepSeek LLM ────────────────────────────────────────────────────

def load_system_prompt():
    """Extract the system prompt from node 134:114 in the template."""
    with open(TEMPLATE_PATH) as f:
        template = json.load(f)
    node = template.get("134:114", {})
    value = node.get("inputs", {}).get("value", "")
    if not value:
        # Fallback: try widgets_values
        value = node.get("widgets_values", [""])[0] if node.get("widgets_values") else ""
    return value


def strip_thinking(text):
    """Remove <think>...</think> blocks from DeepSeek response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def deepseek_transform(user_prompt, _system_prompt_unused="", width=1024, height=1024):
    """Call OpenCode to transform a prompt into Ideogram 4 JSON schema."""
    if not OPENCODE_BIN.exists():
        print(f"  WARNING: {OPENCODE_BIN} not found, using raw prompt")
        return user_prompt

    schema_prompt = f"""You are an expert image prompt engineer for Ideogram 4. Convert the user's idea into a structured JSON caption.

OUTPUT — exactly one single-line minified JSON, no markdown, no commentary:
{{"high_level_description":"vivid one-two sentence summary","style_description":{{"aesthetics":"style keywords","lighting":"lighting description","photo":"camera/lens details","medium":"photograph","color_palette":["#RRGGBB"]}},"compositional_deconstruction":{{"background":"scene shell description","elements":[{{"type":"obj","bbox":[y_min,x_min,y_max,x_max],"desc":"detailed element description"}}]}}}}

RULES:
- Output ONLY the JSON, no markdown fences, no commentary
- Bbox normalized 0-1000, [y_min, x_min, y_max, x_max]
- style_description key order: aesthetics, lighting, photo (or art_style), medium, color_palette (optional, last)
- elements: one per distinct subject. Parts belong in desc
- color_palette: uppercase #RRGGBB
- Keep high_level_description under 50 words
- Keep element desc 30-60 words
- background describes the scene SHELL only

TARGET RESOLUTION: {width}x{height}

USER IDEA: {user_prompt}

JSON:"""

    try:
        result = subprocess.run(
            [str(OPENCODE_BIN), "run", "-m", OPENCODE_MODEL, schema_prompt],
            capture_output=True, text=True, timeout=90,
        )
        output = result.stdout.strip()
        # Extract JSON from output (opencode may add metadata lines)
        lines = output.split("\n")
        json_lines = [l for l in lines if l.strip().startswith("{")]
        cleaned = json_lines[-1].strip() if json_lines else output.strip()
        # Strip thinking tags if any
        cleaned = strip_thinking(cleaned)
        parsed = json.loads(cleaned)
        assert "compositional_deconstruction" in parsed, "Missing compositional_deconstruction"
        return cleaned
    except json.JSONDecodeError as e:
        print(f"  LLM returned invalid JSON: {e}")
        print(f"  Raw: {cleaned[:200]}")
        return user_prompt
    except subprocess.TimeoutExpired:
        print(f"  LLM timed out (120s)")
        return user_prompt
    except Exception as e:
        print(f"  LLM error: {e}")
        return user_prompt


# ── Experiment grid ──────────────────────────────────────────────────
PROMPTS = [
    # ── LANDSCAPE ──
    {"name": "landscape_ocean",     "prompt": "Crashing waves on a rocky coastline at golden hour, sea spray catching the light, distant lighthouse on a cliff, dramatic sky with layered clouds"},
    {"name": "landscape_mountain",  "prompt": "Alpine mountain valley with a turquoise glacial lake, wildflowers in the foreground, snow-capped peaks, clear blue sky, late morning light"},
    {"name": "landscape_desert",    "prompt": "Vast sand dunes stretching to the horizon at sunset, warm orange and red tones, a lone camel caravan in silhouette, clear gradient sky"},
    {"name": "landscape_forest",    "prompt": "Ancient redwood forest with sunbeams piercing through the canopy, moss-covered fallen log, ferns and mist, ethereal morning atmosphere"},
    {"name": "landscape_arctic",    "prompt": "Northern lights dancing over a frozen lake in Iceland, snow-covered mountains, ice formations in foreground, starry sky, long exposure feel"},
    {"name": "landscape_urban",     "prompt": "Tokyo skyline at twilight viewed from a hillside park, city lights beginning to glow, Mount Fuji in the far distance, deep blue hour sky"},
    # ── PORTRAIT ──
    {"name": "portrait_studio",     "prompt": "Professional studio portrait of a middle-aged man with salt-and-pepper beard, dramatic Rembrandt lighting, dark backdrop, wearing a charcoal suit"},
    {"name": "portrait_outdoor",    "prompt": "Candid portrait of a young woman laughing in a sunflower field, golden hour backlighting, wind in her hair, shallow depth of field"},
    {"name": "portrait_elderly",    "prompt": "Close-up portrait of an elderly woman with deep wrinkles and kind eyes, natural window light, simple white blouse, warm neutral background"},
    {"name": "portrait_child",      "prompt": "Portrait of a curious six-year-old boy exploring tide pools, crouching on wet rocks, morning light, candid expression of wonder"},
    {"name": "portrait_group",      "prompt": "Group portrait of three friends sitting on a park bench, casual autumn clothing, laughing together, soft overcast daylight, urban park background"},
    {"name": "portrait_profile",    "prompt": "Side profile portrait of a woman with braided hair against a deep blue backdrop, studio rim lighting, elegant and minimal composition"},
    # ── ANIMALS ──
    {"name": "animals_wildlife",    "prompt": "African elephant walking across the savanna at sunset, dust kicked up by its feet, acacia trees in silhouette, warm amber light"},
    {"name": "animals_pet",         "prompt": "A ginger tabby cat curled up on a windowsill watching rain fall outside, cozy indoor setting, soft diffused light, shallow focus"},
    {"name": "animals_bird",        "prompt": "Bald eagle in flight over a mountain lake, wings fully spread, reflection visible in still water, crisp morning light, wildlife photography"},
    {"name": "animals_marine",      "prompt": "Sea turtle gliding through crystal clear tropical water, coral reef below, sunlight filtering through the surface, underwater photography"},
    {"name": "animals_insect",      "prompt": "Monarch butterfly perched on a purple coneflower, morning dew on wings, macro photography, soft bokeh background of a garden"},
    {"name": "animals_fantasy",     "prompt": "A majestic silver dragon perched on a mountain peak, scales reflecting moonlight, storm clouds gathering, fantasy art illustration style"},
    # ── ABSURD ──
    {"name": "absurd_surreal",      "prompt": "A grand piano floating in mid-air above a wheat field, keys playing themselves, sheet music scattering in the wind, photorealistic"},
    {"name": "absurd_scale",        "prompt": "A tiny person standing inside a giant teacup, steam rising around them like fog, surreal miniature world, whimsical documentary style"},
    {"name": "absurd_mashup",       "prompt": "A Victorian-era living room inside a giant seashell on a beach, ocean waves visible through the shell opening, photorealistic"},
    {"name": "absurd_physics",      "prompt": "Rain falling upward from the ground into dark clouds, people with upside-down umbrellas looking confused, overcast city street, photorealistic"},
    {"name": "absurd_time",         "prompt": "Ancient Roman colosseum with modern LED billboards and people in contemporary clothing, anachronistic mashup, golden hour, photorealistic"},
    {"name": "absurd_dreamscape",   "prompt": "A staircase made of floating books leading into clouds, a figure climbing upward, impossible architecture, warm ethereal light, photorealistic"},
    # ── STILL LIFE ──
    {"name": "stilllife_food",      "prompt": "Fresh pasta being hand-rolled on a floured wooden surface, herbs and olive oil nearby, warm kitchen light from a window, overhead angle"},
    {"name": "stilllife_flowers",   "prompt": "Overflowing bouquet of peonies and roses in a ceramic vase, petals scattered on a marble surface, soft natural sidelight, fine art photography"},
    {"name": "stilllife_vintage",   "prompt": "Antique brass compass and old leather-bound journal on a dark oak desk, candlelight, vintage exploration theme, moody chiaroscuro lighting"},
    {"name": "stilllife_organic",   "prompt": "Cut open pomegranate revealing jewel-like seeds, scattered on a dark slate surface, dramatic side lighting, food photography, macro detail"},
    {"name": "stilllife_tech",      "prompt": "Disassembled vintage camera parts arranged neatly on a light wooden surface, overhead flat lay, clean minimal aesthetic, soft even lighting"},
    {"name": "stilllife_mineral",   "prompt": "Collection of raw amethyst and quartz crystals on black velvet, studio spotlight creating dramatic shadows and internal reflections"},
    # ── ARCHITECTURE ──
    {"name": "arch_modern",         "prompt": "Sleek glass and steel skyscraper reflecting clouds, dramatic low-angle perspective, blue sky, minimalist modern architecture photography"},
    {"name": "arch_ancient",        "prompt": "Greek marble temple ruins at sunrise, columns casting long shadows, wild grass growing between stones, clear sky, fine art photography"},
    {"name": "arch_interior",       "prompt": "Grand library interior with floor-to-ceiling bookshelves, reading lamps glowing warm, spiral staircase, symmetrical composition, rich wood tones"},
    {"name": "arch_exterior",       "prompt": "Colorful row of Amsterdam canal houses reflected in still water, overcast sky, bicycles parked along the bridge, travel photography"},
    {"name": "arch_ruin",           "prompt": "Abandoned Art Deco theater with crumbling ornate plasterwork, shafts of light through broken roof, nature reclaiming, moody atmospheric"},
    {"name": "arch_futuristic",     "prompt": "Futuristic parametric architecture building with flowing organic curves, white composite material, reflecting pool, clear sky, architectural photography"},
]

SETTINGS = [
    {"name": "turbo_cfg3",   "cfg": 3.0,  "preset": "Turbo"},
    {"name": "turbo_cfg5",   "cfg": 5.0,  "preset": "Turbo"},
    {"name": "default_cfg3", "cfg": 3.0,  "preset": "Default"},
    {"name": "default_cfg7", "cfg": 7.0,  "preset": "Default"},
]

PRESET_MAP = {
    "Quality": {"index": 0},
    "Default": {"index": 1},
    "Turbo":   {"index": 2},
}

# ── Workflow manipulation ────────────────────────────────────────────

def load_template():
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def patch_prompt(template, prompt_text, setting):
    """Patch the prompt dict in-place for one experiment run."""
    p = json.loads(json.dumps(template))  # deep copy

    # Try to parse prompt_text as Ideogram 4 JSON schema
    try:
        caption = json.loads(prompt_text)
        is_json = isinstance(caption, dict) and "compositional_deconstruction" in caption
    except (json.JSONDecodeError, TypeError):
        is_json = False

    if "209" in p:
        if is_json:
            # Map JSON schema to node 209 inputs
            p["209"]["inputs"]["high_level_description"] = caption.get("high_level_description", "")
            style = caption.get("style_description", {})
            p["209"]["inputs"]["aesthetics"] = style.get("aesthetics", "")
            p["209"]["inputs"]["lighting"] = style.get("lighting", "")
            p["209"]["inputs"]["medium"] = style.get("medium", "")
            # DynamicCombo: plain string selection + dotted sub-input keys
            photo_val = style.get("photo", "")
            art_val = style.get("art_style", "")
            if photo_val:
                p["209"]["inputs"]["style"] = "photo"
                p["209"]["inputs"]["style.photo"] = photo_val
            elif art_val:
                p["209"]["inputs"]["style"] = "art_style"
                p["209"]["inputs"]["style.art_style"] = art_val
            else:
                p["209"]["inputs"]["style"] = "none"
            palette = style.get("color_palette", [])
            p["209"]["inputs"]["style_palette_data"] = json.dumps(palette) if palette else ""
            # Compositional deconstruction
            cd = caption.get("compositional_deconstruction", {})
            p["209"]["inputs"]["background"] = cd.get("background", "")
            elements = cd.get("elements", [])
            # Convert elements to the format the node expects (list of dicts with x,y,w,h)
            node_elements = []
            for el in elements:
                bbox = el.get("bbox", [])
                if len(bbox) == 4:
                    y1, x1, y2, x2 = bbox
                    node_el = {
                        "x": x1 / 1000.0,
                        "y": y1 / 1000.0,
                        "w": (x2 - x1) / 1000.0,
                        "h": (y2 - y1) / 1000.0,
                        "type": el.get("type", "obj"),
                        "text": el.get("text", ""),
                        "desc": el.get("desc", ""),
                        "palette": el.get("color_palette", []),
                    }
                else:
                    node_el = {
                        "x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8,
                        "type": el.get("type", "obj"),
                        "text": el.get("text", ""),
                        "desc": el.get("desc", ""),
                        "palette": el.get("color_palette", []),
                    }
                node_elements.append(node_el)
            p["209"]["inputs"]["elements_data"] = json.dumps(node_elements, separators=(",", ":"))
            p["209"]["inputs"]["width"] = 1024
            p["209"]["inputs"]["height"] = 1024
        else:
            # Plain text prompt — use as HLD directly
            p["209"]["inputs"]["high_level_description"] = prompt_text
            p["209"]["inputs"]["elements_data"] = ""
            p["209"]["inputs"]["width"] = 1024
            p["209"]["inputs"]["height"] = 1024
            p["209"]["inputs"]["background"] = ""
            p["209"]["inputs"]["style"] = "none"
            p["209"]["inputs"]["aesthetics"] = ""
            p["209"]["inputs"]["lighting"] = ""
            p["209"]["inputs"]["medium"] = ""

    # Node 134:115: User Prompt — clear
    if "134:115" in p:
        p["134:115"]["inputs"]["value"] = ""

    # Node 98:157: CFGOverride
    if "98:157" in p:
        p["98:157"]["inputs"]["cfg"] = setting["cfg"]

    # Node 98:156: CustomCombo
    if "98:156" in p:
        preset_info = PRESET_MAP[setting["preset"]]
        p["98:156"]["inputs"]["choice"] = setting["preset"]
        p["98:156"]["inputs"]["index"] = preset_info["index"]

    # Node 98:18: RandomNoise
    if "98:18" in p:
        p["98:18"]["inputs"]["noise_seed"] = random.randint(1, 2**48)

    return p


# ── ComfyUI API ─────────────────────────────────────────────────────

def queue_prompt(prompt_dict):
    """Queue a prompt dict and return prompt_id."""
    client_id = str(uuid.uuid4())
    payload = json.dumps({
        "prompt": prompt_dict,
        "client_id": client_id,
    }).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    return result["prompt_id"]


def poll_history(prompt_id, timeout=300):
    """Poll /history until prompt_id appears, return output info."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}")
            data = json.loads(resp.read())
            if prompt_id in data:
                status = data[prompt_id].get("status", {})
                if status.get("completed"):
                    return data[prompt_id]
        except Exception:
            pass
        time.sleep(2)
    return None


def download_image(filename, subfolder="", img_type="output"):
    params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": img_type})
    resp = urllib.request.urlopen(f"{COMFYUI_URL}/view?{params}")
    return resp.read()


# ── Contact sheet ────────────────────────────────────────────────────

def build_contact_sheet(results, cell_w=320, cell_h=256, padding=16, font_size=14):
    n_cols = len(SETTINGS)
    n_rows = len(PROMPTS)
    label_h = 80

    sheet_w = 160 + padding + n_cols * (cell_w + padding)
    sheet_h = padding + 50 + n_rows * (cell_h + label_h + padding)

    sheet = Image.new("RGB", (sheet_w, sheet_h), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size - 2)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size + 2)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size + 6)
    except Exception:
        font = font_small = font_header = font_title = ImageFont.load_default()

    # Title
    draw.text((padding, padding // 2), "Ideogram 4.0 Parameter Sweep", fill=(220, 220, 220), font=font_title)

    header_y = padding + 30

    # Column headers
    for col_idx, setting in enumerate(SETTINGS):
        x = 160 + padding + col_idx * (cell_w + padding)
        cfg_label = f"cfg={setting['cfg']:.0f}"
        draw.text((x + cell_w // 2, header_y), setting["preset"], fill=(180, 180, 180), font=font_header, anchor="mt")
        draw.text((x + cell_w // 2, header_y + 18), cfg_label, fill=(130, 130, 130), font=font_small, anchor="mt")

    # Rows
    for row_idx, prompt_info in enumerate(PROMPTS):
        y_row = header_y + 40 + row_idx * (cell_h + label_h + padding)

        # Row label
        draw.text((padding + 150, y_row + cell_h // 2), prompt_info["name"], fill=(180, 180, 180), font=font_header, anchor="rm")

        for col_idx, setting in enumerate(SETTINGS):
            key = (prompt_info["name"], setting["name"])
            img_path = results.get(key)

            x = 160 + padding + col_idx * (cell_w + padding)
            y_img = y_row

            if img_path and img_path.exists():
                thumb = Image.open(img_path).convert("RGB")
                thumb.thumbnail((cell_w, cell_h))
                tx = x + (cell_w - thumb.width) // 2
                ty = y_img + (cell_h - thumb.height) // 2
                sheet.paste(thumb, (tx, ty))
            else:
                draw.rectangle([x, y_img, x + cell_w, y_img + cell_h], outline=(60, 60, 60))
                status = "FAILED" if key in results else "PENDING"
                draw.text((x + cell_w // 2, y_img + cell_h // 2), status, fill=(255, 80, 80), font=font, anchor="mm")

            # Label
            y_text = y_img + cell_h + 4
            snippet = prompt_info["prompt"][:55] + "..." if len(prompt_info["prompt"]) > 55 else prompt_info["prompt"]
            draw.text((x + 4, y_text), snippet, fill=(100, 100, 100), font=font_small)

    return sheet


# ── HTML contact sheet (copy-able prompts) ───────────────────────────

def build_html_contact_sheet(results, transformed_prompts=None):
    """Build an HTML file with embedded images and copy-able prompt text."""
    import base64

    if transformed_prompts is None:
        transformed_prompts = {}

    rows_html = ""
    for prompt_info in PROMPTS:
        cells = ""
        for setting in SETTINGS:
            key = (prompt_info["name"], setting["name"])
            img_path = results.get(key)
            if img_path and img_path.exists():
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                img_tag = f'<img src="data:image/png;base64,{b64}" />'
            else:
                img_tag = '<div class="placeholder">NO IMAGE</div>'

            raw_escaped = prompt_info["prompt"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("\n", "<br>")
            transformed = transformed_prompts.get(prompt_info["name"], prompt_info["prompt"])
            trans_escaped = transformed.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("\n", "<br>")

            cells += f"""
            <td>
              <div class="cell">
                <div class="img-wrap">{img_tag}</div>
                <div class="meta">
                  <div class="settings">{setting['preset']} | cfg={setting['cfg']:.0f}</div>
                  <div class="prompt-label">RAW:</div>
                  <div class="prompt" onclick="navigator.clipboard.writeText(this.dataset.prompt)" data-prompt="{prompt_info['prompt'].replace('"', '&quot;')}" title="Click to copy">{raw_escaped}</div>
                  <div class="prompt-label">LLM:</div>
                  <div class="prompt transformed" onclick="navigator.clipboard.writeText(this.dataset.prompt)" data-prompt="{transformed.replace('"', '&quot;')}" title="Click to copy">{trans_escaped}</div>
                </div>
              </div>
            </td>"""
        rows_html += f"""
        <tr>
          <td class="row-label">{prompt_info['name']}</td>
          {cells}
        </tr>"""

    header_cells = "".join(
        f'<th>{s["preset"]}<br><small>cfg={s["cfg"]:.0f}</small></th>'
        for s in SETTINGS
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Ideogram 4.0 Parameter Sweep</title>
<style>
  body {{ background: #1a1a1a; color: #ddd; font-family: 'DejaVu Sans', sans-serif; margin: 20px; }}
  h1 {{ font-size: 18px; color: #eee; margin-bottom: 4px; }}
  p.hint {{ font-size: 12px; color: #888; margin-top: 0; }}
  table {{ border-collapse: collapse; }}
  th {{ padding: 8px 12px; text-align: center; font-size: 14px; color: #bbb; border-bottom: 1px solid #333; }}
  th small {{ color: #888; }}
  td.row-label {{ writing-mode: vertical-lr; text-orientation: mixed; transform: rotate(180deg);
                   font-size: 14px; font-weight: bold; color: #bbb; padding: 8px 4px;
                   text-align: center; white-space: nowrap; }}
  .cell {{ width: 340px; margin: 6px; }}
  .img-wrap {{ background: #111; border-radius: 4px; overflow: hidden; text-align: center; }}
  .img-wrap img {{ max-width: 100%; max-height: 280px; display: block; margin: 0 auto; }}
  .placeholder {{ height: 200px; line-height: 200px; color: #555; text-align: center; }}
  .meta {{ padding: 6px 0; }}
  .settings {{ font-size: 11px; color: #999; margin-bottom: 4px; font-weight: bold; }}
  .prompt {{ font-size: 12px; color: #aaa; cursor: pointer; padding: 4px 6px;
             border-radius: 3px; transition: background 0.15s; word-break: break-word;
             max-height: 120px; overflow-y: auto; }}
  .prompt:hover {{ background: #2a2a3a; color: #ddd; }}
  .prompt:active {{ background: #3a3a5a; }}
  .prompt.transformed {{ font-size: 11px; color: #8a8; background: #1a1f1a; }}
  .prompt.transformed:hover {{ background: #2a3a2a; color: #afa; }}
  .prompt-label {{ font-size: 10px; color: #666; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
</style>
</head>
<body>
  <h1>Ideogram 4.0 Parameter Sweep</h1>
  <p class="hint">Click any prompt to copy it to clipboard.</p>
  <table>
    <tr><th></th>{header_cells}</tr>
    {rows_html}
  </table>
</body>
</html>"""
    return html


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    template = load_template()

    total = len(PROMPTS) * len(SETTINGS)
    print(f"Ideogram 4.0 Parameter Sweep")
    print(f"  {len(PROMPTS)} prompts x {len(SETTINGS)} settings = {total} runs")
    print(f"  LLM: OpenCode ({OPENCODE_MODEL})" if OPENCODE_BIN.exists() else "  LLM: none (raw prompts)")
    print(f"  Output: {OUTPUT_DIR}\n")

    # Transform prompts with LLM — load from cache, fill missing
    transformed = {}
    if PROMPTS_CACHE.exists():
        with open(PROMPTS_CACHE) as f:
            cache = json.load(f)
        for k, v in cache.items():
            if isinstance(v, dict):
                transformed[k] = v.get("llm", v.get("transformed", ""))
            else:
                transformed[k] = v
        print(f"Loaded {len(transformed)} cached prompts from {PROMPTS_CACHE}\n")
    else:
        cache = {}

    # Fill any missing or empty prompts via LLM
    missing_prompts = [p for p in PROMPTS if not transformed.get(p["name"], "").strip()]
    if missing_prompts:
        print(f"Transforming {len(missing_prompts)} missing prompts with LLM...")
        for prompt_info in missing_prompts:
            raw = prompt_info["prompt"]
            print(f"  {prompt_info['name']}...", end=" ", flush=True)
            result = deepseek_transform(raw, "")
            transformed[prompt_info["name"]] = result
            cache[prompt_info["name"]] = {"raw": raw, "llm": result}
            # Save after each prompt (crash-safe)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(PROMPTS_CACHE, "w") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            try:
                parsed = json.loads(result)
                n_elements = len(parsed.get("compositional_deconstruction", {}).get("elements", []))
                print(f"OK ({n_elements} elements)")
            except Exception:
                print(f"OK ({len(result)} chars)")
            time.sleep(0.5)
        print(f"Prompts saved to {PROMPTS_CACHE}\n")

    # Process jobs sequentially — queue one, wait, queue next
    print(f"\nRunning {total} jobs sequentially...")
    results = {}
    done = 0
    for prompt_info in PROMPTS:
        hld = transformed[prompt_info["name"]]
        for setting in SETTINGS:
            done += 1
            label = f"{prompt_info['name']}_{setting['name']}"
            out_path = OUTPUT_DIR / f"{label}.png"

            # Skip if already generated
            if out_path.exists() and out_path.stat().st_size > 1000:
                results[(prompt_info["name"], setting["name"])] = out_path
                print(f"  [{done}/{total}] {label}... SKIP (exists)")
                continue

            p = patch_prompt(template, hld, setting)
            prompt_id = queue_prompt(p)
            print(f"  [{done}/{total}] {label}...", end=" ", flush=True)

            # Wait for this specific job to complete
            history = poll_history(prompt_id, timeout=600)
            if history and "outputs" in history:
                found = False
                # Prefer node 158 (SaveImage from Ideogram 4), skip SUPIR
                out = history["outputs"].get("158", {})
                if "images" in out:
                    for img_info in out["images"]:
                        img_data = download_image(
                            img_info["filename"],
                            img_info.get("subfolder", ""),
                            img_info.get("type", "output"),
                        )
                        out_path = OUTPUT_DIR / f"{label}.png"
                        with open(out_path, "wb") as f:
                            f.write(img_data)
                        results[(prompt_info["name"], setting["name"])] = out_path
                        print(f"OK ({len(img_data)//1024}KB)")
                        found = True
                        break
                if not found:
                    print("NO IMAGES")
                    results[(prompt_info["name"], setting["name"])] = None
            else:
                print("TIMEOUT")
                results[(prompt_info["name"], setting["name"])] = None

    print(f"\n{done} jobs complete.")

    # Build contact sheets
    print("\nBuilding contact sheets...")
    sheet = build_contact_sheet(results)
    sheet.save(str(CONTACT_SHEET_PATH), quality=95)

    html = build_html_contact_sheet(results, transformed)
    html_path = OUTPUT_DIR / "contact_sheet.html"
    with open(html_path, "w") as f:
        f.write(html)

    print(f"PNG:  {CONTACT_SHEET_PATH}")
    print(f"HTML: {html_path}  (open in browser, click prompts to copy)")
    print(f"Images: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
