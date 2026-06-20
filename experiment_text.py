#!/usr/bin/env python3
"""
Ideogram 4.0 Text Rendering Experiment
Tests literal text rendering in images: posters, banners, signs, packaging, etc.
Single setting: Turbo preset, CFG 3.0
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
OUTPUT_DIR = Path.home() / "ComfyUI" / "output" / "experiment_text"
PROMPTS_CACHE = OUTPUT_DIR / "prompts.json"
CONTACT_SHEET_PATH = OUTPUT_DIR / "contact_sheet.png"
OPENCODE_BIN = Path.home() / ".opencode" / "bin" / "opencode"
OPENCODE_MODEL = "opencode-go/mimo-v2.5"

SETTING = {"name": "turbo_cfg3", "cfg": 3.0, "preset": "Turbo"}

PRESET_MAP = {
    "Quality": {"index": 0},
    "Default": {"index": 1},
    "Turbo":   {"index": 2},
}

# ── 30 text-heavy prompts ────────────────────────────────────────────
# Categories: posters, banners, signs, packaging, covers, jerseys, tickets, etc.

PROMPTS = [
    # ── MOVIE / EVENT POSTERS ──
    {"name": "poster_scifi",
     "prompt": "A cinematic movie poster for a sci-fi film called 'ECHOES OF TITAN'. A lone astronaut silhouette against Saturn's rings. Bold sans-serif title at top, 'COMING SOON' at bottom, PG-13 rating badge. Dark moody color palette.",
     "text_elements": ["ECHOES OF TITAN", "COMING SOON", "PG-13"]},

    {"name": "poster_horror",
     "prompt": "A horror movie poster for 'THE HOLLOW'. Creepy dark forest with a single lit window in a distant cabin. Title in distressed serif font across the top, tagline 'FEAR HAS A NEW ADDRESS' below. Red and black color scheme.",
     "text_elements": ["THE HOLLOW", "FEAR HAS A NEW ADDRESS"]},

    {"name": "poster_comedy",
     "prompt": "A comedy film poster for 'ACCIDENTAL DAD'. A flustered man juggling baby bottles, car keys and a briefcase. Bright yellow background, playful rounded title font, 'THIS SUMMER' tagline. Fun vibrant colors.",
     "text_elements": ["ACCIDENTAL DAD", "THIS SUMMER"]},

    {"name": "poster_concert",
     "prompt": "A concert poster for 'MIDNIGHT ECHO LIVE IN BERLIN'. Neon typography on black background, electric blue and magenta gradient text, date 'DEC 15 2026' and venue 'WAREHOUSE 42'. Art deco border装饰.",
     "text_elements": ["MIDNIGHT ECHO", "LIVE IN BERLIN", "DEC 15 2026", "WAREHOUSE 42"]},

    {"name": "poster_festival",
     "prompt": "A music festival poster for 'SOLSTICE FEST 2026'. Colorful psychedelic lettering, lineup listing multiple band names in varying sizes, 'JUNE 20-22' dates, 'DESERT VALLEY CAMPGROUND'. Tie-dye watercolor background.",
     "text_elements": ["SOLSTICE FEST 2026", "JUNE 20-22", "DESERT VALLEY CAMPGROUND"]},

    # ── BOOK / ALBUM COVERS ──
    {"name": "bookcover_thriller",
     "prompt": "A thriller novel book cover for 'THE LAST WITNESS' by Sarah Chen. Dark alley at night with a lone figure, bold white title text across center, author name at bottom in smaller serif font. Penguin Books style layout.",
     "text_elements": ["THE LAST WITNESS", "Sarah Chen"]},

    {"name": "bookcover_fantasy",
     "prompt": "A fantasy book cover for 'THRONE OF BROKEN CROWS'. Ornate gold lettering on dark leather-textured background, dragon silhouette, 'A SONG OF EMPIRES BOOK ONE' series subtitle. Elaborate decorative border.",
     "text_elements": ["THRONE OF BROKEN CROWS", "A SONG OF EMPIRES BOOK ONE"]},

    {"name": "albumcover_hiphop",
     "prompt": "A hip-hop album cover for 'GOLDEN HOUR' by KAE. Warm sunset gradient background, silhouette of a city skyline, bold blocky white title text, 'FEAT. LIL BREEZE' in small text. Clean minimalist design.",
     "text_elements": ["GOLDEN HOUR", "KAE", "FEAT. LIL BREEZE"]},

    {"name": "albumcover_electronic",
     "prompt": "An electronic music album cover for 'SYNTHWAVE DREAMS' by NEON PULSE. Retro 80s grid landscape with neon pink and cyan, chrome title text with glow effect, 'OUT NOW' badge. VHS aesthetic.",
     "text_elements": ["SYNTHWAVE DREAMS", "NEON PULSE", "OUT NOW"]},

    {"name": "magazine_cover",
     "prompt": "A Vogue magazine cover featuring a woman in red lipstick, masthead 'VOGUE' in large serif letters at top, cover lines 'THE NEW RULES OF BEAUTY' and 'SPRING COLLECTION 2026'. Clean high-fashion layout.",
     "text_elements": ["VOGUE", "THE NEW RULES OF BEAUTY", "SPRING COLLECTION 2026"]},

    # ── SIGNS / NEON ──
    {"name": "neon_bar",
     "prompt": "A neon bar sign glowing 'THE RUSTY ANCHOR' in cursive orange neon tubes against a dark brick wall. Below it a smaller green neon 'OPEN 24 HOURS'. Wet reflections on pavement below. Moody night atmosphere.",
     "text_elements": ["THE RUSTY ANCHOR", "OPEN 24 HOURS"]},

    {"name": "neon_diner",
     "prompt": "A retro 1950s diner sign reading 'ROSIE'S DINER' in pink and white neon, with 'BEST PIE IN TOWN' smaller neon below. Classic chrome diner visible through window. Nostalgic Americana color palette.",
     "text_elements": ["ROSIE'S DINER", "BEST PIE IN TOWN"]},

    {"name": "storefront_bakery",
     "prompt": "A charming bakery storefront window with gold-leaf lettering on glass reading 'LA PETITE BOULANGERIE'. Below in smaller text 'ARTISAN BREADS & PASTRIES SINCE 1987'. Warm golden interior light visible through window.",
     "text_elements": ["LA PETITE BOULANGERIE", "ARTISAN BREADS & PASTRIES SINCE 1987"]},

    {"name": "street_sign_old",
     "prompt": "An old weathered street sign reading 'WALL ST' in white on green metal, slightly rusted edges. Financial district buildings blurred in background. Photorealistic urban photography.",
     "text_elements": ["WALL ST"]},

    {"name": "protest_sign",
     "prompt": "A hand-painted protest sign held above a crowd reading 'THE FUTURE IS FEMALE' in bold black marker on white cardboard. Diverse crowd in soft focus behind. Documentary photography style.",
     "text_elements": ["THE FUTURE IS FEMALE"]},

    # ── PACKAGING / PRODUCTS ──
    {"name": "wine_label",
     "prompt": "A wine bottle label for 'CHATEAU MIDNIGHT' 2024 Cabernet Sauvignon. Elegant serif typography on cream parchment label, ornate border, 'NAPA VALLEY' appellation, 'RESERVE' in gold foil. Photorealistic product shot.",
     "text_elements": ["CHATEAU MIDNIGHT", "2024 Cabernet Sauvignon", "NAPA VALLEY", "RESERVE"]},

    {"name": "coffee_bag",
     "prompt": "A craft coffee bean bag with kraft paper texture, reading 'BLUE MOUNTAIN ROASTERS' in bold stamped black ink. 'SINGLE ORIGIN COLOMBIA' below, 'NET WT 12 OZ' at bottom. Artisanal minimalist design.",
     "text_elements": ["BLUE MOUNTAIN ROASTERS", "SINGLE ORIGIN COLOMBIA", "NET WT 12 OZ"]},

    {"name": "soda_can",
     "prompt": "A retro soda can design for 'FIZZ POP COLA'. Red and white vertical stripes, bold cursive logo text, 'ESTABLISHED 1952' in small text, '12 FL OZ' at bottom. Classic Americana product photography.",
     "text_elements": ["FIZZ POP COLA", "ESTABLISHED 1952", "12 FL OZ"]},

    {"name": "cereal_box",
     "prompt": "A colorful kids cereal box for 'STAR CRUNCH'. Cartoon star mascot, bold yellow and blue title text, 'WITH REAL HONEY' callout badge, 'NET WT 14 OZ' at bottom. Bright playful packaging design.",
     "text_elements": ["STAR CRUNCH", "WITH REAL HONEY", "NET WT 14 OZ"]},

    {"name": "tech_box",
     "prompt": "A premium tech product box for 'NOVA PRO' wireless headphones. Clean white minimalist box, product image on front, 'NOVA PRO' in sleek sans-serif, '40H BATTERY · ANC · HI-RES' specs line. Apple-style packaging.",
     "text_elements": ["NOVA PRO", "40H BATTERY · ANC · HI-RES"]},

    # ── TICKETS / PASSES / CARDS ──
    {"name": "concert_ticket",
     "prompt": "A concert ticket for 'MIDNIGHT ECHO - LIVE'. Holographic foil effect, 'ADMIT ONE' at top, seat info 'SECTION A · ROW 12 · SEAT 7', date 'DEC 15 2026', barcode at bottom. Premium event ticket design.",
     "text_elements": ["MIDNIGHT ECHO - LIVE", "ADMIT ONE", "SECTION A · ROW 12 · SEAT 7", "DEC 15 2026"]},

    {"name": "boarding_pass",
     "prompt": "An airline boarding pass for 'SKYLINE AIRWAYS'. Flight 'SL 2847' from 'NEW YORK (JFK)' to 'TOKYO (NRT)', gate 'B42', seat '14A', 'BOARDING TIME 10:30 AM'. Clean airline design with blue accent stripe.",
     "text_elements": ["SKYLINE AIRWAYS", "SL 2847", "NEW YORK (JFK) → TOKYO (NRT)", "GATE B42", "SEAT 14A", "BOARDING TIME 10:30 AM"]},

    {"name": "business_card",
     "prompt": "A luxury business card for 'VICTORIA STERLING' at 'STERLING & ASSOCIATES'. Gold foil text on matte black card, 'FOUNDING PARTNER' title, phone and email in small clean font. Elegant minimal design.",
     "text_elements": ["VICTORIA STERLING", "STERLING & ASSOCIATES", "FOUNDING PARTNER"]},

    {"name": "movie_ticket_stub",
     "prompt": "A vintage movie ticket stub for 'CINEMA PARADISO'. Perforated edge, 'ADMIT ONE' and 'SEAT G7' printed, date 'MAR 15', theater number 'SCREEN 3'. Retro halftone print texture, faded red and cream.",
     "text_elements": ["CINEMA PARADISO", "ADMIT ONE", "SEAT G7", "SCREEN 3"]},

    # ── BANNERS / SIGNAGE ──
    {"name": "grand_opening",
     "prompt": "A 'GRAND OPENING' banner across a storefront. Red and gold letters on white vinyl banner, 'NOW OPEN' smaller text below, confetti graphics. Celebratory bunting and balloons around it.",
     "text_elements": ["GRAND OPENING", "NOW OPEN"]},

    {"name": "marathon_banner",
     "prompt": "A finish line banner for 'CITY MARATHON 2026'. Large text 'FINISH' stretched across the arch, '42.195 KM' distance marker, sponsor logos along bottom. Bright sunny day, crowd in background.",
     "text_elements": ["FINISH", "CITY MARATHON 2026", "42.195 KM"]},

    {"name": "welcome_mat",
     "prompt": "A doormat with the text 'OH HELLO' in bold friendly serif font. Coir texture mat on a wooden porch doorstep, potted plant beside it. Warm homey atmosphere, overhead angle.",
     "text_elements": ["OH HELLO"]},

    # ── DIGITAL / UI ──
    {"name": "app_icon",
     "prompt": "A mobile app icon design for 'MEDITO'. Clean gradient from deep purple to teal, minimalist lotus flower icon, app name 'MEDITO' in thin white sans-serif below. Flat design, rounded corners. App Store style.",
     "text_elements": ["MEDITO"]},

    {"name": "game_cover",
     "prompt": "A video game cover for 'PHANTOM EDGE'. Cyberpunk cityscape background, protagonist with glowing visor, bold metallic title 'PHANTOM EDGE' across top, 'RATED M' badge, 'PS5' platform banner. AAA game art style.",
     "text_elements": ["PHANTOM EDGE", "RATED M", "PS5"]},

    {"name": "emoji_tshirt",
     "prompt": "A t-shirt graphic design with text 'KEEP IT WEIRD' in wavy psychedelic rainbow lettering on a black t-shirt. Retro 70s style typography with small star and moon decorative elements around it.",
     "text_elements": ["KEEP IT WEIRD"]},

    # ── REAL LIFE SCENES (10) ──
    {"name": "scene_tokyo_street",
     "prompt": "A bustling Tokyo street at night seen from a pedestrian's perspective. Overhead neon signs reading 'カラオケ' and '居酒屋' in glowing kanji, a Family Mart convenience store with 'ファミリーマート' signage. Rain-slicked asphalt reflecting colorful lights, crowds with umbrellas. Photorealistic street photography.",
     "text_elements": ["カラオケ", "居酒屋", "ファミリーマート"]},

    {"name": "scene_diner_interior",
     "prompt": "Interior of a cluttered American diner at breakfast time. A chalkboard menu on the wall reads 'DAILY SPECIALS: PANCAKES $8.99, BACON & EGGS $10.50, HASH BROWNS $4.99'. Salt and pepper shakers on red vinyl booth, waitress carrying plates. Warm fluorescent lighting, photorealistic.",
     "text_elements": ["DAILY SPECIALS", "PANCAKES $8.99", "BACON & EGGS $10.50", "HASH BROWNS $4.99"]},

    {"name": "scene_construction_site",
     "prompt": "A rainy urban construction site with orange safety fencing. A yellow warning sign reads 'DANGER HARD HAT AREA AUTHORIZED PERSONNEL ONLY', another sign shows 'XYZ CONSTRUCTION LLC' with phone number '555-0142'. Cranes and scaffolding in foggy background. Gritty documentary photography.",
     "text_elements": ["DANGER HARD HAT AREA AUTHORIZED PERSONNEL ONLY", "XYZ CONSTRUCTION LLC", "555-0142"]},

    {"name": "scene_farmers_market",
     "prompt": "A sunny outdoor farmer's market stall overflowing with fresh produce. Handwritten cardboard price signs reading 'HEIRLOOM TOMATOES $4.50/lb', 'FRESH BASIL $2.00', 'ORGANIC STRAWBERRIES $6.00'. Wooden crates, burlap sacks, shoppers browsing. Warm natural light, food photography.",
     "text_elements": ["HEIRLOOM TOMATOES $4.50/lb", "FRESH BASIL $2.00", "ORGANIC STRAWBERRIES $6.00"]},

    {"name": "scene_record_store",
     "prompt": "Interior of a cramped vintage record store. Walls covered floor to ceiling with vinyl album covers. A hand-written sale sign reads 'EVERYTHING MUST GO 50% OFF'. A neon 'RECORDS' sign glows in the window. Staff member behind counter with 'CASH ONLY' register sign. Nostalgic warm tungsten lighting.",
     "text_elements": ["EVERYTHING MUST GO 50% OFF", "RECORDS", "CASH ONLY"]},

    {"name": "scene_bus_stop_rain",
     "prompt": "A rainy city bus stop at night with an illuminated ad poster behind glass. The poster shows 'SUMMER SALE 40% OFF EVERYTHING' with a smiling model. The bus route sign above reads 'ROUTE 42 - DOWNTOWN'. Wet bench, puddles reflecting neon from nearby shops. Moody cinematic atmosphere.",
     "text_elements": ["SUMMER SALE 40% OFF EVERYTHING", "ROUTE 42 - DOWNTOWN"]},

    {"name": "scene_gym_wall",
     "prompt": "A gritty CrossFit gym interior with exposed brick walls. A large painted mural reads 'NO EXCUSES' in bold white capital letters across the back wall. Dumbbells, barbells, chalk buckets. An athlete mid-deadlift. Dramatic overhead industrial lighting, sweat and chalk dust in the air.",
     "text_elements": ["NO EXCUSES"]},

    {"name": "scene_laundromat",
     "prompt": "A late-night laundromat with rows of spinning washing machines. A faded sign above the counter reads 'WASH $4.00 DRY $3.00 OPEN 24 HOURS'. A chalkboard says 'OUT OF ORDER — SORRY!'. Fluorescent lights, linoleum floor, a woman reading a magazine on a plastic chair. Lonely urban atmosphere.",
     "text_elements": ["WASH $4.00 DRY $3.00 OPEN 24 HOURS", "OUT OF ORDER — SORRY!"]},

    {"name": "scene_bookstore",
     "prompt": "A cozy independent bookstore interior with floor-to-ceiling shelves. A hand-lettered sign on the counter reads 'BOOK CLUB MEETS FRIDAY 7PM — NEW MEMBERS WELCOME'. Staff picks shelf labeled 'OUR FAVORITES THIS MONTH'. Warm lamp light, stacked books on every surface, cat sleeping on a chair.",
     "text_elements": ["BOOK CLUB MEETS FRIDAY 7PM — NEW MEMBERS WELCOME", "OUR FAVORITES THIS MONTH"]},

    {"name": "scene_airport_gate",
     "prompt": "A busy airport departure gate viewed from a passenger seat. The flight information display reads 'FLIGHT AA 1247 → LOS ANGELES GATE B12 BOARDING 2:45 PM DELAYED'. Passengers with luggage, duty-free shop in background with 'TAX FREE' signage. Cool fluorescent terminal lighting.",
     "text_elements": ["FLIGHT AA 1247 → LOS ANGELES", "GATE B12", "BOARDING 2:45 PM", "DELAYED", "TAX FREE"]},
]


# ── Text-specialized LLM transform ───────────────────────────────────

def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def text_render_transform(user_prompt, width=1024, height=1024):
    """Transform a prompt into Ideogram 4 JSON with emphasis on literal text rendering."""
    if not OPENCODE_BIN.exists():
        print(f"  WARNING: {OPENCODE_BIN} not found, using raw prompt")
        return user_prompt

    schema_prompt = f"""You are an expert Ideogram 4 prompt engineer specializing in TEXT RENDERING. Convert the user's idea into a structured JSON caption.

OUTPUT — exactly one single-line minified JSON, no markdown, no commentary:
{{"high_level_description":"vivid one-two sentence summary","style_description":{{"aesthetics":"style keywords","lighting":"lighting description","art_style":"art style description","medium":"graphic_design","color_palette":["#RRGGBB"]}},"compositional_deconstruction":{{"background":"scene shell description","elements":[{{"type":"text","bbox":[y_min,x_min,y_max,x_max],"text":"THE EXACT LITERAL TEXT TO RENDER","desc":"description of font style, size, color, material","color_palette":["#RRGGBB"]}},{{"type":"obj","bbox":[y_min,x_min,y_max,x_max],"desc":"object description"}}]}}}}

CRITICAL RULES FOR TEXT RENDERING:
- Use "type":"text" elements with "text" field containing the EXACT literal text to render
- The "text" field is what appears ON the image — spell it exactly right
- For graphic design (posters, banners, signs, packaging), use "medium":"graphic_design" and "art_style" (NOT "photo")
- Key order in style_description: aesthetics, lighting, medium, art_style, color_palette (optional last)
- Each text element needs a "desc" describing the font style, color, and material
- Use "color_palette" per text element for precise color control
- Bbox normalized 0-1000, [y_min, x_min, y_max, x_max], origin top-left
- Include BOTH text elements AND object elements (background imagery, illustrations, etc.)

TEXT ELEMENT PATTERN:
{{"type":"text","bbox":[50,100,150,900],"text":"YOUR TEXT HERE","desc":"bold white sans-serif font, centered, large size","color_palette":["#FFFFFF","#000000"]}}

STYLE FOR DIFFERENT FORMATS:
- Movie posters: dark cinematic, bold title typography, "medium":"graphic_design"
- Neon signs: dark background, glowing text, "medium":"photograph" with neon art_style
- Product packaging: clean product photography, "medium":"photograph"
- Book covers: "medium":"graphic_design", "art_style":"book cover design"
- Banners: "medium":"graphic_design", "art_style":"event banner"
- Magazine covers: "medium":"photograph" with overlay text elements
- Tickets/boarding passes: "medium":"graphic_design", "art_style":"ticket design"

TARGET RESOLUTION: {width}x{height}

USER IDEA: {user_prompt}

JSON:"""

    try:
        result = subprocess.run(
            [str(OPENCODE_BIN), "run", "-m", OPENCODE_MODEL, schema_prompt],
            capture_output=True, text=True, timeout=90,
        )
        output = result.stdout.strip()
        lines = output.split("\n")
        json_lines = [l for l in lines if l.strip().startswith("{")]
        cleaned = json_lines[-1].strip() if json_lines else output.strip()
        cleaned = strip_thinking(cleaned)
        parsed = json.loads(cleaned)
        assert "compositional_deconstruction" in parsed, "Missing compositional_deconstruction"
        # Verify at least one text element exists
        elements = parsed.get("compositional_deconstruction", {}).get("elements", [])
        has_text = any(e.get("type") == "text" for e in elements)
        if not has_text:
            print(f"  WARNING: No text elements in LLM output, adding manually")
        return cleaned
    except json.JSONDecodeError as e:
        print(f"  LLM returned invalid JSON: {e}")
        print(f"  Raw: {cleaned[:200]}")
        return user_prompt
    except subprocess.TimeoutExpired:
        print(f"  LLM timed out (90s)")
        return user_prompt
    except Exception as e:
        print(f"  LLM error: {e}")
        return user_prompt


# ── Workflow manipulation ────────────────────────────────────────────

def load_template():
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def patch_prompt(template, prompt_text, setting):
    """Patch the prompt dict for one text experiment run."""
    p = json.loads(json.dumps(template))  # deep copy

    try:
        caption = json.loads(prompt_text)
        is_json = isinstance(caption, dict) and "compositional_deconstruction" in caption
    except (json.JSONDecodeError, TypeError):
        is_json = False

    if "209" in p:
        if is_json:
            p["209"]["inputs"]["high_level_description"] = caption.get("high_level_description", "")
            style = caption.get("style_description", {})
            p["209"]["inputs"]["aesthetics"] = style.get("aesthetics", "")
            p["209"]["inputs"]["lighting"] = style.get("lighting", "")
            p["209"]["inputs"]["medium"] = style.get("medium", "")
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
            cd = caption.get("compositional_deconstruction", {})
            p["209"]["inputs"]["background"] = cd.get("background", "")
            elements = cd.get("elements", [])
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
            p["209"]["inputs"]["high_level_description"] = prompt_text
            p["209"]["inputs"]["elements_data"] = ""
            p["209"]["inputs"]["width"] = 1024
            p["209"]["inputs"]["height"] = 1024
            p["209"]["inputs"]["background"] = ""
            p["209"]["inputs"]["style"] = "none"
            p["209"]["inputs"]["aesthetics"] = ""
            p["209"]["inputs"]["lighting"] = ""
            p["209"]["inputs"]["medium"] = ""

    if "134:115" in p:
        p["134:115"]["inputs"]["value"] = ""
    if "98:157" in p:
        p["98:157"]["inputs"]["cfg"] = setting["cfg"]
    if "98:156" in p:
        p["98:156"]["inputs"]["choice"] = setting["preset"]
        p["98:156"]["inputs"]["index"] = PRESET_MAP[setting["preset"]]["index"]
    if "98:18" in p:
        p["98:18"]["inputs"]["noise_seed"] = random.randint(1, 2**48)

    return p


# ── ComfyUI API ─────────────────────────────────────────────────────

def queue_prompt(prompt_dict):
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": prompt_dict, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    return result["prompt_id"]


def poll_history(prompt_id, timeout=300):
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

def build_contact_sheet(results, cell_w=400, cell_h=320, padding=16, font_size=14):
    n_cols = 1  # single setting
    n_rows = len(PROMPTS)

    sheet_w = 200 + padding + n_cols * (cell_w + padding)
    sheet_h = padding + 50 + n_rows * (cell_h + 40 + padding)

    sheet = Image.new("RGB", (sheet_w, sheet_h), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size - 2)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size + 2)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size + 6)
    except Exception:
        font = font_small = font_header = font_title = ImageFont.load_default()

    draw.text((padding, padding // 2), "Ideogram 4.0 Text Rendering Experiment (Turbo CFG 3)", fill=(220, 220, 220), font=font_title)

    header_y = padding + 30
    draw.text((200 + padding + cell_w // 2, header_y), "Turbo | CFG 3.0", fill=(180, 180, 180), font=font_header, anchor="mt")

    for row_idx, prompt_info in enumerate(PROMPTS):
        y_row = header_y + 40 + row_idx * (cell_h + 40 + padding)
        name_label = prompt_info["name"]
        if len(name_label) > 22:
            name_label = name_label[:20] + ".."
        draw.text((padding + 190, y_row + cell_h // 2), name_label, fill=(180, 180, 180), font=font_header, anchor="rm")

        key = prompt_info["name"]
        img_path = results.get(key)

        x = 200 + padding
        if img_path and img_path.exists():
            thumb = Image.open(img_path).convert("RGB")
            thumb.thumbnail((cell_w, cell_h))
            tx = x + (cell_w - thumb.width) // 2
            ty = y_row + (cell_h - thumb.height) // 2
            sheet.paste(thumb, (tx, ty))
        else:
            draw.rectangle([x, y_row, x + cell_w, y_row + cell_h], outline=(60, 60, 60))
            status = "FAILED" if key in results else "PENDING"
            draw.text((x + cell_w // 2, y_row + cell_h // 2), status, fill=(255, 80, 80), font=font, anchor="mm")

        y_text = y_row + cell_h + 4
        texts = prompt_info.get("text_elements", [])
        text_label = " | ".join(texts[:4])
        if len(text_label) > 70:
            text_label = text_label[:68] + ".."
        draw.text((x + 4, y_text), text_label, fill=(100, 100, 100), font=font_small)

    return sheet


# ── HTML contact sheet ───────────────────────────────────────────────

def build_html_contact_sheet(results, transformed_prompts=None):
    import base64

    if transformed_prompts is None:
        transformed_prompts = {}

    rows_html = ""
    for prompt_info in PROMPTS:
        key = prompt_info["name"]
        img_path = results.get(key)
        if img_path and img_path.exists():
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            img_tag = f'<img src="data:image/png;base64,{b64}" />'
        else:
            img_tag = '<div class="placeholder">NO IMAGE</div>'

        raw_escaped = prompt_info["prompt"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        transformed = transformed_prompts.get(prompt_info["name"], prompt_info["prompt"])
        trans_escaped = transformed.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

        texts = " | ".join(prompt_info.get("text_elements", []))

        rows_html += f"""
        <tr>
          <td class="row-label">{prompt_info['name']}</td>
          <td>
            <div class="cell">
              <div class="img-wrap">{img_tag}</div>
              <div class="meta">
                <div class="text-elements">TEXT: {texts}</div>
                <div class="prompt-label">RAW:</div>
                <div class="prompt" onclick="navigator.clipboard.writeText(this.dataset.prompt)" data-prompt="{prompt_info['prompt'].replace('"', '&quot;')}" title="Click to copy">{raw_escaped}</div>
                <div class="prompt-label">LLM JSON:</div>
                <div class="prompt transformed" onclick="navigator.clipboard.writeText(this.dataset.prompt)" data-prompt="{transformed.replace('"', '&quot;')}" title="Click to copy">{trans_escaped}</div>
              </div>
            </div>
          </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Ideogram 4.0 Text Rendering Experiment</title>
<style>
  body {{ background: #1a1a1a; color: #ddd; font-family: 'DejaVu Sans', sans-serif; margin: 20px; }}
  h1 {{ font-size: 18px; color: #eee; margin-bottom: 4px; }}
  p.hint {{ font-size: 12px; color: #888; margin-top: 0; }}
  table {{ border-collapse: collapse; }}
  th {{ padding: 8px 12px; text-align: center; font-size: 14px; color: #bbb; border-bottom: 1px solid #333; }}
  td.row-label {{ writing-mode: vertical-lr; text-orientation: mixed; transform: rotate(180deg);
                   font-size: 13px; font-weight: bold; color: #bbb; padding: 8px 4px;
                   text-align: center; white-space: nowrap; }}
  .cell {{ width: 420px; margin: 6px; }}
  .img-wrap {{ background: #111; border-radius: 4px; overflow: hidden; text-align: center; }}
  .img-wrap img {{ max-width: 100%; max-height: 360px; display: block; margin: 0 auto; }}
  .placeholder {{ height: 280px; line-height: 280px; color: #555; text-align: center; }}
  .meta {{ padding: 6px 0; }}
  .text-elements {{ font-size: 12px; color: #8af; margin-bottom: 6px; font-weight: bold; }}
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
  <h1>Ideogram 4.0 Text Rendering Experiment</h1>
  <p class="hint">Click any prompt to copy. Setting: Turbo preset, CFG 3.0. 30 prompts testing literal text rendering.</p>
  <table>
    <tr><th></th><th>Turbo | CFG 3.0</th></tr>
    {rows_html}
  </table>
</body>
</html>"""
    return html


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    template = load_template()

    print(f"Ideogram 4.0 Text Rendering Experiment")
    print(f"  {len(PROMPTS)} prompts x 1 setting = {len(PROMPTS)} runs")
    print(f"  Setting: Turbo, CFG 3.0")
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

    missing_prompts = [p for p in PROMPTS if not transformed.get(p["name"], "").strip()]
    if missing_prompts:
        print(f"Transforming {len(missing_prompts)} missing prompts with LLM...")
        for prompt_info in missing_prompts:
            raw = prompt_info["prompt"]
            print(f"  {prompt_info['name']}...", end=" ", flush=True)
            result = text_render_transform(raw)
            transformed[prompt_info["name"]] = result
            cache[prompt_info["name"]] = {"raw": raw, "llm": result}
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(PROMPTS_CACHE, "w") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            try:
                parsed = json.loads(result)
                n_text = sum(1 for e in parsed.get("compositional_deconstruction", {}).get("elements", []) if e.get("type") == "text")
                n_obj = sum(1 for e in parsed.get("compositional_deconstruction", {}).get("elements", []) if e.get("type") == "obj")
                print(f"OK ({n_text} text, {n_obj} obj)")
            except Exception:
                print(f"OK ({len(result)} chars)")
            time.sleep(0.5)
        print(f"Prompts saved to {PROMPTS_CACHE}\n")

    # Run jobs sequentially
    print(f"\nRunning {len(PROMPTS)} jobs sequentially...")
    results = {}
    done = 0
    for prompt_info in PROMPTS:
        done += 1
        hld = transformed[prompt_info["name"]]
        label = prompt_info["name"]
        out_path = OUTPUT_DIR / f"{label}.png"

        if out_path.exists() and out_path.stat().st_size > 1000:
            results[label] = out_path
            print(f"  [{done}/{len(PROMPTS)}] {label}... SKIP (exists)")
            continue

        p = patch_prompt(template, hld, SETTING)
        prompt_id = queue_prompt(p)
        print(f"  [{done}/{len(PROMPTS)}] {label}...", end=" ", flush=True)

        history = poll_history(prompt_id, timeout=600)
        if history and "outputs" in history:
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
                    results[label] = out_path
                    print(f"OK ({len(img_data)//1024}KB)")
                    break
                else:
                    print("NO IMAGES")
                    results[label] = None
            else:
                print("NO IMAGES")
                results[label] = None
        else:
            print("TIMEOUT")
            results[label] = None

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
