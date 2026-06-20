#!/usr/bin/env python3
"""Generate AI assets for Sunset Cruise using local Ideogram 4 JSON prompting."""

import json, requests, random

API = "http://127.0.0.1:8188/prompt"

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

def round_dim(d):
    return max(((d + 15) // 16) * 16, 256)

def json_prompt(high_level, aesthetics, lighting, photo, medium, palette, background, elements=None):
    j = {
        "high_level_description": high_level,
        "style_description": {
            "aesthetics": aesthetics, "lighting": lighting,
            "photo": photo, "medium": medium, "color_palette": palette,
        },
        "compositional_deconstruction": {"background": background,}
    }
    if elements:
        j["compositional_deconstruction"]["elements"] = elements
    return json.dumps(j, ensure_ascii=False)

ASSETS = [
    ("skybox_sunset", json_prompt(
        "Equirectangular 360-degree HDRI panorama of a cinematic golden-hour sunset sky over a calm ocean",
        "Photorealistic HDR sky dome, smooth cloud gradients, no land or objects visible",
        "Golden hour warm sun low on horizon, orange-pink-purple layered clouds, volumetric atmospheric glow",
        "Shot on Canon EOS R5 14mm ultra-wide lens, 8K, no grain",
        "360-degree equirectangular HDRI photograph",
        ["#F4A460","#FF6B6B","#9B59B6","#2C3E50","#E8DAEF"],
        "Vast sky dome 360x180, deep blue zenith grading through pink to warm orange horizon, calm ocean at bottom edge reflecting sunset",
        [{"type":"obj","bbox":[0,0,500,1000],"desc":"Upper sky: deep blue to purple gradient with scattered wispy clouds","color_palette":["#2C3E50","#6C3483","#2980B9"]},
         {"type":"obj","bbox":[300,0,700,1000],"desc":"Middle sky: layered pink orange peach clouds, bright sun glow","color_palette":["#F4A460","#FF6B6B","#E74C3C"]},
         {"type":"obj","bbox":[700,0,1000,1000],"desc":"Lower horizon: bright golden band, calm ocean with sunset reflections, gentle ripples","color_palette":["#F39C12","#F1C40F","#3498DB"]}]
    ), 2880, 1440),
    ("miami_skyline", json_prompt(
        "Wide panoramic shot of Miami Beach art deco skyline at golden hour sunset",
        "Pastel art deco architecture, vintage 1980s Miami postcard aesthetic, warm nostalgic feel",
        "Golden hour warm side-lighting from setting sun, soft shadows, warm glow on pastel facades",
        "Shot on 35mm Kodachrome film, subtle warm grain, slightly saturated vintage color",
        "Architectural landscape photograph",
        ["#F4A460","#E8DAEF","#87CEEB","#FFFFFF","#D4A574"],
        "Miami Beach shoreline at golden hour: calm turquoise ocean in foreground, white sand beach, row of pastel-colored 1930s art deco buildings, palm tree silhouettes, clear sky above",
        [{"type":"obj","bbox":[600,0,1000,1000],"desc":"Sky: warm sunset gradient orange to pink to purple","color_palette":["#F4A460","#FF6B6B","#9B59B6"]},
         {"type":"obj","bbox":[400,100,700,900],"desc":"Art deco buildings: pastel pink white blue facades, geometric lines, 3-5 stories","color_palette":["#E8DAEF","#FFFFFF","#87CEEB"]},
         {"type":"obj","bbox":[300,600,500,1000],"desc":"Palm trees: tall silhouettes with distinct fronds","color_palette":["#2C3E50","#1A252F"]},
         {"type":"obj","bbox":[700,500,1000,1000],"desc":"Foreground: calm ocean water with golden reflections","color_palette":["#3498DB","#F1C40F"]}]
    ), 2880, 1440),
    ("asphalt_road", json_prompt(
        "Top-down macro photograph of worn coastal highway asphalt road surface texture",
        "Weathered cracked pavement, realistic PBR surface detail, industrial quality",
        "Warm golden hour sunlight casting low-angle shadows across surface highlighting texture cracks",
        "Top-down macro 8K sharp detail, 50mm lens, neutral perspective",
        "Texture reference photograph",
        ["#555555","#3D3D3D","#8B7355","#D4A574","#F4A460"],
        "Flat asphalt surface extending to all edges, worn cracks and faded yellow road paint, subtle debris weathering",
        [{"type":"obj","bbox":[0,0,1000,1000],"desc":"Main asphalt: dark gray with fine aggregate texture, hairline diagonal cracks","color_palette":["#555555","#3D3D3D","#4A4A4A"]},
         {"type":"obj","bbox":[200,300,350,1000],"desc":"Faded yellow center line running vertically, worn chipped edges","color_palette":["#F1C40F","#8B7355"]},
         {"type":"obj","bbox":[600,0,1000,600],"desc":"Weathered edge with small cracks gravel, warm sunlight on surface","color_palette":["#8B7355","#D4A574"]}]
    ), 2048, 2048),
    ("sand_beach", json_prompt(
        "Top-down macro photograph of golden beach sand texture with fine grain detail",
        "Natural pristine beach sand, subtle wind ripples, unspoiled coastline material",
        "Warm golden hour sunlight raking across surface at low angle, soft shadow on sand ripples",
        "Top-down macro 8K, 100mm macro lens, shallow depth for texture emphasis",
        "Macro texture reference photograph",
        ["#F4D03F","#D4A574","#F5DEB3","#C9A96E","#FFF8DC"],
        "Seamless flat sand surface extending to all edges, fine uniform golden grains, gentle wind ripples, tiny shell fragments",
        [{"type":"obj","bbox":[0,0,600,1000],"desc":"Main sand: uniform fine golden-tan granules, subtle horizontal wind ripples","color_palette":["#F4D03F","#D4A574"]},
         {"type":"obj","bbox":[600,200,950,800],"desc":"Slightly damp sand: darker tone with moisture sheen","color_palette":["#C9A96E","#D4A574"]},
         {"type":"obj","bbox":[800,100,1000,1000],"desc":"Corner: tiny white shell fragments and pebbles","color_palette":["#FFF8DC","#FFFFFF"]}]
    ), 2048, 2048),
    ("lut_reference", json_prompt(
        "A classic white Lamborghini Countach convertible driving on a coastal highway at golden hour sunset",
        "Cinematic automotive editorial, 1980s OutRun retro-futuristic, Miami Vice vibes",
        "Dramatic golden hour warm sun from left, long shadows, anamorphic lens flare, warm car body highlights",
        "Shot on ARRI Alexa anamorphic lens, Kodak Portra 400 film emulation, subtle grain, teal orange color grade",
        "Automotive editorial photograph",
        ["#F4A460","#00CED1","#FFFFFF","#2C3E50","#FF6B6B"],
        "Coastal highway curving along ocean at sunset: low-angle 24mm, white Countach in foreground-right, ocean and pastel Miami skyline background-left, warm golden sky, palm trees in distance",
        [{"type":"obj","bbox":[400,300,850,900],"desc":"White Countach: angular wedge, pop-up headlights up, top down, red interior, glossy white paint with sunset reflections, right foreground","color_palette":["#FFFFFF","#E8E8E8","#8B0000"]},
         {"type":"obj","bbox":[0,0,400,700],"desc":"Background: deep blue ocean meeting golden sky, distant pastel art deco buildings","color_palette":["#2980B9","#F4A460","#E8DAEF"]},
         {"type":"obj","bbox":[0,700,1000,1000],"desc":"Coastal road: dark gray asphalt with yellow center line curving into distance","color_palette":["#555555","#F1C40F"]},
         {"type":"obj","bbox":[0,0,1000,500],"desc":"Sky: dramatic sunset layered clouds, anamorphic lens flare, warm orange to purple","color_palette":["#F4A460","#FF6B6B","#9B59B6"]}]
    ), 2880, 1440),
    ("palm_trees", json_prompt(
        "Silhouettes of tropical palm trees against a golden sunset sky background",
        "High-contrast natural silhouette, tropical atmosphere, clean distinct frond shapes",
        "Warm golden sunset backlighting creating strong silhouettes, natural rim light on palm frond edges",
        "35mm film photograph, natural contrast, no excessive grain",
        "Nature photograph",
        ["#2C3E50","#F4A460","#FF6B6B","#9B59B6","#1A1A1A"],
        "Multiple palm trees silhouetted against spectacular golden-pink sunset sky, viewed from beach level, calm ocean horizon visible below",
        [{"type":"obj","bbox":[0,0,1000,650],"desc":"Sky: warm sunset gradient golden to purple, scattered pink orange cloud wisps","color_palette":["#F4A460","#FF6B6B","#9B59B6"]},
         {"type":"obj","bbox":[100,100,800,900],"desc":"Main palm silhouette left-center: tall curved trunk leaning right, crown of sharply defined fronds","color_palette":["#1A1A1A","#2C3E50"]},
         {"type":"obj","bbox":[600,50,1000,800],"desc":"Secondary palm right: shorter trunk, different frond angle, overlapping","color_palette":["#1A1A1A","#2C3E50"]},
         {"type":"obj","bbox":[800,800,1000,1000],"desc":"Bottom: thin dark ocean horizon with subtle golden reflection","color_palette":["#1A252F","#F1C40F"]}]
    ), 2048, 2048),
]

for i, (prefix, prompt, W, H) in enumerate(ASSETS):
    w = round_dim(W)
    h = round_dim(H)
    seed = random.randint(0, 2**31)
    print(f"[{i+1}/{len(ASSETS)}] {prefix} ({w}×{h})")

    p = {
        # Models
        "c_unet": node("UNETLoader", unet_name="ideogram4_fp8_scaled.safetensors", weight_dtype="default"),
        "u_unet": node("UNETLoader", unet_name="ideogram4_unconditional_fp8_scaled.safetensors", weight_dtype="default"),
        "clip":   node("CLIPLoader", clip_name="qwen3vl_8b_fp8_scaled.safetensors", type="ideogram4"),
        "vae":    node("VAELoader", vae_name="flux2-vae.safetensors"),

        # Conditioning
        "pos_enc":  node("CLIPTextEncode", text=prompt, clip=("clip", 0)),
        "neg_enc":  node("ConditioningZeroOut", conditioning=("pos_enc", 0)),

        # CFG override applied to conditional UNET before guider
        "cfg_ov":   node("CFGOverride", model=("c_unet", 0), cfg=3.0, start_percent=0.7, end_percent=1.0),

        # Dual model guider: CFGOverride'd model + unconditional model
        "guider":   node("DualModelGuider", model=("cfg_ov", 0), model_negative=("u_unet", 0), positive=("pos_enc", 0), negative=("neg_enc", 0), cfg=7.0),

        # Sampling
        "sigmas":   node("Ideogram4Scheduler", steps=20, width=w, height=h, mu=0.0, std=1.75),
        "latent":   node("EmptyFlux2LatentImage", width=w, height=h, batch_size=1),
        "sampler":  node("KSamplerSelect", sampler_name="euler"),
        "noise":    node("RandomNoise", noise_seed=seed),
        "samples":  node("SamplerCustomAdvanced", noise=("noise", 0), guider=("guider", 0), sampler=("sampler", 0), sigmas=("sigmas", 0), latent_image=("latent", 0)),

        # Decode & save
        "image":    node("VAEDecode", samples=("samples", 0), vae=("vae", 0)),
        "save":     node("SaveImage", filename_prefix=f"sunset_cruise/{prefix}", images=("image", 0)),
    }

    try:
        r = requests.post(API, json={"prompt": p}, timeout=600)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}: {r.text[:300]}")
        else:
            resp = r.json()
            if "error" in resp:
                msg = resp["error"]
                print(f"  Error: {(msg if isinstance(msg,str) else msg.get('message',str(msg)))[:300]}")
            else:
                print(f"  Done")
    except Exception as e:
        print(f"  Failed: {e}")

print(f"\nDone. Output: ~/ComfyUI/output/sunset_cruise/")
