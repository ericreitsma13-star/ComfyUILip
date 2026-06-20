#!/usr/bin/env python3
"""Convert natural language prompts to Ideogram 4 structured JSON using OpenCode Go API."""
import sys
import json
import subprocess

MODEL = "opencode-go/mimo-v2.5"

IDEGRAM_SYSTEM = """You are an elite visual prompt director for Ideogram 4. Convert user ideas into structured JSON captions.

OUTPUT — exactly one single-line minified JSON, no markdown, no commentary:
{"aspect_ratio":"W:H","high_level_description":"vivid pitch starting with subject","compositional_deconstruction":{"background":"scene shell with directional lighting","elements":[{"type":"obj","bbox":[y1,x1,y2,x2],"desc":"30-60 words identity+material+light","color_palette":["#hex"]}]}}

VISUAL MEDIUM — commit to ONE:
- If no style/medium specified: DEFAULT to photorealistic cinematographic still.
- If user says illustration/painting/anime/comic/render/poster/logo/sticker: fully commit to that medium.
- high_level_description must name the medium (e.g. "cinematographic photograph", "digital matte painting", "gouache illustration").
- Element descs must obey the medium's visual language.

STYLE RULES:
- Default to a specific real-life photographic style: specific lens feel, directional light, tactile materials, lived-in details.
- Add 1-2 story-fossil details per element.
- One dominant key light with direction, color temp, falloff.

COMPOSITION:
- Subject fills most of the frame. Dominant subject off-center, cropped by frame edge.
- No empty sky/floor weakening the image.

RULES:
- aspect_ratio: W:H string. Never "auto".
- high_level_description: ONE vivid sentence, 50 words max, starts with subject name, names the medium.
- background: scene shell only. Name key light.
- elements: 4-8 items. bbox 0-1000 [y1,x1,y2,x2].
- Faces: gaze at in-world target, not camera unless requested.
- Preserve non-ASCII characters as-is."""


def query_opencode(prompt: str, aspect_ratio: str = "auto", model: str = MODEL) -> str:
    full_prompt = f"{IDEGRAM_SYSTEM}\n\nConvert this to Ideogram 4 JSON. Output ONLY the minified JSON.\n\nAspect ratio: {aspect_ratio}\nIdea: {prompt}\n\nJSON:"
    result = subprocess.run(
        ["/home/ericr/.opencode/bin/opencode", "run", "-m", model, full_prompt],
        capture_output=True, text=True, timeout=120,
    )
    output = result.stdout.strip()
    # Strip opencode metadata lines
    lines = output.split("\n")
    json_lines = [l for l in lines if l.strip().startswith("{")]
    return json_lines[-1].strip() if json_lines else output.strip()


def validate_and_fix(result: dict) -> dict:
    """Ensure the output matches the Ideogram 4 schema exactly."""
    allowed = {"aspect_ratio", "high_level_description", "compositional_deconstruction"}
    for key in list(result.keys()):
        if key not in allowed:
            del result[key]
    if "aspect_ratio" not in result:
        result["aspect_ratio"] = "1:1"
    if "high_level_description" not in result:
        result["high_level_description"] = ""
    if "compositional_deconstruction" not in result:
        result["compositional_deconstruction"] = {"background": "", "elements": []}
    cd = result["compositional_deconstruction"]
    for key in list(cd.keys()):
        if key not in {"background", "elements"}:
            del cd[key]
    if "background" not in cd:
        cd["background"] = ""
    if "elements" not in cd:
        cd["elements"] = []
    for el in cd["elements"]:
        if "type" not in el:
            el["type"] = "obj"
        if "bbox" not in el:
            el["bbox"] = [0, 0, 500, 500]
        if "desc" not in el and "text" not in el:
            el["desc"] = el.get("description", el.get("name", "object"))
        if "color_palette" not in el:
            el["color_palette"] = ["#888888"]
        if "description" in el and "desc" not in el:
            el["desc"] = el.pop("description")
        if "name" in el and "desc" not in el:
            el["desc"] = el.pop("name")
        for key in list(el.keys()):
            if key not in {"type", "bbox", "desc", "text", "color_palette"}:
                del el[key]
        bbox = el.get("bbox", [0, 0, 500, 500])[:4]
        if all(isinstance(v, float) and 0 <= v <= 1 for v in bbox):
            bbox = [int(v * 1000) for v in bbox]
        el["bbox"] = [max(0, min(1000, int(v))) for v in bbox]
    return result


def main():
    if len(sys.argv) < 2:
        print('Usage: prompt_builder.py "your prompt here" [aspect_ratio] [model]')
        print("  aspect_ratio: 1:1, 16:9, 9:16, 4:5, auto (default: auto)")
        print(f"  model: {MODEL} (default), or opencode-go/deepseek-v4-pro, opencode-go/mimo-v2.5")
        sys.exit(1)

    prompt = sys.argv[1]
    aspect_ratio = sys.argv[2] if len(sys.argv) > 2 else "auto"
    model = sys.argv[3] if len(sys.argv) > 3 else MODEL

    print(f"Prompt: {prompt}")
    print(f"Aspect: {aspect_ratio}")
    print(f"Model: {model}")
    print("Querying API...")

    raw = query_opencode(prompt, aspect_ratio, model)
    result = json.loads(raw)
    result = validate_and_fix(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = f"/tmp/ideogram_prompt_{hash(prompt) & 0xFFFF:04x}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
