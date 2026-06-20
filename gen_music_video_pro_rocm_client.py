#!/usr/bin/env python3
"""
ROCm / Strix Halo Edition — Pro Music Video Pipeline
Uses Q6_K for LTX video, higher quality settings, benefits from 96GB unified memory.

Prerequisites (one-time):
  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
  # Then launch ComfyUI normally — it auto-detects ROCm torch
"""
import json, urllib.request, time, subprocess, os, sys, math, gc, shutil

COMFY = "http://192.168.169.2:8188"
OUT = "/home/ericr/ComfyUI/output"
INP = "/home/ericr/ComfyUI/input"
RENDER_DIR = os.path.join(OUT, "pro_mv_rocm")
os.makedirs(RENDER_DIR, exist_ok=True)

# ── Music Generation Config (HeartMuLa) ──────────────────────────────────
HEARTMULA_TAGS = "hardcore rap, aggressive power rap, heavy 808 bass, pounding kick drum, sharp snare, dark orchestral stabs, 85 BPM, female rapper, powerful delivery, intense energy, cinematic"
HEARTMULA_LYRICS = """(Intro)
Yeah... Step back...
Watch the throne...

(Verse 1)
Built this empire from the concrete cracks
Every scar on my body's a story that hits back
They told me I was nothing, just a voice in the void
Now every stage I touch gets completely destroyed
Iron will, steel spine, I was forged in the flame
Every loss that I took just added weight to my name
You hear that rumble? That's the sound of the crown
Coming home to the head that was born to wear it down

(Chorus)
I am the storm, I am the fire, I am the reckoning
Every word that I spit turns a crown into offering
Bow down or get burned, this is my ascension
Years of blood and tears built this empire of attention

(Verse 2)
They want the old me back, the one who played it safe
But you can't cage a wildfire and expect it to behave
I turned rejection into fuel, hate into gasoline
Now I'm burning through the industry like a killing machine
This ain't a comeback, this is a takeover
Every skeptical face just makes me grow stronger
From the underground to the top, I never lost my way
I'm not just running the game — I'm rewriting how it's played

(Chorus)
I am the storm, I am the fire, I am the reckoning
Every word that I spit turns a crown into offering
Bow down or get burned, this is my ascension
Years of blood and tears built this empire of attention

(Bridge)
They said be grateful for the scraps they threw my way
I said build your own table and eat gourmet
Now I'm sitting at the head with a crown of nails
Every doubt that they had is a link in my chain mail

(Outro)
Storm's here...
Feel the thunder...
I am the reckoning..."""

MUSIC_DURATION = 60

# ── Video Config (higher quality for ROCm) ──────────────────────────────
NUM_SCENES = 8
SCENE_DURATION = 7.5
FPS = 24
W, H = 960, 544  # higher resolution with 96GB
WORKFLOW_TEMPLATE = "/home/ericr/ComfyUI/workflow_local_gguf_rocm.json"

Z_UNET = "z-image-turbo-Q6_K.gguf"
Z_CLIP = "qwen_3_4b.safetensors"
Z_VAE = "ae.safetensors"
Z_LORA_DARK = "DarkB ZIT lora.safetensors"
Z_LORA_DAEMON = "REDZ15_DetailDaemonZ_lora_v1.1.safetensors"

NEG = "cartoon, anime, illustration, painting, 3d render, cgi, animal, furry, disney, pixar, deformed, ugly, blurry, low quality, distorted face, bad anatomy, extra limbs, skinny, frail, thin"
CHAR = "powerful muscular female rapper, 30 years old, athletic build, defined arm muscles, broad shoulders, dark skin tone, shaved head, gold hoop earrings, black tank top, cargo pants, combat boots, gold chains, intense fierce expression, commanding presence, tattoos on neck and hands, confident posture"

scenes = [
    {"prompt": f"{CHAR} standing on a rooftop overlooking a city at night, holding a microphone, powerful rapping pose, city skyline lit up behind, steam rising from vents, dramatic cinematic lighting, photorealistic, 8k", "seed": 200},
    {"prompt": f"{CHAR} in a dark warehouse with spotlights, flexing muscles, surrounded by shadow, dramatic side lighting, sweat on skin, intense expression, power stance, gritty atmosphere", "seed": 201},
    {"prompt": f"{CHAR} performing on a massive stage under dramatic spotlights, huge crowd silhouetted in smoke, gripping mic stand with both hands, raw power, concert photography style", "seed": 202},
    {"prompt": f"{CHAR} sitting on a throne-like chair in an empty arena, crown on head, looking down with authority, chains visible, dramatic low angle shot, cinematic lighting", "seed": 203},
    {"prompt": f"{CHAR} walking through a rain-slicked city street at night with her crew of muscular women, street lamps reflecting on wet pavement, powerful stride, candid photography style", "seed": 204},
    {"prompt": f"{CHAR} in a boxing gym, shadow boxing, sweat glistening, focused intense expression, ropes and punching bags in background, training montage aesthetic, documentary style", "seed": 205},
    {"prompt": f"{CHAR} extreme close up of face, intense commanding expression, gold light casting across her features, sweat on skin, hyperrealistic portrait, pores visible, cinematic", "seed": 206},
    {"prompt": f"{CHAR} standing victorious on a rooftop at golden hour, arms crossed, city skyline behind, sunrise lighting, powerful triumphant pose, epic hero shot", "seed": 207},
]

# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def free_vram():
    try:
        req = urllib.request.Request(f"{COMFY}/free", data=json.dumps({"free_memory": True}).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)
    except: pass

def queue(prompt_wf):
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": prompt_wf}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["prompt_id"]

def wait(pid, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        except: time.sleep(3); continue
        if pid in h: return h[pid]
        time.sleep(3)
    return None

# ══════════════════════════════════════════════════════════════════════════
# STAGE 0: GENERATE MUSIC (HeartMuLa via ComfyUI API)
# ══════════════════════════════════════════════════════════════════════════

def generate_music_heartmula(lyrics, tags, duration=60, output_path="pro_music.wav"):
    print(f"Generating rap with HeartMuLa: {duration}s ...")
    wf = {
        "1": {
            "class_type": "HeartMuLa_Generate",
            "inputs": {
                "lyrics": lyrics, "tags": tags,
                "version": "3B", "codec_version": "oss",
                "seed": 0, "max_audio_length_seconds": duration,
                "topk": 50, "temperature": 1.0, "cfg_scale": 1.5,
                "keep_model_loaded": False, "quantize_4bit": True,
                "use_compile": False, "tf32_matmul": True,
                "cudnn_benchmark": True, "flash_attention": True,
            }
        },
        "2": {
            "class_type": "SaveAudio",
            "inputs": {"audio": ["1", 0], "filename_prefix": "heartmula_rap_rocm"}
        }
    }
    pid = queue(wf)
    print(f"  Queued HeartMuLa ({pid[:8]}...)")
    res = wait(pid, timeout=900)
    if not res or res['status']['status_str'] != 'success':
        raise RuntimeError(f"HeartMuLa generation failed: {res}")

    out_path = None
    for node_id, outs in res.get("outputs", {}).items():
        for key, val in outs.items():
            if isinstance(val, str) and val.endswith(".wav"):
                out_path = val
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item.endswith(".wav"):
                        out_path = item

    if not out_path:
        search_dirs = [os.path.join(os.path.dirname(OUT), "temp"), OUT]
        for d in search_dirs:
            if not os.path.isdir(d): continue
            candidates = sorted(
                [os.path.join(d, f) for f in os.listdir(d)
                 if f.startswith("heartmula_") and f.endswith((".wav", ".flac"))],
                key=os.path.getmtime
            )
            if candidates: out_path = candidates[-1]; break

    if not out_path or not os.path.exists(out_path):
        raise FileNotFoundError("Could not locate generated HeartMuLa audio file")
    shutil.copy2(out_path, output_path)
    print(f"  -> {output_path}")
    return output_path

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1: SPLIT AUDIO
# ══════════════════════════════════════════════════════════════════════════

def split_audio(audio_path, num_scenes, duration):
    print(f"Splitting audio into {num_scenes} segments...")
    segs = []
    for i in range(num_scenes):
        seg = os.path.join(INP, f"pro_scene_{i:03d}.wav")
        start = i * duration
        subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ss", str(start),
            "-t", str(duration), seg], capture_output=True)
        segs.append(seg)
    return segs

# ══════════════════════════════════════════════════════════════════════════
# STAGE 2: GENERATE Z-IMAGE REFERENCES
# ══════════════════════════════════════════════════════════════════════════

def generate_refs():
    print(f"\nGenerating Z-Image references...")
    refs = []
    for i, s in enumerate(scenes):
        prompt = s["prompt"]
        wf = {
            "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "qwen_image"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1",0]}},
            "2n": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1",0]}},
            "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
            "3a": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["3",0], "lora_name": Z_LORA_DARK, "strength_model": 0.6}},
            "3b": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["3a",0], "lora_name": Z_LORA_DAEMON, "strength_model": 0.4}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
            "5": {"class_type": "VAELoader", "inputs": {"vae_name": Z_VAE}},
            "8": {"class_type": "KSampler", "inputs": {
                "model": ["3b",0], "seed": s["seed"], "steps": 4, "cfg": 1.5,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "positive": ["2",0], "negative": ["2n",0], "latent_image": ["4",0]
            }},
            "9": {"class_type": "VAEDecode", "inputs": {"vae": ["5",0], "samples": ["8",0]}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9",0], "filename_prefix": f"pro_ref_{i:03d}"}},
        }
        pid = queue(wf)
        print(f"  Scene {i+1}/{len(scenes)}: queued")
        res = wait(pid, 180)
        free_vram()
        if res and res['status']['status_str'] == 'success':
            for no, out in res.get("outputs",{}).items():
                for img in out.get("images",[]):
                    fp = os.path.join(OUT, img["filename"])
                    if os.path.exists(fp):
                        dst = os.path.join(INP, f"pro_ref_{i:03d}.png")
                        shutil.copy(fp, dst)
                        refs.append(dst)
                        print(f"    -> ref_{i:03d}.png")
        else:
            if os.path.exists(f"{INP}/ref_2k.png"):
                shutil.copy(f"{INP}/ref_2k.png", f"{INP}/pro_ref_{i:03d}.png")
                refs.append(f"{INP}/pro_ref_{i:03d}.png")
    return refs

# ══════════════════════════════════════════════════════════════════════════
# STAGE 3: GENERATE LTX VIDEOS (ROCm — Q6_K, more steps, higher res)
# ══════════════════════════════════════════════════════════════════════════

def generate_videos(refs, audio_segs):
    with open(WORKFLOW_TEMPLATE) as f:
        base_wf = json.load(f)

    clips = []
    for i, (s, ref, seg) in enumerate(zip(scenes, refs, audio_segs)):
        wf = json.loads(json.dumps(base_wf))
        prompt = f"{s['prompt']}, rapping powerfully, cinematic, sharp"

        wf["30"]["inputs"]["text"] = prompt
        wf["40"]["inputs"]["image"] = os.path.basename(ref)
        wf["41"]["inputs"]["audio"] = os.path.basename(seg)
        wf["47"]["inputs"]["end_time"] = float(SCENE_DURATION)

        fc = max(9, ((int(round(SCENE_DURATION * FPS)) - 1) // 8) * 8 + 1)
        wf["43"]["inputs"]["length"] = fc
        # Q6_K + 15 steps keeps KV cache manageable on 96GB unified
        wf["70"]["inputs"]["filename_prefix"] = f"pro_clip_{i:03d}"

        pid = queue(wf)
        print(f"  Scene {i+1}/{len(scenes)}: queued")
        res = wait(pid, 900)
        free_vram()
        if res and res['status']['status_str'] == 'success':
            for no, out in res.get("outputs",{}).items():
                for v in out.get("gifs", out.get("images", out.get("media", []))):
                    fp = os.path.join(OUT, v.get("filename", ""))
                    if fp and os.path.exists(fp):
                        clips.append(fp)
                        print(f"    -> {os.path.basename(fp)}")
        else:
            print(f"    FAILED")
    return clips

# ══════════════════════════════════════════════════════════════════════════
# STAGE 4: STITCH
# ══════════════════════════════════════════════════════════════════════════

def stitch(clips, audio_path, output="pro_mv_final_rocm.mp4"):
    print(f"\nStitching {len(clips)} clips...")
    if len(clips) >= 2:
        with open("/tmp/pro_stitch_rocm.txt", "w") as f:
            for c in clips: f.write(f"file '{c}'\n")
        raw = os.path.join(RENDER_DIR, "pro_raw.mp4")
        subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i","/tmp/pro_stitch_rocm.txt",
            "-c","copy",raw], check=True, capture_output=True)
        final = os.path.join(RENDER_DIR, output)
        subprocess.run(["ffmpeg","-y","-i",raw,"-i",audio_path,
            "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",final],
            check=False, capture_output=True)
    elif len(clips) == 1:
        final = clips[0]
    else:
        print("No clips to stitch")
        return None
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration,size",
        "-of","csv=p=0",final], capture_output=True, text=True)
    print(f"  {output} ({r.stdout.strip()})")
    return final

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start_all = time.time()

    audio_path = os.path.join(RENDER_DIR, "pro_music.wav")
    generate_music_heartmula(HEARTMULA_LYRICS, HEARTMULA_TAGS, MUSIC_DURATION, audio_path)

    segs = split_audio(audio_path, NUM_SCENES, SCENE_DURATION)

    refs = generate_refs()
    if len(refs) < NUM_SCENES:
        print(f"Only got {len(refs)}/{NUM_SCENES} references, continuing anyway")

    clips = generate_videos(refs, segs)

    if clips:
        stitch(clips, audio_path)

    elapsed = (time.time() - start_all) / 60
    print(f"\nTotal: {elapsed:.0f}m")
    print(f"Output: {RENDER_DIR}")
