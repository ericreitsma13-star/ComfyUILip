#!/usr/bin/env python3
"""
LTX 2.3 Lip-Sync — Fixed Architecture
- IC-LoRA for character consistency
- SOTAI zero-value audio mask (preserves audio signal)
- Proper resolution (divisible by 32, height/32 even)
- Better negative prompt
- Memory-efficient tiled decode
"""
import json, uuid, time, urllib.request, urllib.parse, os, sys
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8188"

MODELS = {
    "unet_gguf": "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
    "text_encoder": "gemma_3_12B_it_fp8_e4m3fn.safetensors",
    "text_projection": "ltx-2-3-22b-text_encoder.safetensors",
    "ic_lora": "ltx-2.3-22b-ic-lora-lipdub.safetensors",
    "distilled_lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    "audio_vae_ckpt": "ltx-2.3-22b-distilled_audio_vae.safetensors",
    "video_vae": "ltx-2.3-22b-distilled_video_vae.safetensors",
    "upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}

NEGATIVE_PROMPT = (
    "ugly, distorted face, blurry, cartoon, bad anatomy, crooked head, "
    "low quality, artifacts, double face, floating face, face deformation, "
    "asymmetrical eyes, wrong eyes, noise, flickering, glitching"
)


def build_workflow(image_path, audio_path,
                   prompt="A person speaking with natural lip movements, high quality, detailed face",
                   width=704, height=1280, fps=24, duration=4, seed=42,
                   lora_strength=0.2, i2v_strength=0.5):
    fc = int(((duration * fps - 1) / 8) * 8 + 1)
    lw = width // 2
    lh = height // 2

    sigmas = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    n_steps = 8

    wf = {
        # ── Loaders ──
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": MODELS["unet_gguf"]},
              "_meta": {"title": "UNET GGUF"}},
        "2": {"class_type": "LTXAVTextEncoderLoader",
              "inputs": {"text_encoder": MODELS["text_encoder"],
                         "ckpt_name": MODELS["text_projection"], "device": "default"},
              "_meta": {"title": "Text Encoder"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["video_vae"]},
              "_meta": {"title": "Video VAE"}},
        "4": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": MODELS["audio_vae_ckpt"]},
              "_meta": {"title": "Audio VAE"}},
        "5": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": MODELS["distilled_lora"],
                         "strength_model": lora_strength},
              "_meta": {"title": "Distilled LoRA"}},
        "6": {"class_type": "LTXICLoRALoaderModelOnly",
              "inputs": {"model": ["5", 0], "lora_name": MODELS["ic_lora"],
                         "strength_model": 1.0},
              "_meta": {"title": "IC-LoRA"}},

        # ── Inputs ──
        "10": {"class_type": "LoadImage", "inputs": {"image": image_path},
               "_meta": {"title": "Reference Image"}},
        "11": {"class_type": "LoadAudio", "inputs": {"audio": audio_path},
               "_meta": {"title": "Audio"}},

        # ── Preprocessing ──
        "12": {"class_type": "LTXVPreprocess",
               "inputs": {"image": ["10", 0], "frames": fc, "img_compression": 90},
               "_meta": {"title": "Preprocess"}},

        # ── Audio Latent ──
        "20": {"class_type": "LTXVAudioVAEEncode",
               "inputs": {"audio": ["11", 0], "audio_vae": ["4", 0]},
               "_meta": {"title": "Encode Audio"}},

        # ── Video Latent ──
        "30": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": lw, "height": lh, "length": fc, "batch_size": 1},
               "_meta": {"title": "Empty Video"}},
        "35": {"class_type": "LTXVImgToVideoInplace",
               "inputs": {"vae": ["3", 0], "image": ["12", 0], "latent": ["30", 0],
                          "strength": i2v_strength, "bypass": False},
               "_meta": {"title": "I2V"}},

        # ── Conditioning ──
        "50": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["2", 0], "text": prompt},
               "_meta": {"title": "Positive"}},
        "51": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["2", 0], "text": NEGATIVE_PROMPT},
               "_meta": {"title": "Negative"}},

        # ── IC-LoRA Guide (end anchor only — preserves expression, fixes end) ──
        "36": {"class_type": "LTXAddVideoICLoRAGuide",
               "inputs": {
                   "positive": ["50", 0], "negative": ["51", 0],
                   "vae": ["3", 0], "latent": ["35", 0],
                   "image": ["10", 0],
                   "frame_idx": 88, "strength": 0.35,
                   "latent_downscale_factor": ["6", 1],
                   "crop": "center",
                   "use_tiled_encode": True, "tile_size": 256, "tile_overlap": 64,
               },
               "_meta": {"title": "IC-LoRA Guide end"}},

        # ── Concat AV ──
        "45": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["36", 2], "audio_latent": ["20", 0]},
               "_meta": {"title": "Concat AV"}},

        # ── SOTAI Zero-Value Audio Mask ──
        "46": {"class_type": "LTXVSetAudioVideoMaskByTime",
               "inputs": {
                   "av_latent": ["45", 0],
                   "positive": ["36", 0], "negative": ["36", 1],
                   "model": ["6", 0],
                   "vae": ["3", 0], "audio_vae": ["4", 0],
                   "start_time": 0.0, "end_time": float(duration),
                   "video_fps": float(fps),
                   "mask_video": True, "mask_audio": False,
                   "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0,
                   "slope_len": 3,
               },
               "_meta": {"title": "SOTAI Audio Mask"}},

        # ── Frame-rate Conditioning ──
        "52": {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["46", 0], "negative": ["46", 1],
                          "frame_rate": float(fps)},
               "_meta": {"title": "Frame Rate"}},

        # ── Sampler ──
        "60": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed},
               "_meta": {"title": "Noise"}},
        "61": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral_cfg_pp"},
               "_meta": {"title": "Sampler"}},
        "62": {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas, "steps": n_steps},
               "_meta": {"title": "Sigmas"}},
        "63": {"class_type": "CFGGuider",
               "inputs": {"model": ["6", 0], "positive": ["52", 0], "negative": ["52", 1],
                          "cfg": 1.0},
               "_meta": {"title": "CFG"}},
        "64": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["60", 0], "guider": ["63", 0],
                          "sampler": ["61", 0], "sigmas": ["62", 0],
                          "latent_image": ["46", 2]},
               "_meta": {"title": "Sample"}},

        # ── Decode ──
        "70": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["64", 0]},
               "_meta": {"title": "Separate AV"}},
        "71": {"class_type": "LTXVSpatioTemporalTiledVAEDecode",
               "inputs": {
                   "vae": ["3", 0], "latents": ["70", 0],
                   "spatial_tiles": 2, "spatial_overlap": 2,
                   "temporal_tile_length": 16, "temporal_overlap": 2,
                   "last_frame_fix": True,
                   "working_device": "auto", "working_dtype": "auto",
               },
               "_meta": {"title": "Decode Tiled"}},
        "72": {"class_type": "LTXVAudioVAEDecode",
               "inputs": {"samples": ["70", 1], "audio_vae": ["4", 0]},
               "_meta": {"title": "Decode Audio"}},

        # ── Output ──
        "80": {"class_type": "CreateVideo",
               "inputs": {"images": ["71", 0], "audio": ["72", 0], "fps": float(fps)},
               "_meta": {"title": "Create Video"}},
        "81": {"class_type": "LTXMotionSaveVideo",
               "inputs": {"video": ["80", 0], "filename_prefix": "ltx_lipsync"},
               "_meta": {"title": "Save"}},
    }
    return wf


def queue_prompt(wf):
    payload = json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt", data=payload,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req).read())["prompt_id"]


def poll_history(pid, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = urllib.request.urlopen(f"{COMFYUI_URL}/history/{pid}")
            d = json.loads(r.read())
            if pid in d and d[pid].get("status", {}).get("completed"):
                return d[pid]
        except: pass
        time.sleep(3)
    return None


def download_output(history, output_dir):
    outputs = history.get("outputs", {})
    for nid, no in outputs.items():
        if "gifs" in no:
            for g in no["gifs"]:
                p = urllib.parse.urlencode({"filename": g["filename"],
                                            "subfolder": g.get("subfolder", ""),
                                            "type": g.get("type", "output")})
                data = urllib.request.urlopen(f"{COMFYUI_URL}/view?{p}").read()
                out = Path(output_dir) / g["filename"]
                out.write_bytes(data)
                print(f"  Saved: {out} ({len(data)//1024}KB)")
                return out
        if "text" in no:
            for t in no["text"]:
              s = str(t)
              if "Saved final video:" in s:
                fn = s.replace("Saved final video: ", "").strip()
                for sub in ["", "video/"]:
                    c = Path(output_dir).parent / (sub + fn)
                    if c.exists():
                        print(f"  Found: {c} ({c.stat().st_size//1024}KB)")
                        return c
                print(f"  Video saved as: {fn}")
                return None
    print("  No output found")
    return None


def main():
    import argparse
    p = argparse.ArgumentParser(description="LTX 2.3 Fixed Lip-Sync")
    p.add_argument("--image", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--prompt", default="A person speaking with natural lip movements, high quality, detailed face")
    p.add_argument("--width", type=int, default=704)
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--duration", type=float, default=4, help="Video duration in seconds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora", type=float, default=0.2)
    p.add_argument("--i2v", type=float, default=0.5)
    p.add_argument("--output", default="/home/ericr/ComfyUI/output/ltx_lipsync")
    args = p.parse_args()

    # Strip input/ prefix — ComfyUI LoadImage/LoadAudio want just the filename
    for attr in ('image', 'audio'):
        val = getattr(args, attr)
        if val.startswith('input/'):
            setattr(args, attr, val[len('input/'):])
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'input', getattr(args, attr))):
            print(f"ERROR: {attr} not found in input/: {getattr(args, attr)}"); sys.exit(1)
    if args.width % 32 != 0:
        print(f"ERROR: Width {args.width} not divisible by 32 (use 704, 832, 960)")
        sys.exit(1)
    if args.height % 32 != 0:
        print(f"ERROR: Height {args.height} not divisible by 32")
        sys.exit(1)
    if (args.height // 32) % 2 != 0:
        print(f"WARNING: height/32={args.height//32} is odd. IC-LoRA needs even. "
              f"Try height=1024, 1152, or 1280")

    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats")
    except:
        print("ERROR: ComfyUI not running"); sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"LTX 2.3 Fixed Lip-Sync")
    print(f"  Image: {args.image}")
    print(f"  Audio: {args.audio}")
    print(f"  {args.width}x{args.height} @ {args.fps}fps, {args.duration}s")
    print(f"  LoRA: {args.lora}, I2V: {args.i2v}, Seed: {args.seed}")
    print(f"  IC-LoRA: enabled, SOTAI mask: enabled")
    print()

    wf = build_workflow(
        args.image, args.audio, args.prompt,
        args.width, args.height, args.fps, args.duration, args.seed,
        args.lora, args.i2v,
    )

    print("Queuing...")
    pid = queue_prompt(wf)
    print(f"  prompt_id: {pid}")
    print("  Waiting (timeout 15 min)...")

    history = poll_history(pid, timeout=900)
    if history:
        print("  Complete!")
        download_output(history, args.output)
    else:
        print("  TIMEOUT or FAILED")
        # Try to get error
        try:
            r = urllib.request.urlopen(f"{COMFYUI_URL}/history/{pid}")
            d = json.loads(r.read())
            if pid in d:
                errors = d[pid].get("status", {}).get("messages", [])
                for msg in errors:
                    print(f"  {msg}")
        except: pass
        sys.exit(1)


if __name__ == "__main__":
    main()
