#!/usr/bin/env python3
"""
LTX 2.3 Lip-Sync — Working Pipeline
Based on workflow_local_gguf_rocm.json (the one that produced pro_clip outputs).
"""
import json, uuid, time, urllib.request, urllib.parse, os, sys
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8188"

MODELS = {
    "unet_gguf": "LTX-2.3-22B-distilled-1.1-Q6_K.gguf",
    "text_encoder": "gemma_3_12B_it_fp8_e4m3fn.safetensors",
    "text_projection": "ltx-2-3-22b-text_encoder.safetensors",
    "audio_vae_ckpt": "ltx-2.3-22b-distilled-1.1.safetensors",
    "video_vae": "ltx-2.3-22b-distilled_video_vae.safetensors",
    "distilled_lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
}

NEGATIVE_PROMPT = (
    "headshot, close up, portrait, from behind, back view, "
    "ugly, deformed, blurry, low quality, cartoon"
)


def build_workflow(image_path, audio_path,
                   prompt="",
                   width=960, height=544, fps=24, duration=7.5, seed=42,
                   lora_strength=0.8, i2v_strength=0.7, cfg=3.5):
    fc = int(((duration * fps - 1) / 8) * 8 + 1)

    wf = {
        "10": {"class_type": "UnetLoaderGGUF",
               "inputs": {"unet_name": MODELS["unet_gguf"]}},
        "11": {"class_type": "VAELoader",
               "inputs": {"vae_name": MODELS["video_vae"]}},
        "12": {"class_type": "LTXVAudioVAELoader",
               "inputs": {"ckpt_name": MODELS["audio_vae_ckpt"]}},
        "13": {"class_type": "LTXAVTextEncoderLoader",
               "inputs": {"text_encoder": MODELS["text_encoder"],
                          "ckpt_name": MODELS["text_projection"],
                          "device": "default"}},
        "20": {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["10", 0],
                          "lora_name": MODELS["distilled_lora"],
                          "strength_model": lora_strength}},
        "30": {"class_type": "CLIPTextEncode",
               "inputs": {"text": prompt, "clip": ["13", 0]}},
        "31": {"class_type": "CLIPTextEncode",
               "inputs": {"text": NEGATIVE_PROMPT, "clip": ["13", 0]}},
        "32": {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["30", 0], "negative": ["31", 0],
                          "frame_rate": float(fps)}},
        "40": {"class_type": "LoadImage",
               "inputs": {"image": image_path}},
        "41": {"class_type": "LoadAudio",
               "inputs": {"audio": audio_path}},
        "42": {"class_type": "LTXVAudioVAEEncode",
               "inputs": {"audio": ["41", 0], "audio_vae": ["12", 0]}},
        "43": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": width, "height": height,
                          "length": fc, "batch_size": 1}},
        "44": {"class_type": "LTXVImgToVideoInplace",
               "inputs": {"vae": ["11", 0], "image": ["40", 0],
                          "latent": ["43", 0], "strength": i2v_strength,
                          "bypass": False}},
        "46": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["44", 0],
                          "audio_latent": ["42", 0]}},
        "47": {"class_type": "LTXVSetAudioVideoMaskByTime",
               "inputs": {"av_latent": ["46", 0],
                          "positive": ["32", 0], "negative": ["32", 1],
                          "model": ["20", 0], "vae": ["11", 0],
                          "audio_vae": ["12", 0],
                          "start_time": 0.0, "end_time": float(duration),
                          "video_fps": float(fps),
                          "mask_video": True, "mask_audio": False,
                          "mask_init_value_video": 0.0,
                          "mask_init_value_audio": 0.0,
                          "slope_len": 3}},
        "50": {"class_type": "RandomNoise",
               "inputs": {"noise_seed": seed}},
        "51": {"class_type": "KSamplerSelect",
               "inputs": {"sampler_name": "euler"}},
        "52": {"class_type": "BasicScheduler",
               "inputs": {"model": ["20", 0],
                          "scheduler": "linear_quadratic",
                          "steps": 15, "denoise": 1.0}},
        "53": {"class_type": "CFGGuider",
               "inputs": {"model": ["20", 0],
                          "positive": ["47", 0], "negative": ["47", 1],
                          "cfg": cfg}},
        "54": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["50", 0], "guider": ["53", 0],
                          "sampler": ["51", 0], "sigmas": ["52", 0],
                          "latent_image": ["47", 2]}},
        "60": {"class_type": "LTXVSeparateAVLatent",
               "inputs": {"av_latent": ["54", 0]}},
        "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode",
               "inputs": {"vae": ["11", 0], "latents": ["60", 0],
                          "spatial_tiles": 2, "spatial_overlap": 4,
                          "temporal_tile_length": 16,
                          "temporal_overlap": 4,
                          "last_frame_fix": False,
                          "working_device": "auto",
                          "working_dtype": "auto"}},
        "62": {"class_type": "LTXVAudioVAEDecode",
               "inputs": {"samples": ["60", 1],
                          "audio_vae": ["12", 0]}},
        "70": {"class_type": "VHS_VideoCombine",
               "inputs": {"images": ["61", 0], "frame_rate": fps,
                          "loop_count": 0,
                          "filename_prefix": "ltx_lipsync",
                          "format": "video/h264-mp4",
                          "pingpong": False, "save_output": True,
                          "crf": 18, "pix_fmt": "yuv420p",
                          "audio": ["62", 0]}},
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
    print("  No output found")
    return None


def main():
    import argparse
    p = argparse.ArgumentParser(description="LTX 2.3 Lip-Sync (working pipeline)")
    p.add_argument("--image", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=544)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--duration", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora", type=float, default=0.8)
    p.add_argument("--i2v", type=float, default=0.7)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument("--output", default="/home/ericr/ComfyUI/output/ltx_lipsync")
    args = p.parse_args()

    for attr in ('image', 'audio'):
        val = getattr(args, attr)
        if val.startswith('input/'):
            setattr(args, attr, val[len('input/'):])
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'input', getattr(args, attr))):
            print(f"ERROR: {attr} not found in input/: {getattr(args, attr)}"); sys.exit(1)

    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats")
    except:
        print("ERROR: ComfyUI not running"); sys.exit(1)

    print(f"LTX 2.3 Lip-Sync (working pipeline)")
    print(f"  Image: {args.image}")
    print(f"  Audio: {args.audio}")
    print(f"  {args.width}x{args.height} @ {args.fps}fps, {args.duration}s")
    print(f"  LoRA: {args.lora}, I2V: {args.i2v}, CFG: {args.cfg}, Seed: {args.seed}")
    print(f"  Model: Q6_K, Sampler: euler, Scheduler: linear_quadratic, Steps: 15")
    print()

    wf = build_workflow(
        args.image, args.audio, args.prompt,
        args.width, args.height, args.fps, args.duration, args.seed,
        args.lora, args.i2v, args.cfg,
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
