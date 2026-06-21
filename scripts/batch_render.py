#!/usr/bin/env python3
"""batch_render.py — Render multiple segments with the working LTX pipeline.

Usage:
    python batch_render.py --segments input/segments.json --ref-image pro_ref_003.png --prompt "..." --output-dir output/batch/
"""
import argparse, json, os, sys, time, urllib.request, urllib.parse
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

def build_workflow(image, audio, prompt, width, height, fps, duration, seed, lora, i2v, cfg):
    fc = int(((duration * fps - 1) / 8) * 8 + 1)
    wf = {
        "10": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": MODELS["unet_gguf"]}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["video_vae"]}},
        "12": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": MODELS["audio_vae_ckpt"]}},
        "13": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": MODELS["text_encoder"], "ckpt_name": MODELS["text_projection"], "device": "default"}},
        "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10", 0], "lora_name": MODELS["distilled_lora"], "strength_model": lora}},
        "30": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["13", 0]}},
        "31": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE_PROMPT, "clip": ["13", 0]}},
        "32": {"class_type": "LTXVConditioning", "inputs": {"positive": ["30", 0], "negative": ["31", 0], "frame_rate": float(fps)}},
        "40": {"class_type": "LoadImage", "inputs": {"image": image}},
        "41": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "42": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["41", 0], "audio_vae": ["12", 0]}},
        "43": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width, "height": height, "length": fc, "batch_size": 1}},
        "44": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["11", 0], "image": ["40", 0], "latent": ["43", 0], "strength": i2v, "bypass": False}},
        "46": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["44", 0], "audio_latent": ["42", 0]}},
        "47": {"class_type": "LTXVSetAudioVideoMaskByTime", "inputs": {"av_latent": ["46", 0], "positive": ["32", 0], "negative": ["32", 1], "model": ["20", 0], "vae": ["11", 0], "audio_vae": ["12", 0], "start_time": 0.0, "end_time": float(duration), "video_fps": float(fps), "mask_video": True, "mask_audio": False, "mask_init_value_video": 0.0, "mask_init_value_audio": 0.0, "slope_len": 3}},
        "50": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "51": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "52": {"class_type": "BasicScheduler", "inputs": {"model": ["20", 0], "scheduler": "linear_quadratic", "steps": 15, "denoise": 1.0}},
        "53": {"class_type": "CFGGuider", "inputs": {"model": ["20", 0], "positive": ["47", 0], "negative": ["47", 1], "cfg": cfg}},
        "54": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["50", 0], "guider": ["53", 0], "sampler": ["51", 0], "sigmas": ["52", 0], "latent_image": ["47", 2]}},
        "60": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["54", 0]}},
        "61": {"class_type": "LTXVSpatioTemporalTiledVAEDecode", "inputs": {"vae": ["11", 0], "latents": ["60", 0], "spatial_tiles": 2, "spatial_overlap": 4, "temporal_tile_length": 16, "temporal_overlap": 4, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "62": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["60", 1], "audio_vae": ["12", 0]}},
        "70": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["61", 0], "frame_rate": fps, "loop_count": 0, "filename_prefix": f"batch_clip", "format": "video/h264-mp4", "pingpong": False, "save_output": True, "crf": 18, "pix_fmt": "yuv420p", "audio": ["62", 0]}},
    }
    return wf

def queue_and_wait(wf, timeout=900):
    payload = json.dumps({"prompt": wf, "client_id": "batch_render"}).encode()
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=payload, headers={"Content-Type": "application/json"})
    pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    start = time.time()
    while time.time() - start < timeout:
        try:
            d = json.loads(urllib.request.urlopen(f"{COMFYUI_URL}/history/{pid}", timeout=10).read())
            if pid in d and d[pid].get("status",{}).get("completed"):
                outputs = d[pid].get("outputs",{})
                for nid, no in outputs.items():
                    for k, v in no.items():
                        if isinstance(v, list):
                            for item in v:
                                if isinstance(item, dict) and "filename" in item:
                                    return item["filename"]
        except: pass
        time.sleep(3)
    return None

def download_output(filename, output_dir):
    qs = urllib.parse.urlencode({"filename": filename, "subfolder": "", "type": "output"})
    data = urllib.request.urlopen(f"{COMFYUI_URL}/view?{qs}").read()
    out = Path(output_dir) / filename
    out.write_bytes(data)
    return out

def main():
    p = argparse.ArgumentParser(description="Batch render LTX segments")
    p.add_argument("--segments", required=True, help="Path to segments.json from split_audio.py")
    p.add_argument("--ref-image", required=True, help="Reference image filename (in input/)")
    p.add_argument("--prompt", default="female singer performing, singing powerfully, emotional expression, cinematic lighting, photorealistic")
    p.add_argument("--output-dir", default="output/batch")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=544)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora", type=float, default=0.8)
    p.add_argument("--i2v", type=float, default=0.7)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument("--only-index", type=int, default=None)
    args = p.parse_args()

    manifest = json.load(open(args.segments))
    segments = manifest["segments"]
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy ref image to ComfyUI input
    import shutil
    comfy_input = "/home/ericr/ComfyUI/input"
    src = Path(comfy_input) / args.ref_image
    if not src.exists():
        print(f"ERROR: Reference image not found: {src}")
        sys.exit(1)

    log = {"segments": [], "total": len(segments)}

    for seg in segments:
        idx = seg["index"]
        if args.only_index is not None and idx != args.only_index:
            continue

        audio_file = seg["audio_file"]
        audio_path = Path(comfy_input) / audio_file
        if not audio_path.exists():
            print(f"  SKIP segment {idx}: audio not found ({audio_file})")
            continue

        print(f"\nSegment {idx}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s ({seg['duration_sec']:.1f}s)")
        seed = args.seed + idx
        duration = seg["duration_sec"]

        wf = build_workflow(args.ref_image, audio_file, args.prompt,
                           args.width, args.height, args.fps, duration,
                           seed, args.lora, args.i2v, args.cfg)

        t0 = time.time()
        filename = queue_and_wait(wf, timeout=900)
        elapsed = time.time() - t0

        if filename:
            out = download_output(filename, args.output_dir)
            print(f"  OK in {elapsed:.0f}s -> {out}")
            log["segments"].append({"index": idx, "status": "ok", "file": str(out), "time_sec": round(elapsed, 1)})
        else:
            print(f"  FAILED (timeout or error)")
            log["segments"].append({"index": idx, "status": "failed"})

        # Save log after each segment (crash-safe)
        log_path = Path(args.output_dir) / "batch_log.json"
        log_path.write_text(json.dumps(log, indent=2))

    ok = sum(1 for s in log["segments"] if s["status"] == "ok")
    print(f"\nDone: {ok}/{len(segments)} segments OK")

if __name__ == "__main__":
    main()
