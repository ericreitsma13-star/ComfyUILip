#!/usr/bin/env python3
"""
LTX 2.3 GGUF Lip-Sync Test Script
Minimal workflow: reference image + audio → lip-synced video
Uses GGUF Q6_K model + Gemma GGUF for low VRAM
"""

import json
import uuid
import time
import urllib.request
import urllib.parse
import argparse
import os
import sys
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8188"

# Default model paths (relative to models/)
MODELS = {
    "unet_gguf": "LTX-2.3-22B-distilled-1.1-Q6_K.gguf",
    "text_encoder": "gemma_3_12B_it_fp8_e4m3fn.safetensors",
    "text_projection": "ltx-2-3-22b-text_encoder.safetensors",
    "lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    "audio_vae_ckpt": "ltx-2.3-22b-distilled_audio_vae.safetensors",
    "video_vae": "ltx-2.3-22b-distilled_video_vae.safetensors",
    "upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}


def build_workflow(image_path, audio_path, prompt="A person speaking with natural lip movements",
                   width=720, height=1280, fps=24, duration=5, seed=42, mode="turbo",
                   lora_strength=0.5, i2v_strength=0.6):
    """Build the lip-sync workflow as a ComfyUI API prompt dict."""

    frame_count = int(((duration * fps - 1) / 8) * 8 + 1)
    latent_w = width // 2
    latent_h = height // 2

    # Mode-specific sampling: turbo=8 steps, default=12 steps (denser schedule)
    if mode == "default":
        sigmas = "1.0, 0.99375, 0.9875, 0.975, 0.9375, 0.8125, 0.625, 0.4375, 0.25, 0.125, 0.0625, 0.0"
        n_steps = 12
    else:
        sigmas = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
        n_steps = 8

    workflow = {
        # ── Loaders ──
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": MODELS["unet_gguf"],
            },
            "_meta": {"title": "Load LTX 2.3 GGUF UNET"}
        },
        "2": {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {
                "text_encoder": MODELS["text_encoder"],
                "ckpt_name": MODELS["text_projection"],
                "device": "default",
            },
            "_meta": {"title": "Load Gemma 3 12B Text Encoder"}
        },
        "3": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": MODELS["lora"],
                "strength_model": lora_strength,
            },
            "_meta": {"title": "Load Distilled LoRA"}
        },
        "4": {
            "class_type": "LTXVAudioVAELoader",
            "inputs": {
                "ckpt_name": MODELS["audio_vae_ckpt"],
            },
            "_meta": {"title": "Load Audio VAE"}
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": MODELS["video_vae"],
            },
            "_meta": {"title": "Load Video VAE"}
        },

        # ── Inputs ──
        "10": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_path,
            },
            "_meta": {"title": "Load Reference Image"}
        },
        "11": {
            "class_type": "LoadAudio",
            "inputs": {
                "audio": audio_path,
            },
            "_meta": {"title": "Load Audio"}
        },

        # ── Audio Pipeline ──
        "20": {
            "class_type": "LTXVAudioVAEEncode",
            "inputs": {
                "audio": ["11", 0],
                "audio_vae": ["4", 0],
            },
            "_meta": {"title": "Encode Audio to Latent"}
        },

        # ── Image Preprocessing ──
        "25": {
            "class_type": "LTXVPreprocess",
            "inputs": {
                "image": ["10", 0],
                "frames": frame_count,
                "img_compression": 90,
            },
            "_meta": {"title": "Preprocess Image"}
        },

        # ── Video Latent Setup ──
        "30": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": latent_w,
                "height": latent_h,
                "length": frame_count,
                "batch_size": 1,
            },
            "_meta": {"title": "Empty Video Latent"}
        },

        # ── Image-to-Video ──
        "35": {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "vae": ["5", 0],
                "image": ["25", 0],
                "latent": ["30", 0],
                "strength": i2v_strength,
                "bypass": False,
            },
            "_meta": {"title": "Image to Video Inplace"}
        },

        # ── Audio Mask ──
        "40": {
            "class_type": "SolidMask",
            "inputs": {
                "width": width,
                "height": height,
                "value": 0,
            },
            "_meta": {"title": "Solid Mask for Audio"}
        },
        "41": {
            "class_type": "SetLatentNoiseMask",
            "inputs": {
                "samples": ["20", 0],
                "mask": ["40", 0],
            },
            "_meta": {"title": "Set Audio Noise Mask"}
        },

        # ── Concat AV Latent ──
        "45": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {
                "video_latent": ["35", 0],
                "audio_latent": ["41", 0],
            },
            "_meta": {"title": "Concat Audio+Video Latent"}
        },

        # ── Conditioning ──
        "50": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": prompt,
            },
            "_meta": {"title": "Positive Prompt"}
        },
        "51": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": "pc game, console game, video game, cartoon, childish, ugly, blurry, distorted face",
            },
            "_meta": {"title": "Negative Prompt"}
        },
        "52": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["50", 0],
                "negative": ["51", 0],
                "frame_rate": float(fps),
            },
            "_meta": {"title": "Set Frame Rate"}
        },

        # ── Sampler ──
        "60": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed,
            },
            "_meta": {"title": "Random Noise"}
        },
        "61": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler_ancestral_cfg_pp",
            },
            "_meta": {"title": "Sampler"}
        },
        "62": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": sigmas,
                "steps": n_steps,
            },
            "_meta": {"title": "Sampling Sigmas"}
        },
        "63": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["3", 0],
                "positive": ["52", 0],
                "negative": ["52", 1],
                "cfg": 1.0,
            },
            "_meta": {"title": "CFG Guider"}
        },
        "64": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["60", 0],
                "guider": ["63", 0],
                "sampler": ["61", 0],
                "sigmas": ["62", 0],
                "latent_image": ["45", 0],
            },
            "_meta": {"title": "Sample"}
        },

        # ── Decode ──
        "70": {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {
                "av_latent": ["64", 0],
            },
            "_meta": {"title": "Separate A/V Latent"}
        },
        "71": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["70", 0],
                "vae": ["5", 0],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
            "_meta": {"title": "Decode Video (Tiled)"}
        },
        "72": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {
                "samples": ["70", 1],
                "audio_vae": ["4", 0],
            },
            "_meta": {"title": "Decode Audio"}
        },

        # ── Output ──
        "80": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["71", 0],
                "audio": ["72", 0],
                "fps": float(fps),
            },
            "_meta": {"title": "Create Video"}
        },
        "81": {
            "class_type": "LTXMotionSaveVideo",
            "inputs": {
                "video": ["80", 0],
                "filename_prefix": "ltx_lipsync_test",
            },
            "_meta": {"title": "Save Video"}
        },
    }

    return workflow


def queue_prompt(workflow):
    """Queue a workflow and return prompt_id."""
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    return result["prompt_id"]


def poll_history(prompt_id, timeout=600):
    """Poll until prompt completes."""
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
        time.sleep(3)
    return None


def download_output(history, output_dir):
    """Download output video from completed job."""
    outputs = history.get("outputs", {})

    # Check for SaveVideo/LTXMotionSaveVideo output (gifs or text)
    for node_id, node_out in outputs.items():
        if "gifs" in node_out:
            for gif_info in node_out["gifs"]:
                params = urllib.parse.urlencode({
                    "filename": gif_info["filename"],
                    "subfolder": gif_info.get("subfolder", ""),
                    "type": gif_info.get("type", "output"),
                })
                data = urllib.request.urlopen(f"{COMFYUI_URL}/view?{params}").read()
                out_path = Path(output_dir) / gif_info["filename"]
                with open(out_path, "wb") as f:
                    f.write(data)
                print(f"  Saved: {out_path} ({len(data)//1024}KB)")
                return out_path

        # LTXMotionSaveVideo returns text, not gifs - file is already saved by ComfyUI
        if "text" in node_out:
            for text_info in node_out["text"]:
                if "Saved final video:" in str(text_info):
                    filename = str(text_info).replace("Saved final video: ", "").strip()
                    # Check common output locations
                    for subdir in ["", "video/"]:
                        candidate = Path(output_dir).parent / (subdir + filename)
                        if candidate.exists():
                            print(f"  Found: {candidate} ({candidate.stat().st_size//1024}KB)")
                            return candidate
                    print(f"  Video saved by ComfyUI as: {filename}")
                    return None

    print("  No output found in history")
    return None


def main():
    parser = argparse.ArgumentParser(description="LTX 2.3 GGUF Lip-Sync Test")
    parser.add_argument("--image", required=True, help="Reference image path")
    parser.add_argument("--audio", required=True, help="Audio file path (WAV/MP3)")
    parser.add_argument("--prompt", default="A person speaking with natural lip movements, high quality, detailed face",
                        help="Text prompt")
    parser.add_argument("--width", type=int, default=720, help="Output width")
    parser.add_argument("--height", type=int, default=1280, help="Output height")
    parser.add_argument("--fps", type=int, default=24, help="Frame rate")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", choices=["turbo", "default"], default="turbo", help="Sampling mode: turbo (8 steps) or default (12 steps)")
    parser.add_argument("--lora", type=float, default=0.5, help="LoRA strength (0 to disable)")
    parser.add_argument("--i2v", type=float, default=0.6, help="Image-to-video strength")
    parser.add_argument("--output", default="/home/ericr/ComfyUI/output/ltx_lipsync",
                        help="Output directory")
    args = parser.parse_args()

    # Validate inputs exist
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)
    if not os.path.exists(args.audio):
        print(f"ERROR: Audio not found: {args.audio}")
        sys.exit(1)

    # Check ComfyUI is running
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats")
    except Exception:
        print("ERROR: ComfyUI not running at", COMFYUI_URL)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"LTX 2.3 GGUF Lip-Sync Test")
    print(f"  Image: {args.image}")
    print(f"  Audio: {args.audio}")
    print(f"  Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"  Duration: {args.duration}s")
    print(f"  Mode: {args.mode}")
    print(f"  LoRA: {args.lora}, I2V: {args.i2v}")
    print(f"  Seed: {args.seed}")
    print()

    # Build and queue workflow
    workflow = build_workflow(
        image_path=args.image,
        audio_path=args.audio,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        seed=args.seed,
        mode=args.mode,
        lora_strength=args.lora,
        i2v_strength=args.i2v,
    )

    print("Queuing workflow...")
    prompt_id = queue_prompt(workflow)
    print(f"  prompt_id: {prompt_id}")
    print("  Waiting for completion (timeout 10 min)...")

    history = poll_history(prompt_id, timeout=600)
    if history:
        print("  Complete!")
        download_output(history, args.output)
    else:
        print("  TIMEOUT or FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
