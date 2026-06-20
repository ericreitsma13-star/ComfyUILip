#!/usr/bin/env python3
"""
LTX 2.3 Full Checkpoint Lip-Sync Test
Uses full safetensors checkpoint with lowvram offloading
"""
import json, uuid, time, urllib.request, urllib.parse, random, sys

COMFYUI_URL = "http://127.0.0.1:8188"

def build_workflow(image, audio, prompt, w=704, h=1280, fps=24, dur=8, seed=42):
    fc = int(((dur * fps - 1) / 8) * 8 + 1)
    return {
        "1": {"class_type": "LowVRAMCheckpointLoader", "inputs": {"ckpt_name": "ltx-2.3-22b-distilled-1.1.safetensors"}, "_meta": {"title": "Load Full Checkpoint (lowvram)"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "strength_model": 0.3}, "_meta": {"title": "LoRA"}},
        "3": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltx-2.3-22b-distilled_audio_vae.safetensors"}, "_meta": {"title": "Audio VAE"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.3-22b-distilled_video_vae.safetensors"}, "_meta": {"title": "Video VAE"}},
        "5": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": "gemma_3_12B_it_fp8_e4m3fn.safetensors", "ckpt_name": "ltx-2-3-22b-text_encoder.safetensors", "device": "default"}, "_meta": {"title": "Text Encoder"}},
        "10": {"class_type": "LoadImage", "inputs": {"image": image}, "_meta": {"title": "Reference"}},
        "11": {"class_type": "LoadAudio", "inputs": {"audio": audio}, "_meta": {"title": "Audio"}},
        "20": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["11", 0], "audio_vae": ["3", 0]}, "_meta": {"title": "Encode Audio"}},
        "25": {"class_type": "LTXVPreprocess", "inputs": {"image": ["10", 0], "frames": fc, "img_compression": 90}, "_meta": {"title": "Preprocess"}},
        "30": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w // 2, "height": h // 2, "length": fc, "batch_size": 1}, "_meta": {"title": "Empty Latent"}},
        "35": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["4", 0], "image": ["25", 0], "latent": ["30", 0], "strength": 0.7, "bypass": False}, "_meta": {"title": "I2V"}},
        "40": {"class_type": "SolidMask", "inputs": {"width": w, "height": h, "value": 0}, "_meta": {"title": "Mask"}},
        "41": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["20", 0], "mask": ["40", 0]}, "_meta": {"title": "Set Mask"}},
        "45": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["35", 0], "audio_latent": ["41", 0]}, "_meta": {"title": "Concat"}},
        "50": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": prompt}, "_meta": {"title": "Positive"}},
        "51": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": "ugly, distorted face, blurry, cartoon, bad anatomy, crooked"}, "_meta": {"title": "Negative"}},
        "52": {"class_type": "LTXVConditioning", "inputs": {"positive": ["50", 0], "negative": ["51", 0], "frame_rate": float(fps)}, "_meta": {"title": "Conditioning"}},
        "60": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}, "_meta": {"title": "Noise"}},
        "61": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}, "_meta": {"title": "Sampler"}},
        "62": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0", "steps": 8}, "_meta": {"title": "Sigmas"}},
        "63": {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["52", 0], "negative": ["52", 1], "cfg": 1.0}, "_meta": {"title": "Guider"}},
        "64": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["60", 0], "guider": ["63", 0], "sampler": ["61", 0], "sigmas": ["62", 0], "latent_image": ["45", 0]}, "_meta": {"title": "Sample"}},
        "70": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["64", 0]}, "_meta": {"title": "Separate"}},
        "71": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["70", 0], "vae": ["4", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}, "_meta": {"title": "Decode Video"}},
        "72": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["70", 1], "audio_vae": ["3", 0]}, "_meta": {"title": "Decode Audio"}},
        "80": {"class_type": "CreateVideo", "inputs": {"images": ["71", 0], "audio": ["72", 0], "fps": float(fps)}, "_meta": {"title": "Create Video"}},
        "81": {"class_type": "LTXMotionSaveVideo", "inputs": {"video": ["80", 0], "filename_prefix": "ltx_full_ckpt"}, "_meta": {"title": "Save"}},
    }

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--prompt", default="A person speaking, natural lip movements, detailed face")
    p.add_argument("--width", type=int, default=704)
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--duration", type=float, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Full Checkpoint Test (lowvram offload)")
    print(f"  {args.width}x{args.height} @ {args.fps}fps, {args.duration}s")

    wf = build_workflow(args.image, args.audio, args.prompt, args.width, args.height, args.fps, args.duration, args.seed)
    payload = json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=payload, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    pid = json.loads(resp.read())["prompt_id"]
    print(f"  Queued: {pid}")

    start = time.time()
    while time.time() - start < 900:
        try:
            r = urllib.request.urlopen(f"{COMFYUI_URL}/history/{pid}")
            d = json.loads(r.read())
            if pid in d and d[pid].get("status", {}).get("completed"):
                out = d[pid]["outputs"].get("81", {})
                print(f"  Complete! ({time.time()-start:.0f}s)")
                break
        except: pass
        time.sleep(3)
    else:
        print("  TIMEOUT")
        sys.exit(1)

if __name__ == "__main__":
    main()
