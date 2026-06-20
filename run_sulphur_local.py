import json, requests, random, sys

API = "http://127.0.0.1:8188/prompt"
prompt_text = sys.argv[1] if len(sys.argv) > 1 else "a cinematic scene of a cyberpunk city at night with neon lights and rain"
audio_file = sys.argv[2] if len(sys.argv) > 2 else None

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

p = {
    "10": node("UNETLoader", unet_name="ltx-2.3-22b-distilled-1.1.safetensors", weight_dtype="default"),
    "11": node("LoraLoaderModelOnly", model=("10", 0), lora_name="ltx-2.3-22b-distilled-lora-384-1.1.safetensors", strength_model=0.5),
    "12": node("LoraLoaderModelOnly", model=("11", 0), lora_name="sulphur_final.safetensors", strength_model=1.0),
    # Use fp8 Gemma for full quality text encoding
    "332": node("DualCLIPLoaderGGUF", clip_name1="gemma-3-12b-it-qat-UD-Q4_K_XL.gguf", clip_name2="ltx-2-3-22b-text_encoder.safetensors", type="ltxv"),
    "2": node("CLIPTextEncode", text=prompt_text, clip=("332", 0)),
    "3": node("CLIPTextEncode", text="blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text", clip=("332", 0)),
    "5": node("LTXVConditioning", positive=("2", 0), negative=("3", 0), frame_rate=24),
    "48": node("VAELoader", vae_name="ltx-2-3-22b-VAE.safetensors"),
    "389": node("VAELoader", vae_name="ltx-2-3-22b-audio_vae.safetensors"),
    "4": node("EmptyLTXVLatentVideo", width=1280, height=720, length=97, batch_size=1),
}

if audio_file:
    p["351"] = node("LoadAudio", audio=audio_file)
    p["352"] = node("TrimAudioDuration", audio=("351", 0), start_time=0, end_time=10)
    p["364"] = node("LTXVAudioVAEEncode", audio=("352", 0), audio_vae=("389", 0))
    p["321"] = node("LTXVConcatAVLatent", video_latent=("4", 0), audio_latent=("364", 0))
    p["6"] = node("KSampler", model=("12", 0), positive=("5", 0), negative=("5", 1), latent_image=("321", 0), seed=random.randint(0, 2**63), steps=8, cfg=2.0, sampler_name="euler", scheduler="beta", denoise=1.0)
    p["432"] = node("LTXVSeparateAVLatent", av_latent=("6", 0))
    p["7"] = node("VAEDecode", samples=("432", 0), vae=("48", 0))
    p["424"] = node("LTXVAudioVAEDecode", samples=("432", 1), audio_vae=("389", 0))
    p["8"] = node("CreateVideo", images=("7", 0), audio=("424", 0), fps=24)
else:
    p["6"] = node("KSampler", model=("12", 0), positive=("5", 0), negative=("5", 1), latent_image=("4", 0), seed=random.randint(0, 2**63), steps=8, cfg=2.0, sampler_name="euler", scheduler="beta", denoise=1.0)
    p["7"] = node("VAEDecode", samples=("6", 0), vae=("48", 0))
    p["8"] = node("CreateVideo", images=("7", 0), fps=24)

p["9"] = node("SaveVideo", video=("8", 0), filename_prefix="sulphur_local", format="auto", codec="auto")

r = requests.post(API, json={"prompt": p})
print(f"Status: {r.status_code}")
resp = r.json()
if "error" in resp:
    print(f"Error: {resp['error'].get('message', resp)}")
