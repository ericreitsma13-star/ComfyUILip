import json, requests, random, sys, os

API = "http://127.0.0.1:8188/prompt"
API_KEY = "ltxv_GJNim3DP2LwmLgF2VdTV-YV5hX9pxoIZOnlKNKTVkU2MRlesiTcBs-881s5gQ5RYi06kmcV-oYZMIUJxa82loKetMpHEV9JieH-g0r6UFe7_kJzxpfJ8wouidUA_doQoaYGPOvyrw2Kl7LjNiEZ1BGtRzYRT7GAtIB8VaHKOmHb3"

prompt_text = sys.argv[1] if len(sys.argv) > 1 else "a cinematic scene of a cyberpunk city at night with neon lights"

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}

prompt = {
    "10": node("UNETLoader",
        unet_name="ltx-2.3-22b-distilled-1.1.safetensors",
        weight_dtype="default"),
    "11": node("LoraLoaderModelOnly",
        model=("10", 0),
        lora_name="ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        strength_model=0.5),
    "12": node("LoraLoaderModelOnly",
        model=("11", 0),
        lora_name="sulphur_final.safetensors",
        strength_model=0.5),
    "40": node("LTXAPITextEncode",
        api_key=API_KEY,
        prompt=prompt_text,
        ckpt_name="sulphur_dev_fp8mixed.safetensors",
        enhance_prompt=False),
    "41": node("LTXAPITextEncode",
        api_key=API_KEY,
        prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text, logo, signature, oversaturated, overexposed, underexposed, grainy, noisy",
        ckpt_name="sulphur_dev_fp8mixed.safetensors",
        enhance_prompt=False),
    "42": node("EmptyLTXVLatentVideo",
        width=1280, height=720, length=97, batch_size=1),
    "48": node("VAELoader",
        vae_name="ltx-2.3-22b-distilled_video_vae.safetensors"),
    "43": node("ModelSamplingLTXV",
        model=("12", 0),
        max_shift=2.05, base_shift=0.95),
    "44": node("KSampler",
        model=("43", 0),
        positive=("40", 0), negative=("41", 0),
        latent_image=("42", 0),
        seed=random.randint(0, 2**63),
        steps=8, cfg=2.0, sampler_name="euler", scheduler="beta", denoise=1.0),
    "45": node("VAEDecode",
        samples=("44", 0), vae=("48", 0)),
    "46": node("CreateVideo",
        images=("45", 0), fps=24),
    "47": node("SaveVideo",
        video=("46", 0), filename_prefix="sulphur_t2v", format="auto", codec="auto"),
}

r = requests.post(API, json={"prompt": prompt})
print(r.status_code, r.text[:500])
