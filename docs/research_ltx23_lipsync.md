# LTX 2.3 LipSync Research

## Video 1: "Lip Sync Any AI Model with LTX 2.3 Audio to Video in WAN2GP!"
- **Link**: https://www.youtube.com/watch?v=TDZmMRkhHTk
- **Author**: The Local Lab
- **Date**: May 2026

### Key Takeaways
- Uses **Waifu2x GP** (WAN2GP) to run LTX 2.3 for lip-syncing singing characters
- **16 GB VRAM** config: use distilled GGUF Q4KM model quant
- **Resolution**: 720p for best results, 10-second clips in ~10 min
- **Critical LoRA**: Must use "camera control static LoRA" (LTX 2.0 compatible with 2.3) for motion — without it, output is stiff/still
- **Audio setting**: checkbox "ignore background music" forces model to focus on vocals
- **Process**: Image-to-video mode + upload audio + text prompt = sung lip sync
- For longer music videos: model handles up to 20s coherently, stitch segments

### Architecture
1. Select model → LTX 2.3 distilled GGUF Q4KM
2. Mode: Image-to-video (upload reference photo)
3. Control video: "generate based on soundtrack and text prompt"
4. Upload audio clip (must match video duration)
5. LoRA: camera control static LoRA for motion
6. Resolution/Frame count must match audio duration
7. Upscaler: spatial (not temporal)

---

## Video 2: "LTX 2.3 Custom Audio Workflow: Perfect Lip Sync + Pro Optimization Tips!"
- **Link**: https://www.youtube.com/watch?v=wZfUchcM6h4
- **Author**: SOTAI
- **Date**: May 2026

### Key Takeaways
- Builds custom ComfyUI workflow using native **ComfyUI-LTXVideo** nodes (not third-party)
- Core technique: encode real audio into latent using `LTXVAudioVAEEncode` instead of `EmptyLTXVLatentAudio`
- Uses **MelBand Roformer** for vocal separation (optional but recommended)
- **Zero-value mask** via `LTXVSetAudioVideoMaskByTime` to preserve encoded audio latent during sampling

### Implementation Flow
1. **Stage 1**: Load model, LoRA, VAE, audio VAE, image, audio
2. **Stage 2**: 
   - Encode audio with `LTXVAudioVAEEncode(audio, audio_vae)` → audio_latent
   - `EmptyLTXVLatentVideo` + `LTXVImgToVideoInplace` → video_latent
   - `LTXVConcatAVLatent(video_latent, audio_latent)` → NestedTensor
   - `LTXVSetAudioVideoMaskByTime` with `mask_init_value_audio=0` to preserve audio latent
3. **Stage 3**: Sample with combined AV latent + conditioning
4. **Stage 4**: Decode video with tiled VAE, optionally decode audio with `LTXVAudioVAEDecode`

### Optimization Tips
- **Use LoRA v1.1** (not v1.0) for better facial details + motion stability
- **Avoid face-enhancement LoRAs** in audio workflows — they hurt lip sync (override mouth)
- **Steps**: 15 (not 8) for best balance of speed + lip sync quality
- **Resolution**: Use LTX's recommended shape (832×480, 768×512, etc.) within constraints
- Try different resolutions if lip sync is weak — may help model pick up audio signal
- Last resort: increase steps to 30
- Audio quality matters — clearer vocals = better lip sync; try vocal separation first

### Architecture
- Uses official ComfyUI-LTXVideo nodes (already installed in our setup):
  - `LTXVAudioVAELoader` — loads audio VAE from main checkpoint
  - `LTXVAudioVAEEncode` — encodes audio waveform into latent
  - `LTXVConcatAVLatent` — merges video + audio latents into NestedTensor
  - `LTXVSeparateAVLatent` — splits NestedTensor back into video + audio
  - `LTXVSetAudioVideoMaskByTime` — masks to preserve audio signal

---

## Alternative Approaches Comparison

| Feature | Geekatplay LipSync | SOTAI Native | LipDub | ID-LoRA | SoundTracks |
|---------|--------------------|-------------|--------|---------|-------------|
| Lip Sync | ✅ | ✅ | ✅ | — | — |
| Audio Sync | ✅ | ✅ | ✅ AI | ✅ | ✅ Motion |
| Consistency | ✅ | ✅ | ✅ | ✅ Strong | ✅ |
| IC-LoRA | — | — | ✅ | ✅ | ✅ |
| Notes | Custom nodes | Native LTXVideo | Gemini AI | Identity | Audio motion |

### Key Insight
**SOTAI's native approach** (tested & working) is preferred because:
1. Uses built-in ComfyUI-LTXVideo nodes (no custom node fragility)
2. Compatible with CondSafe LoRA (preserves conditioning)
3. Full control over audio conditioning
4. Can run per-clip or full song with storyboard rotation
