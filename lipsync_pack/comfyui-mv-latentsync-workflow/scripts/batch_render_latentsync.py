#!/usr/bin/env python3
"""
batch_render_latentsync.py - Orchestrate ComfyUI API for the LatentSync pipeline.

For each scene:
  - If scene_type == "singer_latentsync":
      1. Extract vocal segment for this scene's time range
      2. Render base singer video (workflow 01 LTX or 02 Wan + IP-Adapter FaceID)
      3. Apply LatentSync 1.5 to the base video (workflow 05)
      4. Run face-restore post-FX (workflow 06)
  - If scene_type == "broll":
      1. Render B-roll (workflow 03 LTX or 04 Wan)
      2. Run standard post-FX (workflow 07)

USAGE:
    export COMFYUI_INPUT_DIR="/path/to/ComfyUI/input"
    python batch_render_latentsync.py \\
        --shotlist shotlist.json \\
        --singer-dir singer_ref_images/ \\
        --broll-dir broll_ref_images/ \\
        --vocals separated/vocals.wav \\
        --output-dir output/scenes/ \\
        --post-fx

NOTES:
    • All workflow JSONs must be in API format (Save (API Format) in ComfyUI).
    • Saves render_log.json after every scene (crash-safe).
    • Each singer scene produces 3 intermediate files:
        base_singer_NNN.mp4 → lipsync_latentsync_NNN.mp4 → scene_NNN_final.mp4
      All kept for debugging; only scene_NNN_final.mp4 is used by assemble_mv.py.
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: missing 'requests'. Run: pip install requests")
    sys.exit(1)


# -------- ComfyUI API helpers --------

def comfyui_queue_prompt(server_url, workflow_api, client_id="batch_render_latentsync"):
    url = f"{server_url.rstrip('/')}/prompt"
    payload = {"prompt": workflow_api, "client_id": client_id}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def comfyui_get_history(server_url, prompt_id):
    url = f"{server_url.rstrip('/')}/history/{prompt_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get(prompt_id)


def comfyui_wait_for_job(server_url, prompt_id, poll_interval=3.0, timeout=1800):
    start = time.time()
    while True:
        history = comfyui_get_history(server_url, prompt_id)
        if history is not None:
            return history
        if time.time() - start > timeout:
            raise TimeoutError(f"ComfyUI job {prompt_id} did not complete within {timeout}s")
        time.sleep(poll_interval)


def comfyui_find_output_file(history):
    outputs = history.get("outputs", {})
    for node_id, node_out in outputs.items():
        for key in ("gifs", "images", "video"):
            if key in node_out:
                for item in node_out[key]:
                    fn = item.get("filename") or item.get("file_name")
                    subfolder = item.get("subfolder", "")
                    if fn:
                        return os.path.join(subfolder, fn) if subfolder else fn
    return None


def comfyui_download_output(server_url, remote_path, local_path):
    parts = Path(remote_path).parts
    filename = parts[-1]
    subfolder = "/".join(parts[:-1]) if len(parts) > 1 else ""
    qs = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
    url = f"{server_url.rstrip('/')}/view?{qs}"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                f.write(chunk)


# -------- Vocal segment extraction --------

def extract_vocal_segment(vocals_path, start_sec, end_sec, output_path):
    duration = max(0.1, end_sec - start_sec)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration:.3f}",
        "-i", vocals_path,
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


# -------- Workflow patching --------

def patch_base_singer_workflow(workflow_api, scene, ref_image_path, output_prefix, seed):
    """Patch workflow 01 (LTX) or 02 (Wan) with scene values for base singer video."""
    wf = copy.deepcopy(workflow_api)
    ref_filename = Path(ref_image_path).name

    nodes_by_type = {}
    for node_id, node in wf.items():
        ct = node.get("class_type", "")
        nodes_by_type.setdefault(ct, []).append(node_id)

    # LoadImage
    for ct in ("LoadImage", "ETN_LoadImageBase64"):
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["image"] = ref_filename

    # CLIPTextEncode (positive then negative)
    text_node_ids = nodes_by_type.get("CLIPTextEncode", [])
    if len(text_node_ids) >= 2:
        wf[text_node_ids[0]]["inputs"]["text"] = scene.get("prompt", "")
        wf[text_node_ids[1]]["inputs"]["text"] = scene.get("negative_prompt", "")
    elif len(text_node_ids) == 1:
        wf[text_node_ids[0]]["inputs"]["text"] = scene.get("prompt", "")

    # Seed (on RandomNoise node if present, else on I2V node)
    for ct in ("RandomNoise",):
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["noise_seed"] = int(seed)

    # I2V node — set LOW motion for LatentSync stability
    motion = float(scene.get("motion_strength", 0.3))
    for ct in ("LTXVImgToVideoInplace",):
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                # strength: 1.0 = full motion from image, 0.5 = moderate
                # For LatentSync stability, use lower motion
                wf[nid]["inputs"]["strength"] = round(max(0.3, 1.0 - motion), 2)
    for ct in ("WanImageToVideo", "WanVideoImageToVideo"):
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["motion_strength"] = motion

    # VHS_VideoCombine
    for ct in ("VHS_VideoCombine", "VHS_VideoCombineMux"):
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["filename_prefix"] = output_prefix
    return wf


def patch_broll_workflow(workflow_api, scene, ref_image_path, output_prefix, seed):
    """Patch workflow 03 (LTX) or 04 (Wan) with scene values for B-roll."""
    wf = copy.deepcopy(workflow_api)
    ref_filename = Path(ref_image_path).name

    nodes_by_type = {}
    for node_id, node in wf.items():
        ct = node.get("class_type", "")
        nodes_by_type.setdefault(ct, []).append(node_id)

    for ct in ("LoadImage", "ETN_LoadImageBase64"):
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["image"] = ref_filename

    text_node_ids = nodes_by_type.get("CLIPTextEncode", [])
    if len(text_node_ids) >= 2:
        wf[text_node_ids[0]]["inputs"]["text"] = scene.get("prompt", "")
        wf[text_node_ids[1]]["inputs"]["text"] = scene.get("negative_prompt", "")
    elif len(text_node_ids) == 1:
        wf[text_node_ids[0]]["inputs"]["text"] = scene.get("prompt", "")

    # Seed
    for ct in ("RandomNoise",):
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["noise_seed"] = int(seed)

    # I2V motion
    motion = float(scene.get("motion_strength", 0.5))
    for ct in ("LTXVImgToVideoInplace",):
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["strength"] = round(max(0.3, 1.0 - motion), 2)
    for ct in ("WanImageToVideo", "WanVideoImageToVideo"):
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["motion_strength"] = motion

    for ct in ("VHS_VideoCombine", "VHS_VideoCombineMux"):
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["filename_prefix"] = output_prefix
    return wf


def patch_latentsync_workflow(workflow_api, base_video_filename, vocal_segment_filename, output_prefix, scene):
    """Patch workflow 05 (LatentSync apply) for the LatentSyncNode + VideoLengthAdjuster chain.

    Expected API node types:
      VHS_LoadVideo      → loads base singer video
      LoadAudio          → loads vocal segment
      VideoLengthAdjuster → syncs video/audio lengths
      LatentSyncNode     → runs lip-sync inference
      VHS_VideoCombine   → saves output MP4
    """
    wf = copy.deepcopy(workflow_api)
    seed = int(scene.get("latentsync_seed", 42))
    lips_expression = float(scene.get("latentsync_lips_expression", 1.5))
    inference_steps = int(scene.get("latentsync_inference_steps", 20))

    for node_id, node in wf.items():
        ct = node.get("class_type", "")
        if ct in ("VHS_LoadVideo", "VHS_LoadVideoFFmpeg"):
            node["inputs"]["video"] = base_video_filename
            if "format" in node["inputs"]:
                node["inputs"]["format"] = "None"
        elif ct == "LoadAudio":
            node["inputs"]["audio"] = vocal_segment_filename
        elif ct == "LatentSyncNode":
            node["inputs"]["seed"] = seed
            node["inputs"]["lips_expression"] = lips_expression
            node["inputs"]["inference_steps"] = inference_steps
        elif ct == "VideoLengthAdjuster":
            node["inputs"]["fps"] = 25
            node["inputs"]["mode"] = "normal"
        elif ct in ("VHS_VideoCombine",):
            node["inputs"]["filename_prefix"] = output_prefix
    return wf


def patch_postfx_workflow(workflow_api, input_video_filename, output_prefix):
    """Patch workflow 06 or 07 (post-FX) with input video filename."""
    wf = copy.deepcopy(workflow_api)
    for node_id, node in wf.items():
        ct = node.get("class_type", "")
        if ct in ("VHS_LoadVideo", "VHS_LoadVideoFFmpeg"):
            node["inputs"]["video"] = input_video_filename
            if "format" in node["inputs"]:
                node["inputs"]["format"] = "None"
        elif ct in ("VHS_VideoCombine",):
            node["inputs"]["filename_prefix"] = output_prefix
    return wf


# -------- Render helpers --------

def render_workflow(server_url, workflow_api, output_dir, expected_filename_prefix, poll_interval, timeout, comfyui_input_dir, local_input_file=None):
    """Queue a workflow, wait, download output. Returns (local_path, render_time)."""
    t0 = time.time()
    resp = comfyui_queue_prompt(server_url, workflow_api)
    prompt_id = resp["prompt_id"]
    history = comfyui_wait_for_job(server_url, prompt_id, poll_interval=poll_interval, timeout=timeout)
    remote_file = comfyui_find_output_file(history)
    if not remote_file:
        raise RuntimeError(f"No output file for prompt {prompt_id}")
    # Determine local output path — use the expected prefix
    remote_filename = Path(remote_file).name
    local_path = str(Path(output_dir) / remote_filename)
    comfyui_download_output(server_url, remote_file, local_path)
    render_time = time.time() - t0
    return local_path, render_time


def main():
    p = argparse.ArgumentParser(description="Batch-render LatentSync + broll scenes via ComfyUI API.")
    p.add_argument("--shotlist", required=True)
    p.add_argument("--singer-dir", required=True)
    p.add_argument("--broll-dir", required=True)
    p.add_argument("--vocals", required=True, help="Path to isolated vocals.wav from vocal_isolation.py")
    p.add_argument("--comfyui-url", default="http://127.0.0.1:8188")
    p.add_argument("--workflow-base-ltx", default="workflows/01_base_singer_LTX_IPAdapter_api.json")
    p.add_argument("--workflow-base-wan", default="workflows/02_base_singer_Wan_IPAdapter_api.json")
    p.add_argument("--workflow-broll-ltx", default="workflows/03_broll_render_LTX_api.json")
    p.add_argument("--workflow-broll-wan", default="workflows/04_broll_render_Wan_api.json")
    p.add_argument("--workflow-latentsync", default="workflows/05_latentsync_apply_api.json")
    p.add_argument("--workflow-postfx-singer", default="workflows/06_face_restore_postfx_api.json")
    p.add_argument("--workflow-postfx-broll",   default="workflows/07_standard_postfx_api.json")
    p.add_argument("--output-dir", default="output/scenes")
    p.add_argument("--post-fx", action="store_true",
                   help="After each scene, run the appropriate post-FX workflow on it.")
    p.add_argument("--only-index", type=int, default=None)
    p.add_argument("--only-type", choices=("singer_latentsync", "broll"), default=None)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--poll-interval", type=float, default=3.0)
    p.add_argument("--per-job-timeout", type=float, default=1800.0)
    args = p.parse_args()

    with open(args.shotlist, encoding="utf-8") as f:
        shotlist = json.load(f)
    scenes = shotlist["scenes"]
    print(f"[batch_render_latentsync] Loaded {len(scenes)} scenes")

    if not os.path.isfile(args.vocals):
        print(f"ERROR: vocals file not found: {args.vocals}")
        sys.exit(1)

    # Load workflow templates
    wf_base_ltx = json.load(open(args.workflow_base_ltx, encoding="utf-8"))
    wf_base_wan = json.load(open(args.workflow_base_wan, encoding="utf-8"))
    wf_broll_ltx = json.load(open(args.workflow_broll_ltx, encoding="utf-8"))
    wf_broll_wan = json.load(open(args.workflow_broll_wan, encoding="utf-8"))
    wf_latentsync = json.load(open(args.workflow_latentsync, encoding="utf-8"))
    wf_postfx_singer = json.load(open(args.workflow_postfx_singer, encoding="utf-8")) if args.post_fx else None
    wf_postfx_broll = json.load(open(args.workflow_postfx_broll, encoding="utf-8")) if args.post_fx else None

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    comfyui_input_dir = os.environ.get("COMFYUI_INPUT_DIR", "")
    if not comfyui_input_dir or not Path(comfyui_input_dir).is_dir():
        print("[batch_render_latentsync] ERROR: COMFYUI_INPUT_DIR env var not set or not a directory.")
        sys.exit(1)
    print(f"[batch_render_latentsync] ComfyUI input dir: {comfyui_input_dir}")

    # Pre-sync singer ref image
    singer_ref = shotlist.get("singer_ref_image")
    if singer_ref:
        src = Path(args.singer_dir) / singer_ref
        if src.is_file():
            dst = Path(comfyui_input_dir) / singer_ref
            if not dst.exists():
                shutil.copy2(src, dst)
            print(f"[batch_render_latentsync] Synced singer ref: {singer_ref}")

    log = {
        "rendered_at": datetime.now(timezone.utc).isoformat(),
        "total_scenes": len(scenes),
        "scenes": [],
    }

    for scene in scenes:
        idx = scene["index"]
        if args.only_index is not None and idx != args.only_index:
            continue
        if args.only_type is not None and scene.get("scene_type") != args.only_type:
            continue

        scene_type = scene.get("scene_type", "broll")
        seed = idx * 1000 + args.seed_offset

        if scene_type == "singer_latentsync":
            # === SINGER SCENE (3-stage: base → latentsync → postfx) ===
            base_model = scene.get("base_model", "wan")
            print(f"\n[batch_render_latentsync] Scene {idx:2d}  SINGER/{base_model:3s}  seed={seed}")
            print(f"                            dur={scene.get('duration_sec', 0):.1f}s  motion={scene.get('motion_strength', 0.3):.2f}")

            # Stage 1: Extract vocal segment
            vocal_seg_filename = scene.get("vocal_audio_segment", f"vocal_segment_{idx:03d}.wav")
            vocal_seg_local = Path(args.output_dir) / "vocal_segments" / vocal_seg_filename
            vocal_seg_local.parent.mkdir(parents=True, exist_ok=True)
            vocal_seg_comfy = Path(comfyui_input_dir) / vocal_seg_filename

            try:
                extract_vocal_segment(
                    args.vocals,
                    float(scene["start_time"]),
                    float(scene["end_time"]),
                    str(vocal_seg_local),
                )
                shutil.copy2(vocal_seg_local, vocal_seg_comfy)
                print(f"                            [1/3] vocal segment: {vocal_seg_filename} ({vocal_seg_local.stat().st_size/1024:.1f} KB)")
            except subprocess.CalledProcessError as e:
                print(f"                            FAILED vocal extract: {e}")
                log["scenes"].append({
                    "index": idx, "scene_type": "singer_latentsync",
                    "status": f"vocal_extract_error: {e}",
                })
                with open(Path(args.output_dir) / "render_log.json", "w") as f:
                    json.dump(log, f, indent=2)
                continue

            # Stage 2: Render base singer video
            base_prefix = f"base_singer_{idx:03d}"
            base_out_path = Path(args.output_dir) / f"{base_prefix}.mp4"
            if base_out_path.exists():
                print(f"                            [2/3] base singer SKIP (exists)")
                base_render_time = 0.0
            else:
                try:
                    wf_base_template = wf_base_ltx if base_model == "ltx" else wf_base_wan
                    wf_patched = patch_base_singer_workflow(
                        wf_base_template, scene,
                        ref_image_path=str(Path(args.singer_dir) / scene["ref_image"]),
                        output_prefix=base_prefix,
                        seed=seed,
                    )
                    base_local, base_render_time = render_workflow(
                        args.comfyui_url, wf_patched, args.output_dir, base_prefix,
                        args.poll_interval, args.per_job_timeout, comfyui_input_dir,
                    )
                    # Rename to expected name
                    if Path(base_local).resolve() != base_out_path.resolve():
                        shutil.move(base_local, base_out_path)
                    print(f"                            [2/3] base singer DONE in {base_render_time:.1f}s → {base_out_path.name}")
                except Exception as e:
                    print(f"                            [2/3] base singer FAILED: {e}")
                    log["scenes"].append({
                        "index": idx, "scene_type": "singer_latentsync",
                        "status": f"base_render_error: {e}",
                    })
                    with open(Path(args.output_dir) / "render_log.json", "w") as f:
                        json.dump(log, f, indent=2)
                    continue

            # Stage 3: Apply LatentSync
            lipsync_prefix = f"lipsync_latentsync_{idx:03d}"
            lipsync_out_path = Path(args.output_dir) / f"{lipsync_prefix}.mp4"
            if lipsync_out_path.exists():
                print(f"                            [3/3] LatentSync SKIP (exists)")
                lipsync_render_time = 0.0
            else:
                try:
                    # Copy base video into ComfyUI input for VHS_LoadVideo
                    shutil.copy2(base_out_path, Path(comfyui_input_dir) / base_out_path.name)
                    wf_patched = patch_latentsync_workflow(
                        wf_latentsync,
                        base_video_filename=base_out_path.name,
                        vocal_segment_filename=vocal_seg_filename,
                        output_prefix=lipsync_prefix,
                        scene=scene,
                    )
                    lipsync_local, lipsync_render_time = render_workflow(
                        args.comfyui_url, wf_patched, args.output_dir, lipsync_prefix,
                        args.poll_interval, args.per_job_timeout * 2, comfyui_input_dir,
                    )
                    if Path(lipsync_local).resolve() != lipsync_out_path.resolve():
                        shutil.move(lipsync_local, lipsync_out_path)
                    print(f"                            [3/3] LatentSync DONE in {lipsync_render_time:.1f}s → {lipsync_out_path.name}")
                except Exception as e:
                    print(f"                            [3/3] LatentSync FAILED: {e}")
                    log["scenes"].append({
                        "index": idx, "scene_type": "singer_latentsync",
                        "status": f"latentsync_error: {e}",
                    })
                    with open(Path(args.output_dir) / "render_log.json", "w") as f:
                        json.dump(log, f, indent=2)
                    continue

            # Stage 4 (optional): Face-restore post-FX
            postfx_out_path = Path(args.output_dir) / f"scene_{idx:03d}_final.mp4"
            postfx_done = False
            if args.post_fx and wf_postfx_singer is not None:
                if postfx_out_path.exists():
                    print(f"                            [postfx] SKIP (exists)")
                    postfx_done = True
                else:
                    try:
                        shutil.copy2(lipsync_out_path, Path(comfyui_input_dir) / lipsync_out_path.name)
                        wf_post = patch_postfx_workflow(
                            wf_postfx_singer,
                            input_video_filename=lipsync_out_path.name,
                            output_prefix=f"scene_{idx:03d}_final",
                        )
                        postfx_local, postfx_render_time = render_workflow(
                            args.comfyui_url, wf_post, args.output_dir, f"scene_{idx:03d}_final",
                            args.poll_interval, args.per_job_timeout * 2, comfyui_input_dir,
                        )
                        if Path(postfx_local).resolve() != postfx_out_path.resolve():
                            shutil.move(postfx_local, postfx_out_path)
                        print(f"                            [postfx] DONE in {postfx_render_time:.1f}s → {postfx_out_path.name}")
                        postfx_done = True
                    except Exception as e:
                        print(f"                            [postfx] FAILED: {e} (continuing with lipsync output)")

            # If post-FX failed, use the lipsync output as the final
            final_path = str(postfx_out_path) if postfx_done else str(lipsync_out_path)
            log["scenes"].append({
                "index": idx, "scene_type": "singer_latentsync", "base_model": base_model,
                "base_path": str(base_out_path),
                "lipsync_path": str(lipsync_out_path),
                "output_path": final_path,
                "postfx_path": str(postfx_out_path) if postfx_done else None,
                "vocal_segment": vocal_seg_filename,
                "duration_sec": scene.get("duration_sec"),
                "render_time_sec": round(base_render_time + lipsync_render_time, 2),
                "status": "ok",
            })

        else:
            # === B-ROLL SCENE ===
            model = scene.get("model", "wan")
            print(f"\n[batch_render_latentsync] Scene {idx:2d}  BROLL/{model:3s}  seed={seed}")

            prefix = f"scene_{idx:03d}_broll_{model}"
            out_path = Path(args.output_dir) / f"{prefix}.mp4"
            postfx_out_path = Path(args.output_dir) / f"scene_{idx:03d}_final.mp4"

            if out_path.exists():
                print(f"                            SKIP (exists)")
                log["scenes"].append({
                    "index": idx, "scene_type": "broll", "model": model,
                    "output_path": str(out_path),
                    "postfx_path": str(postfx_out_path) if args.post_fx and postfx_out_path.exists() else None,
                    "duration_sec": scene.get("duration_sec"),
                    "render_time_sec": 0.0,
                    "status": "skipped_existing",
                })
            else:
                # Sync broll ref
                src = Path(args.broll_dir) / scene["ref_image"]
                if src.is_file():
                    dst = Path(comfyui_input_dir) / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)

                try:
                    wf_template = wf_broll_ltx if model == "ltx" else wf_broll_wan
                    wf_patched = patch_broll_workflow(
                        wf_template, scene,
                        ref_image_path=str(src),
                        output_prefix=prefix,
                        seed=seed,
                    )
                    local_path, render_time = render_workflow(
                        args.comfyui_url, wf_patched, args.output_dir, prefix,
                        args.poll_interval, args.per_job_timeout, comfyui_input_dir,
                    )
                    if Path(local_path).resolve() != out_path.resolve():
                        shutil.move(local_path, out_path)
                    print(f"                            DONE in {render_time:.1f}s → {out_path.name}")
                except Exception as e:
                    print(f"                            FAILED: {e}")
                    log["scenes"].append({
                        "index": idx, "scene_type": "broll", "model": model,
                        "status": f"error: {e}",
                    })
                    with open(Path(args.output_dir) / "render_log.json", "w") as f:
                        json.dump(log, f, indent=2)
                    continue

                # Post-FX
                postfx_done = False
                if args.post_fx and wf_postfx_broll is not None:
                    try:
                        shutil.copy2(out_path, Path(comfyui_input_dir) / out_path.name)
                        wf_post = patch_postfx_workflow(
                            wf_postfx_broll,
                            input_video_filename=out_path.name,
                            output_prefix=f"scene_{idx:03d}_final",
                        )
                        postfx_local, postfx_render_time = render_workflow(
                            args.comfyui_url, wf_post, args.output_dir, f"scene_{idx:03d}_final",
                            args.poll_interval, args.per_job_timeout * 2, comfyui_input_dir,
                        )
                        if Path(postfx_local).resolve() != postfx_out_path.resolve():
                            shutil.move(postfx_local, postfx_out_path)
                        print(f"                            [postfx] DONE → {postfx_out_path.name}")
                        postfx_done = True
                    except Exception as e:
                        print(f"                            [postfx] FAILED: {e}")

                final_path = str(postfx_out_path) if postfx_done else str(out_path)
                log["scenes"].append({
                    "index": idx, "scene_type": "broll", "model": model,
                    "output_path": final_path,
                    "postfx_path": str(postfx_out_path) if postfx_done else None,
                    "duration_sec": scene.get("duration_sec"),
                    "render_time_sec": round(render_time, 2),
                    "status": "ok",
                })

        with open(Path(args.output_dir) / "render_log.json", "w") as f:
            json.dump(log, f, indent=2)

    print(f"\n[batch_render_latentsync] Done. Render log: {Path(args.output_dir)/'render_log.json'}")
    singer_ok = sum(1 for s in log["scenes"] if s.get("scene_type") == "singer_latentsync" and s.get("status") == "ok")
    broll_ok = sum(1 for s in log["scenes"] if s.get("scene_type") == "broll" and s.get("status") == "ok")
    print(f"[batch_render_latentsync]   Singer (LatentSync) OK: {singer_ok}")
    print(f"[batch_render_latentsync]   B-roll OK:             {broll_ok}")


if __name__ == "__main__":
    main()
