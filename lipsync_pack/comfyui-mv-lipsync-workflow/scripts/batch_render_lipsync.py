#!/usr/bin/env python3
"""
batch_render_lipsync.py - Orchestrate ComfyUI API calls to render lipsync + broll scenes.

Reads:
  - shotlist.json     (from llm_shotlist_lipsync.py — has scene_type field per scene)
  - singer_ref_images/   (where the singer reference image lives)
  - broll_ref_images/    (where B-roll reference images live)
  - separated/vocals.wav (from vocal_isolation.py — used to extract per-scene vocal segments)

For each scene:
  - If scene_type == "lipsync":
      1. Extract the vocal segment for this scene's time range (ffmpeg trim of vocals.wav)
      2. Copy singer ref image + vocal segment into ComfyUI/input/
      3. Patch the Sonic workflow (01_lipsync_render_Sonic_api.json) with scene values
      4. Queue + wait + download output MP4
      5. Run the face-restore post-FX workflow (04_face_restore_postfx_api.json)
  - If scene_type == "broll":
      1. Copy broll ref image into ComfyUI/input/
      2. Patch the appropriate B-roll workflow (02 LTX or 03 Wan)
      3. Queue + wait + download output MP4
      4. Run the standard post-FX workflow (05_standard_postfx_api.json)

USAGE:
    python batch_render_lipsync.py \\
        --shotlist shotlist.json \\
        --singer-dir singer_ref_images/ \\
        --broll-dir broll_ref_images/ \\
        --vocals separated/vocals.wav \\
        --comfyui-url http://127.0.0.1:8188 \\
        --workflow-lipsync workflows/01_lipsync_render_Sonic_api.json \\
        --workflow-ltx workflows/02_broll_render_LTX_api.json \\
        --workflow-wan workflows/03_broll_render_Wan_api.json \\
        --workflow-postfx-lipsync workflows/04_face_restore_postfx_api.json \\
        --workflow-postfx-broll workflows/05_standard_postfx_api.json \\
        --output-dir output/scenes/ \\
        --post-fx

NOTES:
    • All workflow JSONs must be in API format (Save (API Format) in ComfyUI UI).
    • ComfyUI must be running and reachable.
    • The COMFYUI_INPUT_DIR env var must point at your ComfyUI/input/ folder so
      reference images + vocal segments can be loaded by the LoadImage / VHS_LoadAudio nodes.
    • The script saves render_log.json after every scene (crash-safe resume).
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: missing 'requests'. Run: pip install requests")
    sys.exit(1)


# -------- ComfyUI API helpers (reused from B-roll pack) --------

def comfyui_queue_prompt(server_url: str, workflow_api: dict, client_id: str = "batch_render_lipsync") -> dict:
    url = f"{server_url.rstrip('/')}/prompt"
    payload = {"prompt": workflow_api, "client_id": client_id}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def comfyui_get_history(server_url: str, prompt_id: str) -> dict | None:
    url = f"{server_url.rstrip('/')}/history/{prompt_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get(prompt_id)


def comfyui_wait_for_job(server_url: str, prompt_id: str, poll_interval: float = 2.0, timeout: float = 1800) -> dict:
    start = time.time()
    while True:
        history = comfyui_get_history(server_url, prompt_id)
        if history is not None:
            return history
        if time.time() - start > timeout:
            raise TimeoutError(f"ComfyUI job {prompt_id} did not complete within {timeout}s")
        time.sleep(poll_interval)


def comfyui_find_output_file(history: dict) -> str | None:
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


def comfyui_download_output(server_url: str, remote_path: str, local_path: str):
    parts = Path(remote_path).parts
    filename = parts[-1]
    subfolder = "/".join(parts[:-1]) if len(parts) > 1 else ""
    import urllib.parse
    qs = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
    url = f"{server_url.rstrip('/')}/view?{qs}"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                f.write(chunk)


# -------- Vocal segment extraction (NEW for lip-sync) --------

def extract_vocal_segment(vocals_path: str, start_sec: float, end_sec: float, output_path: str):
    """Use ffmpeg to extract a segment of the isolated vocal track for a lip-sync scene."""
    duration = max(0.1, end_sec - start_sec)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration:.3f}",
        "-i", vocals_path,
        "-ac", "1",           # mono — Sonic prefers mono audio input
        "-ar", "16000",       # 16kHz — Sonic's expected sample rate
        "-c:a", "pcm_s16le",  # 16-bit PCM WAV
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


# -------- Workflow patching --------

def patch_lipsync_workflow(workflow_api: dict, scene: dict, ref_image_path: str,
                            vocal_segment_path: str, output_prefix: str, seed: int) -> dict:
    """Patch Sonic lip-sync workflow with scene-specific values."""
    wf = copy.deepcopy(workflow_api)
    ref_filename = Path(ref_image_path).name
    vocal_filename = Path(vocal_segment_path).name

    for node_id, node in wf.items():
        ct = node.get("class_type", "")
        if ct == "LoadImage":
            node["inputs"]["image"] = ref_filename
        elif ct in ("VHS_LoadAudio", "VHS_LoadAudioUpload", "LoadAudio"):
            node["inputs"]["audio_file"] = vocal_filename
        elif ct == "SonicSampler":
            node["inputs"]["seed"] = int(seed)
            if "motion_strength" in node["inputs"]:
                node["inputs"]["motion_strength"] = float(scene.get("motion_strength", 0.6))
            if "lip_sync_strength" in node["inputs"]:
                node["inputs"]["lip_sync_strength"] = float(scene.get("lip_sync_strength", 0.7))
        elif ct in ("VHS_VideoCombine",):
            node["inputs"]["filename_prefix"] = output_prefix
    return wf


def patch_broll_workflow(workflow_api: dict, scene: dict, ref_image_path: str,
                          output_prefix: str, seed: int) -> dict:
    """Patch LTX or Wan B-roll workflow with scene-specific values."""
    wf = copy.deepcopy(workflow_api)
    ref_filename = Path(ref_image_path).name

    nodes_by_type: dict[str, list[str]] = {}
    for node_id, node in wf.items():
        ct = node.get("class_type", "")
        nodes_by_type.setdefault(ct, []).append(node_id)

    # LoadImage
    for ct in ("LoadImage", "ETN_LoadImageBase64", "LoadImageBatch"):
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

    # I2V node
    i2v_types = ("LTXImageToVideo", "WanImageToVideo", "WanVideoImageToVideo", "LTXVImageToVideo")
    motion = float(scene.get("motion_strength", 0.5))
    for ct in i2v_types:
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["seed"] = int(seed)
                if ct.startswith("LTX"):
                    wf[nid]["inputs"]["context_schedule"] = round(0.3 + motion * 0.7, 3)
                elif ct.startswith("Wan"):
                    wf[nid]["inputs"]["motion_strength"] = motion

    # VHS_VideoCombine
    for ct in ("VHS_VideoCombine", "VHS_VideoCombineMux", "SaveAnimatedWEBP"):
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["filename_prefix"] = output_prefix
    return wf


def patch_postfx_workflow(workflow_api: dict, input_video_filename: str, output_prefix: str) -> dict:
    """Patch a post-FX workflow (04 or 05) with the input video filename."""
    wf = copy.deepcopy(workflow_api)
    for node_id, node in wf.items():
        ct = node.get("class_type", "")
        if ct in ("VHS_LoadVideo", "VHS_LoadVideoFFmpeg"):
            node["inputs"]["video"] = input_video_filename
        elif ct in ("VHS_VideoCombine",):
            node["inputs"]["filename_prefix"] = output_prefix
    return wf


# -------- Main --------

def main():
    p = argparse.ArgumentParser(description="Batch-render lipsync + broll scenes via ComfyUI API.")
    p.add_argument("--shotlist", required=True)
    p.add_argument("--singer-dir", required=True)
    p.add_argument("--broll-dir", required=True)
    p.add_argument("--vocals", required=True, help="Path to isolated vocals.wav from vocal_isolation.py")
    p.add_argument("--comfyui-url", default="http://127.0.0.1:8188")
    p.add_argument("--workflow-lipsync", default="workflows/01_lipsync_render_Sonic_api.json")
    p.add_argument("--workflow-ltx",      default="workflows/02_broll_render_LTX_api.json")
    p.add_argument("--workflow-wan",      default="workflows/03_broll_render_Wan_api.json")
    p.add_argument("--workflow-postfx-lipsync", default="workflows/04_face_restore_postfx_api.json")
    p.add_argument("--workflow-postfx-broll",   default="workflows/05_standard_postfx_api.json")
    p.add_argument("--output-dir", default="output/scenes")
    p.add_argument("--post-fx", action="store_true",
                   help="After each scene renders, run the appropriate post-FX workflow on it.")
    p.add_argument("--only-index", type=int, default=None,
                   help="Render only the scene with this index (useful for retries).")
    p.add_argument("--only-type", choices=("lipsync", "broll"), default=None,
                   help="Render only scenes of this type.")
    p.add_argument("--only-model", choices=("ltx", "wan"), default=None,
                   help="Override model for all B-roll scenes.")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--poll-interval", type=float, default=3.0)
    p.add_argument("--per-job-timeout", type=float, default=1800.0)
    args = p.parse_args()

    with open(args.shotlist, encoding="utf-8") as f:
        shotlist = json.load(f)
    scenes = shotlist["scenes"]
    print(f"[batch_render_lipsync] Loaded {len(scenes)} scenes from {args.shotlist}")

    if not os.path.isfile(args.vocals):
        print(f"ERROR: vocals file not found: {args.vocals}")
        print("       Run vocal_isolation.py first.")
        sys.exit(1)

    # Load workflow templates
    wf_lipsync = json.load(open(args.workflow_lipsync, encoding="utf-8"))
    wf_ltx = json.load(open(args.workflow_ltx, encoding="utf-8"))
    wf_wan = json.load(open(args.workflow_wan, encoding="utf-8"))
    wf_postfx_lipsync = json.load(open(args.workflow_postfx_lipsync, encoding="utf-8")) \
        if args.post_fx and Path(args.workflow_postfx_lipsync).exists() else None
    wf_postfx_broll = json.load(open(args.workflow_postfx_broll, encoding="utf-8")) \
        if args.post_fx and Path(args.workflow_postfx_broll).exists() else None

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    comfyui_input_dir = os.environ.get("COMFYUI_INPUT_DIR", "")
    if not comfyui_input_dir or not Path(comfyui_input_dir).is_dir():
        print("[batch_render_lipsync] ERROR: COMFYUI_INPUT_DIR env var not set or not a directory.")
        print("                       Set it to your ComfyUI/input/ folder so reference images and")
        print("                       vocal segments can be loaded by LoadImage / VHS_LoadAudio.")
        sys.exit(1)
    print(f"[batch_render_lipsync] ComfyUI input dir: {comfyui_input_dir}")

    # Pre-sync singer ref image (used by all lipsync scenes)
    singer_ref = shotlist.get("singer_ref_image")
    if singer_ref:
        src = Path(args.singer_dir) / singer_ref
        if src.is_file():
            dst = Path(comfyui_input_dir) / singer_ref
            if not dst.exists():
                shutil.copy2(src, dst)
            print(f"[batch_render_lipsync] Synced singer ref: {singer_ref}")

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

        if scene_type == "lipsync":
            # === LIP-SYNC SCENE ===
            prefix = f"scene_{idx:03d}_lipsync"
            out_path = Path(args.output_dir) / f"{prefix}.mp4"
            postfx_out_path = Path(args.output_dir) / f"scene_{idx:03d}_final.mp4"

            if out_path.exists():
                print(f"[batch_render_lipsync] Scene {idx:2d}  LIPSYNC  SKIP (exists)")
                log["scenes"].append({
                    "index": idx, "scene_type": "lipsync",
                    "output_path": str(out_path),
                    "postfx_path": str(postfx_out_path) if args.post_fx and postfx_out_path.exists() else None,
                    "duration_sec": scene.get("duration_sec"),
                    "render_time_sec": 0.0,
                    "status": "skipped_existing",
                })
                continue

            print(f"[batch_render_lipsync] Scene {idx:2d}  LIPSYNC  seed={seed}  dur={scene.get('duration_sec', 0):.1f}s")

            # Extract vocal segment
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
                print(f"               vocal segment: {vocal_seg_filename} ({vocal_seg_local.stat().st_size/1024:.1f} KB)")
            except subprocess.CalledProcessError as e:
                print(f"               FAILED to extract vocal segment: {e}")
                log["scenes"].append({
                    "index": idx, "scene_type": "lipsync", "status": f"vocal_extract_error: {e}",
                })
                continue

            # Patch + queue Sonic workflow
            t0 = time.time()
            try:
                wf_patched = patch_lipsync_workflow(
                    wf_lipsync, scene,
                    ref_image_path=str(Path(args.singer_dir) / scene["ref_image"]),
                    vocal_segment_path=str(vocal_seg_local),
                    output_prefix=prefix,
                    seed=seed,
                )
                resp = comfyui_queue_prompt(args.comfyui_url, wf_patched)
                prompt_id = resp["prompt_id"]
                print(f"               queued as {prompt_id}; waiting ...")
                history = comfyui_wait_for_job(args.comfyui_url, prompt_id,
                                                poll_interval=args.poll_interval,
                                                timeout=args.per_job_timeout)
                remote_file = comfyui_find_output_file(history)
                if not remote_file:
                    raise RuntimeError(f"No output file found for prompt {prompt_id}")
                comfyui_download_output(args.comfyui_url, remote_file, str(out_path))
                render_time = time.time() - t0
                print(f"               DONE in {render_time:.1f}s → {out_path.name}")
            except Exception as e:
                render_time = time.time() - t0
                print(f"               FAILED after {render_time:.1f}s: {e}")
                log["scenes"].append({
                    "index": idx, "scene_type": "lipsync",
                    "output_path": None, "postfx_path": None,
                    "duration_sec": scene.get("duration_sec"),
                    "render_time_sec": render_time, "status": f"error: {e}",
                })
                # Save log and continue
                with open(Path(args.output_dir) / "render_log.json", "w") as f:
                    json.dump(log, f, indent=2)
                continue

            # Post-FX (face-restore pipeline for lipsync)
            postfx_done = False
            if args.post_fx and wf_postfx_lipsync is not None:
                print(f"               running face-restore post-FX ...")
                try:
                    shutil.copy2(out_path, Path(comfyui_input_dir) / out_path.name)
                    wf_post = patch_postfx_workflow(wf_postfx_lipsync, out_path.name, f"scene_{idx:03d}_final")
                    resp = comfyui_queue_prompt(args.comfyui_url, wf_post)
                    history = comfyui_wait_for_job(args.comfyui_url, resp["prompt_id"],
                                                    poll_interval=args.poll_interval,
                                                    timeout=args.per_job_timeout * 2)
                    remote_file = comfyui_find_output_file(history)
                    if remote_file:
                        comfyui_download_output(args.comfyui_url, remote_file, str(postfx_out_path))
                        print(f"               post-FX DONE → {postfx_out_path.name}")
                        postfx_done = True
                except Exception as e:
                    print(f"               post-FX FAILED: {e} (continuing with raw render)")

            log["scenes"].append({
                "index": idx, "scene_type": "lipsync",
                "output_path": str(out_path),
                "postfx_path": str(postfx_out_path) if postfx_done else None,
                "vocal_segment": vocal_seg_filename,
                "duration_sec": scene.get("duration_sec"),
                "render_time_sec": round(render_time, 2),
                "status": "ok",
            })

        else:
            # === B-ROLL SCENE ===
            model = args.only_model or scene.get("model", "wan")
            wf_template = wf_ltx if model == "ltx" else wf_wan
            prefix = f"scene_{idx:03d}_broll_{model}"
            out_path = Path(args.output_dir) / f"{prefix}.mp4"
            postfx_out_path = Path(args.output_dir) / f"scene_{idx:03d}_final.mp4"

            if out_path.exists():
                print(f"[batch_render_lipsync] Scene {idx:2d}  BROLL/{model:3s}  SKIP (exists)")
                log["scenes"].append({
                    "index": idx, "scene_type": "broll", "model": model,
                    "output_path": str(out_path),
                    "postfx_path": str(postfx_out_path) if args.post_fx and postfx_out_path.exists() else None,
                    "duration_sec": scene.get("duration_sec"),
                    "render_time_sec": 0.0,
                    "status": "skipped_existing",
                })
                continue

            print(f"[batch_render_lipsync] Scene {idx:2d}  BROLL/{model:3s}  seed={seed}  motion={scene.get('motion_strength', 0.5):.2f}")

            # Sync broll ref image
            src = Path(args.broll_dir) / scene["ref_image"]
            if src.is_file():
                dst = Path(comfyui_input_dir) / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

            t0 = time.time()
            try:
                wf_patched = patch_broll_workflow(
                    wf_template, scene,
                    ref_image_path=str(src),
                    output_prefix=prefix,
                    seed=seed,
                )
                resp = comfyui_queue_prompt(args.comfyui_url, wf_patched)
                prompt_id = resp["prompt_id"]
                print(f"               queued as {prompt_id}; waiting ...")
                history = comfyui_wait_for_job(args.comfyui_url, prompt_id,
                                                poll_interval=args.poll_interval,
                                                timeout=args.per_job_timeout)
                remote_file = comfyui_find_output_file(history)
                if not remote_file:
                    raise RuntimeError(f"No output file found for prompt {prompt_id}")
                comfyui_download_output(args.comfyui_url, remote_file, str(out_path))
                render_time = time.time() - t0
                print(f"               DONE in {render_time:.1f}s → {out_path.name}")
            except Exception as e:
                render_time = time.time() - t0
                print(f"               FAILED after {render_time:.1f}s: {e}")
                log["scenes"].append({
                    "index": idx, "scene_type": "broll", "model": model,
                    "output_path": None, "postfx_path": None,
                    "duration_sec": scene.get("duration_sec"),
                    "render_time_sec": render_time, "status": f"error: {e}",
                })
                with open(Path(args.output_dir) / "render_log.json", "w") as f:
                    json.dump(log, f, indent=2)
                continue

            # Post-FX (standard pipeline for broll)
            postfx_done = False
            if args.post_fx and wf_postfx_broll is not None:
                print(f"               running standard post-FX ...")
                try:
                    shutil.copy2(out_path, Path(comfyui_input_dir) / out_path.name)
                    wf_post = patch_postfx_workflow(wf_postfx_broll, out_path.name, f"scene_{idx:03d}_final")
                    resp = comfyui_queue_prompt(args.comfyui_url, wf_post)
                    history = comfyui_wait_for_job(args.comfyui_url, resp["prompt_id"],
                                                    poll_interval=args.poll_interval,
                                                    timeout=args.per_job_timeout * 2)
                    remote_file = comfyui_find_output_file(history)
                    if remote_file:
                        comfyui_download_output(args.comfyui_url, remote_file, str(postfx_out_path))
                        print(f"               post-FX DONE → {postfx_out_path.name}")
                        postfx_done = True
                except Exception as e:
                    print(f"               post-FX FAILED: {e} (continuing with raw render)")

            log["scenes"].append({
                "index": idx, "scene_type": "broll", "model": model,
                "output_path": str(out_path),
                "postfx_path": str(postfx_out_path) if postfx_done else None,
                "duration_sec": scene.get("duration_sec"),
                "render_time_sec": round(render_time, 2),
                "status": "ok",
            })

        # Save log after every scene
        with open(Path(args.output_dir) / "render_log.json", "w") as f:
            json.dump(log, f, indent=2)

    print(f"\n[batch_render_lipsync] Done. Render log: {Path(args.output_dir)/'render_log.json'}")
    lipsync_n = sum(1 for s in log["scenes"] if s.get("scene_type") == "lipsync" and s.get("status") == "ok")
    broll_n = sum(1 for s in log["scenes"] if s.get("scene_type") == "broll" and s.get("status") == "ok")
    print(f"[batch_render_lipsync]   Lip-sync rendered OK: {lipsync_n}")
    print(f"[batch_render_lipsync]   B-roll rendered OK:   {broll_n}")


if __name__ == "__main__":
    main()
