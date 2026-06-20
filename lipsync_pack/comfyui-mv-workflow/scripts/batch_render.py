#!/usr/bin/env python3
"""
batch_render.py - Orchestrate ComfyUI API calls to render every scene in a shot list.

Reads:
  - shotlist.json   (from llm_shotlist.py)
  - reference_images/  (where each ref_image filename lives)

For each scene:
  1. Loads the matching workflow template (LTX or Wan) as a ComfyUI API-format JSON
  2. Patches it with scene-specific values (ref image, prompt, motion, seed, output prefix)
  3. POSTs to ComfyUI /prompt endpoint
  4. Polls /history until the job completes
  5. Records the output MP4 path per scene

USAGE:
    python batch_render.py \
        --shotlist shotlist.json \
        --ref-dir reference_images/ \
        --comfyui-url http://127.0.0.1:8188 \
        --workflow-ltx workflows/01_scene_render_LTX.json \
        --workflow-wan workflows/02_scene_render_Wan.json \
        --output-dir output/scenes/ \
        --post-fx                            # chain post-FX workflow after each render
        --workflow-postfx workflows/03_post_fx_pipeline.json

NOTES:
    - ComfyUI must be running with --enable-cors-header and the workflows must already be saved as API-format JSON (via "Save (API Format)" menu in ComfyUI). The provided workflow JSON files in this pack are UI-format; you must convert them once via the ComfyUI UI (right-click → "Save (API Format)") before this script can use them.
    - For headless operation, see README "API mode" section.

OUTPUT SCHEMA (render_log.json):
{
  "rendered_at": "2026-06-17T22:13:00Z",
  "total_scenes": 12,
  "scenes": [
    {
      "index": 0,
      "model": "ltx",
      "output_path": "output/scenes/scene_000_ltx.mp4",
      "postfx_path": "output/scenes/scene_000_final.mp4",
      "duration_sec": 12.5,
      "render_time_sec": 38.2,
      "status": "ok"
    },
    ...
  ]
}
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: missing 'requests'. Run: pip install requests")
    sys.exit(1)


# -------- ComfyUI API helpers --------

def comfyui_queue_prompt(server_url: str, workflow_api: dict, client_id: str = "batch_render") -> dict:
    """POST a workflow (API-format JSON) to /prompt. Returns the response containing prompt_id."""
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
    """Block until the ComfyUI job finishes (or timeout). Returns the history entry."""
    start = time.time()
    while True:
        history = comfyui_get_history(server_url, prompt_id)
        if history is not None:
            return history
        if time.time() - start > timeout:
            raise TimeoutError(f"ComfyUI job {prompt_id} did not complete within {timeout}s")
        time.sleep(poll_interval)


def comfyui_find_output_file(history: dict) -> str | None:
    """Find the first output filename in a completed job's history entry."""
    outputs = history.get("outputs", {})
    for node_id, node_out in outputs.items():
        # VHS_VideoCombine stores under "gifs" or "images"
        for key in ("gifs", "images", "video"):
            if key in node_out:
                for item in node_out[key]:
                    fn = item.get("filename") or item.get("file_name")
                    subfolder = item.get("subfolder", "")
                    if fn:
                        return os.path.join(subfolder, fn) if subfolder else fn
    return None


def comfyui_download_output(server_url: str, remote_path: str, local_path: str):
    """Download /view?filename=...&type=output to local_path."""
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


# -------- Workflow patching --------

def patch_workflow_for_scene(workflow_api: dict, scene: dict, ref_image_path: str,
                              output_prefix: str, seed: int) -> dict:
    """
    Patch an API-format workflow JSON with scene-specific values.

    This implementation uses heuristics that work with the workflows shipped in this pack:
      - Find the LoadImage node and set its `image` widget to the ref image filename
      - Find CLIPTextEncode nodes (in order: positive, negative) and set their text
      - Find the I2V node (LTXImageToVideo / WanImageToVideo) and tweak motion widget
      - Find VHS_VideoCombine and set `filename_prefix`
      - Set all `seed` widgets to the provided seed

    For robustness, this tries multiple common node-class names. If your workflow uses
    custom nodes with different class names, you may need to extend the lookups below.
    """
    import copy
    wf = copy.deepcopy(workflow_api)

    # Build maps of node id -> node dict by class_type
    nodes_by_type: dict[str, list[str]] = {}
    for node_id, node in wf.items():
        ct = node.get("class_type", "")
        nodes_by_type.setdefault(ct, []).append(node_id)

    # --- 1. LoadImage: set ref image filename ---
    ref_filename = Path(ref_image_path).name
    load_image_types = ("LoadImage", "ETN_LoadImageBase64", "LoadImageBatch")
    found_loader = False
    for ct in load_image_types:
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["image"] = ref_filename
                found_loader = True
    if not found_loader:
        print(f"  [WARN] No LoadImage node found in workflow; ref image won't be set automatically")

    # --- 2. CLIPTextEncode: positive first, negative second ---
    text_node_ids = nodes_by_type.get("CLIPTextEncode", [])
    if len(text_node_ids) >= 2:
        # Positive
        wf[text_node_ids[0]]["inputs"]["text"] = scene.get("prompt", "")
        # Negative
        wf[text_node_ids[1]]["inputs"]["text"] = scene.get("negative_prompt", "")
    elif len(text_node_ids) == 1:
        wf[text_node_ids[0]]["inputs"]["text"] = scene.get("prompt", "")

    # --- 3. I2V node: motion strength + seed ---
    i2v_types = ("LTXImageToVideo", "WanImageToVideo", "WanVideoImageToVideo", "LTXVImageToVideo")
    motion = float(scene.get("motion_strength", 0.5))
    for ct in i2v_types:
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                # Set seed (last numeric widget typically)
                wf[nid]["inputs"]["seed"] = int(seed)
                # Motion strength: LTX uses context_schedule, Wan uses motion_strength
                if ct.startswith("LTX"):
                    # Map 0-1 motion → 0.3-1.0 context_schedule (heuristic)
                    wf[nid]["inputs"]["context_schedule"] = round(0.3 + motion * 0.7, 3)
                elif ct.startswith("Wan"):
                    wf[nid]["inputs"]["motion_strength"] = motion

    # --- 4. VHS_VideoCombine: filename_prefix ---
    vhs_types = ("VHS_VideoCombine", "VHS_VideoCombineMux", "SaveAnimatedWEBP")
    for ct in vhs_types:
        if ct in nodes_by_type:
            for nid in nodes_by_type[ct]:
                wf[nid]["inputs"]["filename_prefix"] = output_prefix

    return wf


# -------- Main --------

def main():
    p = argparse.ArgumentParser(description="Batch-render every scene in a shot list via ComfyUI API.")
    p.add_argument("--shotlist", required=True, help="shotlist.json from llm_shotlist.py")
    p.add_argument("--ref-dir", required=True, help="Directory containing reference images.")
    p.add_argument("--comfyui-url", default="http://127.0.0.1:8188")
    p.add_argument("--workflow-ltx", default="workflows/01_scene_render_LTX_api.json",
                   help="LTX workflow in API-format JSON. Convert the UI-format file via ComfyUI 'Save (API Format)' once.")
    p.add_argument("--workflow-wan", default="workflows/02_scene_render_Wan_api.json")
    p.add_argument("--workflow-postfx", default="workflows/03_post_fx_pipeline_api.json")
    p.add_argument("--output-dir", default="output/scenes", help="Where to save rendered scene MP4s.")
    p.add_argument("--post-fx", action="store_true",
                   help="After each scene renders, run the post-FX workflow on it and save the result.")
    p.add_argument("--only-index", type=int, default=None,
                   help="Render only the scene with this index (useful for retries).")
    p.add_argument("--only-model", choices=("ltx", "wan"), default=None,
                   help="Override model for all scenes (ignore shotlist's per-scene choice).")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Add to each scene's seed. Default 0 = use scene index as seed.")
    p.add_argument("--poll-interval", type=float, default=3.0)
    p.add_argument("--per-job-timeout", type=float, default=1800.0)
    args = p.parse_args()

    # Load shot list
    with open(args.shotlist, encoding="utf-8") as f:
        shotlist = json.load(f)
    scenes = shotlist["scenes"]
    print(f"[batch_render] Loaded {len(scenes)} scenes from {args.shotlist}")

    # Load workflow templates (each as a dict of node_id -> node)
    wf_ltx = json.load(open(args.workflow_ltx, encoding="utf-8"))
    wf_wan = json.load(open(args.workflow_wan, encoding="utf-8"))
    wf_post = json.load(open(args.workflow_postfx, encoding="utf-8")) if args.post_fx and Path(args.workflow_postfx).exists() else None

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # We also need ComfyUI to "see" the reference images — they must be in ComfyUI's input folder.
    # We try to copy them into the default ComfyUI/input/ folder via filesystem (if local).
    # If ComfyUI is remote, use the /upload/image endpoint instead (not implemented here for brevity).
    comfyui_input_dir = os.environ.get("COMFYUI_INPUT_DIR", "")
    if comfyui_input_dir and Path(comfyui_input_dir).is_dir():
        print(f"[batch_render] Syncing reference images to ComfyUI input dir: {comfyui_input_dir}")
        for scene in scenes:
            src = Path(args.ref_dir) / scene["ref_image"]
            if src.is_file():
                dst = Path(comfyui_input_dir) / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
    else:
        print("[batch_render] WARN: COMFYUI_INPUT_DIR env var not set or not a directory.")
        print("                If ComfyUI runs locally, set it to e.g. /path/to/ComfyUI/input so reference")
        print("                images can be loaded by LoadImage. Otherwise, manually copy ref images there.")

    # Render each scene
    log = {
        "rendered_at": datetime.now(timezone.utc).isoformat(),
        "total_scenes": len(scenes),
        "scenes": [],
    }
    for scene in scenes:
        idx = scene["index"]
        if args.only_index is not None and idx != args.only_index:
            continue
        model = args.only_model or scene.get("model", "wan")
        wf_template = wf_ltx if model == "ltx" else wf_wan
        seed = idx * 1000 + args.seed_offset

        prefix = f"scene_{idx:03d}_{model}"
        out_path = Path(args.output_dir) / f"{prefix}.mp4"
        postfx_out_path = Path(args.output_dir) / f"scene_{idx:03d}_final.mp4"

        if out_path.exists():
            print(f"[batch_render] Scene {idx:2d}  model={model:3s}  SKIP (output exists: {out_path.name})")
            log["scenes"].append({
                "index": idx, "model": model,
                "output_path": str(out_path),
                "postfx_path": str(postfx_out_path) if args.post_fx and postfx_out_path.exists() else None,
                "duration_sec": scene.get("duration_sec"),
                "render_time_sec": 0.0,
                "status": "skipped_existing",
            })
            continue

        print(f"[batch_render] Scene {idx:2d}  model={model:3s}  seed={seed}  motion={scene.get('motion_strength', 0.5):.2f}")
        print(f"               ref={scene['ref_image']}  prompt={scene.get('prompt', '')[:80]}...")

        # Patch workflow
        wf_patched = patch_workflow_for_scene(
            wf_template, scene,
            ref_image_path=str(Path(args.ref_dir) / scene["ref_image"]),
            output_prefix=prefix,
            seed=seed,
        )

        # Queue + wait
        t0 = time.time()
        try:
            resp = comfyui_queue_prompt(args.comfyui_url, wf_patched)
            prompt_id = resp["prompt_id"]
            print(f"               queued as {prompt_id}; waiting ...")
            history = comfyui_wait_for_job(args.comfyui_url, prompt_id,
                                            poll_interval=args.poll_interval,
                                            timeout=args.per_job_timeout)
            remote_file = comfyui_find_output_file(history)
            if not remote_file:
                raise RuntimeError(f"No output file found in history for prompt {prompt_id}")
            comfyui_download_output(args.comfyui_url, remote_file, str(out_path))
            render_time = time.time() - t0
            print(f"               DONE in {render_time:.1f}s → {out_path.name}")
        except Exception as e:
            render_time = time.time() - t0
            print(f"               FAILED after {render_time:.1f}s: {e}")
            log["scenes"].append({
                "index": idx, "model": model,
                "output_path": None,
                "postfx_path": None,
                "duration_sec": scene.get("duration_sec"),
                "render_time_sec": render_time,
                "status": f"error: {e}",
            })
            continue

        postfx_done = False
        if args.post_fx and wf_post is not None:
            print(f"               running post-FX ...")
            try:
                # The post-FX workflow starts from VHS_LoadVideo — we need to set its 'video' input to the scene MP4.
                # The scene MP4 must also be in ComfyUI's input folder.
                if comfyui_input_dir and Path(comfyui_input_dir).is_dir():
                    shutil.copy2(out_path, Path(comfyui_input_dir) / out_path.name)
                # Patch post-FX workflow
                import copy
                wf_post_patched = copy.deepcopy(wf_post)
                for nid, node in wf_post_patched.items():
                    if node.get("class_type") in ("VHS_LoadVideo", "VHS_LoadVideoFFmpeg"):
                        node["inputs"]["video"] = out_path.name
                    if node.get("class_type") in ("VHS_VideoCombine",):
                        node["inputs"]["filename_prefix"] = f"scene_{idx:03d}_final"
                resp = comfyui_queue_prompt(args.comfyui_url, wf_post_patched)
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
            "index": idx, "model": model,
            "output_path": str(out_path),
            "postfx_path": str(postfx_out_path) if postfx_done else None,
            "duration_sec": scene.get("duration_sec"),
            "render_time_sec": round(render_time, 2),
            "status": "ok",
        })

        # Save the running log after every scene (so a crash doesn't lose progress)
        with open(Path(args.output_dir) / "render_log.json", "w") as f:
            json.dump(log, f, indent=2)

    print(f"[batch_render] Done. Render log: {Path(args.output_dir)/'render_log.json'}")


if __name__ == "__main__":
    main()
