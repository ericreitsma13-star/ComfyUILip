#!/usr/bin/env python3
"""log_render_metadata.py — Log render params for reproducibility."""
import json, os, subprocess, sys
from pathlib import Path

def get_git_sha(path="."):
    try:
        return subprocess.run(["git","rev-parse","HEAD"],cwd=path,capture_output=True,text=True).stdout.strip()
    except: return "unknown"

def log_metadata(scene, out_dir, checkpoint, seed, sampler, steps, cfg, loras=None, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    meta = {"scene": scene, "checkpoint": checkpoint, "seed": seed, "sampler": sampler,
            "steps": steps, "cfg": cfg, "loras": loras or [], "git_sha": get_git_sha(),
            "extra": extra or {}}
    p = Path(out_dir) / f"{scene}_metadata.json"
    p.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {p}")
    return p

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--scene", required=True); a.add_argument("--output-dir", required=True)
    a.add_argument("--checkpoint", required=True); a.add_argument("--seed", type=int, required=True)
    a.add_argument("--sampler", required=True); a.add_argument("--steps", type=int, required=True)
    a.add_argument("--cfg", type=float, required=True)
    args = a.parse_args()
    log_metadata(args.scene, args.output_dir, args.checkpoint, args.seed, args.sampler, args.steps, args.cfg)
