#!/usr/bin/env python3
"""
assemble_mv.py - Final assembly for lip-sync music videos.

Same as the B-roll pack's assemble_mv.py but tested to work with the mixed
lipsync + broll scene outputs from batch_render_lipsync.py. No logic changes
were needed — the script is scene-type-agnostic; it just looks at render_log.json
to find each scene's output path and concatenates them in index order.

USAGE:
    python assemble_mv.py \\
        --shotlist shotlist.json \\
        --render-log output/scenes/render_log.json \\
        --audio suno_song.mp3 \\
        --output output/music_video_final.mp4 \\
        --title "Static" \\
        --artist "Your Name"

REQUIRES:
    - ffmpeg + ffprobe on PATH
    - Python 3.10+

OUTPUT:
    output/music_video_final.mp4      (1920x1080 H.264 CRF 16 + AAC 320kbps)
    output/music_video_final_square.mp4      (1080x1080)
    output/music_video_final_vertical.mp4    (1080x1920)
    output/music_video_final_wide_720.mp4    (1280x720)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def probe_duration(path: str) -> float:
    r = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ])
    return float(r.stdout.strip())


def make_scene_segment(scene_path: str, target_duration: float, fps: int,
                        width: int, height: int, tmp_dir: Path, index: int) -> Path:
    """Normalize a scene MP4 to uniform resolution/fps/duration."""
    out_path = tmp_dir / f"segment_{index:03d}.mp4"
    src_dur = probe_duration(scene_path)

    # For lip-sync scenes, we want to PRESERVE the original duration (don't trim/loop)
    # because the audio sync would break. If the source is within 0.5s of target, just normalize.
    if abs(src_dur - target_duration) < 0.5:
        # Just normalize format/resolution
        vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps={fps},format=yuv420p"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", scene_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "16",
            "-an",
            str(out_path),
        ]
    elif src_dur + 0.05 < target_duration:
        # Source is shorter — loop to fill (freeze last frame is safer than loop for lip-sync)
        # For B-roll: use loop. For lip-sync: extend with freeze frame (handled by -t with -shortest)
        loop_count = int(target_duration // src_dur) + 1
        vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps={fps},format=yuv420p"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-stream_loop", str(loop_count),
            "-i", scene_path,
            "-t", f"{target_duration:.3f}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "16",
            "-an",
            str(out_path),
        ]
    else:
        # Trim
        vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps={fps},format=yuv420p"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", scene_path,
            "-t", f"{target_duration:.3f}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "16",
            "-an",
            str(out_path),
        ]
    run(cmd)
    return out_path


def write_concat_list(segments: list[Path], list_path: Path):
    with open(list_path, "w") as f:
        for s in segments:
            f.write(f"file '{s.resolve()}'\n")


def render_social_variants(master: str, out_dir: Path, base_name: str):
    variants = [
        ("square",   1080, 1080, "crop=1080:1080:(iw-1080)/2:0,scale=1080:1080"),
        ("vertical", 1080, 1920, "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"),
        ("wide_720", 1280, 720,  "scale=1280:720"),
    ]
    for name, w, h, vf in variants:
        out_path = out_dir / f"{base_name}_{name}.mp4"
        if out_path.exists():
            print(f"  [social] {name} exists, skipping")
            continue
        print(f"  [social] rendering {name} ({w}x{h}) ...")
        run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", master,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_path),
        ])


def burn_subtitles(master: str, srt_path: str, out_path: str):
    srt_escaped = srt_path.replace("\\", "/").replace(":", r"\:")
    vf = f"subtitles='{srt_escaped}':force_style='Fontname=Arial,Fontsize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1'"
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", master,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        "-c:a", "copy",
        "-movflags", "+faststart",
        out_path,
    ])


def main():
    p = argparse.ArgumentParser(description="Assemble final lip-sync music video from rendered scenes + Suno audio.")
    p.add_argument("--shotlist", required=True)
    p.add_argument("--render-log", required=True)
    p.add_argument("--audio", required=True, help="ORIGINAL Suno song (NOT isolated vocals).")
    p.add_argument("--output", required=True)
    p.add_argument("--title", default="Music Video")
    p.add_argument("--artist", default="Unknown Artist")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=48)
    p.add_argument("--prefer-postfx", action="store_true", default=True)
    p.add_argument("--srt", help="Optional .srt lyrics subtitle file.")
    p.add_argument("--no-social", action="store_true")
    args = p.parse_args()

    with open(args.shotlist, encoding="utf-8") as f:
        shotlist = json.load(f)
    with open(args.render_log, encoding="utf-8") as f:
        render_log = json.load(f)

    rendered_map = {}
    for sc in render_log["scenes"]:
        idx = sc["index"]
        path = None
        if args.prefer_postfx and sc.get("postfx_path") and Path(sc["postfx_path"]).exists():
            path = sc["postfx_path"]
        elif sc.get("output_path") and Path(sc["output_path"]).exists():
            path = sc["output_path"]
        rendered_map[idx] = path

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="mv_lipsync_assemble_"))
    print(f"[assemble_mv] Working in temp dir: {tmp_dir}")

    segments = []
    for scene in shotlist["scenes"]:
        idx = scene["index"]
        target_dur = float(scene["duration_sec"])
        path = rendered_map.get(idx)
        if not path:
            print(f"[assemble_mv] WARN: scene {idx} not rendered (skipping)")
            continue
        stype = scene.get("scene_type", "broll")
        print(f"[assemble_mv] Preparing segment {idx:2d}  [{stype:7s}]  dur={target_dur:.2f}s  src={Path(path).name}")
        seg = make_scene_segment(path, target_dur, args.fps,
                                  args.width, args.height, tmp_dir, idx)
        segments.append(seg)

    if not segments:
        print("[assemble_mv] ERROR: no segments to assemble")
        sys.exit(1)

    concat_list = tmp_dir / "concat.txt"
    write_concat_list(segments, concat_list)
    concat_out = tmp_dir / "concat_master.mp4"
    print(f"[assemble_mv] Concatenating {len(segments)} segments ...")
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy",
        str(concat_out),
    ])

    print(f"[assemble_mv] Muxing audio + final encode ...")
    video_dur = probe_duration(str(concat_out))
    fade_out_start = max(0.0, min(video_dur, video_dur - 2.0))
    af = f"atrim=0:{video_dur:.3f},asetpts=PTS-STARTPTS,afade=t=out:st={fade_out_start:.3f}:d=2.0"

    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(concat_out),
        "-i", args.audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        "-metadata", f"title={args.title}",
        "-metadata", f"artist={args.artist}",
        "-metadata", f"comment=Generated via comfyui-mv-lipsync-workflow on {video_dur:.1f}s of audio",
        "-shortest",
        args.output,
    ])
    print(f"[assemble_mv] Master video: {args.output}  ({probe_duration(args.output):.2f}s)")

    if args.srt and Path(args.srt).is_file():
        sub_path = str(out_dir / f"{Path(args.output).stem}_subtitled.mp4")
        print(f"[assemble_mv] Burning subtitles from {args.srt} ...")
        burn_subtitles(args.output, args.srt, sub_path)
        print(f"[assemble_mv] Subtitled video: {sub_path}")

    if not args.no_social:
        print(f"[assemble_mv] Rendering social variants ...")
        base = Path(args.output).stem
        render_social_variants(args.output, out_dir, base)

    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception:
        pass
    print(f"[assemble_mv] Done. Master: {args.output}")


if __name__ == "__main__":
    main()
