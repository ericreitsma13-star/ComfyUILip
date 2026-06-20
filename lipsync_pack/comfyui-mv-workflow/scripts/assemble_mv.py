#!/usr/bin/env python3
"""
assemble_mv.py - Final assembly: concat scenes + mux Suno audio + re-encode for delivery.

Reads:
  - shotlist.json     (for scene order + per-scene durations)
  - render_log.json   (from batch_render.py, for actual rendered scene paths)
  - the original Suno audio file

Produces:
  - a final MP4 with H.264 video + AAC audio
  - optional social-media variants (1080p square, 1080x1920 vertical, 1080p wide)

USAGE:
    python assemble_mv.py \
        --shotlist shotlist.json \
        --render-log output/scenes/render_log.json \
        --audio suno_song.mp3 \
        --output output/music_video_final.mp4 \
        --title "Song Title" \
        --artist "Artist Name"

REQUIRES:
    - ffmpeg + ffprobe on PATH
    - Python 3.10+

OUTPUT FILES (in --output parent directory):
    music_video_final.mp4      (1920x1080 H.264 CRF 16 + AAC 320kbps)
    music_video_1080p_wide.mp4 (1920x1080 — same as final)
    music_video_square.mp4     (1080x1080 — Instagram feed)
    music_video_vertical.mp4   (1080x1920 — TikTok / Reels / Shorts)
    music_video_subtitled.mp4  (optional — burns in lyrics .srt if --srt provided)

BEHAVIOR:
    1. Each scene MP4 is trimmed/padded to match the shotlist's duration_sec (within 0.1s tolerance)
    2. Scenes are concatenated via ffmpeg concat demuxer (no re-encode if all same fps/codec)
    3. Suno audio is muxed in, trimmed to video duration, fade-out 2s
    4. Final re-encode to H.264 CRF 16 (visually lossless) + AAC 320kbps stereo
    5. Title + artist metadata embedded
    6. Social variants rendered via ffmpeg scale+crop filters
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command, printing it for debugging."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def probe_duration(path: str) -> float:
    """Use ffprobe to get file duration in seconds."""
    r = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ])
    return float(r.stdout.strip())


def make_scene_segment(scene_path: str, target_duration: float, fps: int,
                        width: int, height: int, tmp_dir: Path, index: int) -> Path:
    """
    Prepare a single scene as a normalized MP4 with exact target duration, fps, resolution.
    Uses ffmpeg to:
      - Trim or loop/extend to target_duration (loop if shorter, trim if longer)
      - Set fixed fps, 1920x1080, yuv420p, CRF 16 (visually lossless intermediate)
      - No audio (we mux audio at the end)
    Returns the path to the normalized segment.
    """
    out_path = tmp_dir / f"segment_{index:03d}.mp4"
    # Use loop filter if source is shorter than target, else trim
    src_dur = probe_duration(scene_path)

    if src_dur + 0.05 < target_duration:
        # Loop to fill
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
    """Render square + vertical variants from the master wide MP4."""
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
    """Burn an SRT subtitle file into the video (lyrics)."""
    # Use ffmpeg's subtitles filter; need to escape colons in path on Windows
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
    p = argparse.ArgumentParser(description="Assemble final music video from rendered scenes + Suno audio.")
    p.add_argument("--shotlist", required=True, help="shotlist.json")
    p.add_argument("--render-log", required=True, help="render_log.json (from batch_render.py)")
    p.add_argument("--audio", required=True, help="Suno audio file (mp3/wav).")
    p.add_argument("--output", required=True, help="Final master MP4 path.")
    p.add_argument("--title", default="Music Video")
    p.add_argument("--artist", default="Unknown Artist")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=48)
    p.add_argument("--prefer-postfx", action="store_true", default=True,
                   help="Use postfx_path when available, else fall back to raw output_path.")
    p.add_argument("--srt", help="Optional .srt lyrics subtitle file to burn in.")
    p.add_argument("--no-social", action="store_true", help="Skip rendering square/vertical variants.")
    args = p.parse_args()

    # Load metadata
    with open(args.shotlist, encoding="utf-8") as f:
        shotlist = json.load(f)
    with open(args.render_log, encoding="utf-8") as f:
        render_log = json.load(f)

    # Map index → rendered file path
    rendered_map = {}
    for sc in render_log["scenes"]:
        idx = sc["index"]
        path = None
        if args.prefer_postfx and sc.get("postfx_path") and Path(sc["postfx_path"]).exists():
            path = sc["postfx_path"]
        elif sc.get("output_path") and Path(sc["output_path"]).exists():
            path = sc["output_path"]
        rendered_map[idx] = path

    # Build scene segments
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="mv_assemble_"))
    print(f"[assemble_mv] Working in temp dir: {tmp_dir}")

    segments = []
    for scene in shotlist["scenes"]:
        idx = scene["index"]
        target_dur = float(scene["duration_sec"])
        path = rendered_map.get(idx)
        if not path:
            print(f"[assemble_mv] WARN: scene {idx} not rendered (will skip)")
            continue
        print(f"[assemble_mv] Preparing segment {idx:2d}  dur={target_dur:.2f}s  src={Path(path).name}")
        seg = make_scene_segment(path, target_dur, args.fps,
                                  args.width, args.height, tmp_dir, idx)
        segments.append(seg)

    if not segments:
        print("[assemble_mv] ERROR: no segments to assemble")
        sys.exit(1)

    # Concat segments
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

    # Mux audio + final encode + metadata
    print(f"[assemble_mv] Muxing audio + final encode ...")
    video_dur = probe_duration(str(concat_out))
    audio_dur = probe_duration(args.audio)
    fade_out_start = max(0.0, min(video_dur, video_dur - 2.0))

    # Build audio filter: trim to video duration, 2s fade-out at the end
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
        "-metadata", f"comment=Generated via comfyui-mv-workflow on {video_dur:.1f}s of audio",
        "-shortest",
        args.output,
    ])
    print(f"[assemble_mv] Master video: {args.output}  ({probe_duration(args.output):.2f}s)")

    # Optional: burn subtitles
    if args.srt and Path(args.srt).is_file():
        sub_path = str(out_dir / f"{Path(args.output).stem}_subtitled.mp4")
        print(f"[assemble_mv] Burning subtitles from {args.srt} ...")
        burn_subtitles(args.output, args.srt, sub_path)
        print(f"[assemble_mv] Subtitled video: {sub_path}")

    # Render social variants
    if not args.no_social:
        print(f"[assemble_mv] Rendering social variants ...")
        base = Path(args.output).stem
        render_social_variants(args.output, out_dir, base)

    # Cleanup
    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception:
        pass
    print(f"[assemble_mv] Done. Master: {args.output}")


if __name__ == "__main__":
    main()
