#!/usr/bin/env python3
"""split_audio.py — Split a long audio file into segments for LTX rendering.

Usage:
    python split_audio.py --input Heatwave.wav --output-dir input/ --segment-sec 7.5
"""
import argparse, os, subprocess, sys
from pathlib import Path

def split_audio(input_path, output_dir, segment_sec=7.5, fps=24, overlap=0.0):
    """Split audio into segments for LTX rendering."""
    os.makedirs(output_dir, exist_ok=True)

    # Get total duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", input_path],
        capture_output=True, text=True
    )
    total = float(result.stdout.strip())
    print(f"Input: {input_path} ({total:.1f}s)")

    segments = []
    start = 0.0
    idx = 0
    while start < total:
        end = min(start + segment_sec, total)
        duration = end - start
        # Ensure frame count is valid for LTX (fc = ((dur*fps - 1) / 8) * 8 + 1)
        fc = int(((duration * fps - 1) / 8) * 8 + 1)
        actual_duration = fc / fps

        out_file = Path(output_dir) / f"segment_{idx:03d}.wav"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-t", f"{actual_duration:.3f}",
            "-i", input_path,
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out_file),
        ]
        subprocess.run(cmd, check=True)

        segments.append({
            "index": idx,
            "start_time": round(start, 3),
            "end_time": round(start + actual_duration, 3),
            "duration_sec": round(actual_duration, 3),
            "frame_count": fc,
            "audio_file": str(out_file.name),
        })
        print(f"  Segment {idx:3d}: {start:.1f}s - {start+actual_duration:.1f}s ({actual_duration:.1f}s, {fc} frames) -> {out_file.name}")

        start += segment_sec - overlap
        idx += 1

    # Save manifest
    import json
    manifest = {
        "input": str(input_path),
        "total_duration": total,
        "segment_duration": segment_sec,
        "fps": fps,
        "segments": segments,
    }
    manifest_path = Path(output_dir) / "segments.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")
    print(f"Total segments: {len(segments)}")
    return segments

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Split audio for LTX rendering")
    p.add_argument("--input", required=True, help="Input audio file")
    p.add_argument("--output-dir", default="input/", help="Output directory for segments")
    p.add_argument("--segment-sec", type=float, default=7.5, help="Segment duration in seconds")
    p.add_argument("--fps", type=int, default=24, help="Video FPS")
    p.add_argument("--overlap", type=float, default=0.0, help="Overlap between segments in seconds")
    args = p.parse_args()
    split_audio(args.input, args.output_dir, args.segment_sec, args.fps, args.overlap)
