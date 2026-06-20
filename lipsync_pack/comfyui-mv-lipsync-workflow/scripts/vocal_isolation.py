#!/usr/bin/env python3
"""
vocal_isolation.py - Separate vocals from a Suno-generated song using Demucs.

Suno songs are mixed (vocals + instrumental). For lip-sync, we need the ISOLATED
vocal track — otherwise the singer's mouth would move during instrumental breaks.

This script:
  1. Runs Demucs htdemucs on the input Suno song
  2. Extracts the 'vocals' stem → vocals.wav
  3. Extracts the 'no_vocals' stem (drums+bass+other mixed) → instrumental.wav
  4. (Optional) Saves all 4 stems separately for remixing

USAGE:
    python vocal_isolation.py --audio suno_song.mp3 --output-dir separated/

REQUIREMENTS:
    pip install demucs torch torchaudio

OUTPUT:
    separated/
    ├── vocals.wav          (isolated vocal track — drives lip-sync)
    ├── instrumental.wav    (instrumental only — useful for karaoke / re-mix)
    ├── drums.wav           (optional, with --save-stems)
    ├── bass.wav            (optional, with --save-stems)
    └── other.wav           (optional, with --save-stems)

NOTES:
    • Demucs htdemucs is the recommended model for Suno songs (trained on modern mixes)
    • On RTX 4090 mobile: ~30s processing for a 3-4 minute song
    • On CPU only: ~5-10 minutes (auto-fallback if CUDA unavailable)
    • Output is 44.1kHz stereo WAV (lossless)
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def check_demucs_installed():
    """Verify demucs is importable. If not, print install instructions."""
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        print("ERROR: demucs not installed. Run:")
        print("  pip install demucs")
        print("  pip install torch torchaudio  # if not already installed")
        return False


def run_demucs(audio_path: str, output_dir: str, model: str = "htdemucs",
                device: str = "cuda", shifts: int = 1, overlap: float = 0.25,
                save_stems: bool = False, verbose: bool = True):
    """
    Run Demucs via the CLI (most stable interface).
    Returns the path to the separated stems directory.
    """
    # Demucs CLI: python -m demucs --two-stems vocals -n htdemucs -o output_dir audio.mp3
    # --two-stems vocals produces just two files: vocals.wav + no_vocals.wav (fastest)
    # Without --two-stems, it produces 4 stems: drums/bass/other/vocals
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", model,
        "--device", device,
        "--shifts", str(shifts),
        "--overlap", str(overlap),
        "-o", output_dir,
    ]
    if not save_stems:
        cmd.extend(["--two-stems", "vocals"])
    cmd.append(audio_path)

    if verbose:
        print(f"[vocal_isolation] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        print(f"[vocal_isolation] Demucs failed (exit {result.returncode})")
        if result.stderr:
            print(f"[vocal_isolation] stderr: {result.stderr}")
        sys.exit(1)

    # Demucs creates: output_dir/<model_name>/<audio_basename>/{vocals.wav, no_vocals.wav}
    # or output_dir/<model_name>/<audio_basename>/{drums.wav, bass.wav, other.wav, vocals.wav}
    audio_basename = Path(audio_path).stem
    stems_dir = Path(output_dir) / model / audio_basename
    if not stems_dir.is_dir():
        print(f"[vocal_isolation] ERROR: expected stems dir not found: {stems_dir}")
        sys.exit(1)
    return stems_dir


def organize_outputs(stems_dir: Path, final_output_dir: Path, save_stems: bool):
    """Copy the separated stems to a flat output directory with clear names."""
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # vocals.wav (always present)
    vocals_src = stems_dir / "vocals.wav"
    if vocals_src.is_file():
        shutil.copy2(vocals_src, final_output_dir / "vocals.wav")
        print(f"[vocal_isolation]   vocals.wav       → {final_output_dir / 'vocals.wav'}")

    # no_vocals.wav (when --two-stems vocals was used) — this is the instrumental
    no_vocals_src = stems_dir / "no_vocals.wav"
    if no_vocals_src.is_file():
        shutil.copy2(no_vocals_src, final_output_dir / "instrumental.wav")
        print(f"[vocal_isolation]   instrumental.wav → {final_output_dir / 'instrumental.wav'}")
    else:
        # 4-stem mode: combine drums+bass+other into instrumental.wav via ffmpeg
        if save_stems:
            drums = stems_dir / "drums.wav"
            bass = stems_dir / "bass.wav"
            other = stems_dir / "other.wav"
            for stem_name, stem_file in [("drums", drums), ("bass", bass), ("other", other)]:
                if stem_file.is_file():
                    shutil.copy2(stem_file, final_output_dir / f"{stem_name}.wav")
                    print(f"[vocal_isolation]   {stem_name}.wav     → {final_output_dir / f'{stem_name}.wav'}")

            # Mix drums+bass+other into instrumental.wav
            instrumental_dst = final_output_dir / "instrumental.wav"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(drums), "-i", str(bass), "-i", str(other),
                "-filter_complex", "[0:a][1:a][2:a]amix=inputs=3:duration=longest",
                "-ac", "2", "-ar", "44100",
                str(instrumental_dst),
            ]
            subprocess.run(cmd, check=True)
            print(f"[vocal_isolation]   instrumental.wav → {instrumental_dst} (mixed from stems)")


def main():
    p = argparse.ArgumentParser(description="Isolate vocals from a Suno song using Demucs htdemucs.")
    p.add_argument("--audio", required=True, help="Path to the Suno song (mp3/wav/flac).")
    p.add_argument("--output-dir", default="separated", help="Where to write vocals.wav + instrumental.wav.")
    p.add_argument("--model", default="htdemucs",
                   help="Demucs model (htdemucs default; htdemucs_ft for higher quality but slower).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="cuda if GPU available, else cpu (much slower).")
    p.add_argument("--shifts", type=int, default=1,
                   help="Number of random shifts for better separation (1=fast, 5=best).")
    p.add_argument("--overlap", type=float, default=0.25,
                   help="Overlap between chunks (0.25 typical).")
    p.add_argument("--save-stems", action="store_true",
                   help="Save all 4 stems separately (drums/bass/other/vocals) instead of just 2.")
    p.add_argument("--force-cpu", action="store_true",
                   help="Force CPU even if CUDA is available (useful if GPU is busy with ComfyUI).")
    args = p.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERROR: audio file not found: {args.audio}")
        sys.exit(1)

    if not check_demucs_installed():
        sys.exit(1)

    # Auto-detect device if not forced
    if not args.force_cpu:
        if HAS_TORCH and torch.cuda.is_available():
            args.device = "cuda"
            print(f"[vocal_isolation] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            args.device = "cpu"
            print("[vocal_isolation] CUDA not available, falling back to CPU (much slower)")
    else:
        args.device = "cpu"
        print("[vocal_isolation] Forced CPU mode")

    print(f"[vocal_isolation] Audio: {args.audio}")
    print(f"[vocal_isolation] Model: {args.model}")
    print(f"[vocal_isolation] Device: {args.device}")
    print(f"[vocal_isolation] Output dir: {args.output_dir}")
    print(f"[vocal_isolation] Save all stems: {args.save_stems}")
    print()

    stems_dir = run_demucs(
        audio_path=args.audio,
        output_dir=args.output_dir,
        model=args.model,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
        save_stems=args.save_stems,
        verbose=True,
    )
    print(f"\n[vocal_isolation] Demucs finished. Organizing outputs ...")
    organize_outputs(stems_dir, Path(args.output_dir), save_stems=args.save_stems)

    print(f"\n[vocal_isolation] DONE.")
    print(f"[vocal_isolation]   Vocals (for lip-sync): {Path(args.output_dir) / 'vocals.wav'}")
    print(f"[vocal_isolation]   Instrumental:          {Path(args.output_dir) / 'instrumental.wav'}")


if __name__ == "__main__":
    main()
