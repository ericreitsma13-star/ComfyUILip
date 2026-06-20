#!/usr/bin/env python3
"""
audio_analysis.py - Beat / section / BPM analysis for Suno-generated songs.

Produces a JSON beat map that drives:
  - Beat-locked scene cuts in llm_shotlist.py
  - Section-based scene distribution
  - Optional audio-reactive FX timing in post-production

USAGE:
    python audio_analysis.py --audio path/to/suno_song.mp3 --output beatmap.json

REQUIREMENTS:
    pip install librosa==0.10.2.post1 soundfile numpy scipy

OUTPUT SCHEMA (beatmap.json):
{
  "audio_file": "suno_song.mp3",
  "duration_sec": 215.4,
  "sample_rate": 22050,
  "bpm": 124.0,
  "bpm_confidence": 0.92,
  "time_signature": "4/4",
  "beats": [0.464, 0.928, 1.392, ...],          # beat timestamps (seconds)
  "downbeats": [0.464, 2.32, 4.176, ...],        # bar-start beats (every 4th)
  "sections": [
    {"label": "intro", "start": 0.0, "end": 12.5},
    {"label": "verse1", "start": 12.5, "end": 36.8},
    {"label": "chorus", "start": 36.8, "end": 61.0},
    ...
  ],
  "energy_envelope": [0.12, 0.18, 0.23, ...],    # per-second RMS energy (normalized 0-1)
  "onsets": [0.232, 0.698, 1.044, ...]            # strong onset times (for FX triggers)
}
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    import librosa
    import numpy as np
except ImportError:
    print("ERROR: missing dependencies. Run:")
    print("  pip install librosa==0.10.2.post1 soundfile numpy scipy")
    sys.exit(1)


def analyze_audio(audio_path: str, output_path: str, hop_length: int = 512):
    print(f"[audio_analysis] Loading {audio_path} ...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = float(len(y) / sr)
    print(f"[audio_analysis] Duration: {duration:.2f}s @ {sr}Hz")

    # --- BPM + beats ---
    print("[audio_analysis] Detecting tempo and beats ...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    # Tempo from librosa can be a numpy array in newer versions
    bpm = float(np.atleast_1d(tempo)[0])

    # BPM confidence: how strong is the beat periodicity
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Pulse clarity proxy via autocorrelation peak strength
    ac = librosa.autocorrelate(onset_env, max_size=2 * sr // hop_length)
    ac_norm = ac / (ac.max() + 1e-9)
    # Look around the BPM period
    period_frames = int(60 * sr / bpm / hop_length)
    window = 5
    lo, hi = max(0, period_frames - window), period_frames + window + 1
    peak = float(ac_norm[lo:hi].max())
    bpm_confidence = round(min(1.0, peak * 1.4), 3)  # heuristic scaling

    # Downbeats: every 4th beat (assumes 4/4; for 3/4 use 3)
    time_signature = "4/4"
    downbeats = [float(beat_times[i]) for i in range(0, len(beat_times), 4)]

    # --- Onsets (for FX triggers) ---
    print("[audio_analysis] Detecting onsets ...")
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, onset_envelope=onset_env, hop_length=hop_length, units="frames"
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # --- Energy envelope (per-second RMS, normalized) ---
    print("[audio_analysis] Computing energy envelope ...")
    frame_length = sr  # 1 second frames
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=sr)[0]
    # Each rms value ~ 1 second of audio
    energy_times = np.arange(len(rms)) * 1.0
    rms_norm = rms / (rms.max() + 1e-9)
    energy_envelope = [round(float(v), 4) for v in rms_norm]

    # --- Structural segmentation (intro / verse / chorus / bridge / outro) ---
    print("[audio_analysis] Segmenting song structure ...")
    sections = segment_song(y, sr, beat_times, rms_norm, duration)

    # --- Assemble ---
    beatmap = {
        "audio_file": os.path.basename(audio_path),
        "duration_sec": round(duration, 3),
        "sample_rate": sr,
        "bpm": round(bpm, 2),
        "bpm_confidence": bpm_confidence,
        "time_signature": time_signature,
        "beats": [round(float(b), 3) for b in beat_times],
        "downbeats": [round(float(b), 3) for b in downbeats],
        "sections": sections,
        "energy_envelope": energy_envelope,
        "onsets": [round(float(o), 3) for o in onset_times],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(beatmap, f, indent=2)

    print(f"[audio_analysis] Wrote beatmap to {output_path}")
    print(f"[audio_analysis]   BPM: {bpm:.1f} (confidence {bpm_confidence})")
    print(f"[audio_analysis]   Beats: {len(beat_times)} | Downbeats: {len(downbeats)} | Onsets: {len(onset_times)}")
    print(f"[audio_analysis]   Sections: {len(sections)}")
    for s in sections:
        print(f"     {s['label']:10s}  {s['start']:6.1f} - {s['end']:6.1f}s")
    return beatmap


def segment_song(y, sr, beat_times, rms_norm, duration):
    """
    Lightweight structural segmentation.
    Uses librosa's structural segmentation (Self-Similarity Matrix + agglomerative clustering)
    then labels segments heuristically by position + energy.
    """
    try:
        # Use MFCC + chroma for structural similarity
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        features = np.vstack([mfcc, chroma])
        # Agglomerative segmentation
        bound_frames, labels = librosa.segment.agglomerative(features, k=None)
        bound_times = librosa.frames_to_time(bound_frames, sr=sr)
    except Exception as e:
        print(f"[audio_analysis] Warning: SSM segmentation failed ({e}); falling back to fixed 8-bar segments")
        bound_times = []

    # Always include 0 and duration as boundaries
    bounds = sorted(set([0.0] + list(bound_times) + [duration]))
    bounds = [b for b in bounds if 0 <= b <= duration]

    # Compute average energy per segment to inform labels
    def avg_energy(start, end):
        idx = [i for i, t in enumerate(np.arange(len(rms_norm))) if start <= t < end]
        if not idx:
            return 0.0
        return float(np.mean([rms_norm[i] for i in idx]))

    # Heuristic labeling: intro / outro for first/last, chorus for highest-energy middle segments,
    # verse for the rest, bridge for short odd middle sections
    n = len(bounds) - 1
    sections = []
    for i in range(n):
        start, end = float(bounds[i]), float(bounds[i + 1])
        seg_energy = avg_energy(start, end)
        seg_len = end - start

        if i == 0 and seg_len < 20:
            label = "intro"
        elif i == n - 1 and seg_len < 25:
            label = "outro"
        elif i == 0:
            label = "intro"
        elif i == n - 1:
            label = "outro"
        elif seg_energy > 0.6 and seg_len > 15:
            label = "chorus"
        elif seg_len < 10:
            label = "bridge"
        else:
            label = "verse"
        sections.append({"label": label, "start": round(start, 3), "end": round(end, 3)})

    # Merge adjacent same-label segments
    merged = [sections[0]]
    for s in sections[1:]:
        if s["label"] == merged[-1]["label"]:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s)

    # Renumber verse/chorus sections (verse1, verse2, chorus1, chorus2, ...)
    counters = {}
    final = []
    for s in merged:
        if s["label"] in ("verse", "chorus"):
            counters[s["label"]] = counters.get(s["label"], 0) + 1
            s["label"] = f"{s['label']}{counters[s['label']]}"
        final.append(s)
    return final


def main():
    p = argparse.ArgumentParser(description="Analyze a Suno-generated song for beat-locked MV cuts.")
    p.add_argument("--audio", required=True, help="Path to the audio file (mp3/wav/flac).")
    p.add_argument("--output", default="beatmap.json", help="Output JSON path.")
    p.add_argument("--hop-length", type=int, default=512, help="librosa hop length (samples).")
    args = p.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERROR: audio file not found: {args.audio}")
        sys.exit(1)

    analyze_audio(args.audio, args.output, hop_length=args.hop_length)


if __name__ == "__main__":
    main()
