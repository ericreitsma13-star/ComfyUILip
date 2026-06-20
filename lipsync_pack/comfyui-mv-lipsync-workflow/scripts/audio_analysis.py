#!/usr/bin/env python3
"""
audio_analysis.py - Beat / section / BPM analysis for Suno-generated songs,
with vocal-energy emphasis for lip-sync scene planning.

This is an EXTENDED version of the audio_analysis.py from the B-roll pack:
  • Same beat / BPM / section / onset detection
  • PLUS: per-second vocal energy (computed from the isolated vocal track)
  • PLUS: vocal_active flag per second (true if singer is singing)
  • PLUS: vocal_section boundaries (segments where singer is active vs silent)

The vocal-active map lets the LLM in llm_shotlist_lipsync.py decide which
sections should be lip-sync shots (singer on camera) vs B-roll (instrumental).

USAGE:
    # After running vocal_isolation.py, you have vocals.wav + instrumental.wav
    python audio_analysis.py \\
        --audio suno_song.mp3 \\
        --vocals separated/vocals.wav \\
        --output beatmap.json

REQUIREMENTS:
    pip install librosa==0.10.2.post1 soundfile numpy scipy

OUTPUT SCHEMA (beatmap.json):
{
  "audio_file": "suno_song.mp3",
  "vocals_file": "separated/vocals.wav",
  "duration_sec": 215.4,
  "sample_rate": 22050,
  "bpm": 124.0,
  "bpm_confidence": 0.92,
  "time_signature": "4/4",
  "beats": [0.464, 0.928, 1.392, ...],
  "downbeats": [0.464, 2.32, 4.176, ...],
  "sections": [
    {"label": "intro",      "start": 0.0,   "end": 12.5,  "vocal_active": false},
    {"label": "verse1",     "start": 12.5,  "end": 36.8,  "vocal_active": true},
    {"label": "chorus1",    "start": 36.8,  "end": 61.0,  "vocal_active": true},
    ...
  ],
  "energy_envelope": [0.12, 0.18, 0.23, ...],
  "vocal_energy_envelope": [0.02, 0.01, 0.45, 0.78, ...],
  "vocal_active_per_sec": [false, false, true, true, ...],
  "onsets": [0.232, 0.698, 1.044, ...]
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


def analyze_audio(audio_path: str, output_path: str, vocals_path: str | None = None,
                   hop_length: int = 512):
    print(f"[audio_analysis] Loading {audio_path} ...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = float(len(y) / sr)
    print(f"[audio_analysis] Duration: {duration:.2f}s @ {sr}Hz")

    # --- BPM + beats ---
    print("[audio_analysis] Detecting tempo and beats ...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    bpm = float(np.atleast_1d(tempo)[0])

    # BPM confidence
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    ac = librosa.autocorrelate(onset_env, max_size=2 * sr // hop_length)
    ac_norm = ac / (ac.max() + 1e-9)
    period_frames = int(60 * sr / bpm / hop_length)
    window = 5
    lo, hi = max(0, period_frames - window), period_frames + window + 1
    peak = float(ac_norm[lo:hi].max())
    bpm_confidence = round(min(1.0, peak * 1.4), 3)

    # Downbeats
    time_signature = "4/4"
    downbeats = [float(beat_times[i]) for i in range(0, len(beat_times), 4)]

    # --- Onsets ---
    print("[audio_analysis] Detecting onsets ...")
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, onset_envelope=onset_env, hop_length=hop_length, units="frames"
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # --- Energy envelope ---
    print("[audio_analysis] Computing energy envelope ...")
    frame_length = sr
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=sr)[0]
    rms_norm = rms / (rms.max() + 1e-9)
    energy_envelope = [round(float(v), 4) for v in rms_norm]

    # --- VOCAL ENERGY ENVELOPE (NEW for lip-sync pack) ---
    vocal_energy_envelope = None
    vocal_active_per_sec = None
    if vocals_path and os.path.isfile(vocals_path):
        print(f"[audio_analysis] Loading isolated vocals: {vocals_path} ...")
        y_voc, sr_voc = librosa.load(vocals_path, sr=sr, mono=True)
        # Pad / trim vocal track to same length as full mix
        if len(y_voc) < len(y):
            y_voc = np.pad(y_voc, (0, len(y) - len(y_voc)))
        elif len(y_voc) > len(y):
            y_voc = y_voc[:len(y)]
        # Per-second RMS on the vocal track
        rms_voc = librosa.feature.rms(y=y_voc, frame_length=frame_length, hop_length=sr)[0]
        rms_voc_norm = rms_voc / (rms_voc.max() + 1e-9)
        vocal_energy_envelope = [round(float(v), 4) for v in rms_voc_norm]
        # A second is "vocal active" if vocal energy > 0.15 (heuristic threshold)
        vocal_active_per_sec = [bool(v > 0.15) for v in vocal_energy_envelope]
        active_secs = sum(vocal_active_per_sec)
        total_secs = len(vocal_active_per_sec)
        print(f"[audio_analysis]   Vocal active: {active_secs}/{total_secs}s "
              f"({100*active_secs/max(1,total_secs):.1f}%)")
    else:
        print("[audio_analysis] No isolated vocals provided — skipping vocal-active analysis.")
        print("                  Run vocal_isolation.py first for lip-sync-aware section planning.")

    # --- Structural segmentation ---
    print("[audio_analysis] Segmenting song structure ...")
    sections = segment_song(y, sr, beat_times, rms_norm, duration,
                             vocal_active_per_sec=vocal_active_per_sec)

    # --- Assemble ---
    beatmap = {
        "audio_file": os.path.basename(audio_path),
        "vocals_file": os.path.basename(vocals_path) if vocals_path else None,
        "duration_sec": round(duration, 3),
        "sample_rate": sr,
        "bpm": round(bpm, 2),
        "bpm_confidence": bpm_confidence,
        "time_signature": time_signature,
        "beats": [round(float(b), 3) for b in beat_times],
        "downbeats": [round(float(b), 3) for b in downbeats],
        "sections": sections,
        "energy_envelope": energy_envelope,
        "vocal_energy_envelope": vocal_energy_envelope,
        "vocal_active_per_sec": vocal_active_per_sec,
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
        va = "🎤" if s.get("vocal_active") else "  "
        print(f"     {va} {s['label']:10s}  {s['start']:6.1f} - {s['end']:6.1f}s")
    return beatmap


def segment_song(y, sr, beat_times, rms_norm, duration, vocal_active_per_sec=None):
    """Structural segmentation with vocal_active flag per section."""
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        features = np.vstack([mfcc, chroma])
        bound_frames, labels = librosa.segment.agglomerative(features, k=None)
        bound_times = librosa.frames_to_time(bound_frames, sr=sr)
    except Exception as e:
        print(f"[audio_analysis] Warning: SSM segmentation failed ({e}); falling back to fixed 8-bar segments")
        bound_times = []

    bounds = sorted(set([0.0] + list(bound_times) + [duration]))
    bounds = [b for b in bounds if 0 <= b <= duration]

    def avg_energy(start, end):
        idx = [i for i, t in enumerate(np.arange(len(rms_norm))) if start <= t < end]
        if not idx:
            return 0.0
        return float(np.mean([rms_norm[i] for i in idx]))

    def is_vocal_active(start, end):
        if not vocal_active_per_sec:
            return None
        # A section is "vocal active" if >40% of its seconds have vocal activity
        sec_indices = [i for i, t in enumerate(np.arange(len(vocal_active_per_sec))) if start <= t < end]
        if not sec_indices:
            return False
        active_count = sum(1 for i in sec_indices if vocal_active_per_sec[i])
        return active_count / len(sec_indices) > 0.4

    n = len(bounds) - 1
    sections = []
    for i in range(n):
        start, end = float(bounds[i]), float(bounds[i + 1])
        seg_energy = avg_energy(start, end)
        seg_len = end - start
        vocal_active = is_vocal_active(start, end)

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
        sections.append({
            "label": label, "start": round(start, 3), "end": round(end, 3),
            "vocal_active": vocal_active,
        })

    # Merge adjacent same-label sections
    merged = [sections[0]]
    for s in sections[1:]:
        if s["label"] == merged[-1]["label"]:
            merged[-1]["end"] = s["end"]
            # Re-evaluate vocal_active for the merged segment
            if vocal_active_per_sec:
                merged[-1]["vocal_active"] = is_vocal_active(merged[-1]["start"], merged[-1]["end"])
        else:
            merged.append(s)

    # Renumber verse/chorus sections
    counters = {}
    final = []
    for s in merged:
        if s["label"] in ("verse", "chorus"):
            counters[s["label"]] = counters.get(s["label"], 0) + 1
            s["label"] = f"{s['label']}{counters[s['label']]}"
        final.append(s)
    return final


def main():
    p = argparse.ArgumentParser(
        description="Analyze a Suno song for beat-locked MV cuts, with vocal-section detection for lip-sync planning."
    )
    p.add_argument("--audio", required=True, help="Path to the full Suno song (mp3/wav/flac).")
    p.add_argument("--vocals", help="Path to isolated vocals.wav from vocal_isolation.py (optional but recommended for lip-sync).")
    p.add_argument("--output", default="beatmap.json", help="Output JSON path.")
    p.add_argument("--hop-length", type=int, default=512, help="librosa hop length (samples).")
    args = p.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERROR: audio file not found: {args.audio}")
        sys.exit(1)

    analyze_audio(args.audio, args.output, vocals_path=args.vocals, hop_length=args.hop_length)


if __name__ == "__main__":
    main()
