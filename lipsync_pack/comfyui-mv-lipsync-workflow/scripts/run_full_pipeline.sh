#!/usr/bin/env bash
# run_full_pipeline.sh — one-shot lip-sync MV pipeline runner.
#
# Usage:
#   1. Edit the PATH variables below
#   2. Place your Suno song at SONG_PATH
#   3. Place your lyrics at LYRICS_PATH
#   4. Place ONE singer reference image in SINGER_DIR (front-facing portrait)
#   5. Place B-roll reference images in BROLL_DIR (landscapes, objects, abstract)
#   6. Run:  bash run_full_pipeline.sh
#
# Output:  OUTPUT_DIR/music_video_final.mp4 + social variants

set -euo pipefail

# ============================================================================
# USER CONFIG
# ============================================================================
SONG_PATH="/home/you/music/suno_song.mp3"
LYRICS_PATH="/home/you/music/lyrics.txt"
SINGER_DIR="/home/you/music/singer_ref_images"
BROLL_DIR="/home/you/music/broll_ref_images"
OUTPUT_DIR="/home/you/music/output"

# ============================================================================
# DERIVED PATHS
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SEPARATED_DIR="$OUTPUT_DIR/separated"
VOCALS_PATH="$SEPARATED_DIR/vocals.wav"
BEATMAP="$OUTPUT_DIR/beatmap.json"
SHOTLIST="$OUTPUT_DIR/shotlist.json"
SCENES_DIR="$OUTPUT_DIR/scenes"
RENDER_LOG="$SCENES_DIR/render_log.json"
FINAL_MP4="$OUTPUT_DIR/music_video_final.mp4"

# LLM provider — set in env or override here
# export ZAI_API_KEY="..."

# Where ComfyUI looks for input images / vocal segments
export COMFYUI_INPUT_DIR="${COMFYUI_INPUT_DIR:-/path/to/ComfyUI/input}"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================
echo "============================================================"
echo "  COMFYUI LIP-SYNC MUSIC VIDEO PIPELINE"
echo "============================================================"
echo "Song:        $SONG_PATH"
echo "Lyrics:      $LYRICS_PATH"
echo "Singer refs: $SINGER_DIR"
echo "B-roll refs: $BROLL_DIR"
echo "Output:      $OUTPUT_DIR"
echo "ComfyUI:     $COMFYUI_INPUT_DIR (input dir)"
echo "============================================================"

mkdir -p "$OUTPUT_DIR" "$SCENES_DIR" "$SEPARATED_DIR"

[ -f "$SONG_PATH" ]   || { echo "ERROR: SONG_PATH not found"; exit 1; }
[ -f "$LYRICS_PATH" ] || { echo "ERROR: LYRICS_PATH not found"; exit 1; }
[ -d "$SINGER_DIR" ]  || { echo "ERROR: SINGER_DIR not found"; exit 1; }
[ -d "$BROLL_DIR" ]   || { echo "ERROR: BROLL_DIR not found"; exit 1; }

command -v ffmpeg    >/dev/null || { echo "ERROR: ffmpeg not installed"; exit 1; }
command -v ffprobe   >/dev/null || { echo "ERROR: ffprobe not installed"; exit 1; }
command -v python3   >/dev/null || { echo "ERROR: python3 not installed"; exit 1; }

# ============================================================================
# STAGE 0: VOCAL ISOLATION (Demucs)
# ============================================================================
echo ""
echo "[0/5] Vocal isolation (Demucs) → $VOCALS_PATH"
if [ -f "$VOCALS_PATH" ]; then
    echo "      SKIP (vocals already isolated)"
else
    python3 "$SCRIPT_DIR/vocal_isolation.py" \
        --audio "$SONG_PATH" \
        --output-dir "$SEPARATED_DIR"
fi

# ============================================================================
# STAGE 1: AUDIO ANALYSIS (with vocal-active detection)
# ============================================================================
echo ""
echo "[1/5] Audio analysis (with vocal-active flags) → $BEATMAP"
python3 "$SCRIPT_DIR/audio_analysis.py" \
    --audio "$SONG_PATH" \
    --vocals "$VOCALS_PATH" \
    --output "$BEATMAP"

# ============================================================================
# STAGE 2: LLM-DIRECTED SHOT LIST (lipsync + broll scene types)
# ============================================================================
echo ""
echo "[2/5] LLM shot list (lipsync + broll) → $SHOTLIST"
python3 "$SCRIPT_DIR/llm_shotlist_lipsync.py" \
    --beatmap "$BEATMAP" \
    --lyrics "$LYRICS_PATH" \
    --singer-dir "$SINGER_DIR" \
    --broll-dir "$BROLL_DIR" \
    --output "$SHOTLIST" \
    --send-images

# ============================================================================
# STAGE 3: BATCH RENDER (lipsync + broll + post-FX)
# ============================================================================
echo ""
echo "[3/5] Batch render scenes → $SCENES_DIR/"
python3 "$SCRIPT_DIR/batch_render_lipsync.py" \
    --shotlist "$SHOTLIST" \
    --singer-dir "$SINGER_DIR" \
    --broll-dir "$BROLL_DIR" \
    --vocals "$VOCALS_PATH" \
    --output-dir "$SCENES_DIR" \
    --post-fx

# ============================================================================
# STAGE 4: FINAL ASSEMBLY
# ============================================================================
echo ""
echo "[4/5] Final assembly → $FINAL_MP4"
python3 "$SCRIPT_DIR/assemble_mv.py" \
    --shotlist "$SHOTLIST" \
    --render-log "$RENDER_LOG" \
    --audio "$SONG_PATH" \
    --output "$FINAL_MP4" \
    --title "Music Video" \
    --artist "Unknown"

echo ""
echo "============================================================"
echo "  DONE"
echo "  Final master: $FINAL_MP4"
echo "  Social variants: $(dirname "$FINAL_MP4")/music_video_final_*.mp4"
echo "============================================================"
