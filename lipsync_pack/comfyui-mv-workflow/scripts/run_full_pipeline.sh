#!/usr/bin/env bash
# run_full_pipeline.sh — one-shot end-to-end pipeline runner.
#
# Usage:
#   1. Edit the 4 PATH variables below to match your setup
#   2. Place your Suno song at SONG_PATH (mp3/wav)
#   3. Place your lyrics at LYRICS_PATH (txt, UTF-8)
#   4. Place your reference images in REF_DIR (png/jpg/webp)
#   5. Run:  bash run_full_pipeline.sh
#
# Output lands in:  OUTPUT_DIR/music_video_final.mp4
#                    OUTPUT_DIR/music_video_final_square.mp4
#                    OUTPUT_DIR/music_video_final_vertical.mp4

set -euo pipefail

# ============================================================================
# USER CONFIG — edit these 4 paths
# ============================================================================
SONG_PATH="/home/you/music/suno_song.mp3"
LYRICS_PATH="/home/you/music/lyrics.txt"
REF_DIR="/home/you/music/ref_images"
OUTPUT_DIR="/home/you/music/output"

# ============================================================================
# DERIVED PATHS — usually no need to edit
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"   # parent of scripts/

BEATMAP="$OUTPUT_DIR/beatmap.json"
SHOTLIST="$OUTPUT_DIR/shotlist.json"
SCENES_DIR="$OUTPUT_DIR/scenes"
RENDER_LOG="$SCENES_DIR/render_log.json"
FINAL_MP4="$OUTPUT_DIR/music_video_final.mp4"

# LLM provider — set in env or override here
# export ZAI_API_KEY="..."     # for Z.ai / GLM
# export OPENAI_API_KEY="..."  # for OpenAI / GPT-4o

# Where ComfyUI looks for input images (so LoadImage can find them)
export COMFYUI_INPUT_DIR="${COMFYUI_INPUT_DIR:-/path/to/ComfyUI/input}"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================
echo "============================================================"
echo "  COMFYUI MUSIC VIDEO PIPELINE"
echo "============================================================"
echo "Song:     $SONG_PATH"
echo "Lyrics:   $LYRICS_PATH"
echo "Refs:     $REF_DIR"
echo "Output:   $OUTPUT_DIR"
echo "ComfyUI input: $COMFYUI_INPUT_DIR"
echo "============================================================"

mkdir -p "$OUTPUT_DIR" "$SCENES_DIR"

[ -f "$SONG_PATH" ]     || { echo "ERROR: SONG_PATH not found: $SONG_PATH"; exit 1; }
[ -f "$LYRICS_PATH" ]   || { echo "ERROR: LYRICS_PATH not found: $LYRICS_PATH"; exit 1; }
[ -d "$REF_DIR" ]       || { echo "ERROR: REF_DIR not found: $REF_DIR"; exit 1; }

command -v ffmpeg    >/dev/null || { echo "ERROR: ffmpeg not installed"; exit 1; }
command -v ffprobe   >/dev/null || { echo "ERROR: ffprobe not installed"; exit 1; }
command -v python3   >/dev/null || { echo "ERROR: python3 not installed"; exit 1; }

# ============================================================================
# STAGE 1: AUDIO ANALYSIS
# ============================================================================
echo ""
echo "[1/4] Audio analysis → $BEATMAP"
python3 "$SCRIPT_DIR/audio_analysis.py" \
    --audio "$SONG_PATH" \
    --output "$BEATMAP"

# ============================================================================
# STAGE 2: LLM-DIRECTED SHOT LIST
# ============================================================================
echo ""
echo "[2/4] LLM shot list → $SHOTLIST"
python3 "$SCRIPT_DIR/llm_shotlist.py" \
    --beatmap "$BEATMAP" \
    --lyrics "$LYRICS_PATH" \
    --ref-dir "$REF_DIR" \
    --output "$SHOTLIST" \
    --send-images

# ============================================================================
# STAGE 3: BATCH RENDER
# ============================================================================
echo ""
echo "[3/4] Batch render scenes → $SCENES_DIR/"
python3 "$SCRIPT_DIR/batch_render.py" \
    --shotlist "$SHOTLIST" \
    --ref-dir "$REF_DIR" \
    --output-dir "$SCENES_DIR" \
    --post-fx

# ============================================================================
# STAGE 4: FINAL ASSEMBLY
# ============================================================================
echo ""
echo "[4/4] Final assembly → $FINAL_MP4"
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
