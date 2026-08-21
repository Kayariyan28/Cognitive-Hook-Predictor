#!/usr/bin/env bash
# Synthesize a deterministic 14-second test clip, so the pipeline can be
# exercised end to end without supplying real creator media.
#
# The audio is deliberately silent for the first 1.4 seconds, which is past the
# hook readout's 0.8-second opening-silence convention. That makes the flagged
# check reproducible, and makes the recut assistant's effect measurable.
set -euo pipefail

OUT="${1:-./demo-clip.mp4}"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required"; exit 1; }

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc2=size=540x960:rate=25:duration=14" \
  -f lavfi -i "sine=frequency=330:duration=14" \
  -filter_complex "[1:a]volume='if(lt(t,1.4),0,if(lt(t,1.6),1.0,0.35))':eval=frame[a]" \
  -map 0:v -map "[a]" \
  -c:v libx264 -preset veryfast -crf 26 -pix_fmt yuv420p \
  -c:a aac -b:a 96k -movflags +faststart \
  "$OUT"

echo "Wrote $OUT"
echo "Expected: the hook readout flags ~1.44 s of silence before the first measured sound peak."
