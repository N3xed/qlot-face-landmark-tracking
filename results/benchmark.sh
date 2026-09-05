#!/usr/bin/env bash
#
# profile_sweep.sh
#
# Run `profile_model.py` over a sweep of `--num-queries` values and print
# one line per run: "<num-queries> <latency_ms>".
#
# Defaults to a log-spaced sweep from 1 to 1000 (25 samples).
#
# Tunables via env vars:
#   MODEL        path to .onnx file        (../models/model-a4s2-dyn.onnx)
#   PROVIDER     onnxruntime provider      (cuda)
#   MIN_Q        minimum num_queries       (1)
#   MAX_Q        maximum num_queries       (1000)
#   NUM_POINTS   log-spaced sample count   (25)
#   WARMUP       warmup iterations         (20)
#   ITERATIONS   measurement iterations    (2000)
#
set -euo pipefail
export LC_ALL=C

MODEL="${MODEL:-../models/model-a4s2-dyn.onnx}"
PROVIDER="${PROVIDER:-cpu}"
MIN_Q="${MIN_Q:-1}"
MAX_Q="${MAX_Q:-1000}"
NUM_POINTS="${NUM_POINTS:-50}"
WARMUP="${WARMUP:-20}"
ITERATIONS="${ITERATIONS:-1000}"

# Build the log-spaced integer sweep, rounded and de-duplicated.
mapfile -t QUERIES < <(uv run python -c "
import numpy as np
v = np.unique(
    np.round(
        np.logspace(np.log10(${MIN_Q}), np.log10(${MAX_Q}), ${NUM_POINTS})
    ).astype(int)
)
print('\n'.join(map(str, v)))
")

# Brief header on stderr so stdout stays clean for piping/redirection.
{
    echo "# model:      $MODEL"
    echo "# provider:   $PROVIDER"
    echo "# warmup:     $WARMUP"
    echo "# iterations: $ITERATIONS"
    echo "# samples:    ${#QUERIES[@]}"
} >&2

for nq in "${QUERIES[@]}"; do
    if ! output=$(
        uv run profile_model.py "$MODEL" --benchmark \
            --warmup "$WARMUP" --iterations "$ITERATIONS" \
            --num-queries "$nq" --provider "$PROVIDER" 2>&1
    ); then
        printf '%s FAILED\n' "$nq"
        continue
    fi

    # Pull the latency (ms) from the data row immediately under
    # "│ Fixed Batch Size: ..." in the latency table.
    latency=$(printf '%s\n' "$output" | awk '
        /Fixed Batch Size/ { seen = 1; next }
        seen {
            for (i = 1; i <= NF; i++) {
                if ($i == "ms" && i > 1) { print $(i - 1); exit }
            }
        }
    ')

    if [[ -n "$latency" ]]; then
        printf '%s %s\n' "$nq" "$latency"
    else
        printf '%s FAILED\n' "$nq"
    fi
done