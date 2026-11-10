#!/usr/bin/env bash
# Copyright (c) Unconfirmed Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Native measurements of walrus-blob-id: best of REPEAT runs per configuration, every blob ID
# checked against walrus-core, one JSON line per configuration (see scripts/measure.py).
#
# usage: BUILD=native|ssse3 [REPEAT=3] scripts/bench_native.sh <mode>... -- <files...>
#   modes: stream:<MiB> | memory:<threads>
#   BUILD=native compiles with -C target-cpu=native (AVX2 Reed-Solomon and BLAKE2b on AVX2
#   CPUs); BUILD=ssse3 with +ssse3,+sse4.1 and the sse41 BLAKE2b backend (128-bit everything).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD:-native}
REPEAT=${REPEAT:-3}
case $BUILD in
  native) FLAGS="-C target-cpu=native"; BACKEND=() ;;
  ssse3) FLAGS="-C target-feature=+ssse3,+sse4.1"; BACKEND=(--backend sse41) ;;
  *) echo "unknown BUILD=$BUILD" >&2; exit 2 ;;
esac
cargo build -q --release -p walrus-blob-encoder --example upstream_id
RUSTFLAGS="$FLAGS" cargo build -q --release -p walrus-blob-encoder-cli --target-dir "target/$BUILD"
CLI=target/$BUILD/release/walrus-blob-id
modes=()
while [ "$1" != "--" ]; do modes+=("$1"); shift; done
shift
for f in "$@"; do
  expect=$(target/release/examples/upstream_id "$f" | awk '{ print $1 }')
  for mode in "${modes[@]}"; do
    case $mode in
      stream:*) args=(--budget "${mode#stream:}" --stats) ;;
      memory:*) args=(--memory --threads "${mode#memory:}") ;;
      *) echo "unknown mode $mode" >&2; exit 2 ;;
    esac
    python3 scripts/measure.py --repeat "$REPEAT" --expect "$expect" \
      --label "$BUILD $mode $(stat -c %s "$f")" -- "$CLI" "${BACKEND[@]}" "${args[@]}" "$f"
  done
done
