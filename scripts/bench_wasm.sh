#!/usr/bin/env bash
# Copyright (c) Unconfirmed Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# WebAssembly measurements: the simd128 module (FEATURES=relaxed for +relaxed-simd) under Node
# and/or Bun, best of REPEAT runs per configuration, every blob ID checked against walrus-core,
# one JSON line per configuration (see scripts/measure.py; the module's own counters, phase
# times and peak linear memory are under "stats").
#
# usage: [FEATURES=relaxed] [REPEAT=3] [MODE=sync|async] scripts/bench_wasm.sh \
#          <runtime>... -- <budget MiB | memory>... -- <files...>
set -euo pipefail
cd "$(dirname "$0")/.."
REPEAT=${REPEAT:-3}
MODE=${MODE:-sync}
if [ "${FEATURES:-}" = relaxed ]; then
  FLAGS="-C target-feature=+simd128,+relaxed-simd"; DIR=target/wasm-relaxed
else
  FLAGS="-C target-feature=+simd128"; DIR=target/wasm-simd128
fi
cargo build -q --release -p walrus-blob-encoder --example upstream_id
RUSTFLAGS="$FLAGS" cargo +stable build -q -p walrus-blob-encoder-wasm --profile wasm \
  --target wasm32-unknown-unknown --target-dir "$DIR"
WASM=$DIR/wasm32-unknown-unknown/wasm/walrus_blob_encoder_wasm.wasm
RUN=crates/walrus-blob-encoder-wasm/js/run.mjs
runtimes=(); budgets=()
while [ "$1" != "--" ]; do runtimes+=("$1"); shift; done; shift
while [ "$1" != "--" ]; do budgets+=("$1"); shift; done; shift
for f in "$@"; do
  expect=$(target/release/examples/upstream_id "$f" | awk '{ print $1 }')
  for rt in "${runtimes[@]}"; do
    for w in "${budgets[@]}"; do
      if [ "$w" = memory ]; then args=(64 --memory); else args=("$w"); [ "$MODE" = async ] && args+=(--async); fi
      python3 scripts/measure.py --repeat "$REPEAT" --expect "$expect" \
        --label "$rt ${FEATURES:-simd128} $MODE W=$w $(stat -c %s "$f")" -- "$rt" "$RUN" "$WASM" "$f" "${args[@]}"
    done
  done
done
