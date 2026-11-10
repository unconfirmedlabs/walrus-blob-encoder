#!/usr/bin/env bash
# Copyright (c) Unconfirmed Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Cross-checks blob IDs across runtimes: walrus-core (the upstream_id example), the native
# streaming CLI at budgets of 1 and 32 MiB and in memory, and the WebAssembly module under Node
# (synchronous reads, 1 MiB) and Node and Bun (asynchronous and synchronous reads, 32 MiB).
#
# usage: scripts/crosscheck.sh <files...>
set -euo pipefail
cd "$(dirname "$0")/.."
WASM=target/wasm-simd128/wasm32-unknown-unknown/wasm/walrus_blob_encoder_wasm.wasm
RUN=crates/walrus-blob-encoder-wasm/js/run.mjs
cargo build -q --release -p walrus-blob-encoder --example upstream_id
RUSTFLAGS="-C target-cpu=native" cargo build -q --release -p walrus-blob-encoder-cli \
  --target-dir target/native
RUSTFLAGS="-C target-feature=+simd128" cargo +stable build -q -p walrus-blob-encoder-wasm \
  --profile wasm --target wasm32-unknown-unknown --target-dir target/wasm-simd128
CLI=target/native/release/walrus-blob-id
id() { awk '/^blob_id/ { print $2; exit } /^[A-Za-z0-9_-]{43} / { print $1; exit }'; }
status=0
for f in "$@"; do
  up=$(target/release/examples/upstream_id "$f" | id)
  ids=(
    "native-1MiB=$($CLI --budget 1 "$f" | id)"
    "native-32MiB=$($CLI --budget 32 "$f" | id)"
    "native-memory=$($CLI --memory "$f" | id)"
    "node-sync-1MiB=$(node $RUN $WASM "$f" 1 --no-timing | id)"
    "node-async-32MiB=$(node $RUN $WASM "$f" 32 --async --no-timing | id)"
    "bun-sync-32MiB=$(bun $RUN $WASM "$f" 32 --no-timing | id)"
    "bun-async-32MiB=$(bun $RUN $WASM "$f" 32 --async --no-timing | id)"
  )
  verdict=MATCH
  for entry in "${ids[@]}"; do
    [ "${entry#*=}" = "$up" ] || { verdict=MISMATCH; status=1; }
  done
  echo "$verdict $(stat -c %s "$f") $up ${ids[*]}"
done
exit $status
