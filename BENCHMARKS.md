# Benchmarks

Measurements use isolated AWS c6a.2xlarge instances (AMD EPYC 7R13,
8 vCPUs / 4 physical cores, 16 GiB RAM), Rust 1.94.1, release defaults, CPUs 2–5.
These CPUs are four separate physical cores. Times are seconds. The user authorized
this instance after shared-devbox variability made local comparisons unreliable.
Each accepted change is compared with a fresh baseline on the same instance.
AWS reclaimed two Spot instances; iteration 2 uses an approved On-Demand replacement.
Local exploratory results are below.
Each cell lists 0.7 / 17 / 70 / 259 MB (710603 / 17066416 / 69741071 / 259209074 bytes).
Take the minimum per configuration across two runs, each reporting best of five.
Inputs are high-entropy `/dev/urandom` bytes; all IDs are checked against the pinned walrus-core.
Sub-2% deltas are not treated as demonstrated wins.

```sh
CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=4 taskset -c 2-5 cargo run --release --locked --example bench -- --ref --backend all --threads 1,4 --repeat 5 bench/files/*
```

| Iteration | walrus-core ×1 | avx2 ×1 | avx2 ×4 | sse41 ×1 | sse41 ×4 | scalar ×1 | scalar ×4 |
|---|---|---|---|---|---|---|---|
| 0 | 0.495 / 0.524 / 0.862 / 2.053 | 0.200 / 0.233 / 0.400 / 1.041 | 0.053 / 0.062 / 0.103 / 0.280 | 0.310 / 0.339 / 0.572 / 1.525 | 0.080 / 0.088 / 0.149 / 0.394 | 0.883 / 1.288 / 2.764 / 9.364 | 0.224 / 0.326 / 0.699 / 2.373 |
| 1 | 0.494 / 0.523 / 0.861 / 2.050 | 0.199 / 0.233 / 0.398 / 1.091 | 0.053 / 0.062 / 0.104 / 0.283 | 0.308 / 0.340 / 0.580 / 1.515 | 0.080 / 0.088 / 0.150 / 0.387 | 0.882 / 1.283 / 2.773 / 9.365 | 0.225 / 0.326 / 0.704 / 2.376 |
| 2 | 0.499 / 0.525 / 0.860 / 2.099 | 0.199 / 0.233 / 0.404 / 1.045 | 0.053 / 0.061 / 0.105 / 0.282 | 0.310 / 0.338 / 0.572 / 1.500 | 0.080 / 0.088 / 0.151 / 0.393 | 0.884 / 1.284 / 2.760 / 9.371 | 0.224 / 0.325 / 0.699 / 2.371 |

Iteration 0 was measured on the first Spot instance; iteration 1 on its replacement
following an AWS interruption. Iteration 1 retains no library changes (the rebuilt
benchmark was byte-identical to the saved baseline), so cross-instance wall-time
variation is not an encoder performance delta. Iteration 1's 259 MB peak RSS is 427296 KiB (AVX2, four threads).
After the second Spot interruption, the unchanged iteration 1 code was remeasured
on the On-Demand instance; that fresh baseline is the denominator for iteration 2.

Iteration 2 also retains no encoder changes and has the same executable hash as
the baseline. Its wall-time differences are measurement variation, not accepted
speedups or slowdowns. Peak RSS: 427440 KiB (baseline binary: 427448 KiB).

## On-Demand comparison baseline

Same c6a.2xlarge instance used for iteration 2, unchanged iteration 1 code.
This is the denominator for the rejected experiments in `ITERATIONS.md`.

| Measurement | walrus-core ×1 | avx2 ×1 | avx2 ×4 | sse41 ×1 | sse41 ×4 | scalar ×1 | scalar ×4 |
|---|---|---|---|---|---|---|---|
| Before iteration 2 | 0.496 / 0.524 / 0.865 / 2.052 | 0.201 / 0.232 / 0.389 / 1.054 | 0.053 / 0.062 / 0.105 / 0.284 | 0.309 / 0.342 / 0.581 / 1.528 | 0.080 / 0.088 / 0.150 / 0.393 | 0.885 / 1.287 / 2.763 / 9.374 | 0.225 / 0.328 / 0.705 / 2.375 |

## Input SHA-256

- 710603: `0a4e79a43a56b528f85419a9026b91882ea3a4ab26a89a49467343136c6a9305`
- 17066416: `850553a71b1c9585713d2f09c64deac71d0b2a73949f6c886a723bb3e17819f4`
- 69741071: `da8a582739196267def5dc5e33a7dea972ba9f48920c6d0701420088be01c11e`
- 259209074: `814dadf5aba0baac2b3df9ea85c83c793977b412856dda264e2383b229370c0f`

## Original shared-devbox baseline

AMD EPYC 7313P, same inputs and commands. Recorded before cloud authorization;
not used as the denominator for EC2 improvements.

| Iteration | walrus-core ×1 | avx2 ×1 | avx2 ×4 | sse41 ×1 | sse41 ×4 | scalar ×1 | scalar ×4 |
|---|---|---|---|---|---|---|---|
| Local | 0.472 / 0.505 / 0.833 / 1.961 | 0.200 / 0.230 / 0.381 / 1.006 | 0.072 / 0.084 / 0.135 / 0.277 | 0.296 / 0.325 / 0.552 / 1.459 | 0.103 / 0.113 / 0.164 / 0.382 | 0.847 / 1.237 / 2.642 / 9.008 | 0.289 / 0.395 / 0.864 / 2.289 |
