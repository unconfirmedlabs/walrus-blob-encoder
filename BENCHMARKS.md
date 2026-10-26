# Benchmarks

The acceptance baseline uses an isolated AWS c6a.2xlarge Spot instance (AMD EPYC 7R13,
8 vCPUs / 4 physical cores, 16 GiB RAM), Rust 1.94.1, release defaults, CPUs 2–5.
These CPUs are four separate physical cores. Times are seconds. The user authorized
this instance after shared-devbox variability made local comparisons unreliable.
All iteration comparisons use this same instance; local exploratory results are below.
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

Iteration 0 was measured on the first Spot instance; iteration 1 on its replacement
following an AWS interruption. Iteration 1 retains no library changes (the rebuilt
benchmark was byte-identical to the saved baseline), so cross-instance wall-time
variation is not an encoder performance delta. Iteration 1 is the fresh denominator
for subsequent improvements. Its 259 MB peak RSS is 427296 KiB (AVX2, four threads).

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
