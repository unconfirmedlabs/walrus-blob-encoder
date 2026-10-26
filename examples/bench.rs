// Copyright (c) Unconfirmed Labs, LLC
// SPDX-License-Identifier: Apache-2.0

//! Times blob-ID computation per backend and thread count, optionally against walrus-core.
//!
//! ```text
//! cargo run --release --example bench -- [--ref] [--backend all|scalar|avx2|avx512|neon]
//!     [--threads 1,4] [--repeat N] <files...>
//! ```

use std::{
    num::NonZeroU16,
    process::exit,
    sync::Arc,
    time::{Duration, Instant},
};

use rayon::ThreadPoolBuilder;
use walrus_blob_encoder::{Backend, BlobEncoder, BlobId, MAINNET_N_SHARDS};
use walrus_core::{
    DEFAULT_ENCODING,
    encoding::{EncodingConfig, EncodingFactory as _},
};

const USAGE: &str = "usage: bench [--ref] [--backend all|scalar|avx2|avx512|neon] \
                     [--threads 1,4] [--repeat N] <files...>";

struct Options {
    reference: bool,
    backends: Vec<Backend>,
    threads: Vec<usize>,
    repeat: usize,
    files: Vec<String>,
}

fn parse_options() -> Result<Options, String> {
    let mut options = Options {
        reference: false,
        backends: vec![Backend::detect()],
        threads: vec![rayon::current_num_threads()],
        repeat: 3,
        files: Vec::new(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--ref" => options.reference = true,
            "--backend" => {
                options.backends = match value()?.as_str() {
                    "all" => Backend::ALL
                        .iter()
                        .copied()
                        .filter(|backend| backend.is_available())
                        .collect(),
                    name => vec![
                        *Backend::ALL
                            .iter()
                            .find(|backend| backend.name() == name)
                            .ok_or(format!("unknown backend {name}"))?,
                    ],
                }
            }
            "--threads" => {
                options.threads = value()?
                    .split(',')
                    .map(|n| n.parse().map_err(|_| format!("invalid thread count {n}")))
                    .collect::<Result<_, _>>()?
            }
            "--repeat" => {
                options.repeat = value()?
                    .parse()
                    .map_err(|_| "invalid repeat count".to_string())?
            }
            file => options.files.push(file.to_string()),
        }
    }
    if options.files.is_empty() || options.repeat == 0 {
        return Err(USAGE.to_string());
    }
    Ok(options)
}

/// Best wall time of `repeat` runs, and the last result.
fn best_of<T>(repeat: usize, mut run: impl FnMut() -> T) -> (Duration, T) {
    let mut best = Duration::MAX;
    let mut result = None;
    for _ in 0..repeat {
        let start = Instant::now();
        result = Some(run());
        best = best.min(start.elapsed());
    }
    (best, result.expect("repeat is positive"))
}

fn main() {
    let options = parse_options().unwrap_or_else(|error| {
        eprintln!("{error}");
        exit(2);
    });
    println!(
        "| file | MB | implementation | threads | best of {} (s) | speedup |",
        options.repeat
    );
    println!("|---|---|---|---|---|---|");
    for file in &options.files {
        let blob = std::fs::read(file).unwrap_or_else(|error| {
            eprintln!("{file}: {error}");
            exit(1);
        });
        let name = file.rsplit('/').next().unwrap_or(file);
        let mb = blob.len() as f64 / 1e6;
        let reference = options.reference.then(|| {
            let config = EncodingConfig::new(NonZeroU16::from(MAINNET_N_SHARDS))
                .get_for_type(DEFAULT_ENCODING);
            let (time, id) = best_of(options.repeat, || config.compute_blob_id(&blob).unwrap());
            println!(
                "| {name} | {mb:.1} | walrus-core | 1 | {:.3} | 1.00x |",
                time.as_secs_f64()
            );
            (time, BlobId::from(id.0))
        });
        for &backend in &options.backends {
            let encoder = BlobEncoder::mainnet()
                .with_backend(backend)
                .unwrap_or_else(|error| {
                    eprintln!("{error}");
                    exit(1);
                });
            for &threads in &options.threads {
                let pool = ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let encoder = encoder.clone().with_thread_pool(Arc::new(pool));
                let (time, id) = best_of(options.repeat, || encoder.blob_id(&blob).unwrap());
                let speedup = reference.map_or("-".to_string(), |(reference_time, expected)| {
                    assert_eq!(id, expected, "{name}: blob ID differs from walrus-core");
                    format!("{:.2}x", reference_time.as_secs_f64() / time.as_secs_f64())
                });
                println!(
                    "| {name} | {mb:.1} | {backend} | {threads} | {:.3} | {speedup} |",
                    time.as_secs_f64()
                );
            }
        }
    }
}
