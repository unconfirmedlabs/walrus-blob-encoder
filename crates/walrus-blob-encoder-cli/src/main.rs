// Copyright (c) Unconfirmed Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Prints the Walrus blob ID of files, streamed with a memory budget or read into memory.
//!
//! ```text
//! walrus-blob-id [--budget MIB | --memory] [--threads N] [--shards N] [--backend NAME]
//!     [--stats] <files...>
//! ```
//!
//! `--budget` (default 64 MiB) bounds the column batches of the streaming encoder, which reads
//! the file with positioned reads; `--memory` reads the whole file and uses the in-memory
//! encoder on `--threads` rayon threads (default: rayon's). `--stats` adds a JSON line per file
//! with the counters and phase times.

use std::{
    fs::File,
    num::NonZeroU16,
    process::exit,
    sync::Arc,
    time::{Duration, Instant},
};

use walrus_blob_encoder::{Backend, BlobEncoder, Source, StreamStats};

const USAGE: &str = "usage: walrus-blob-id [--budget MIB | --memory] [--threads N] [--shards N] \
                     [--backend NAME] [--stats] <files...>";

struct Options {
    budget: Option<f64>,
    threads: Option<usize>,
    shards: u16,
    backend: Option<Backend>,
    stats: bool,
    sequential: bool,
    parallel: bool,
    files: Vec<String>,
}

fn parse() -> Result<Options, String> {
    let mut options = Options {
        budget: Some(64.0),
        threads: None,
        shards: 1000,
        backend: None,
        stats: false,
        sequential: false,
        parallel: false,
        files: Vec::new(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--budget" => {
                let mib = value()?.parse().map_err(|_| "invalid budget".to_string())?;
                options.budget = Some(mib);
            }
            "--memory" => options.budget = None,
            "--sequential" => {
                options.budget = None;
                options.sequential = true;
            }
            "--parallel" => {
                options.budget = None;
                options.parallel = true;
            }
            "--threads" => {
                let threads = value()?.parse().map_err(|_| "invalid thread count")?;
                options.threads = Some(threads);
            }
            "--shards" => {
                options.shards = value()?.parse().map_err(|_| "invalid shard count")?;
            }
            "--backend" => {
                let backend = value()?.parse().map_err(|e| format!("{e}"))?;
                options.backend = Some(backend);
            }
            "--stats" => options.stats = true,
            "--help" | "-h" => return Err(USAGE.to_string()),
            file => options.files.push(file.to_string()),
        }
    }
    if options.files.is_empty() {
        return Err(USAGE.to_string());
    }
    Ok(options)
}

/// A file read with positioned reads, timing them.
struct FileSource {
    file: File,
    len: u64,
    io: Duration,
}

impl Source for FileSource {
    type Error = std::io::Error;

    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        let start = Instant::now();
        #[cfg(unix)]
        let result = std::os::unix::fs::FileExt::read_exact_at(&self.file, buf, offset);
        #[cfg(not(unix))]
        let result = {
            use std::io::{Read, Seek, SeekFrom};
            self.file
                .seek(SeekFrom::Start(offset))
                .and_then(|_| self.file.read_exact(buf))
        };
        self.io += start.elapsed();
        result
    }
}

fn clock() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e3
}

fn run(encoder: &BlobEncoder, options: &Options, path: &str) -> Result<(), String> {
    let error = |e: &dyn std::fmt::Display| format!("{path}: {e}");
    let start = Instant::now();
    let (metadata, stats, io) = match options.budget {
        Some(mib) => {
            let file = File::open(path).map_err(|e| error(&e))?;
            let len = file.metadata().map_err(|e| error(&e))?.len();
            let mut source = FileSource {
                file,
                len,
                io: Duration::ZERO,
            };
            let budget = (mib * 1_048_576.0) as usize;
            let mut stream = encoder.stream(len, budget).map_err(|e| error(&e))?;
            if options.stats {
                stream = stream.with_clock(clock);
            }
            while let Some(read) = stream.next_read() {
                source
                    .read_at(read.offset, stream.dest())
                    .map_err(|e| error(&e))?;
                stream.complete_read();
            }
            let stats = stream.stats();
            (stream.finish(), Some(stats), source.io)
        }
        None => {
            let read = Instant::now();
            let blob = std::fs::read(path).map_err(|e| error(&e))?;
            let io = read.elapsed();
            let metadata = if options.sequential {
                encoder.metadata_sequential(&blob)
            } else if options.parallel {
                encoder.metadata_parallel(&blob)
            } else {
                encoder.metadata(&blob)
            };
            (metadata.map_err(|e| error(&e))?, None, io)
        }
    };
    let wall = start.elapsed();
    println!("{} {path}", metadata.blob_id());
    if options.stats {
        let StreamStats {
            symbol_size,
            batch_columns,
            batch_bytes,
            row_buffer_bytes,
            systematic_batches,
            full_passes,
            bytes_read,
            read_calls,
            row_encodes,
            column_encodes,
            row_encoding_ms,
            column_encoding_ms,
            hashing_ms,
            ..
        } = stats.unwrap_or_default();
        println!(
            "{{\"len\":{},\"mode\":\"{}\",\"backend\":\"{}\",\"symbol_size\":{symbol_size},\
             \"batch_columns\":{batch_columns},\"batch_bytes\":{batch_bytes},\
             \"row_buffer_bytes\":{row_buffer_bytes},\"systematic_batches\":{systematic_batches},\
             \"full_passes\":{full_passes},\"bytes_read\":{bytes_read},\"read_calls\":{read_calls},\
             \"row_encodes\":{row_encodes},\"column_encodes\":{column_encodes},\
             \"row_encoding_ms\":{row_encoding_ms:.1},\"column_encoding_ms\":{column_encoding_ms:.1},\
             \"hashing_ms\":{hashing_ms:.1},\"io_ms\":{:.1},\"wall_ms\":{:.1}}}",
            metadata.unencoded_length(),
            match options.budget {
                Some(mib) => format!("stream {mib} MiB"),
                None if options.sequential => "sequential".to_string(),
                None if options.parallel => "parallel".to_string(),
                None => "memory".to_string(),
            },
            encoder.backend(),
            io.as_secs_f64() * 1e3,
            wall.as_secs_f64() * 1e3,
        );
    }
    Ok(())
}

fn main() {
    let options = parse().unwrap_or_else(|error| {
        eprintln!("{error}");
        exit(2);
    });
    let n_shards = NonZeroU16::new(options.shards).unwrap_or_else(|| {
        eprintln!("invalid shard count 0");
        exit(2);
    });
    let mut encoder = BlobEncoder::new(n_shards).unwrap_or_else(|error| {
        eprintln!("{error}");
        exit(2);
    });
    if let Some(backend) = options.backend {
        encoder = encoder.with_backend(backend).unwrap_or_else(|error| {
            eprintln!("{error}");
            exit(2);
        });
    }
    if let Some(threads) = options.threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap_or_else(|error| {
                eprintln!("{error}");
                exit(2);
            });
        encoder = encoder.with_thread_pool(Arc::new(pool));
    }
    let mut failed = false;
    for path in &options.files {
        if let Err(error) = run(&encoder, &options, path) {
            eprintln!("{error}");
            failed = true;
        }
    }
    exit(i32::from(failed));
}
