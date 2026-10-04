# Tsiro Independent Benchmark

An independent benchmark of [Tsiro](https://github.com/Daxlia/Tsiro), using 2,016 distinct images from the [RAISE RAW dataset](https://loki.disi.unitn.it/RAISE/index.php).

Tested revision: [36287d4](https://github.com/Daxlia/Tsiro/commit/36287d4d7d753f8a74beb8c5047fe6e5974fcfa6).

## Benchmark

The RAW images were developed to RGB8, resized to a maximum 1,024-pixel edge, then center-cropped to dimensions divisible by 16. The same 2,016 images were tested in both tracks. Duplicate RAW files, sensor data and prepared pixels were removed and replaced before the final totals.

Sizes cover all 2,016 images per track. Speeds cover a separate 12-image subset per track, measured serially after warmup, with three repetitions per codec. Throughput is total pixels divided by the sum of each image's median time.

### JPEG recompression

Each image was JPEG-encoded once at quality 60, 75, 85, 90, 95 or 98, with 4:4:4 or 4:2:0 chroma: 168 images per combination. Tsiro received the decoded RGB pixels. Lossless here means recovering those pixels exactly, not reconstructing the original JPEG file.

| Codec | Total size (MB) | Encode (MP/s) | Decode (MP/s) |
|---|---:|---:|---:|
| Tsiro | 797.33 | 0.40 | 19.58 |
| PNG level 9 | 1,647.85 | 4.68 | 78.47 |
| WebP method 6 | 1,186.32 | 0.25 | 58.64 |
| JPEG XL effort 3 | 1,173.53 | 11.37 | 14.89 |
| JPEG XL effort 7 | 1,059.78 | 1.08 | 9.09 |
| JPEG XL effort 9 | 1,047.63 | 0.13 | 8.32 |
| QOI | 2,047.88 | 156.31 | 218.57 |
| JPEG-LS | 1,679.53 | 31.29 | 38.87 |

### Direct compression

The same prepared RGB8 images, without JPEG encoding. Embedded JPEG previews in the RAW files were not used.

| Codec | Total size (MB) | Encode (MP/s) | Decode (MP/s) |
|---|---:|---:|---:|
| Tsiro | 1,816.18 | 0.39 | 13.24 |
| PNG level 9 | 2,068.57 | 3.75 | 72.31 |
| WebP method 6 | 1,703.77 | 0.25 | 56.28 |
| JPEG XL effort 3 | 1,737.40 | 11.05 | 14.78 |
| JPEG XL effort 7 | 1,671.11 | 0.94 | 8.89 |
| JPEG XL effort 9 | 1,654.00 | 0.12 | 8.19 |
| QOI | 2,617.97 | 152.43 | 215.59 |
| JPEG-LS | 1,873.74 | 27.34 | 34.39 |

MB is decimal and MP/s is million pixels per second. All codec settings are lossless. PNG used Pillow with optimization enabled; JPEG XL used one thread.

Measured on an AMD Ryzen 9 9950X3D running Windows 11. Input loading, hashing and saving output files were excluded from timing. Tsiro's internal temporary JPEG file I/O was included.

All 4,032 Tsiro roundtrips and all 32,256 codec streams reproduced the input pixels exactly. I checked the results again in a separate run, recreating the inputs and decoding every saved output.

## Results

- [JPEG recompression CSV](tsiro%20benchmark/jpeg-recompression-only.csv)
- [Direct compression CSV](tsiro%20benchmark/tsiro-only-compression.csv)
- [Serial timing measurements](tsiro%20benchmark/timings.jsonl)

Each CSV contains 16,128 rows: 2,016 images across eight codec settings. The CSV timing columns came from parallel bulk runs; the speed tables above use the separate serial measurements.

## Run

Requires Python 3.12 and Node.js. Run the PowerShell commands in [commands.txt](tsiro%20benchmark/commands.txt) from the `tsiro benchmark` directory. The first run downloads the RAW inputs.

Package versions are pinned in [requirements.txt](tsiro%20benchmark/requirements.txt). The scripts prepare the inputs, run the codecs, verify decoded pixels and export both CSVs.
