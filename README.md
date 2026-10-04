# Tsiro Independent Benchmark

An independent benchmark of [Tsiro](https://github.com/Daxlia/Tsiro), using 2,016 distinct images from the [RAISE RAW dataset](https://loki.disi.unitn.it/RAISE/index.php).

Tested revision: [36287d4](https://github.com/Daxlia/Tsiro/commit/36287d4d7d753f8a74beb8c5047fe6e5974fcfa6).

I converted the RAW files to 8-bit RGB, resized them to a maximum edge of 1,024 pixels and cropped them to dimensions divisible by 16. Each image was tested directly and again after JPEG compression.

The file sizes cover all 2,016 images in each group. For speed, I tested 12 images from each group, running one codec at a time. Each test had a warmup followed by three timed runs. The speeds below use the total pixel count divided by the sum of each image's median time.

## JPEG recompression

Each image was saved as a JPEG once, at quality 60, 75, 85, 90, 95 or 98 with 4:4:4 or 4:2:0 chroma. There were 168 images per combination. The codecs then compressed the decoded RGB pixels. This checks whether those pixels are recovered exactly; it does not reconstruct the original JPEG file.

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

## Direct compression

The same RGB8 images before JPEG compression. These came from the RAW sensor data. The embedded JPEG previews were not used.

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

MB is decimal and MP/s is million pixels per second. All codecs used lossless settings. PNG used Pillow with optimization enabled, and JPEG XL used one thread.

These tests ran on an AMD Ryzen 9 9950X3D with Windows 11. Loading inputs, hashing and saving outputs were not timed. Tsiro's own temporary JPEG reads and writes were included.

Every decoded image matched its input exactly: 4,032 Tsiro roundtrips and 32,256 codec outputs in total. I checked the results again in a separate run, recreating the inputs and decoding every saved output.

## Results

- [JPEG recompression CSV](tsiro%20benchmark/jpeg-recompression-only.csv)
- [Direct compression CSV](tsiro%20benchmark/tsiro-only-compression.csv)
- [Serial timing measurements](tsiro%20benchmark/timings.jsonl)

Each CSV has 16,128 rows, covering 2,016 images and eight codec settings. The CSV times were recorded while multiple tests ran in parallel. The speed tables above use the separate runs where only one codec was timed at a time.

## Run

You'll need Python 3.12 and Node.js. Open PowerShell in the `tsiro benchmark` directory and run the commands in [commands.txt](tsiro%20benchmark/commands.txt). The first run downloads the RAW inputs.

The package versions are listed in [requirements.txt](tsiro%20benchmark/requirements.txt). The scripts prepare the images, run the codecs, check the decoded pixels and export both CSVs.
