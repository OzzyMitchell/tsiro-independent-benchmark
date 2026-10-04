import argparse, collections, hashlib, html, json, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'results-2026-10-04-raise2000'
CODECS=['Tsiro','PNG','WebP','JXL3','JXL7','JXL9','QOI','JPEGLS']
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(name,obj): (RUN/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def esc(x): return html.escape(str(x))
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def aggregate(cases):
    pixels=sum(c['width']*c['height'] for c in cases)
    totals={k:sum(c['codecs'][k]['bytes'] for c in cases) for k in CODECS}
    tsiro=totals['Tsiro']
    best_jxl=sum(min(c['codecs'][k]['bytes'] for k in ['JXL3','JXL7','JXL9']) for c in cases)
    best_other=sum(min(c['codecs'][k]['bytes'] for k in CODECS if k!='Tsiro') for c in cases)
    jpegs=[c['jpeg']['bytes'] for c in cases if 'jpeg' in c]
    return dict(cases=len(cases),pixels=pixels,codec_bytes=totals,
        bits_per_pixel={k:8*v/pixels for k,v in totals.items()},
        tsiro_saving_vs={k:1-tsiro/v for k,v in totals.items() if k!='Tsiro'},
        best_jxl_bytes=best_jxl,tsiro_saving_vs_best_jxl=1-tsiro/best_jxl,
        best_other_bytes=best_other,tsiro_saving_vs_best_other=1-tsiro/best_other,
        tsiro_wins_vs={k:sum(c['codecs']['Tsiro']['bytes']<c['codecs'][k]['bytes'] for c in cases) for k in CODECS if k!='Tsiro'},
        tsiro_wins_vs_best_jxl=sum(c['codecs']['Tsiro']['bytes']<min(c['codecs'][k]['bytes'] for k in ['JXL3','JXL7','JXL9']) for c in cases),
        modes=dict(collections.Counter(c['codecs']['Tsiro']['mode'] for c in cases)),
        jpeg_bytes=sum(jpegs) if jpegs else None,tsiro_to_jpeg_ratio=tsiro/sum(jpegs) if jpegs else None)

def build(draft=False):
    results=sorted([read(p) for p in (RUN/'source-results').glob('*.json')],key=lambda r:r['rank']) if draft else [read(RUN/'source-results'/(row['File']+'.json')) for row in read(RUN/'qualified-plan.json')]
    failures=[r for r in results if r.get('error') or any(not z.get('exact') for c in r['cases'] for z in c['codecs'].values())]
    if failures:
        print(json.dumps({'failed_sources':[r['source_id'] for r in failures]}))
        if not draft: raise RuntimeError('Qualified sample contains failures')
        return
    cases=[c for r in results for c in r['cases']]
    groups={'original':aggregate([c for c in cases if c['profile']=='original']),
            'jpeg_all':aggregate([c for c in cases if c['profile']!='original'])}
    for profile in sorted({c['profile'] for c in cases if c['profile']!='original'}):
        groups[profile]=aggregate([c for c in cases if c['profile']==profile])
    if draft:
        print(json.dumps({'sources':len(results),'modes':{k:g['modes'] for k,g in groups.items() if k in ('original','jpeg_all')},'saving_vs_best_jxl':{k:g['tsiro_saving_vs_best_jxl'] for k,g in groups.items() if k in ('original','jpeg_all')}})); return
    verification=read(RUN/'verification.json')
    assert len(results)==2016 and len(cases)==4032 and verification['passed']
    assert set(verification['unique'].values())=={2016}, 'Duplicate source images'
    assert {r['source_id'] for r in results}=={r['File'] for r in read(RUN/'qualified-plan.json')}
    timing_records=[json.loads(line) for line in (RUN/'timings.jsonl').read_text().splitlines()]
    assert len(timing_records)==24 and len({r['id'] for r in timing_records})==24
    times={}
    for group in ['original','jpeg_all']:
        selected=[r for r in timing_records if (r['profile']=='original')==(group=='original')]
        assert len(selected)==12
        times[group]={}
        for codec in CODECS:
            for r in selected: assert r['codecs'][codec]['exact']
            pixels=sum(r['pixels'] for r in selected)
            enc=sum(r['codecs'][codec]['encode_median_s'] for r in selected)
            dec=sum(r['codecs'][codec]['decode_median_s'] for r in selected)
            times[group][codec]=dict(cases=12,encode_mpix_s=pixels/1e6/enc,decode_mpix_s=pixels/1e6/dec,encode_s=enc,decode_s=dec)
    summary=dict(sources=len(results),cases=len(cases),codec_streams=verification['streams'],
        camera_counts=dict(collections.Counter(r['camera'] for r in results)),
        raw_quality_counts=dict(collections.Counter(r['source']['raw_quality'] for r in results)),
        unique=verification['unique'],groups=groups,timings=times)
    write('summary.json',summary)
    p=read(RUN/'protocol.json'); orig=groups['original']; jpg=groups['jpeg_all']; qualification=read(RUN/'qualification.json')
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tsiro benchmark: 2,016 images</title>
<style>body{max-width:1080px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui,sans-serif;color:#1d2733;background:#fafafa}h1{font-size:30px;line-height:1.2}h2{font-size:22px;margin-top:36px}h3{font-size:18px}a{color:#1459a5}table{border-collapse:collapse;min-width:650px;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 12px;border-bottom:1px solid #d7dde2;text-align:right;white-space:nowrap}th{background:#e9edf1}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}code{font-size:13px;overflow-wrap:anywhere}.note{color:#4a5664}li{margin:7px 0}section{border-top:1px solid #cbd2da;padding-top:10px}strong{font-weight:650}</style>
<h1>Tsiro benchmark: 2,016 images</h1><p>RGB8 inputs prepared from 2,016 RAISE camera RAW images. Tsiro revision <code>'''+p['tsiro_revision']+'</code>.</p>']
    parts.append(f'<p><strong>{len(results):,} distinct RAW files, sensor mosaics and prepared RGB images.</strong> Each source has one direct-compression case and one controlled JPEG case. All {len(cases):,} Tsiro roundtrips and all {verification["streams"]:,} codec streams passed exact comparisons, including fresh-process checks from disk.</p>')
    parts.append('<p><a href="jpeg-recompression-only.csv">JPEG recompression CSV</a> · <a href="tsiro-only-compression.csv">Tsiro-only compression CSV</a> · <a href="verification.json">Verification results</a> · <a href="summary.json">Summary data</a></p>')
    parts.append('<p>Each CSV contains 2,016 source images and 16,128 codec rows. The two files separate input JPEG history; each includes the same eight codecs for comparison.</p>')
    parts.append('<h2>Input preparation</h2><p>“JPEG recompression” here means compressing the <strong>decoded RGB pixels of a known JPEG</strong>. Tsiro receives those pixels only. It is not given the JPEG coefficients, quality or quantization tables. Exactness is measured against that RGB raster, not the original JPEG file bytes.</p><p>For the direct track, the source path is Nikon sensor mosaic → local RAW development → resized/cropped RGB8 → codec. The input pixels have no JPEG generation. Embedded JPEG previews in the NEF container are identified and never used.</p><p>Full-resolution RAWs were developed locally, reduced with LANCZOS to a maximum 1,024-pixel edge, then center-cropped by at most 15 pixels per axis for 16-pixel alignment. Results apply to these prepared RGB8 inputs. RAW-file reconstruction, full-resolution performance, other dimensions, alpha and higher bit depths were not tested in this run.</p>')
    parts.append('<h2>Compression sizes</h2><p>Totals include each codec’s complete output stream. Lower size and bits per pixel are better. Savings compare sums of bytes, rather than averaging per-image percentages.</p>')
    parts.append(table(['Codec','Direct MiB','Direct bits/pixel','JPEG-derived MiB','JPEG-derived bits/pixel'],[[c,f'{orig["codec_bytes"][c]/2**20:,.2f}',f'{orig["bits_per_pixel"][c]:.3f}',f'{jpg["codec_bytes"][c]/2**20:,.2f}',f'{jpg["bits_per_pixel"][c]:.3f}'] for c in CODECS]))
    parts.append(table(['Input track','Tsiro saving vs PNG','Vs WebP','Vs JXL9','Vs best JXL per image','Wins vs best JXL'],[[label,f'{g["tsiro_saving_vs"]["PNG"]:+.2%}',f'{g["tsiro_saving_vs"]["WebP"]:+.2%}',f'{g["tsiro_saving_vs"]["JXL9"]:+.2%}',f'{g["tsiro_saving_vs_best_jxl"]:+.2%}',f'{g["tsiro_wins_vs_best_jxl"]:,} / {g["cases"]:,}'] for label,g in [('Direct',orig),('JPEG-derived',jpg)]]))
    parts.append('<p class="note">Positive savings mean Tsiro is smaller. “Best JXL” chooses the smallest of efforts 3, 7 and 9 separately for each image; this combines settings and has no single encoding speed.</p>')
    parts.append('<h2>JPEG profiles</h2><p>Quality 60, 75, 85, 90, 95 and 98, each with 4:4:4 and 4:2:0 chroma. The hash-ranked sources were assigned cyclically: 168 different sources per profile. Each source contributes one JPEG. Profiles contain different images, so their results cannot be compared as a paired quality sweep.</p>')
    parts.append(table(['Profile','Sources','Tsiro saving vs best JXL','Tsiro wins','JPEG inversion mode','Tsiro / JPEG size'],[[k.replace('jpeg_',''),g['cases'],f'{g["tsiro_saving_vs_best_jxl"]:+.2%}',g['tsiro_wins_vs_best_jxl'],g['modes'].get('TSOJ',0),f'{g["tsiro_to_jpeg_ratio"]:.2f}×'] for k,g in groups.items() if k.startswith('jpeg_q')]))
    parts.append('<p>JPEG sizes are shown for reference. The lossless codecs preserve the decoded RGB pixels; they do not reconstruct the JPEG bitstream.</p>')
    parts.append('<h2>Verification</h2><ol><li>Selected 2,016 source IDs from the 8,156-row public RAISE catalog by a fixed SHA-256 rank, before observing compression results. This is a custom sample, not the publisher’s RAISE-2k subset.</li><li>Downloaded the original NEFs from the publisher. Inspected the TIFF image directories, required a Bayer CFA sensor image with non-JPEG compression, and checked the actual camera model against the catalog.</li><li>Hashed each original NEF, visible sensor mosaic, prepared RGB array, generated JPEG and compressed stream. Distinctness is checked at the RAW-file, sensor and prepared-pixel levels. Distinct captures can still depict similar scenes.</li><li>Compared dtype, shape and every channel value after every codec roundtrip. All encoded streams and all source files are retained.</li><li>A separate audit process pool reopened every original NEF, redeveloped its sensor data, reproduced the prepared pixels, regenerated the JPEG byte-for-byte from its recorded settings, and decoded all retained streams. It checked actual JPEG quantization tables and component sampling too. The audit uses separate processes and freshly decoded inputs, with the same codec implementations.</li></ol>')
    parts.append(table(['Camera','Distinct sources'],summary['camera_counts'].items()))
    parts.append(f'<p>The initial catalog sample contained {len(qualification["initial_duplicates"])} duplicate captures. They were excluded from the CSVs and replaced by the next unused sources in hash order, retaining the assigned JPEG profiles. {qualification["all_tested_sources"]:,} source IDs were processed to obtain the final 2,016 distinct images. See <a href="qualification.json">duplicate decisions and replacements</a>. Excluded results remain in the source records.</p>')
    parts.append(table(['Publisher RAW quality label','Sources'],summary['raw_quality_counts'].items()))
    parts.append('<p>Some sources use Nikon’s compressed 12-bit RAW format, which is not JPEG and can be lossy. Lossless verification starts at the prepared RGB8 input.</p>')
    parts.append('<h2>Warm serial timing</h2><p>First 12 hash-ranked sources, both tracks: 24 cases, one warmup and three timed repetitions per codec. Codec order rotates. Throughput is total pixels divided by the sum of each case’s median time. Source loading, hashing and saving streams are excluded; Tsiro’s internal temporary JPEG I/O is included. Timing covers 12 sources per track. The CSV bulk times include parallel contention and are unsuitable for speed comparisons.</p>')
    if (RUN/'evidence/primary-exit.txt').exists():
        parts.append('<p class="note">Replacing progress.json failed with Windows access denied after 308 reported completions. Queued workers completed all source records, which passed the separate disk audit. Status writes now retry this error. See <a href="evidence/primary-exit.txt">the exception</a>. Serial timing started after the bulk workers exited.</p>')
    parts.append(table(['Codec','Direct encode MP/s','Direct decode MP/s','JPEG encode MP/s','JPEG decode MP/s'],[[c,f'{times["original"][c]["encode_mpix_s"]:.2f}',f'{times["original"][c]["decode_mpix_s"]:.2f}',f'{times["jpeg_all"][c]["encode_mpix_s"]:.2f}',f'{times["jpeg_all"][c]["decode_mpix_s"]:.2f}'] for c in CODECS]))
    parts.append('<h2>Settings</h2>'+table(['Codec','Settings'],p['settings'].items()))
    parts.append('<p>JPEG generation: Pillow 12.3.0 / libjpeg-turbo 3.1.4.1, nonprogressive, optimize=False. Each CSV row records the exact JPEG quality, chroma layout, full quantization tables, component table assignments, encoder/decoder, hash and path. All JPEG inputs have exactly one JPEG generation after local RAW development.</p><p>RAW development: rawpy 0.27.1 / LibRaw 0.22.1, AHD, camera white balance, sRGB, 8-bit output, no auto-bright, gamma (2.222, 4.5), no rotation. See <a href="protocol.json">protocol</a>, <a href="environment.json">environment</a>, <a href="hardware.json">hardware</a>, <a href="plan.json">exact selection and assignments</a>, <a href="../raise_verify.py">runner and audit code</a>, and <a href="../raise_report.py">analysis code</a>.</p>')
    parts.append('<h2>Limitations and known failures</h2><p>This run covers one dataset and three Nikon camera models. Image dimensions were aligned to Tsiro’s JPEG inversion path. Results may differ on other sources, resolutions and dimensions.</p><p>The earlier tests of this same Tsiro revision found separate failures: odd-sized JPEG-file mode changes dimensions, the CLI converts alpha/high-bit-depth PNG inputs to RGB8 before its exactness check, and full-range uint16 RAW counterexamples fail. These failures were not exercised by the aligned RGB8 inputs used here. The earlier size tables include sources with unknown JPEG history and are excluded from these CSVs.</p>')
    parts.append('<h2>Sources</h2><p>Dang-Nguyen, Pasquini, Conotter and Boato, <em>RAISE — A Raw Images Dataset for Digital Image Forensics</em>, ACM MMSys 2015. <a href="https://loki.disi.unitn.it/RAISE/index.php">Publisher and dataset</a> · <a href="https://loki.disi.unitn.it/RAISE/guide.html">RAW format and metadata guide</a>. The public catalog is pinned to <a href="https://raw.githubusercontent.com/eeerpjw/image-enhancement/acfa7c2e55fa348bd76d2739f7ac5e4df1cf09ce/Datasets/RAISE/RAISE_all.csv">commit acfa7c2</a>; images were downloaded from the publisher’s NEF paths, not the catalog mirror.</p></html>')
    report='\n'.join(parts).replace('href="plan.json">exact selection and assignments','href="qualified-plan.json">exact qualified selection and assignments')
    (RUN/'report.html').write_text(report,encoding='utf-8')
    (RUN/'START-HERE.txt').write_text('TSIRO BENCHMARK: 2,016 IMAGES\n\nRead report.html.\n\nJPEG recompression CSV: jpeg-recompression-only.csv\nDirect input CSV: tsiro-only-compression.csv\nEach CSV: 2,016 distinct sources, 16,128 codec rows.\n4,032 Tsiro roundtrips; 32,256 streams verified again from disk.\nScope: prepared RGB8, maximum 1,024-pixel edge, cropped to 16-pixel alignment.\nJPEG track preserves decoded RGB pixels, not original JPEG file bytes.\n\nEvidence: verification.json, protocol.json, environment.json, plan.json, source-results/, audit-records/, raw/, prepared/, derived/, streams/.\n',encoding='utf-8')
    print(json.dumps({'sources':len(results),'groups':{k:g for k,g in groups.items() if k in ('original','jpeg_all')},'timings':times},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--draft',action='store_true'); args=p.parse_args(); build(args.draft)
